"""Rule-based clinical decision support for the brain tumor classifier.

This is NOT a diagnostic tool. It turns the model's output into structured
talking points (confidence checks, typical imaging features, suggested
follow-up) for a qualified clinician to review.
"""

CLASS_NAMES = {0: "glioma", 1: "meningioma", 2: "no tumor", 3: "pituitary"}

# Thresholds (tune on your validation set)
LOW_CONFIDENCE = 0.70   # top probability below this -> flag
CLOSE_CALL_MARGIN = 0.20  # top-1 minus top-2 below this -> flag

KNOWLEDGE_BASE = {
    "glioma": {
        "title": "Glioma",
        "summary": (
            "Tumor arising from glial cells within the brain parenchyma "
            "(intra-axial). Ranges from low-grade to highly aggressive (glioblastoma)."
        ),
        "imaging_features": [
            "Intra-axial mass, often with ill-defined / infiltrative margins",
            "Heterogeneous signal; high-grade lesions may show irregular or ring enhancement and necrosis",
            "Surrounding T2/FLAIR hyperintensity (edema or tumor infiltration)",
            "Possible mass effect or midline shift",
        ],
        "next_steps": [
            "Multi-sequence contrast-enhanced MRI (T1, T1+gad, T2, FLAIR, DWI)",
            "Advanced imaging if available (perfusion, MR spectroscopy)",
            "Urgent neurosurgery / neuro-oncology referral",
            "Tissue diagnosis with molecular profiling (e.g. IDH, MGMT, 1p/19q) for grading",
        ],
        "differentials": ["Metastasis", "Abscess", "Demyelinating lesion", "Lymphoma"],
    },
    "meningioma": {
        "title": "Meningioma",
        "summary": (
            "Usually slow-growing tumor arising from the meninges (extra-axial). "
            "Most are WHO grade 1, but atypical and anaplastic forms exist."
        ),
        "imaging_features": [
            "Extra-axial, dural-based mass, often well circumscribed",
            "Homogeneous, avid contrast enhancement",
            "'Dural tail' sign; possible calcification or adjacent hyperostosis",
            "Brain compression or edema in larger lesions",
        ],
        "next_steps": [
            "Contrast-enhanced MRI to confirm dural attachment and extent",
            "Small, asymptomatic lesions: consider observation with interval imaging",
            "Symptomatic or growing lesions: neurosurgical / radiation oncology review (surgery or radiosurgery)",
            "Correlate with neurological symptoms (headache, seizures, focal deficits)",
        ],
        "differentials": ["Dural metastasis", "Schwannoma", "Hemangiopericytoma / solitary fibrous tumor", "Lymphoma"],
    },
    "pituitary": {
        "title": "Pituitary tumor",
        "summary": (
            "Mass in the sellar / suprasellar region, most often a pituitary adenoma. "
            "May be hormonally active (e.g. prolactinoma) or non-functioning."
        ),
        "imaging_features": [
            "Sellar mass, possibly extending to the suprasellar cistern",
            "Microadenoma (<10 mm) vs macroadenoma (>=10 mm)",
            "Possible optic chiasm compression or cavernous sinus extension",
            "Typically lower enhancement than normal pituitary gland",
        ],
        "next_steps": [
            "Dedicated pituitary-protocol MRI (thin-slice coronal, pre/post contrast)",
            "Endocrine work-up (e.g. prolactin, IGF-1, cortisol/ACTH, TSH/fT4, gonadal axis)",
            "Formal visual field and acuity testing if suprasellar extension",
            "Endocrinology and neurosurgery referral",
        ],
        "differentials": ["Rathke cleft cyst", "Craniopharyngioma", "Meningioma (tuberculum sellae)", "Hypophysitis"],
    },
    "no tumor": {
        "title": "No tumor pattern detected",
        "summary": (
            "The model found no pattern matching the three tumor classes in this image. "
            "This does not exclude other intracranial pathology."
        ),
        "imaging_features": [
            "No tumor-like pattern recognised by the model on this slice",
            "A single 2D slice may miss small or off-slice lesions",
        ],
        "next_steps": [
            "Review the full multi-slice, multi-sequence study",
            "Correlate with clinical presentation; if symptoms persist, consider contrast-enhanced MRI",
            "Radiologist review for non-tumor findings (vascular, demyelinating, infectious, etc.)",
        ],
        "differentials": [],
    },
}

GENERAL_LIMITATIONS = [
    "Model classifies only four categories and cannot detect other pathologies.",
    "Prediction is based on a single 2D image, not the full 3D study or clinical history.",
    "The model has not been clinically validated; performance on other scanners, protocols or populations is unknown.",
    "Grad-CAM highlights where the model looked, not necessarily the lesion itself.",
]


def analyze(probs):
    """probs: sequence of 4 floats aligned with CLASS_NAMES order."""
    probs = [float(p) for p in probs]
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top, second = order[0], order[1]
    top_p, second_p = probs[top], probs[second]
    margin = top_p - second_p
    label = CLASS_NAMES[top]

    flags = []
    if top_p < LOW_CONFIDENCE:
        flags.append(
            f"Low confidence ({top_p:.0%}). Treat this prediction with caution and prioritise manual review."
        )
    if margin < CLOSE_CALL_MARGIN:
        flags.append(
            f"Close call between {label} ({top_p:.0%}) and {CLASS_NAMES[second]} ({second_p:.0%}). "
            "Consider both possibilities."
        )
    if label == "no tumor" and second_p >= 0.15:
        flags.append(
            f"'No tumor' predicted, but {CLASS_NAMES[second]} still scores {second_p:.0%}. "
            "Do not rule out pathology on this result alone."
        )
    if label != "no tumor" and probs[2] >= 0.25:
        flags.append(f"Model assigns {probs[2]:.0%} to 'no tumor', indicating notable uncertainty.")

    if top_p >= 0.90 and margin >= 0.50:
        confidence_level = "High"
    elif top_p >= LOW_CONFIDENCE and margin >= CLOSE_CALL_MARGIN:
        confidence_level = "Moderate"
    else:
        confidence_level = "Low"

    kb = KNOWLEDGE_BASE[label]
    return {
        "predicted_class": label,
        "confidence": top_p,
        "confidence_level": confidence_level,
        "runner_up": CLASS_NAMES[second],
        "runner_up_prob": second_p,
        "flags": flags,
        "needs_second_review": confidence_level == "Low" or bool(flags),
        "knowledge": kb,
        "differentials": kb["differentials"],
        "limitations": GENERAL_LIMITATIONS,
    }
