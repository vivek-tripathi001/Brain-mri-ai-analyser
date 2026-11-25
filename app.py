# app.py
import streamlit as st
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import io
from datetime import date

from doctor_assist import analyze, CLASS_NAMES
from report import generate_report_pdf





# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
@st.cache_resource
def load_model(checkpoint_path):
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Define preprocessing transform
def preprocess_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# Load and preprocess image
def load_and_preprocess(image):
    # Convert Streamlit uploaded image (PIL) to RGB
    image = image.convert('RGB')
    
    # Preprocess for model input
    transform = preprocess_transform()
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Preprocess for visualization (normalize to 0-1)
    rgb_img = np.array(image.resize((224, 224))) / 255.0
    
    return input_tensor, rgb_img

# Generate Grad-CAM visualization
def generate_gradcam(model, target_layers, input_tensor, rgb_img, class_idx=None):
    cam = GradCAM(model=model, target_layers=target_layers)
    
    if class_idx is None:
        with torch.no_grad():
            output = model(input_tensor)
            class_idx = torch.argmax(output).item()
    
    grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization, class_idx

# Generate multi-target Grad-CAM
def multi_target_gradcam(model, target_layers, input_tensor, rgb_img):
    cam = GradCAM(model=model, target_layers=target_layers)
    
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    fig, axs = plt.subplots(1, 5, figsize=(20,20))
    axs[0].imshow(rgb_img)
    axs[0].set_title("Original Image", fontsize=30)
    axs[0].axis('off')
    
    class_names = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}
    for i, class_name in class_names.items():
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(i)], aug_smooth=True, eigen_smooth=True)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        axs[i+1].imshow(visualization)
        axs[i+1].set_title(f"{class_name}\nprob: {probabilities[i]:.2f}", fontsize=30)
        axs[i+1].axis('off')
    
    plt.tight_layout()
    return fig

# Compare different CAM methods
def compare_methods(model, target_layers, input_tensor, rgb_img):
    methods = {
        "GradCAM": GradCAM,
        "GradCAM++": GradCAMPlusPlus,
        "XGradCAM": XGradCAM,
        "EigenCAM": EigenCAM
    }
    
    with torch.no_grad():
        output = model(input_tensor)
        class_idx = torch.argmax(output).item()
    
    
    fig, axs = plt.subplots(1, len(methods)+1)
    axs[0].imshow(rgb_img)
    axs[0].set_title("\n\nOriginal Image")
    axs[0].axis('off')
    
    class_names = {0: 'glioma', 1: 'meningioma', 2: 'no tumor', 3: 'pituitary'}
    for i, (name, method_class) in enumerate(methods.items()):
        cam = method_class(model=model, target_layers=target_layers)
        grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(class_idx)], aug_smooth=True)[0, :]
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        axs[i+1].imshow(visualization)
        axs[i+1].set_title(f"\n\n{name}")
        axs[i+1].axis('off')
    
    plt.suptitle(f"Comparison for {class_names[class_idx]} tumor")
    plt.tight_layout()
    return fig

# Compute prediction + Grad-CAM once per uploaded file (cached across reruns)
@st.cache_data(show_spinner="Analysing image...")
def run_analysis(_model, _target_layers, file_bytes):
    image = Image.open(io.BytesIO(file_bytes))
    input_tensor, rgb_img = load_and_preprocess(image)
    with torch.no_grad():
        probs = torch.nn.functional.softmax(_model(input_tensor), dim=1)[0].cpu().tolist()
    cam_image, _ = generate_gradcam(_model, _target_layers, input_tensor, rgb_img)
    return probs, cam_image


# ---------------- Streamlit app ----------------
st.set_page_config(page_title="Brain Tumor Classification", layout="wide")
st.title("Brain Tumor Classification with Grad-CAM Visualization")
st.warning("Research prototype. Not a medical device. Outputs must be reviewed by a qualified clinician.")

# Model path (update this to your model path)
model_path = "brain_tumor_resnet18.pth"  # Update with your model path
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}. Please provide the correct path.")
    st.stop()

# Load model
model = load_model(model_path)
target_layers = [model.layer4[-1].conv2]

# Sidebar: patient details used in the report
with st.sidebar:
    st.header("Patient details")
    patient = {
        "name": st.text_input("Name"),
        "id": st.text_input("Patient ID"),
        "age": st.text_input("Age"),
        "sex": st.selectbox("Sex", ["", "Female", "Male", "Other"]),
        "scan_date": st.date_input("Scan date", value=date.today()).isoformat(),
        "indication": st.text_area("Clinical indication", height=80),
    }
    st.header("Clinician")
    clinician_name = st.text_input("Reviewing clinician")

# File uploader
uploaded_file = st.file_uploader("Upload a brain MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    probs, cam_image = run_analysis(model, target_layers, file_bytes)
    assist = analyze(probs)

    tab_result, tab_assist, tab_viz, tab_report = st.tabs(
        ["Result", "Doctor Assist", "Detailed Visualizations", "Report"]
    )

    with tab_result:
        c1, c2 = st.columns(2)
        c1.image(image, caption="Uploaded Image", use_column_width=True)
        c2.image(cam_image, caption=f"Grad-CAM for {assist['predicted_class']}", use_column_width=True)
        st.metric("Predicted class", assist["predicted_class"].title(),
                  f"{assist['confidence']:.1%} confidence ({assist['confidence_level']})", delta_color="off")
        st.bar_chart({CLASS_NAMES[i].title(): probs[i] for i in range(4)})

    with tab_assist:
        kb = assist["knowledge"]
        if assist["needs_second_review"]:
            st.error("Flagged for careful / second review")
            for f in assist["flags"]:
                st.write(f"- {f}")
        else:
            st.success("No uncertainty flags raised")
        st.subheader(kb["title"])
        st.write(kb["summary"])
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Typical imaging features**")
            for f in kb["imaging_features"]:
                st.write(f"- {f}")
        with col_b:
            st.markdown("**Suggested next steps**")
            for s in kb["next_steps"]:
                st.write(f"- {s}")
        if assist["differentials"]:
            st.markdown("**Differentials to consider:** " + ", ".join(assist["differentials"]))
        with st.expander("Model limitations"):
            for l in assist["limitations"]:
                st.write(f"- {l}")

    with tab_viz:
        st.caption("These are slower to compute, so they run on demand.")
        if st.button("Generate detailed visualizations"):
            input_tensor, rgb_img = load_and_preprocess(image)
            st.subheader("Multi-Target Grad-CAM for All Classes")
            st.pyplot(multi_target_gradcam(model, target_layers, input_tensor, rgb_img))
            st.subheader("Comparison of Different CAM Methods")
            st.pyplot(compare_methods(model, target_layers, input_tensor, rgb_img))

    with tab_report:
        notes = st.text_area("Clinician notes (included in report)", height=150)
        pdf_bytes = generate_report_pdf(patient, image, cam_image, probs, assist, notes, clinician_name)
        st.download_button(
            "Download PDF report", data=pdf_bytes, mime="application/pdf",
            file_name=f"brain_mri_report_{patient['id'] or 'patient'}.pdf",
        )
