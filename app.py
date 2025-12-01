import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gdown
import os

# ---------------------------------------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Steel Surface Defect Detection System",
    page_icon="⚙️",
    layout="centered"
)

st.title("Steel Surface Defect Detection System")
st.write("Upload an image of a steel surface to detect potential defects.")

# ---------------------------------------------------------------------
# MODEL DOWNLOAD + LOADING
# ---------------------------------------------------------------------

MODEL_PATH = "model_best.pth"

# 🔹 Replace this with your Google Drive file ID if needed
MODEL_URL = "https://drive.google.com/uc?id=1Ov9tpdU7q8PP6fTKrzbwiaRov0tdRtAh"

@st.cache_resource
def load_model():
    # Download model if not present
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Recreate same model architecture as during training
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 6)  # <-- change 6 if you trained on different number of classes

    # Load state_dict correctly (weights only)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        # If the saved dict is nested
        state_dict = state_dict["state_dict"]

    # Remove "module." prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()

    return model, device

# Load model once
model, device = load_model()

# ---------------------------------------------------------------------
# IMAGE UPLOAD SECTION
# ---------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a steel surface image", type=["jpg", "jpeg", "png"])

# Define transformations (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Define your class names (modify these if your training used different labels)
class_names = [
    "Crazing",
    "Inclusion",
    "Patches",
    "Pitted Surface",
    "Rolled-in Scale",
    "Scratches"
]

# ---------------------------------------------------------------------
# PREDICTION SECTION
# ---------------------------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    st.subheader("Prediction Result")
    st.success(f"**Detected Defect Type:** {class_names[pred_class]}")

    st.subheader("Class Probabilities")
    for i, (cls, p) in enumerate(zip(class_names, probs[0])):
        st.write(f"{cls}: {p.item() * 100:.2f}%")
else:
    st.info("Please upload an image to start defect detection.")