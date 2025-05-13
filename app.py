import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from src.data_loader import get_loaders
from src.cnn_model import SimpleCNN
import os

# Page config
st.set_page_config(page_title="Cat vs Dogs Classifier", layout="wide")
st.title("🐱 vs 🐶 Image Classifier")
st.write("Upload gambar kucing atau anjing, lalu klik **Predict** untuk melihat hasil klasifikasi.")

# Load data and model
@st.cache_data
def load_data(batch_size=32):
    return get_loaders(batch_size)

@st.cache_resource
def load_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=2).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model, device

with st.spinner("Memuat data dan model..."):
    batch_size = st.sidebar.slider("Batch size", 8, 64, 32)
    train_loader, test_loader, train_size, test_size = load_data(batch_size)
    model_path = os.path.join(os.getcwd(), "models", "model_cnn.pth")
    model, device = load_model(model_path)

# Sidebar info
st.sidebar.header("Dataset Info")
st.sidebar.write(f"Train samples: **{train_size}**")
st.sidebar.write(f"Test samples: **{test_size}**")

# Main panel: upload and predict
col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader("Pilih gambar (.png, .jpg)", type=["png", "jpg", "jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Gambar yang diunggah", use_container_width =True)

with col2:
    if uploaded:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        tensor = transform(img).unsqueeze(0).to(device)
        if st.button("Predict", key="predict_button"):
            with torch.no_grad():
                logits = model(tensor)
                pred = logits.argmax(dim=1).item()
            labels = ['Cat', 'Dog']
            st.success(f"Hasil prediksi: **{labels[pred]}**")
    else:
        st.info("Belum ada gambar diunggah.")

# Footer
st.markdown("---")
st.write("Model dilatih pada dataset Cat vs Dogs. Dibuat dengan PyTorch dan Streamlit.")