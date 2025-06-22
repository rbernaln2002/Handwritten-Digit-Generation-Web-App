import streamlit as st
import torch
import torch.nn as nn
import numpy as np

# --- CONFIGURACIÓN ---
z_dim = 100
num_classes = 10
embedding_dim = 50
image_size = 28 * 28
device = torch.device("cpu")  # Streamlit Cloud no tiene GPU

# --- GENERADOR ---
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, embedding_dim, img_dim):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(z_dim + embedding_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        return self.model(x)

# --- CARGAR MODELO ---
@st.cache_resource(show_spinner=False)
def load_generator():
    model = Generator(z_dim, num_classes, embedding_dim, image_size).to(device)
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()
    return model

generator = load_generator()

# --- STREAMLIT UI ---
st.title("🧠 Handwritten Digit Generator")
st.write("Select a digit (0–9) and generate 5 different handwritten-style images using a trained GAN.")

digit = st.selectbox("Select a digit to generate:", list(range(10)))

if st.button("Generate 5 Images"):
    noise = torch.randn(5, z_dim).to(device)
    labels = torch.full((5,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        generated = generator(noise, labels).view(-1, 28, 28).cpu()

    st.subheader(f"Generated Images for Digit: {digit}")
    cols = st.columns(5)
    for i in range(5):
        img = (generated[i].numpy() + 1) / 2  # Convert from [-1, 1] to [0, 1]
        cols[i].image(img, width=100, clamp=True, caption=f"Image {i+1}")
