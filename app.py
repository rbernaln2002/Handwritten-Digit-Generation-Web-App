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

# --- GENERADOR CONV (MISMO QUE EN ENTRENAMIENTO) ---
class Generator(nn.Module):
    def __init__(self, z_dim, num_classes, embedding_dim=50):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, embedding_dim)
        self.init_size = 7
        self.l1 = nn.Sequential(
            nn.Linear(z_dim + embedding_dim, 128 * self.init_size ** 2)
        )
        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),  # 7 -> 14
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),  # 14 -> 28
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_emb(labels)
        x = torch.cat([noise, c], dim=1)
        out = self.l1(x)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img  # retorna (batch, 1, 28, 28)

# --- CARGAR MODELO ---
@st.cache_resource(show_spinner=False)
def load_generator():
    model = Generator(z_dim, num_classes, embedding_dim).to(device)
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()
    return model

generator = load_generator()

# --- STREAMLIT UI ---
st.title("🧠 Handwritten Digit Generator")

digit = st.selectbox("Select a digit to generate (0–9):", list(range(10)))
if st.button("Generate 5 Images"):
    noise = torch.randn(5, z_dim).to(device)
    labels = torch.full((5,), digit, dtype=torch.long).to(device)

    with torch.no_grad():
        generated = generator(noise, labels).cpu()
        generated = (generated + 1) / 2  # Escalar a [0,1]

    st.subheader(f"Generated Images for Digit: {digit}")
    cols = st.columns(5)
    for i in range(5):
        img = generated[i].squeeze().numpy()
        cols[i].image(img, width=100, clamp=True, caption=f"Image {i+1}")
