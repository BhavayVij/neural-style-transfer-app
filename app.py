import streamlit as st
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import io

st.set_page_config(page_title="Neural Style Transfer", layout="wide")

# -------------------------
# Load model (cached)
# -------------------------
@st.cache_resource
def load_model():
    return hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )

model = load_model()

# -------------------------
# UI
# -------------------------
st.title("🎨 Neural Style Transfer")
st.caption("Upload a photo + painting → get stylized result in seconds ⚡")

content_file = st.file_uploader("Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Style Image", type=["jpg", "png", "jpeg"])

# -------------------------
# Image processing
# -------------------------
def load_image(img, max_dim=512):
    img = img.convert("RGB")
    w, h = img.size
    scale = max_dim / max(w, h)
    img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img = np.array(img).astype(np.float32) / 255.0
    return tf.constant(img[np.newaxis, ...])

def preview(img, max_dim=400):
    w, h = img.size
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

# -------------------------
# Main
# -------------------------
if content_file and style_file:

    content_img = Image.open(content_file)
    style_img = Image.open(style_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(preview(content_img), caption="Content")
    with col2:
        st.image(preview(style_img), caption="Style")

    if st.button("✨ Generate"):

        with st.spinner("Applying style..."):

            content = load_image(content_img)
            style = load_image(style_img)

            output = model(content, style)[0]

            image = np.squeeze(output.numpy())
            image = (image * 255).astype(np.uint8)
            image = Image.fromarray(image)

        st.subheader("🎨 Result")
        st.image(image)

        buf = io.BytesIO()
        image.save(buf, format="PNG")

        st.download_button(
            "Download",
            buf.getvalue(),
            "stylized.png",
            "image/png"
        )
        