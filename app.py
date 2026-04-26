import streamlit as st
from PIL import Image
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import io

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Neural Style Transfer", layout="wide")

# -------------------------
# Load Model ONCE (cached)
# -------------------------
@st.cache_resource
def load_model():
    return hub.load(
        "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
    )

model = load_model()

# -------------------------
# Title
# -------------------------
st.title("🎨 Neural Style Transfer App")
st.write("Transform your photos into artistic masterpieces using AI style transfer.")

st.caption("💡 Tip: Use strong paintings (Van Gogh, Picasso) for best results")

# -------------------------
# Upload
# -------------------------
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])


# -------------------------
# Resize (NO DISTORTION)
# -------------------------
def load_image(img, max_dim=512):
    img = img.convert("RGB")

    w, h = img.size
    scale = max_dim / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    img = img.resize((new_w, new_h), Image.LANCZOS)

    img = np.array(img).astype(np.float32) / 255.0
    img = img[np.newaxis, ...]

    return tf.constant(img)


# -------------------------
# Preview Resize (UI only)
# -------------------------
def preview(img, max_dim=400):
    w, h = img.size
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# -------------------------
# Main Logic
# -------------------------
if content_file and style_file:

    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    st.subheader("🖼️ Input Images")

    col1, col2 = st.columns(2)

    with col1:
        st.image(preview(content_img), caption="Content Image")

    with col2:
        st.image(preview(style_img), caption="Style Image")

    # -------------------------
    # Generate Button
    # -------------------------
    if st.button("✨ Generate Stylized Image"):

        progress = st.progress(0)
        status = st.empty()

        try:
            with st.spinner("Generating... ⚡"):

                progress.progress(10)
                status.text("Preparing images...")

                content = load_image(content_img, 512)
                style = load_image(style_img, 512)

                progress.progress(40)
                status.text("Applying style...")

                output = model(content, style)[0]

                progress.progress(80)
                status.text("Rendering result...")

                image = np.squeeze(output.numpy())
                image = (image * 255).astype(np.uint8)
                image = Image.fromarray(image)

                progress.progress(100)
                status.text("Done!")

            progress.empty()
            status.empty()

            # -------------------------
            # Output
            # -------------------------
            st.subheader("🎨 Result")
            st.success("Style Transfer Complete!")

            col3, col4 = st.columns(2)

            with col3:
                st.image(preview(content_img), caption="Original")

            with col4:
                st.image(image, caption="Stylized Output")

            # -------------------------
            # Download
            # -------------------------
            buf = io.BytesIO()
            image.save(buf, format="PNG")

            st.download_button(
                "📥 Download Image",
                buf.getvalue(),
                "stylized_output.png",
                "image/png"
            )

        except Exception as e:
            progress.empty()
            status.empty()
            st.error("❌ Something went wrong")
            st.exception(e)