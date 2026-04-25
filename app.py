import streamlit as st
from PIL import Image
from torchvision.transforms import ToPILImage
from model import run_style_transfer
import io

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Neural Style Transfer", layout="wide")

# -------------------------
# Title + Description
# -------------------------
st.title("🎨 Neural Style Transfer App")
st.write("Transform your photos into artistic masterpieces using AI style transfer.")
st.info("⚠️ Processing may take 1–2 minutes on CPU")

# -------------------------
# Upload Section
# -------------------------
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

# -------------------------
# Slider
# -------------------------
style_weight = st.slider("Style Strength", 100000, 5000000, 1000000)

# -------------------------
# Cache Model Run (IMPORTANT)
# -------------------------
@st.cache_data(show_spinner=False)
def process_image(content_bytes, style_bytes, style_weight):
    content_img = Image.open(io.BytesIO(content_bytes)).convert("RGB")
    style_img = Image.open(io.BytesIO(style_bytes)).convert("RGB")

    output = run_style_transfer(content_img, style_img, style_weight)

    image = output.cpu().clone().squeeze(0)
    image = ToPILImage()(image)

    return image

# -------------------------
# Main Logic
# -------------------------
if content_file and style_file:

    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    st.subheader("🖼️ Input Images")

    col1, col2 = st.columns(2)

    with col1:
        st.image(content_img, caption="Content Image", use_column_width=True)

    with col2:
        st.image(style_img, caption="Style Image", use_column_width=True)

    # -------------------------
    # Generate Button
    # -------------------------
    if st.button("✨ Generate Stylized Image"):

        try:
            with st.spinner("Generating artistic image... please wait ⏳"):

                image = process_image(
                    content_file.getvalue(),
                    style_file.getvalue(),
                    style_weight
                )

            # -------------------------
            # Output Section
            # -------------------------
            st.subheader("🎨 Result")
            st.success("Style Transfer Complete!")

            col3, col4 = st.columns(2)

            with col3:
                st.image(content_img, caption="Original", use_column_width=True)

            with col4:
                st.image(image, caption="Stylized Output", use_column_width=True)

            # -------------------------
            # Download Button
            # -------------------------
            buf = io.BytesIO()
            image.save(buf, format="PNG")

            st.download_button(
                label="📥 Download Stylized Image",
                data=buf.getvalue(),
                file_name="stylized_output.png",
                mime="image/png"
            )

        except Exception as e:
            st.error("❌ Something went wrong during processing.")
            st.exception(e)