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
# Title
# -------------------------
st.title("🎨 Neural Style Transfer App")
st.write("Transform your photos into artistic masterpieces using AI style transfer.")

# -------------------------
# Upload Section
# -------------------------
content_file = st.file_uploader("Upload Content Image", type=["jpg", "png", "jpeg"])
style_file = st.file_uploader("Upload Style Image", type=["jpg", "png", "jpeg"])

# -------------------------
# Controls
# -------------------------
style_weight = st.slider("Style Strength", 100000, 5000000, 1000000)

quality_mode = st.selectbox(
    "Select Quality Mode",
    ["Fast (Recommended)", "High Quality"]
)

# Contextual warning (clean UX)
if content_file and style_file and quality_mode == "High Quality":
    st.warning("⚠️ High Quality mode may take longer on CPU")

# -------------------------
# Main Logic
# -------------------------
if content_file and style_file:

    content_img = Image.open(content_file).convert("RGB")
    style_img = Image.open(style_file).convert("RGB")

    st.subheader("🖼️ Input Images")

    col1, col2 = st.columns(2)

    with col1:
        st.image(content_img, caption="Content Image", use_container_width=True)

    with col2:
        st.image(style_img, caption="Style Image", use_container_width=True)

    # -------------------------
    # Generate Button
    # -------------------------
    if st.button("✨ Generate Stylized Image"):

        # Quality settings
        steps = 80 if quality_mode == "Fast (Recommended)" else 200
        img_size = 256 if quality_mode == "Fast (Recommended)" else 384

        # Progress UI
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(step, total):
            percent = int((step / total) * 100)
            progress_bar.progress(percent)
            status_text.text(f"Processing... {percent}%")

        try:
            with st.spinner(f"Generating image ({quality_mode})... ⏳"):

                output = run_style_transfer(
                    content_img,
                    style_img,
                    style_weight=style_weight,
                    steps=steps,
                    img_size=img_size,
                    callback=update_progress
                )

                image = output.cpu().clone().squeeze(0)
                image = ToPILImage()(image)

            # Clean UI
            progress_bar.empty()
            status_text.empty()

            # -------------------------
            # Output Section
            # -------------------------
            st.subheader("🎨 Result")
            st.success("Style Transfer Complete!")

            col3, col4 = st.columns(2)

            with col3:
                st.image(content_img, caption="Original", use_container_width=True)

            with col4:
                st.image(image, caption="Stylized Output", use_container_width=True)

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
            progress_bar.empty()
            status_text.empty()
            st.error("❌ Something went wrong during processing.")
            st.exception(e)