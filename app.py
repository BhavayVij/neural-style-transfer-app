import streamlit as st
from PIL import Image
import style
import os
from io import BytesIO
import base64
from streamlit_image_comparison import image_comparison
# style image paths:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
root_style = os.path.join(BASE_DIR, "images", "style-images")
root_model = os.path.join(BASE_DIR, "saved_models")


# download image function
def get_image_download_link(img, file_name, style_name):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a style = "color:black" href="data:file/jpg;base64,{img_str}" download="{style_name+"_"+file_name+".jpg"}"><input type="button" value="Download"></a>'
    return href


st.markdown("<h1 style='text-align: center; color: Blue;'>Neural Style Transfer</h1>",
            unsafe_allow_html=True)
st.markdown("<h3 style='text-align: right; color: Blue;'>by Bhavay Vij</h3>",
            unsafe_allow_html=True)



# creating a side bar for picking the style of image
style_name = st.sidebar.selectbox(
    'Select Style',
    ("candy", "mosaic", "rain_princess", "udnie")
)
path_style = os.path.join(root_style, style_name+".jpg")


# Upload image functionality
img = None
uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"])

show_file = st.empty()

# checking if user has uploaded any file
if not uploaded_file:
    show_file.info("Please Upload an Image")
else:
    img = Image.open(uploaded_file)
    # check required here if file is an image file
    st.image(img, caption='Uploaded Image.', use_column_width=True)
    style_image = Image.open(path_style)
    st.image(style_image, caption='Style Image', use_column_width=True)


extensions = [".png", ".jpeg", ".jpg"]

if uploaded_file is not None and any(extension in uploaded_file.name for extension in extensions):

    name_file = uploaded_file.name.split(".")
    root_model = "./saved_models"
    model_path = os.path.join(root_model, style_name+".pth")

    img = img.convert('RGB')
    input_image = img

    root_output = "./images/output-images"
    output_image = os.path.join(
        root_output, style_name+"-"+name_file[0]+".jpg")

    stylize_button = st.button("Stylize")

    if stylize_button:
        model = style.load_model(model_path)
        stylized = style.stylize(model, input_image, output_image)

        input_resized = input_image.resize(stylized.size)

        st.write("### 🔍 Comparison: Original vs Stylized")
        st.caption("Drag the slider to compare the transformation")

        image_comparison(
            img1=input_resized,
            img2=stylized,
            label1="Original",
            label2="Stylized",
            width=700,
            starting_position=50,
            show_labels=True,
            make_responsive=True,
            in_memory=True
        )

        st.write("### Output Image")
        st.image(stylized, use_column_width=True)

        st.markdown(get_image_download_link(
            stylized, name_file[0], style_name), unsafe_allow_html=True)