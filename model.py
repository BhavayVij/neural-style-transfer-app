import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------
# Load model once (cached)
# -------------------------
model = None

def get_model():
    global model
    if model is None:
        model = hub.load(
            "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
        )
    return model


# -------------------------
# Resize WITHOUT distortion
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
# Main function
# -------------------------
def run_style_transfer(
    content_img,
    style_img,
    style_weight=None,
    content_weight=None,
    tv_weight=None,
    steps=None,
    img_size=512,
    callback=None
):
    model = get_model()

    content = load_image(content_img, img_size)
    style = load_image(style_img, img_size)

    # Light progress feedback
    if callback:
        callback(10, 100)

    output = model(content, style)[0]

    if callback:
        callback(100, 100)

    return output