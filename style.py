import torch
import re
from torchvision import transforms
import streamlit as st

import utils
from transformer_net import TransformerNet

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model(model_path):

    with torch.no_grad():
        model = TransformerNet()

        state_dict = torch.load(model_path, map_location=device)

        # Remove deprecated InstanceNorm keys
        for k in list(state_dict.keys()):
            if re.search(r'in\d+\.running_(mean|var)$', k):
                del state_dict[k]

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        return model


# -------------------------
# Stylize image
# -------------------------
def stylize(model, content_image, output_path=None):

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])

    content = transform(content_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(content).cpu()

    # Save if path provided
    if output_path:
        utils.save_image(output_path, output[0])

    return utils.tensor_to_image(output[0])