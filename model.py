import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Normalization
# -------------------------
cnn_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

def normalize(img):
    return (img - cnn_mean.view(1,3,1,1)) / cnn_std.view(1,3,1,1)

def denormalize(img):
    return img * cnn_std.view(1,3,1,1) + cnn_mean.view(1,3,1,1)

# -------------------------
# Loader
# -------------------------
def get_loader(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

# -------------------------
# Gram Matrix
# -------------------------
def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(c, h * w)
    G = torch.mm(features, features.t())
    return G / (c * h * w)

# -------------------------
# Total Variation Loss
# -------------------------
def tv_loss(img):
    return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

# -------------------------
# Main Function
# -------------------------
def run_style_transfer(
    content_img,
    style_img,
    style_weight=1e6,   # tutorial uses high value
    content_weight=1,
    steps=500,
    img_size=512,
    callback=None
):

    loader = get_loader(img_size)

    content = loader(content_img).unsqueeze(0).to(device)
    style = loader(style_img).unsqueeze(0).to(device)

    content = normalize(content)
    style = normalize(style)

    # Start from content (stable like tutorial)
    input_img = content.clone().requires_grad_(True)

    # -------------------------
    # Load VGG19
    # -------------------------
    cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

    content_layers = ['conv_4']
    style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']

    model = nn.Sequential().to(device)

    content_targets = {}
    style_targets = {}

    i = 0

    # -------------------------
    # Build model
    # -------------------------
    for layer in cnn.children():

        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            content_targets[name] = model(content).detach()

        if name in style_layers:
            style_targets[name] = gram_matrix(model(style)).detach()

    # -------------------------
    # LBFGS Optimizer (KEY DIFFERENCE)
    # -------------------------
    optimizer = optim.LBFGS([input_img])

    run = [0]

    # -------------------------
    # Optimization Loop
    # -------------------------
    while run[0] <= steps:

        def closure():
            optimizer.zero_grad()

            input_img.data.clamp_(0, 1)

            x = input_img
            i = 0

            content_loss = 0
            style_loss = 0

            for layer in model.children():
                x = layer(x)

                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = f'conv_{i}'

                    if name in content_targets:
                        content_loss += F.mse_loss(x, content_targets[name])

                    if name in style_targets:
                        G = gram_matrix(x)
                        style_loss += F.mse_loss(G, style_targets[name])

            loss = content_weight * content_loss + style_weight * style_loss

            loss.backward()

            run[0] += 1

            if callback:
                callback(min(run[0], steps), steps)

            return loss

        optimizer.step(closure)

    # -------------------------
    # Final Output
    # -------------------------
    output = denormalize(input_img).clamp(0, 1)

    return output.detach()