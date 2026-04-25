import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights

# -------------------------
# Device (AUTO DETECT)
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Dynamic Loader (Resolution Control)
# -------------------------
def get_loader(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])


# -------------------------
# Gram Matrix
# -------------------------
def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)


# -------------------------
# Main Style Transfer Function
# -------------------------
def run_style_transfer(
    content_img,
    style_img,
    style_weight=1e6,
    steps=80,
    img_size=384,
    callback=None
):
    # Load + preprocess
    loader = get_loader(img_size)

    content_img = loader(content_img).unsqueeze(0).to(device)
    style_img = loader(style_img).unsqueeze(0).to(device)

    input_img = content_img.clone().requires_grad_(True)

    # Load VGG19
    cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

    # Layers
    content_layers = ['conv_4']
    style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential().to(device)

    content_targets = {}
    style_targets = {}

    i = 0

    # -------------------------
    # Build model + store targets ONCE
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

        # Store content target
        if name in content_layers:
            target = model(content_img).detach()
            content_targets[name] = target

        # Store style target (GRAM MATRIX ONCE)
        if name in style_layers:
            target = model(style_img).detach()
            style_targets[name] = gram_matrix(target)

    # -------------------------
    # Optimizer
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

            loss = content_loss + style_weight * style_loss
            loss.backward()

            run[0] += 1

            # -------------------------
            # Progress Callback
            # -------------------------
            if callback:
                callback(run[0], steps)

            return loss

        optimizer.step(closure)

    return input_img.detach()