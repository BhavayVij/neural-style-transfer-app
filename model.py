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
# Normalization Module (MANDATORY)
# -------------------------
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor(std).view(1, 3, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


# ImageNet stats
cnn_mean = [0.485, 0.456, 0.406]
cnn_std = [0.229, 0.224, 0.225]

# -------------------------
# Gram Matrix (CORRECT)
# -------------------------
def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

# -------------------------
# Content Loss
# -------------------------
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# -------------------------
# Style Loss
# -------------------------
class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

# -------------------------
# Total Variation Loss (light smoothing)
# -------------------------
def tv_loss(img):
    return torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])) + \
           torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))

# -------------------------
# Loader (keeps aspect ratio)
# -------------------------
def get_loader(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

# -------------------------
# Main Function
# -------------------------
def run_style_transfer(
    content_img_pil,
    style_img_pil,
    style_weight=1e6,
    content_weight=1,
    tv_weight=1e-6,
    steps=600,
    img_size=512,
    callback=None
):

    loader = get_loader(img_size)

    content = loader(content_img_pil).unsqueeze(0).to(device)
    style = loader(style_img_pil).unsqueeze(0).to(device)

    # Start from content image
    input_img = content.clone().requires_grad_(True)

    # Load VGG19
    cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

    # Layers
    content_layers = ['conv_4']
    style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']

    # Build model
    normalization = Normalization(cnn_mean, cnn_std).to(device)
    model = nn.Sequential(normalization)

    content_losses = []
    style_losses = []

    i = 0
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
            target = model(content).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    # Optimizer (tutorial standard)
    optimizer = optim.LBFGS([input_img])

    run = [0]

    # -------------------------
    # Optimization Loop
    # -------------------------
    while run[0] <= steps:

        def closure():
            optimizer.zero_grad()

            input_img.data.clamp_(0, 1)

            model(input_img)

            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            tv = tv_loss(input_img)

            loss = (
                content_weight * content_score +
                style_weight * style_score +
                tv_weight * tv
            )

            loss.backward()

            run[0] += 1

            if callback:
                callback(min(run[0], steps), steps)

            return loss

        optimizer.step(closure)

    # Final clamp
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img.detach()