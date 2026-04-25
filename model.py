import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import VGG19_Weights

device = torch.device("cpu")

loader = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor()
])

def gram_matrix(x):
    b, c, h, w = x.size()
    features = x.view(b*c, h*w)
    G = torch.mm(features, features.t())
    return G.div(b*c*h*w)

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
    def forward(self, x):
        self.loss = F.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = gram_matrix(target).detach()
    def forward(self, x):
        G = gram_matrix(x)
        self.loss = F.mse_loss(G, self.target)
        return x

def run_style_transfer(content_img, style_img, style_weight=1e6, steps=80):

    content_img = loader(content_img).unsqueeze(0).to(device)
    style_img = loader(style_img).unsqueeze(0).to(device)

    input_img = content_img.clone().requires_grad_(True)

    cnn = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

    content_layers = ['conv_4']
    style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']

    content_losses = []
    style_losses = []

    model = nn.Sequential()
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
            target = model(content_img).detach()
            content_losses.append(ContentLoss(target))

        if name in style_layers:
            target = model(style_img).detach()
            style_losses.append(StyleLoss(target))

    optimizer = optim.LBFGS([input_img])

    run = [0]
    while run[0] <= steps:

        def closure():
            optimizer.zero_grad()
            input_img.data.clamp_(0,1)

            x = input_img
            i = 0
            content_loss = 0
            style_loss = 0

            for layer in model.children():
                x = layer(x)

                if isinstance(layer, nn.Conv2d):
                    i += 1
                    name = f'conv_{i}'

                    if name in content_layers:
                        content_loss += torch.mean((x - content_img) ** 2)

                    if name in style_layers:
                        style_loss += torch.mean((gram_matrix(x) - gram_matrix(style_img)) ** 2)

            loss = content_loss + style_weight * style_loss
            loss.backward()

            run[0] += 1
            return loss

        optimizer.step(closure)

    return input_img