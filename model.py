import torch
import torch.nn as nn
import torchvision.models as models

class ClassifierWrapper(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.backbone(x)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, device='cpu'):
    model = ClassifierWrapper(n_classes=2, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model
