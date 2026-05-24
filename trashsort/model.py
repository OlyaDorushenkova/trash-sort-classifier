import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def create_model(num_classes: int) -> nn.Module:
    """
    Backbone model for classification (ResNet18).
    """

    model = resnet18(weights=ResNet18_Weights.DEFAULT)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )

    return model