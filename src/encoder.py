
import torch
import torch.nn as nn

from torchvision.models import (
    resnet18,
    ResNet18_Weights
)


def get_device():
    """Use Apple GPU through MPS if available."""
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class ResNet18Encoder(nn.Module):

    def __init__(self, d_model=512):

        super().__init__()

        # Load pretrained ResNet18.
        resnet = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Remove average pooling and classification layers.
        # Retain the convolutional feature extractor.
        self.backbone = nn.Sequential(
            *list(resnet.children())[:-2]
        )

        # Freeze all CNN parameters.
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        # Project CNN features into Transformer dimensions.
        self.projection = nn.Linear(512, d_model)

        # Keep frozen CNN in evaluation mode.
        self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)

        # Do not update BatchNorm running statistics.
        self.backbone.eval()

        return self

    def extract_spatial_features(self, images):

        # CNN is frozen, so gradients are unnecessary here.
        with torch.no_grad():
            feature_map = self.backbone(images)

        # (B, 512, 7, 7)
        # Convert spatial dimensions into a sequence.
        features = feature_map.flatten(2)
        # (B, 512, 49) -> (B, 49, 512)
        features = features.transpose(1, 2)

        return features

    def forward(self, images):

        # Extract spatial image features.
        features = self.extract_spatial_features(images)

        # Project into Transformer representation size.
        projected_features = self.projection(features)

        return projected_features
