
import torch
import torch.nn as nn

from src.encoder import ResNet18Encoder
from src.decoder import TransformerDecoder


class ImageCaptioningModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # CNN encoder: extracts image features using ResNet18
        self.encoder = ResNet18Encoder(d_model=512)

        # Transformer decoder: generates caption predictions
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=512,
            num_heads=8,
            hidden_dim=2048
        )

    def forward(self, images, caption_ids):

        # Extract image features: (B, 49, 512)
        image_features = self.encoder(images)

        # Predict vocabulary scores: (B, L, vocab_size)
        output = self.decoder(caption_ids, image_features)

        return output
