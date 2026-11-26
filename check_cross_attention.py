
import torch
from src.decoder import CrossAttention

batch_size = 32
sequence_length = 23
image_positions = 49
embed_dim = 512
num_heads = 8

cross_attention = CrossAttention(embed_dim, num_heads)

# Caption representations from token embedding + positional encoding.
caption_features = torch.randn(
    batch_size, sequence_length, embed_dim
)

# Image representations from CNN encoder + feature projection.
image_features = torch.randn(
    batch_size, image_positions, embed_dim
)

output = cross_attention(caption_features, image_features)

print("Caption features:", caption_features.shape)
print("Image features:", image_features.shape)
print("Output shape:", output.shape)

assert output.shape == caption_features.shape

print("Cross-attention test passed!")
