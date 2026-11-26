
import torch
from src.decoder import DecoderBlock

batch_size = 32
sequence_length = 23
image_positions = 49
embed_dim = 512
num_heads = 8
hidden_dim = 2048

decoder_block = DecoderBlock(
    embed_dim, num_heads, hidden_dim
)

# Simulated caption representations.
caption_features = torch.randn(
    batch_size, sequence_length, embed_dim
)

# Simulated CNN image features.
image_features = torch.randn(
    batch_size, image_positions, embed_dim
)

output = decoder_block(caption_features, image_features)

print("Caption input:", caption_features.shape)
print("Image input:", image_features.shape)
print("Decoder output:", output.shape)

assert output.shape == caption_features.shape

print("Decoder block test passed!")
