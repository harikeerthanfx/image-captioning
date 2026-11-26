
import torch
from src.decoder import TransformerDecoder

vocab_size = 4460
batch_size = 32
sequence_length = 23
embed_dim = 512
image_positions = 49

# Create complete Transformer Decoder.
decoder = TransformerDecoder(vocab_size)

# Simulate caption token IDs.
caption_ids = torch.randint(
    0, vocab_size,
    (batch_size, sequence_length)
)

# Simulate CNN image features.
image_features = torch.randn(
    batch_size, image_positions, embed_dim
)

# Forward pass.
output = decoder(caption_ids, image_features)

print("Caption IDs:", caption_ids.shape)
print("Image features:", image_features.shape)
print("Decoder output:", output.shape)

assert output.shape == (32, 23, 4460)

print("Complete Transformer Decoder test passed!")
