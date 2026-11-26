
import torch
from src.decoder import MaskedSelfAttention

batch_size = 32
sequence_length = 23
embed_dim = 512
num_heads = 8

attention = MaskedSelfAttention(embed_dim, num_heads)

# Simulate position-aware caption embeddings.
x = torch.randn(batch_size, sequence_length, embed_dim)

output = attention(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

print("Masked self-attention test passed!")
