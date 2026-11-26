
import torch
from src.decoder import FeedForward

batch_size = 32
sequence_length = 23
embed_dim = 512
hidden_dim = 2048

feed_forward = FeedForward(embed_dim, hidden_dim)

x = torch.randn(batch_size, sequence_length, embed_dim)

output = feed_forward(x)

print("Input shape:", x.shape)
print("Output shape:", output.shape)

assert output.shape == x.shape

print("Feed-forward test passed!")

