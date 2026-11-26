
import torch
from src.decoder import TokenEmbedding

# Our project settings
vocab_size = 4460
embed_dim = 512
batch_size = 32
sequence_length = 23

# Create embedding layer
embedding = TokenEmbedding(vocab_size, embed_dim)

# Simulate a batch of 32 captions
token_ids = torch.randint(
    0, vocab_size, (batch_size, sequence_length)
)

# Convert token IDs into vectors
output = embedding(token_ids)

print("Input shape:", token_ids.shape)
print("Output shape:", output.shape)
