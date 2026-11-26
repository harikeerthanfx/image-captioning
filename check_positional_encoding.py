
import torch

from src.decoder import TokenEmbedding, PositionalEncoding

vocab_size = 4460
embed_dim = 512
batch_size = 32
sequence_length = 23

# Simulate a batch of token IDs.
token_ids = torch.randint(
    0, vocab_size, (batch_size, sequence_length)
)

# Convert token IDs into embeddings.
embedding = TokenEmbedding(vocab_size, embed_dim)
embedded_tokens = embedding(token_ids)

# Add positional information.
positional_encoding = PositionalEncoding(embed_dim)
output = positional_encoding(embedded_tokens)

print("Token IDs:", token_ids.shape)
print("After embedding:", embedded_tokens.shape)
print("After positional encoding:", output.shape)
