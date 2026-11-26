
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, embed_dim):

        super().__init__()

        # Converts each token ID into a learnable vector.
        self.embedding = nn.Embedding(vocab_size, embed_dim)

    def forward(self, token_ids):

        # Input:  (batch_size, sequence_length)
        # Output: (batch_size, sequence_length, embed_dim)
        return self.embedding(token_ids)


import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim, max_len=100):

        super().__init__()

        # Create a matrix to store positional encodings.
        # Each position gets a vector of embed_dim numbers.
        pe = torch.zeros(max_len, embed_dim)

        # Position indices: 0, 1, 2, 3, ...
        position = torch.arange(0, max_len).unsqueeze(1)

        # Values used to calculate different frequencies.
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2)
            * (-math.log(10000.0) / embed_dim)
        )

        # Apply sine to even dimensions.
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd dimensions.
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension so it can be added to embeddings.
        pe = pe.unsqueeze(0)

        # Store as a buffer: moves with the model but isn't trainable.
        self.register_buffer("pe", pe)

    def forward(self, x):

        # x shape: (batch_size, sequence_length, embed_dim)
        # Add positional information to token embeddings.
        x = x + self.pe[:, :x.size(1), :]

        return x


class MaskedSelfAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        # Multi-head self-attention layer.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, x, padding_mask=None):
        sequence_length = x.size(1)

        # Prevent attention to future tokens
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        output, _ = self.attention(
            x, x, x,
            attn_mask=mask,
            key_padding_mask=padding_mask,
            need_weights=False
        )

        return output


class CrossAttention(nn.Module):

    def __init__(self, embed_dim, num_heads):

        super().__init__()

        # Allows caption tokens to attend to image features.
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

    def forward(self, caption_features, image_features):

        # Query: caption features (what information do we need?)
        # Key: image features (what visual information is available?)
        # Value: image features (the information to retrieve)

        output, _ = self.attention(
            query=caption_features,
            key=image_features,
            value=image_features,
            need_weights=False
        )

        return output


class FeedForward(nn.Module):

    def __init__(self, embed_dim, hidden_dim):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):

        return self.network(x)


class DecoderBlock(nn.Module):

    def __init__(self, embed_dim, num_heads, hidden_dim):

        super().__init__()

        self.self_attention = MaskedSelfAttention(
            embed_dim, num_heads
        )

        self.cross_attention = CrossAttention(
            embed_dim, num_heads
        )

        self.feed_forward = FeedForward(
            embed_dim, hidden_dim
        )

        # Layer normalization after each sub-layer.
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, image_features, padding_mask=None):
        attention_output = self.self_attention(x, padding_mask)
        x = self.norm1(x + attention_output)

        attention_output = self.cross_attention(x, image_features)
        x = self.norm2(x + attention_output)

        ff_output = self.feed_forward(x)
        x = self.norm3(x + ff_output)

        return x


class TransformerDecoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        embed_dim=512,
        num_heads=8,
        hidden_dim=2048,
        max_len=100
    ):

        super().__init__()

        # Convert caption token IDs into vectors.
        self.token_embedding = TokenEmbedding(
            vocab_size, embed_dim
        )

        # Add word-position information.
        self.positional_encoding = PositionalEncoding(
            embed_dim, max_len
        )

        # Main Transformer Decoder Block.
        self.decoder_block = DecoderBlock(
            embed_dim, num_heads, hidden_dim
        )

        # Convert decoder representations into vocabulary scores.
        self.output_layer = nn.Linear(
            embed_dim, vocab_size
        )

    def forward(self, caption_ids, image_features):
        # Identify PAD tokens
        padding_mask = (caption_ids == 0)

        x = self.token_embedding(caption_ids)
        x = self.positional_encoding(x)

        x = self.decoder_block(
            x,
            image_features,
            padding_mask
        )

        output = self.output_layer(x)

        return output
