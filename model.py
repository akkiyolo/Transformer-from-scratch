import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    """
    Converts token indices into dense embedding vectors and scales them.
    This is the standard embedding layer used in the Transformer model
    (Vaswani et al., 2017).
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        Parameters:
        - d_model: Dimension of the embedding vectors (e.g., 512, 768)
        - vocab_size: Total number of unique tokens in the vocabulary
        """
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size

        # Lookup table that maps token IDs → embedding vectors of size d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        Forward pass:
        - x: Tensor of token IDs (batch_size, sequence_length)
        Returns:
        - Embedded representation of shape (batch, seq_len, d_model)
        - Scaled by sqrt(d_model), as described in Transformer paper
        """
        # Multiply by sqrt(d_model) to match positional encoding scale
        return self.embedding(x) * math.sqrt(self.d_model)
