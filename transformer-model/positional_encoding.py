import math
import torch
from torch import nn


def create_matrix(dim, length, temp):
    # Create frequency scalers
    pos = torch.arange(length).unsqueeze(1)
    freqs = torch.exp(torch.arange(0, dim, 2) * (-math.log(temp) / dim))  # shape: [dim//2]

    # Allocate encoding matrix and fill alternating sin/cos
    matrix = torch.zeros(length, dim)
    # even indices
    matrix[:, 0::2] = torch.sin(pos * freqs)
    # odd indices
    matrix[:, 1::2] = torch.cos(pos * freqs)
    return matrix

def create_field(dim, length, temp):
    word_matrix = create_matrix(dim, length, temp)

    # Broadcast and merge row/col encodings
    img_matrix = torch.zeros(length, length, dim)
    for d in range(dim):
        img_matrix[:, :, d] = word_matrix[:, d].unsqueeze(1) + word_matrix[:, d].unsqueeze(0)
    
    # Shape: [span, span, dim]
    return img_matrix

class WordPosEnc(nn.Module):
    # Apply sinusoidal positional information to 1D sequence embeddings 
    # to help transformers distinguish token order in the sequence

    def __init__(self, embed_dim, drop_rate=0.1, max_positions=1000, temp=10000.0):
        super().__init__()

        self.injector = nn.Dropout(p=drop_rate)
        self.register_buffer("encoding", create_matrix(embed_dim, max_positions, temp))

    def forward(self, sequence_tensor):
        # Adds encoding matrix up to the input's sequence length
        _, seq_len, _ = sequence_tensor.shape
        embedding = self.encoding[:seq_len].unsqueeze(0)
        sequence_tensor = sequence_tensor + embedding
        return self.injector(sequence_tensor)


class ImgPosEnc(nn.Module):
    # Adds spatial positional encoding to feature maps (e.g. from CNNs)
    # by combining row and column-wise sinusoidal encodings.

    def __init__(self, channels, drop_rate=0.1, grid_size=30, temp=10000.0):
        super().__init__()
        self.embed_drop = nn.Dropout(drop_rate)
        self.register_buffer("grid_encoding", create_field(channels, grid_size, temp))

    def forward(self, feat_grid):
        # Inject spatial positional encoding to each feature map
        _, height, width, _ = feat_grid.shape
        return self.embed_drop(feat_grid + self.grid_encoding[:height, :width].unsqueeze(0))
