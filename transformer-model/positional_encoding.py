import torch
import torch.nn as nn
import numpy as np

def positional_encoding_1d(seq_len, d_model):
  """
  Generate 1D positional encoding for a sequence.
  
  Args:
    seq_len (int): Length of the sequence.
    d_model (int): Dimensionality of the encoding.
  
  Returns:
    torch.Tensor: Positional encoding of shape (seq_len, d_model).
  """
  position = torch.arange(seq_len).unsqueeze(1)  # Shape: (seq_len, 1)
  div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # Shape: (d_model/2,)
  
  pe = torch.zeros(seq_len, d_model)
  pe[:, 0::2] = torch.sin(position * div_term)  # Apply sine to even indices
  pe[:, 1::2] = torch.cos(position * div_term)  # Apply cosine to odd indices
  
  return pe

def positional_encoding_2d(height, width, d_model):
  """
  Generate 2D positional encoding for a 2D structure (e.g., image).
  
  Args:
    height (int): Height of the 2D structure.
    width (int): Width of the 2D structure.
    d_model (int): Dimensionality of the encoding.
  
  Returns:
    torch.Tensor: Positional encoding of shape (height, width, d_model).
  """
  if d_model % 2 != 0:
    raise ValueError("d_model must be even for 2D positional encoding.")
  
  # Generate 1D positional encodings for rows and columns
  pe_row = positional_encoding_1d(height, d_model // 2)  # Shape: (height, d_model/2)
  pe_col = positional_encoding_1d(width, d_model // 2)   # Shape: (width, d_model/2)
  
  # Expand dimensions to combine row and column encodings
  pe_row = pe_row.unsqueeze(1).repeat(1, width, 1)  # Shape: (height, width, d_model/2)
  pe_col = pe_col.unsqueeze(0).repeat(height, 1, 1) # Shape: (height, width, d_model/2)
  
  # Concatenate row and column encodings along the last dimension
  pe = torch.cat([pe_row, pe_col], dim=-1)  # Shape: (height, width, d_model)
  
  return pe


class PositionalEncoding1D(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = positional_encoding_1d(max_len, d_model)  # Your original function
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return self.dropout(x + self.pe[:seq_len].unsqueeze(0))


class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model, height=32, width=32, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = positional_encoding_2d(height, width, d_model)  # Your original function
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(x + self.pe.unsqueeze(0))  # Broadcast across batch
