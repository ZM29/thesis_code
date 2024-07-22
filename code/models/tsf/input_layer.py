"""
This script implements the input layer of the TSF.
"""

import torch
import torch.nn as nn
import numpy as np

#setting device to GPU or CPU
torch.manual_seed(29)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ResidualBlock(nn.Module):
    def __init__(self, num_features, model_dim, hidden_dim, dropout_rate):
        """
        Class that implements the residual block in the input layer that is described in the paper.
        
        Args:
            num_features (int): Number of features
            model_dim (int): Dimension of the model
            hidden_dim (int): Hidden dimension of the model
            dropout_rate (float): Dropout rate for regularization
        """
        super(ResidualBlock, self).__init__()
        self.layer1 = nn.Linear(num_features, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, model_dim)
        self.skip = nn.Linear(num_features, model_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, mask):
        residual = self.skip(x)

        #checking if masked values are zero
        x = x * mask.unsqueeze(-1)
        assert (x * (~mask.unsqueeze(-1))).sum() == 0.0, "Non-zero values found in padded positions!"

        #performing the process that is described in the paper 
        out = self.layer1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer2(out)
        out += residual
        out = self.layer_norm(out)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        """
        Class that takes care of the positional encoding. 
        This code is originally used in the 'Attention Is All You Need' study.

        Args:
            d_hid (int): Hidden dimension of the model
            n_position (int): Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()

        #creating and register positional encoding table
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        #generating sinusoidal position encoding table
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        #adding positional encoding to input
        return x + self.pos_table[:, :x.size(1)].clone().detach()
    


class InputLayer(nn.Module):
    def __init__(self, input_dim, model_dim, patch_size, max_len, hidden_dim, dropout_rate):
        """
        Class that implements the input layer for the TSF.
        
        Args:
            input_dim (int): Dimension of input features
            model_dim (int): Dimension of the model
            patch_size (int): Size of each patch
            max_len (int): Maximum sequence length in PositionalEncoding
            hidden_dim (int): Hidden dimension of the model in ResidualBlock
            dropout_rate (float): Dropout rate for regularization
        """
        super(InputLayer, self).__init__()
        self.input_dim = input_dim
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.max_len = max_len

        #initializing the needed components
        self.residual_block = ResidualBlock(input_dim, model_dim, hidden_dim, dropout_rate)
        self.numeric_embedding = nn.Linear(model_dim, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)


    def forward(self, numeric_data):
        #padding the input and creating padding mask
        padded_input = self.pad_input(numeric_data).to(device)
        padding_mask = self.padding_mask(padded_input).to(device)

        #splitting into patches
        patches = torch.split(padded_input, self.patch_size, dim = 1)
        patches_padding_masks = torch.split(padding_mask, self.patch_size, dim = 1)

        input_tokens = []
        #processing the patches through residual block and adding positional encoding
        for patch, patch_mask in zip(patches, patches_padding_masks):
            residual_output = self.residual_block(patch, patch_mask)
            input_token = residual_output + self.positional_encoding(residual_output)
            input_tokens.append(input_token)

        numeric_embeddings = input_tokens

        return numeric_embeddings, patches_padding_masks
    
    def pad_input(self, input_data):
        """
        This function pads the input to match the patch size.
        """
        batch_size, seq_len, _ = input_data.shape
        pad_len = (self.patch_size - (seq_len % self.patch_size)) % self.patch_size
        padded_input = torch.cat([input_data, torch.zeros(batch_size, pad_len, input_data.shape[2], device = input_data.device)], dim=1)
        return padded_input
    
    def padding_mask(self, input_data):
        """
        This function creates a padding mask.
        """
        padding_mask = (input_data != 0).any(dim = 2)
        return padding_mask
    
