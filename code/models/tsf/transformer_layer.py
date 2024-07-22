"""
This script implements the transformer layer of the TSF.
"""

import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block, GPT2Config
import torch

#setting device to GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TSFTransformer(nn.Module):
    def __init__(self, num_layers, model_dim, num_heads):
        """
        Class that implements the transformer architecture for the TSF.

        Args:
            num_layers (int): Number of transformer layers
            model_dim (int): Dimension of the model
            num_heads (int): Number of attention heads
        """
        super(TSFTransformer, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.num_heads = num_heads

    
    def forward(self, numeric_embeddings_list, padding_masks):

        #stacking the GPT2 transformer blocks
        transformer_blocks = nn.ModuleList([
            GPT2Block(GPT2Config(n_embd = numeric_embeddings_list[0].shape[2], 
                                 n_head = self.num_heads, n_layer = self.num_layers)).to(device)
            for _ in range(self.num_layers)
        ])


        hidden_states = []

        #processing each set of numeric embeddings and padding masks
        for numeric_embeddings, padding_mask in zip(numeric_embeddings_list, padding_masks):
            num_series, patch_size, embedding_dim = numeric_embeddings.shape

            #creating a causal mask (lower triangular matrix)
            causal_mask = torch.tril(torch.ones(patch_size, patch_size)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
            causal_mask = causal_mask.expand(num_series, 1, -1, -1)
            causal_mask = causal_mask.to(device)

            padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)  
            padding_mask = padding_mask.to(device)

            #combining causal and padding masks to create the attention mask
            attention_mask = causal_mask & padding_mask
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))

            #passing the embeddings through each transformer block
            hidden_state = numeric_embeddings
            for block in transformer_blocks:
                outputs = block(hidden_states = hidden_state, attention_mask = attention_mask)
                hidden_state = outputs[0]
            
            hidden_states.append(hidden_state)

        return hidden_states

class ResidualBlockOutput(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Class that implements the residual block in the output layer that is described in the paper.

        Args:
            input_dim (int): Dimension of the input
            hidden_dim (int): Hidden dimension of the model
            output_dim (int): Dimension of the output
        """
        super(ResidualBlockOutput, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.skip = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = self.skip(x)
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out += residual
        return out
    