import torch.nn as nn
import torch
import math

# Original images       (25, 28, 28)
# Patches (flattened)   (25, num_patches, patch_size * patch_size)
# Patch projection      (25, num_patches, dim_model)
# Encoder output        (25, num_patches, dim_model)
# After mean pooling    (25, dim_model)
# Final logits          (25, 10)


class PatchProjection(nn.Module):
    def __init__(
            self, 
            patch_size, 
            stride, 
            dim_model):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=stride)
        self.linear = nn.Linear(patch_size * patch_size, dim_model, bias=False)
        
    def forward(self, x):
        patches = self.unfold(x)                # (batch_size, patch_size * patch_size, num_patches)
        patches = patches.transpose(-2, -1)     # (batch_size, num_patches, patch_size * patch_size)
        x = self.linear(patches)                # (batch_size, num_patches, dim_model)
        return x

class Attention(nn.Module):
    def __init__(self, dim_model, dim_k, dim_v):
        super().__init__()
        self.dk = dim_k
        self.wq = nn.Linear(dim_model, dim_k, bias=False)
        self.wk = nn.Linear(dim_model, dim_k, bias=False)
        self.wv = nn.Linear(dim_model, dim_v, bias=False)
        self.w_o = nn.Linear(dim_v, dim_model, bias=False)
    
    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        attention = self.scaled_dot_product_attention(q, k, v)
        hidden_state = self.w_o(attention)
        return hidden_state
    
    def scaled_dot_product_attention(self, q, k, v):
        k_transpose = k.transpose(-2, -1)
        attention = (q @ k_transpose) / math.sqrt(self.dk)
        attention = torch.softmax(attention, dim=-1)
        return attention @ v


class Encoder(nn.Module):
    def __init__(
            self, 
            dim_model, 
            dim_k, 
            dim_v,
            has_pre_attention_norm: bool,
            has_post_attention_norm: bool):
        super().__init__()
        self.attention = Attention(dim_model, dim_k, dim_v)
        if has_pre_attention_norm:
            self.pre_attention_norm = nn.LayerNorm(dim_model)
        else:
            self.pre_attention_norm = None
        if has_post_attention_norm:
            self.post_attention_norm = nn.LayerNorm(dim_model)
        else:
            self.post_attention_norm = None

    def forward(self, x):
        original_x = x
        if self.pre_attention_norm is not None:
            x = self.pre_attention_norm(x)
        x = self.attention(x)
        x = x + original_x
        if self.post_attention_norm is not None:
            x = self.post_attention_norm(x)
        return x
    

class Classifier(nn.Module):
    def __init__(
            self, 
            patch_size: int, 
            stride: int, 
            dim_model: int, 
            dim_k: int, 
            dim_v,
            has_positional_encoding: bool,
            has_input_norm: bool,
            has_post_attention_norm: bool,
            has_post_mlp_norm: bool,
            has_pre_attention_norm: bool,
            has_final_norm: bool,
            num_encoders: int):
        super().__init__()
        self.patch_projection = PatchProjection(patch_size=patch_size, stride=stride, dim_model=dim_model)
        if has_positional_encoding:
            self.positional_encoding = PositionalEncoding(dim_model=dim_model, num_patches=4)
        else:
            self.positional_encoding = None
        self.encoders = nn.ModuleList(
            [
                Encoder(
                    dim_model=dim_model, 
                    dim_k=dim_k, 
                    dim_v=dim_v,
                    has_pre_attention_norm=has_pre_attention_norm,
                    has_post_attention_norm=has_post_attention_norm
                ) for _ in range(num_encoders)]
        )
        self.final_projection = nn.Linear(dim_model, 10)
        if has_input_norm:
            self.input_norm = nn.LayerNorm(dim_model)
        else:
            self.input_norm = None

    def forward(self, x):
        x = self.patch_projection(x)                # (batch_size, num_patches, dim_model)
        if self.input_norm is not None:
            x = self.input_norm(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)     # (batch_size, num_patches, dim_model)
        for encoder in self.encoders:
            x = encoder(x)                  # (batch_size, num_patches, dim_model)
        x = x.mean(dim=-2)                  # (batch_size, dim_model)
        return self.final_projection(x)
    
    
class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, num_patches: int):
        super().__init__()
        self.dim_model = dim_model
        self.num_patches = num_patches
        pe = self.generate_positional_encoding(dim_model, num_patches)
        self.register_buffer('pe', pe)

    # input is the input embeddings, which is (batch_size, num_patches, dim_model) without linear projection
    def generate_positional_encoding(self, dim_model: int, num_patches: int):
        pe = torch.zeros(num_patches, dim_model) # (num_patches, dim_model)
        position = torch.arange(0, num_patches, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model)
        )
        broadcast = position * div_term
        pe[:, 0::2] = torch.sin(broadcast)  # Even indices
        pe[:, 1::2] = torch.cos(broadcast)  # Odd indices
        return pe.unsqueeze(0)  # shape: [1, n_positions, d_model]

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1]]
        return x