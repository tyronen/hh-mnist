import torch.nn as nn
import torch
import math

# Original images       (25, 28, 28)
# Patches (flattened)   (25, num_patches, patch_size * patch_size)
# Patch projection      (25, num_patches, dim_model)
# Encoder output        (25, num_patches, dim_model)
# After mean pooling    (25, dim_model)
# Final logits          (25, 10)


class Patchify(nn.Module):
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
        patches = patches.transpose(-2, -1)      # (batch_size, num_patches, patch_size * patch_size)
        x = self.linear(patches)                # (batch_size, num_patches, dim_model)
        return x
    
class Encoder(nn.Module):
    def __init__(
            self, 
            dim_model, 
            dim_k, 
            dim_v):
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

class Classifier(nn.Module):
    def __init__(
            self, 
            patch_size: int, 
            stride: int, 
            dim_model: int, 
            dim_k: int, 
            dim_v):
        super().__init__()
        self.patchify = Patchify(patch_size=patch_size, stride=stride, dim_model=dim_model)
        self.encoders = nn.ModuleList(
            [Encoder(dim_model=dim_model, dim_k=dim_k, dim_v=dim_v)]
        )
        self.final_projection = nn.Linear(dim_model, 10)

    def forward(self, x):
        # x = (batch_size, 1, 28, 28)
        x = self.patchify(x) # (batch_size, num_patches, dim_model)
        for encoder in self.encoders:
            x = encoder(x) # (batch_size, num_patches, dim_model)
        x = x.mean(dim=-2) # (batch_size, dim_model)
        return self.final_projection(x)