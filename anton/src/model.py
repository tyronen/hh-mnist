import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate projection layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, D = x.shape  # Batch, Num tokens, Embedding dim

        # Project inputs to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention: (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, num_heads, N, N)
        attn = scores.softmax(dim=-1)
        attended = attn @ v  # (B, num_heads, N, head_dim)

        # Concatenate heads and project back
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention + residual
        x = x + self.dropout(self.attn(self.norm1(x)))
        # MLP + residual
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    def __init__(self,
                 patch_dim=196,       # Each 14x14 patch flattened to 196
                 embed_dim=64,        # Token embedding size
                 num_heads=4,
                 mlp_dim=128,
                 num_layers=6,
                 num_classes=10,
                 num_patches=4):      # 4 patches per MNIST image
        super().__init__()

        # Patch embedding projection
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Learnable positional encoding (for 4 patches + 1 CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        # Transformer encoder layers
        self.encoder = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_dim)
            for _ in range(num_layers)
        ])

        # Final classifier head (based on CLS token)
        self.head = nn.Linear(embed_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, 4, 196) = 4 patch vectors per image
        B = x.size(0)

        # Project patches to embeddings
        x = self.patch_embed(x)  # (B, 4, 64)

        # Expand CLS token for batch and prepend to sequence
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, 64)
        x = torch.cat([cls_token, x], dim=1)          # (B, 5, 64)

        # Add positional embeddings
        x = x + self.pos_embed  # (B, 5, 64)

        # Pass through Transformer layers
        x = self.encoder(x)     # (B, 5, 64)

        # Extract the CLS token output
        cls_output = x[:, 0]    # (B, 64)

        # Final classification
        return self.head(cls_output)  # (B, 10)
