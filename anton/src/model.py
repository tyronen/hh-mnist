import torch
import torch.nn as nn
import math


# -----------------------------
# Multi-Head Self-Attention
# -----------------------------
class MultiHeadSelfAttention(nn.Module):
    """
    Standard multi-head self-attention mechanism.
    """
    def __init__(self, embed_dim, num_heads, dot_product_norm=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate projection layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dot_product_norm = dot_product_norm

    def forward(self, x, mask=None):
        """
        x: Input tensor of shape (B, N, D)
        mask: Optional attention mask of shape (1, 1, N, N)
        """
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
        scores = (q @ k.transpose(-2, -1)) # (B, num_heads, N, N)
        if self.dot_product_norm:
            scores /= math.sqrt(self.head_dim)

        # Apply mask if provided (for causal masking in decoder)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = scores.softmax(dim=-1)

        # Apply attention weights to values
        attended = attn @ v  # (B, num_heads, N, head_dim)

        # Concatenate heads and project back to original embedding dim
        attended = attended.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(attended)


# TODO Consider merging these two classes: MultiHeadSelfAttention & MultiHeadCrossAttention


# -----------------------------
# Multi-Head Cross-Attention
# -----------------------------
class MultiHeadCrossAttention(nn.Module):
    """
    Cross-attention mechanism where the query comes from the decoder and key/value come from the encoder.
    """
    def __init__(self, embed_dim, num_heads, dot_product_norm=True):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Separate projection layers for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Final linear projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dot_product_norm = dot_product_norm

    def forward(self, query, context):
        """
        query: (B, T_q, D) - from decoder
        context: (B, T_c, D) - from encoder
        """
        B, Nq, D = query.shape  # Batch, Num tokens (for query), Embedding dim
        Nc = context.size(1)    # Num tokens (for context)

        # Project inputs to Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(context)
        v = self.v_proj(context)

        # Reshape for multi-head attention: (B, num_heads, N, head_dim)
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, Nc, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) # (B, num_heads, N, N)
        if self.dot_product_norm:
            scores /= math.sqrt(self.head_dim)

        attn = scores.softmax(dim=-1)

        # Apply attention weights to values
        attended = attn @ v  # (B, num_heads, N, head_dim)

        attended = attended.transpose(1, 2).contiguous().view(B, Nq, D)
        return self.out_proj(attended)


# -----------------------------
# Transformer Encoder Block
# -----------------------------
class TransformerEncoderBlock(nn.Module):
    """
    Standard transformer encoder block: self-attention + feed-forward + residual connections.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0, dot_product_norm=True):
        super().__init__()
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dot_product_norm)
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


# -----------------------------
# Transformer Decoder Block
# -----------------------------
class TransformerDecoderBlock(nn.Module):
    """
    Decoder block with self-attention, cross-attention, and MLP.
    """
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0, dot_product_norm=True):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dot_product_norm)
        self.cross_attn = MultiHeadCrossAttention(embed_dim, num_heads, dot_product_norm)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None):
        # Masked self-attention + residual
        x = x + self.dropout(self.self_attn(self.norm1(x), tgt_mask))
        # Cross-attention (with encoder output) + residual
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_out))
        # MLP + residual
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


# -----------------------------
# Vision Transformer Encoder
# -----------------------------
class ViTEncoder(nn.Module):
    """
    Vision Transformer encoder for patch-based images.
    """
    def __init__(self,
                 patch_dim=196,       # Each 14x14 patch flattened to 196
                 embed_dim=64,        # Token embedding size
                 num_heads=4,
                 mlp_dim=128,
                 num_layers=6,
                 num_patches=4,       # 4 patches per image
                 add_pos_emb=True,
                 dropout=0.0,
                 dot_product_norm=True):
        super().__init__()

        # Patch embedding projection
        self.patch_embed = nn.Linear(patch_dim, embed_dim)

        # Learnable positional encoding (for 4 patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer encoder layers
        self.encoder = nn.Sequential(*[
            TransformerEncoderBlock(embed_dim, num_heads, mlp_dim, dropout, dot_product_norm)
            for _ in range(num_layers)
        ])

        self.add_pos_emb = add_pos_emb

        # Initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # x: (B, N, patch_dim)

        # Project patches to embeddings
        x = self.patch_embed(x)  # (B, N, embed_dim)

        # Add positional embeddings
        if self.add_pos_emb:
            x = x + self.pos_embed  # (B, N, embed_dim)

        # Pass through Transformer layers
        x = self.encoder(x)     # (B, N, embed_dim)

        return x


# -----------------------------
# Single Digit Classifier using Vision Transformer
# -----------------------------
class ViTClassifier(nn.Module):
    """
    Classifier for MNIST digits using Visition Transformer and simple 1-L FF network as head.
    """
    def __init__(self,
                 patch_dim=196,       # Each 14x14 patch flattened to 196
                 embed_dim=64,        # Token embedding size
                 num_heads=4,
                 mlp_dim=128,
                 num_layers=6,
                 num_classes=10,
                 num_patches=4,
                 avg_pooling=False,
                 add_pos_emb=True,
                 dropout=0.0,
                 dot_product_norm=True):      # 4 patches per MNIST image
        super().__init__()

        self.encoder = ViTEncoder(
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            num_patches=num_patches + 1,  # CLS token as an extra patch
            add_pos_emb=add_pos_emb,
            dropout=dropout,
            dot_product_norm=dot_product_norm
        )

        # Learnable class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, patch_dim))

        # Final classifier head (based on CLS token)
        self.head = nn.Linear(embed_dim, num_classes)

        self.avg_pooling = avg_pooling

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: (B, N, patch_dim)

        B = x.size(0)

        # Expand CLS token for batch and prepend to sequence
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, patch_dim)
        x = torch.cat([cls_token, x], dim=1)          # (B, 1+N, patch_dim)

        # Apply the Vision Transformer Encoder
        x = self.encoder(x)

        # Extract the CLS token output
        cls_output = x[:, 0]    # (B, embed_dim)

        # An option to use average pooling of the tokens/patches for prediction
        if self.avg_pooling:
            head_input = x[:, 1:].mean(dim=1)
        else:
            head_input = cls_output

        # Final classification
        return self.head(head_input)  # (B, 10)


# -----------------------------
# Text Transformer Decoder
# -----------------------------
class TransformerDecoder(nn.Module):
    """
    Transformer decoder that generates digit sequences from image representations.
    """
    def __init__(self,
                 vocab_size,
                 max_len,
                 embed_dim=64,
                 num_heads=4,
                 mlp_dim=128,
                 num_layers=4,
                 dropout=0.0):
        super().__init__()

        # Embeds digits
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)

        # Positional encoding for target sequence
        # TODO Replace with BERT-style cos/sin embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))

        # Transformer decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])

        # Predicts logits over digits
        self.output_proj = nn.Linear(embed_dim, vocab_size)

        # Initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, target, enc_out):
        # target: (B, T), enc_out: (B, N, D)

        B, T = target.shape

        # Add token and positional embeddings
        x = self.embed_tokens(target) + self.pos_embed[:, :T]

        # Causal mask: allow attention only to current and previous tokens
        mask = torch.tril(torch.ones(T, T, device=target.device)).unsqueeze(0).unsqueeze(1)  # (1, 1, T, T)

        for layer in self.layers:
            x = layer(x, enc_out, mask)

        return self.output_proj(x)  # (B, T, vocab_size)


# -----------------------------
# Final Vision-to-Sequence Model
# -----------------------------
class VisionToSequence(nn.Module):
    """
    Combines ViT encoder + transformer decoder for digit sequence prediction.
    """
    def __init__(self,
                 patch_dim=196,
                 vocab_size=13,      # 0–9 digits + pad, start and end tokens
                 max_len=6,          # Max digits in image. TODO Do we need this?
                 embed_dim=64,
                 num_heads=4,
                 mlp_dim=128,
                 num_layers_encoder=6,
                 num_layers_decoder=4,
                 num_patches=16):
        super().__init__()

        self.encoder = ViTEncoder(
            patch_dim=patch_dim,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers_encoder,
            num_patches=num_patches
        )

        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            max_len=max_len,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            num_layers=num_layers_decoder
        )

    def forward(self, x, target_seq):
        """
        x: (B, N, patch_dim) - input image patches
        target_seq: (B, T) - target digit sequence (teacher-forced during training)
        """
        enc_out = self.encoder(x)                # (B, N, D)
        logits = self.decoder(target_seq, enc_out)  # (B, T, vocab_size)
        return logits
    
    def encode(self, img_patches):
        return self.encoder(img_patches)
    
    def decode(self, enc_out, seq):
        return self.decoder(seq, enc_out)
