import math

import torch
import torch.nn as nn


PE_MAX_LEN = 64  # max length of pe, i.e. max number of patches we expect from an image


# see disection with o3: https://chatgpt.com/share/685a8e42-8f04-8009-b87a-e30b6fbe56b5
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim: int, max_len: int = PE_MAX_LEN):
        super().__init__()

        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, model_dim, 2) * -(math.log(10_000.0) / model_dim)
        )
        broadcast = position * div_term
        pe[:, 0::2] = torch.sin(broadcast)
        pe[:, 1::2] = torch.cos(broadcast)
        pe = pe.unsqueeze(0)  # add batch dimension
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]  # type: ignore


class Patchify(nn.Module):
    # think of each patch as an image token (i.e. as a word, if this was NLP)
    def __init__(self, patch_size: int, model_dim: int):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(patch_size**2, model_dim, bias=False)

    def forward(self, x):
        patches = self.unfold(x)
        rotated = patches.permute(0, 2, 1)
        return self.linear(rotated)


def attention(k_dim, q, k, v):
    kt = k.permute(0, 1, 3, 2)

    # do attention(Q, K, V) = softmax(Q·K^T / sqrt(dims))·V to get hidden state (where · is dot product)
    attn_dot_product = torch.matmul(q, kt)
    attn_scaled = attn_dot_product / math.sqrt(k_dim)
    attn_probs = torch.softmax(attn_scaled, dim=1)
    return torch.matmul(attn_probs, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.k_dim = model_dim // num_heads
        self.wqkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.endmulti = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(self.model_dim, dim=-1)
        qh = q.reshape(B, self.num_heads, L, self.k_dim)
        kh = k.reshape(B, self.num_heads, L, self.k_dim)
        vh = v.reshape(B, self.num_heads, L, self.k_dim)
        attended = attention(self.k_dim, qh, kh, vh)
        concatted = attended.transpose(1, 2).reshape(B, L, self.model_dim)
        return self.endmulti(concatted)


class FeedForward(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int):
        super().__init__()
        self.sequence = nn.Sequential(
            nn.Linear(model_dim, ffn_dim, bias=True),
            nn.ReLU(),
            nn.Linear(ffn_dim, model_dim, bias=True),
        )

    def forward(self, x):
        return self.sequence(x)


class Encoder(nn.Module):
    def __init__(
        self,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
    ):
        super().__init__()
        self.mha = MultiHeadAttention(model_dim=model_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, ffn_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        mhead = self.mha(x)
        addnormed = self.norm1(x + mhead)

        # pass attention output through feed-forward sub-layer (basic MLP)
        ffned = self.ffn(addnormed)
        return self.norm2(addnormed + ffned)


class BaseClassifier(nn.Module):
    def __init__(
        self,
        patch_size: int,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_encoders: int,
        use_pe: bool,
    ):
        super().__init__()
        self.patchify = Patchify(patch_size, model_dim)
        self.use_pe = use_pe
        self.pe = PositionalEncoding(model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))
        # here, 'multi-head dot-product self attention blocks [...] completely replace convolutions' (see 16x16)
        self.encoders = nn.ModuleList(
            [
                Encoder(
                    model_dim=model_dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                )
                for _ in range(num_encoders)
            ]
        )

    def forward(self, x):
        B = x.size(0)
        patched = self.patchify(x)
        D = patched.size(2)
        if self.use_pe:
            patched = self.pe(patched)
        cls_expanded = self.cls_token.expand(B, 1, D)
        withcls = torch.cat([cls_expanded, patched], dim=1)
        for encoder in self.encoders:
            withcls = encoder(withcls)
        return withcls


class Classifier(BaseClassifier):
    def __init__(
        self,
        patch_size: int = 14,
        model_dim: int = 384,
        ffn_dim: int = 64,
        num_heads: int = 4,
        num_encoders: int = 3,
        use_pe: bool = True,
    ):
        super().__init__(
            patch_size=patch_size,
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_encoders=num_encoders,
            use_pe=use_pe,
        )
        self.linear = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 10),
        )

    def forward(self, x):
        base = super().forward(x)
        return self.linear(base).mean(dim=1)


class Decoder(nn.Module):
    def __init__(self, tfeatures, model_dim):
        super().__init__()
        self.wq = nn.Linear(tfeatures, 24, bias=False)
        self.wk = nn.Linear(model_dim, 24, bias=False)
        self.wv = nn.Linear(model_dim, 24, bias=False)
        self.wh = nn.Linear(24, tfeatures, bias=False)

    def forward(self, encoded, text):
        q = self.wq(text)
        k = self.wk(encoded)
        v = self.wv(encoded)
        kt = k.permute(0, 2, 1)
        a = torch.matmul(q, kt)
        a = a / math.sqrt(24)
        a = torch.softmax(a, dim=1)
        h1 = torch.matmul(a, v)
        return self.wh(h1)


class Predictor(BaseClassifier):
    def __init__(
        self,
        patch_size: int = 14,
        model_dim: int = 384,
        ffn_dim: int = 64,
        tfeatures: int = 11,
        num_heads: int = 4,
        num_encoders: int = 3,
        use_pe: bool = True,
    ):
        super().__init__(
            patch_size=patch_size,
            model_dim=model_dim,
            num_encoders=num_encoders,
            use_pe=use_pe,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
        )
        self.decoders = nn.ModuleList(
            [Decoder(tfeatures, model_dim) for _ in range(num_encoders)]
        )
        self.linear = nn.Linear(tfeatures, 13)

    def forward(self, x, input_seqs):
        encoded = super().forward(x)
        x = input_seqs
        for decoder in self.decoders:
            x = decoder(encoded, x)
        return self.linear(x).mean(dim=1)
