import math
import torch
import torch.nn as nn

VOCAB_SIZE = 13  # digits 0-9, plus start, finish, pad
PE_MAX_LEN = 64  # max length of pe, i.e. max number of patches we expect from an image


# TODO: experiment with making the PE learnable
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


def attention(k_dim, q, k, v, mask_tensor):
    kt = k.transpose(-2, -1)
    # do attention(Q, K, V) = softmax(Q·K^T / sqrt(dims))·V to get hidden state (where · is dot product)
    attn_dot_product = torch.matmul(q, kt)
    attn_scaled = attn_dot_product / math.sqrt(k_dim)
    if mask_tensor is not None:
        attn_scaled = attn_scaled.masked_fill(mask_tensor, -torch.inf)
    attn_probs = torch.softmax(attn_scaled, dim=-1)
    return torch.matmul(attn_probs, v)


class SelfAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, mask: bool):
        super().__init__()
        self.num_heads = num_heads
        self.model_dim = model_dim
        self.k_dim = model_dim // num_heads
        self.wqkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.endmulti = nn.Linear(model_dim, model_dim, bias=False)
        self.mask = mask

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.wqkv(x)
        q, k, v = qkv.split(self.model_dim, dim=-1)
        qh = q.reshape(B, self.num_heads, L, self.k_dim)
        kh = k.reshape(B, self.num_heads, L, self.k_dim)
        vh = v.reshape(B, self.num_heads, L, self.k_dim)
        mask_tensor = None
        if self.mask:
            mask_tensor = torch.triu(
                torch.ones(L, L, device=x.device), diagonal=1
            ).bool()

        attended = attention(self.k_dim, qh, kh, vh, mask_tensor=mask_tensor)
        concatted = attended.transpose(1, 2).reshape(B, L, self.model_dim)
        return self.endmulti(concatted)


class CrossAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.k_dim = model_dim // num_heads
        self.wq = nn.Linear(model_dim, model_dim, bias=False)
        self.wk = nn.Linear(model_dim, model_dim, bias=False)
        self.wv = nn.Linear(model_dim, model_dim, bias=False)
        self.endmulti = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, images, texts):
        B_image, L_image, D_image = images.shape
        B_text, L_text, D_text = texts.shape
        q = self.wq(texts)
        k = self.wk(images)
        v = self.wv(images)
        qh = q.reshape(B_text, self.num_heads, L_text, self.k_dim)
        kh = k.reshape(B_image, self.num_heads, L_image, self.k_dim)
        vh = v.reshape(B_image, self.num_heads, L_image, self.k_dim)
        attended = attention(self.k_dim, qh, kh, vh, mask_tensor=None)
        concatted = attended.transpose(1, 2).reshape(B_text, L_text, self.model_dim)
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
        # here, 'multi-head dot-product self attention blocks [...] completely replace convolutions' (see 16x16)
        self.mha = SelfAttention(model_dim=model_dim, num_heads=num_heads, mask=False)
        self.norm1 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim, ffn_dim)
        self.norm2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        mhead = self.mha(x)
        addnormed = self.norm1(x + mhead)

        # pass attention output through feed-forward sub-layer (basic MLP)
        ffned = self.ffn(addnormed)
        return self.norm2(addnormed + ffned)


class Decoder(nn.Module):
    def __init__(self, model_dim: int, ffn_dim: int, num_heads: int):
        super().__init__()
        self.masked_self_mha = SelfAttention(
            model_dim=model_dim, num_heads=num_heads, mask=True
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.cross_mha = CrossAttention(model_dim=model_dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(model_dim)
        self.ffn = FeedForward(model_dim=model_dim, ffn_dim=ffn_dim)
        self.norm3 = nn.LayerNorm(model_dim)

    def forward(self, images, text):
        stage1 = self.masked_self_mha(text)
        addnormed_text = self.norm1(text + stage1)
        stage2 = self.cross_mha(images, addnormed_text)
        addnormed_stage2 = self.norm2(addnormed_text + stage2)
        ffned = self.ffn(addnormed_stage2)
        return self.norm3(addnormed_stage2 + ffned)


class BaseTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_encoders: int,
        use_cls: bool,
    ):
        super().__init__()
        self.patchify = Patchify(patch_size, model_dim)
        self.use_cls = use_cls
        self.pe = PositionalEncoding(model_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, model_dim))

        def make_encoder() -> nn.Module:
            return Encoder(model_dim=model_dim, num_heads=num_heads, ffn_dim=ffn_dim)

        self.encoder_series = nn.ModuleList(
            [make_encoder() for _ in range(num_encoders)]
        )

    def forward(self, x):
        B = x.size(0)
        patched = self.patchify(x)
        D = patched.size(-1)
        out = self.pe(patched)
        if self.use_cls:
            cls_expanded = self.cls_token.expand(B, 1, D)
            out = torch.cat([cls_expanded, out], dim=1)
        for encoder in self.encoder_series:
            out = encoder(out)
        return out


class VitTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_encoders: int,
    ):
        super().__init__()
        self.base_transformer = BaseTransformer(
            patch_size=patch_size,
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_encoders=num_encoders,
            use_cls=True,
        )
        self.linear = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, model_dim),
            nn.GELU(),
            nn.Linear(model_dim, 10),
        )

    def forward(self, x):
        base = self.base_transformer(x)
        cls = base[:, 0, :]
        return self.linear(cls)


class ComplexTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        model_dim: int,
        ffn_dim: int,
        num_heads: int,
        num_coders: int,
    ):
        super().__init__()
        self.base_transformer = BaseTransformer(
            patch_size=patch_size,
            model_dim=model_dim,
            ffn_dim=ffn_dim,
            num_heads=num_heads,
            num_encoders=num_coders,
            use_cls=False,
        )
        self.embedding = nn.Embedding(
            num_embeddings=VOCAB_SIZE, embedding_dim=model_dim
        )

        def make_decoder() -> nn.Module:
            return Decoder(model_dim=model_dim, ffn_dim=ffn_dim, num_heads=num_heads)

        self.decoder_series = nn.ModuleList([make_decoder() for _ in range(num_coders)])

        self.linear = nn.Linear(model_dim, VOCAB_SIZE)

    def forward(self, images, input_seqs):
        encoded = self.base_transformer(images)
        embedded = self.embedding(input_seqs)
        for decoder in self.decoder_series:
            embedded = decoder(encoded, embedded)
        return self.linear(embedded)
