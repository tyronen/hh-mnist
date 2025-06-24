import math

import torch
import torch.nn as nn


PE_MAX_LEN = 128  # max length of positional encoding, i.e. max number of patches from an image
K_DIM = 24
V_DIM = 32


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=PE_MAX_LEN):
        super().__init__()

        self.pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * -(math.log(10_000.0) / model_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # add batch dimension

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Patchify(nn.Module):
    # think of each patch as an image token (i.e. as a word, if this was NLP)
    def __init__(self, patch_size=7, model_dim=64):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(patch_size**2, model_dim, bias=False)

    def forward(self, x):
        patches = self.unfold(x)
        rotated = patches.permute(0, 2, 1)
        return self.linear(rotated)


class Encoder(nn.Module):
    def __init__(self, model_dim=64, ffn_dim=64):
        super().__init__()
        self.model_dim = model_dim
        self.wq = nn.Linear(model_dim, K_DIM, bias=False)
        self.wk = nn.Linear(model_dim, K_DIM, bias=False)
        self.wv = nn.Linear(model_dim, V_DIM, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(V_DIM, ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim, model_dim),
        )

    # TODO: add residual connection and layer normalization logic
    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        kt = k.permute(0, 2, 1)

        # do attention(Q, K, V) = softmax(Q·K^T / sqrt(dims))·V to get hidden state (where · is dot product)
        attn_dot_product = torch.matmul(q, kt)
        attn_scaled = attn_dot_product / math.sqrt(K_DIM)
        attn_probs = torch.softmax(attn_scaled, dim=1)
        hidden = torch.matmul(attn_probs, v)

        # pass attention output through feed-forward sub-layer (basic MLP)
        return self.ffn(hidden)


class BaseClassifier(nn.Module):
    def __init__(
        self,
        patch_size: int = 7,
        model_dim: int = 64,
        num_encoders: int = 6,
        use_pe: bool = True,
    ):
        super().__init__()
        self.patchify = Patchify(patch_size, model_dim)
        self.use_pe = use_pe
        self.pe = PositionalEncoding(model_dim)
        # here, 'multi-head dot-product self attention blocks [...] completely replace convolutions' (see 16x16)
        # TODO: use multi-head attention (currently have single head)
        self.encoders = nn.ModuleList([Encoder(model_dim) for _ in range(num_encoders)])

    def forward(self, x):
        patched = self.patchify(x)
        if self.use_pe:
            patched = self.pe(patched)
        for encoder in self.encoders:
            patched = encoder(patched)
        return patched


class Classifier(BaseClassifier):
    def __init__(
        self,
        patch_size: int = 7,
        model_dim: int = 64,
        num_encoders: int = 6,
        use_pe: bool = True,
    ):
        super().__init__(num_encoders, patch_size, model_dim, use_pe)
        self.linear = nn.Linear(model_dim, 10)

    def forward(self, x):
        base = super().forward(x)
        return self.linear(base).mean(dim=1)


class Decoder(nn.Module):
    def __init__(self, tfeatures, efeatures):
        super().__init__()
        self.wq = nn.Linear(tfeatures, 24, bias=False)
        self.wk = nn.Linear(efeatures, 24, bias=False)
        self.wv = nn.Linear(efeatures, 24, bias=False)
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
    def __init__(self, num_coders=6, patch_size=14, tfeatures=11, efeatures=64):
        super().__init__(num_encoders=num_coders, patch_size=patch_size, model_dim=efeatures)
        self.decoders = nn.ModuleList([Decoder(tfeatures, efeatures) for _ in range(num_coders)])
        self.linear = nn.Linear(tfeatures, 13)

    def forward(self, images, input_seqs):
        encoded = super().forward(images)
        x = input_seqs
        for decoder in self.decoders:
            x = decoder(encoded, x)
        return self.linear(x).mean(dim=1)
