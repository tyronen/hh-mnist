import math

import torch
import torch.nn as nn
import math


class Patchify(nn.Module):
    def __init__(self, patch_size=14, model_dim=64):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(patch_size**2, model_dim, bias=False)

    def forward(self, x):
        patches = self.unfold(x)
        rotated = patches.permute(0, 2, 1)
        return self.linear(rotated)


# TODO: use multi-head attention (currently have single head)
class Encoder(nn.Module):
    def __init__(self, model_dim=64):
        super().__init__()
        self.model_dim = model_dim
        self.wq = nn.Linear(model_dim, 24, bias=False)
        self.wk = nn.Linear(model_dim, 24, bias=False)
        self.wv = nn.Linear(model_dim, 32, bias=False)
        # TODO: extend wh into full 'feed-forward network' (i.e. two linear layers with ReLU in between)
        self.wh = nn.Linear(32, model_dim, bias=False)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        kt = k.permute(0, 2, 1)

        # do attention(Q, K, V) = softmax(QK^T / sqrt(dims)) · V
        attn_dot_product = torch.matmul(q, kt)
        attn_scaled = attn_dot_product / math.sqrt(self.model_dim)
        attn_probs = torch.softmax(attn_scaled, dim=1)

        h1 = torch.matmul(attn_probs, v)
        return self.wh(h1)


class BaseClassifier(nn.Module):
    def __init__(self, patch_size=14, model_dim=64, num_encoders=6):
        super().__init__()
        self.patchify = Patchify(patch_size, model_dim)
        self.encoders = nn.ModuleList([Encoder(model_dim) for _ in range(num_encoders)])

    def forward(self, x):
        patched = self.patchify(x)
        for encoder in self.encoders:
            patched = encoder(patched)
        return patched


class Classifier(BaseClassifier):
    def __init__(self, num_encoders=6, patch_size=14, features=64):
        super().__init__(num_encoders, patch_size, features)
        self.linear = nn.Linear(features, 10)

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
        super().__init__(num_encoders=num_coders, patch_size=patch_size, features=efeatures)
        self.decoders = nn.ModuleList([Decoder(tfeatures, efeatures) for _ in range(num_coders)])
        self.linear = nn.Linear(tfeatures, 13)

    def forward(self, images, input_seqs):
        encoded = super().forward(images)
        x = input_seqs
        for decoder in self.decoders:
            x = decoder(encoded, x)
        return self.linear(x).mean(dim=1)
