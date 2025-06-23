import torch
import torch.nn as nn


class Patchify(nn.Module):
    def __init__(self, patch_size, features):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.linear = nn.Linear(patch_size**2, features, bias=False)

    def forward(self, x):
        patches = self.unfold(x)
        rotated = patches.permute(0, 2, 1)
        return self.linear(rotated)


class Encoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.wq = nn.Linear(features, 24, bias=False)
        self.wk = nn.Linear(features, 24, bias=False)
        self.wv = nn.Linear(features, 32, bias=False)
        self.wh = nn.Linear(32, features, bias=False)

    def forward(self, x):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)
        kt = k.permute(0, 2, 1)
        a = torch.matmul(q, kt)
        a = torch.softmax(a, dim=1)
        h1 = torch.matmul(a, v)
        return self.wh(h1)


class BaseClassifier(nn.Module):
    def __init__(self, num_encoders, patch_size, features):
        super().__init__()
        self.patchify = Patchify(patch_size, features)
        self.encoders = nn.ModuleList([Encoder(features) for _ in range(num_encoders)])

    def forward(self, x):
        patched = self.patchify(x)
        for encoder in self.encoders:
            patched = encoder(patched)
        return patched


class Classifier(BaseClassifier):
    def __init__(self, num_encoders=6, patch_size=14, features=64):
        super().__init__(num_encoders, patch_size, features)
        self.linear = nn.Linear(features, 10)

    def forward(self, images, input_seqs):
        base = super().forward(images)
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
        a = torch.softmax(a, dim=1)
        h1 = torch.matmul(a, v)
        return self.wh(h1)


class Predictor(BaseClassifier):
    def __init__(self, num_coders=6, patch_size=14, tfeatures=11, efeatures=64):
        super().__init__(
            num_encoders=num_coders, patch_size=patch_size, features=efeatures
        )
        self.decoders = nn.ModuleList(
            [Decoder(tfeatures, efeatures) for _ in range(num_coders)]
        )
        self.linear = nn.Linear(tfeatures, 13)

    def forward(self, images, input_seqs):
        encoded = super().forward(images)
        x = input_seqs
        for decoder in self.decoders:
            x = decoder(encoded, x)
        return self.linear(x).mean(dim=1)
