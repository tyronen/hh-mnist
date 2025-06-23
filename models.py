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


class Classifier(nn.Module):
    def __init__(self, num_encoders=6, patch_size=14, features=64):
        super().__init__()
        self.patchify = Patchify(patch_size, features)
        self.encoders = nn.ModuleList([Encoder(features) for _ in range(num_encoders)])
        self.linear = nn.Linear(features, 10)

    def forward(self, x):
        patched = self.patchify(x)
        for encoder in self.encoders:
            patched = encoder(patched)
        return self.linear(patched).mean(dim=1)
