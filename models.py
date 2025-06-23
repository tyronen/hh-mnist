import torch
import torch.nn as nn

class Patchify(nn.Module):
    def __init__(self):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=14, stride=14)
        self.linear = nn.Linear(196, 64, bias=False)

    def forward(self, x):
        patches = self.unfold(x)
        rotated = patches.permute(0, 2, 1)
        return self.linear(rotated)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.wq = nn.Linear(64, 24, bias=False)
        self.wk = nn.Linear(64, 24, bias=False)
        self.wv = nn.Linear(64, 32, bias=False)
        self.wh = nn.Linear(32, 64, bias=False)

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
    def __init__(self, num_encoders=6):
        super().__init__()
        self.patchify = Patchify()
        self.encoders = nn.ModuleList([Encoder() for _ in range(num_encoders)])
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        patched = self.patchify(x)
        for encoder in self.encoders:
            patched = encoder(patched)
        return self.linear(patched).mean(dim=1)