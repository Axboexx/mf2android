"""
@Time : 2021/11/8 17:13
@Author : Axboexx
@File : bridge.py
@Software: PyCharm
"""
import torch
from torch import nn, einsum
from einops import rearrange


class Mobile2Former(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Mobile2Former, self).__init__()
        inner_dim = int(heads * channel / 2)
        self.heads = heads
        self.scale = channel ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        z1 = z[:, :int(m / 2), :]
        z2 = z[:, int(m / 2):, :]
        x1 = x[:, :int(c / 2), :, :]
        x2 = x[:, int(c / 2):, :, :]

        b1, m1, d1 = z1.shape
        b1, c1, h1, w1 = x1.shape
        b2, m2, d2 = z2.shape
        b2, c2, h2, w2 = x2.shape

        x1 = x1.reshape(b1, c1, h1 * w1).transpose(1, 2).unsqueeze(1)
        q1 = self.to_q(z1)
        q1 = q1.view(b1, self.heads, m1, c1)
        dots1 = q1 @ x1.transpose(2, 3) * self.scale
        attn1 = self.attend(dots1)
        out1 = attn1 @ x1
        out1 = rearrange(out1, 'b h m c -> b m (h c)')

        x2 = x2.reshape(b2, c2, h2 * w2).transpose(1, 2).unsqueeze(1)
        q2 = self.to_q(z2)
        q2 = q2.view(b2, self.heads, m2, c2)
        dots2 = q2 @ x2.transpose(2, 3) * self.scale
        attn2 = self.attend(dots2)
        out2 = attn2 @ x2
        out2 = rearrange(out2, 'b h m c -> b m (h c)')

        out = torch.cat((out1, out2), dim=1)

        return z + self.to_out(out)


# #M2F test
# m = Mobile2Former(192, 8, 4, 0.2)
# x = torch.randn(2, 4, 224, 224)
# z = torch.randn(2, 6, 192)
# y = m(x, z)
# print(y.shape)

class Former2Mobile(nn.Module):
    def __init__(self, dim, heads, channel, dropout=0.):
        super(Former2Mobile, self).__init__()
        inner_dim = heads * channel
        self.heads = heads
        self.to_k = nn.Linear(dim, inner_dim)
        self.to_v = nn.Linear(dim, inner_dim)
        self.attend = nn.Softmax(dim=-1)
        self.scale = channel ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, channel),
            nn.Dropout(dropout)
        )

    def forward(self, x, z):
        b, m, d = z.shape
        b, c, h, w = x.shape
        q = x.reshape(b, c, h * w).transpose(1, 2).unsqueeze(1)
        k = self.to_k(z).view(b, self.heads, m, c)
        v = self.to_v(z).view(b, self.heads, m, c)
        dots = q @ k.transpose(2, 3) * self.scale
        attn = self.attend(dots)
        out = attn @ v
        out = rearrange(out, 'b h l c -> b l (h c)')
        out = self.to_out(out)
        out = out.view(b, c, h, w)
        return x + out
