import torch
import torch.nn as nn
from torch.nn import init
from utils.mobile import Mobile, hswish, MobileDown
from utils.former import Former1 as Former
from utils.bridge import Mobile2Former, Former2Mobile
from utils.config import config_294, config_508, config_52


# from torchsummary import summary


class gernatetokens(nn.Module):
    def __init__(self, in_channel, token_num, token_dim):
        super(gernatetokens, self).__init__()
        self.in_channel = in_channel,
        self.token_num = token_num,
        self.token_dim = token_dim,
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=token_dim, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=token_dim, out_channels=token_dim, kernel_size=3, stride=3)
        self.linear = nn.Linear(in_features=576, out_features=token_num)

    def forward(self, x):
        b, _, _, _ = x.shape

        x = self.conv1(x)
        x = self.conv2(x).reshape(b, 150, 24 * 24)
        x = self.linear(x).transpose(1, 2)
        return x


class BaseBlock(nn.Module):
    def __init__(self, inp, exp, out, se, stride, heads, dim):
        super(BaseBlock, self).__init__()
        if stride == 2:
            self.mobile = MobileDown(3, inp, exp, out, se, stride, dim)
        else:
            self.mobile = Mobile(3, inp, exp, out, se, stride, dim)
        self.mobile2former = Mobile2Former(dim=dim, heads=heads, channel=inp)
        self.former = Former(dim=dim)
        self.former2mobile = Former2Mobile(dim=dim, heads=heads, channel=out)

    def forward(self, inputs):
        # print("Forward")
        x, z = inputs
        z_hid = self.mobile2former(x, z)
        z_out = self.former(z_hid)
        x_hid = self.mobile(x, z_out)
        x_out = self.former2mobile(x_hid, z_out)
        return [x_out, z_out]


class MobileFormer(nn.Module):
    def __init__(self, cfg):
        super(MobileFormer, self).__init__()
        self.gernatetoken = gernatetokens(3, 6, 150)
        # self.token = nn.Parameter(nn.Parameter(torch.randn(1, cfg['token'], cfg['embed'])))
        # stem 3 224 224 -> 16 112 112
        self.stem = nn.Sequential(
            nn.Conv2d(3, cfg['stem'], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(cfg['stem']),
            hswish(),
        )
        # bneck
        self.bneck = nn.Sequential(
            nn.Conv2d(cfg['stem'], cfg['bneck']['e'], 3, stride=cfg['bneck']['s'], padding=1, groups=cfg['stem']),
            hswish(),
            nn.Conv2d(cfg['bneck']['e'], cfg['bneck']['o'], kernel_size=1, stride=1),
            nn.BatchNorm2d(cfg['bneck']['o'])
        )

        # body
        self.block = nn.ModuleList()
        for kwargs in cfg['body']:
            # self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))
            self.block.append(BaseBlock(**kwargs, dim=cfg['embed']))
        inp = cfg['body'][-1]['out']
        exp = cfg['body'][-1]['exp']
        self.conv = nn.Conv2d(inp, exp, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(exp)
        self.avg = nn.AvgPool2d((7, 7))
        self.head = nn.Sequential(
            nn.Linear(exp + cfg['embed'], cfg['fc1']),
            hswish(),
            nn.Linear(cfg['fc1'], cfg['fc2'])
        )
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, _, _, _ = x.shape

        # z = self.token.repeat(b, 1, 1)
        z = self.gernatetoken(x)
        x = self.bneck(self.stem(x))
        for m in self.block:
            x, z = m([x, z])
        # x, z = self.block([x, z])
        x = self.avg(self.bn(self.conv(x))).view(b, -1)
        z = z[:, 0, :].view(b, -1)
        out = torch.cat((x, z), -1)
        return self.head(out)
        # return x, z


def gernate_mf_294():
    return MobileFormer(config_294)


if __name__ == "__main__":
    model = MobileFormer(config_294)
    input = torch.randn(2, 3, 224, 224)
