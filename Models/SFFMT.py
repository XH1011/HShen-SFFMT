import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import random

# 获取项目根目录（论文目录）的绝对路径
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# # 将根目录添加到sys.path
# sys.path.append(project_root)
import sys
sys.path.append('/hy-tmp')  # 手动加入 hy-tmp 的路径
from SFFMT_utils.attention_block import Flatten
from SFFMT_utils.attention_block import Flatten, ECA_Layer, CoordAtt


class MixFeature(nn.Module):

    def __init__(self, p=0.5, alpha=0.5, eps=1e-6):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha

    def forward(self, x):
        if not self.training:
            return x

        if random.random() > self.p:
            return x

        batch_size = x.size(0)

        mu = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        sigma = (var + self.eps).sqrt()
        mu, sigma = mu.detach(), sigma.detach()
        x_normed = (x - mu) / sigma

        interpolation = self.beta.sample((batch_size, 1, 1))
        interpolation = interpolation.to(x.device)

        # split into two halves and swap the order
        perm = torch.arange(batch_size - 1, -1, -1)  # inverse index
        perm_b, perm_a = perm.chunk(2)
        perm_b = perm_b[torch.randperm(batch_size // 2)]
        perm_a = perm_a[torch.randperm(batch_size // 2)]
        perm = torch.cat([perm_b, perm_a], 0)

        mu_perm, sigma_perm = mu[perm], sigma[perm]
        mu_mix = mu * interpolation + mu_perm * (1 - interpolation)
        sigma_mix = sigma * interpolation + sigma_perm * (1 - interpolation)

        return x_normed * sigma_mix + mu_mix

def conv_layer(channel, kernel, use_mixfeature=False, p=0.5, alpha=0.5):
    layers = [
        nn.Conv1d(in_channels=channel[0], out_channels=channel[1], kernel_size=kernel, padding=kernel // 2),
        nn.BatchNorm1d(channel[1]),
        nn.ReLU()
    ]

    if use_mixfeature:
        layers.append(MixFeature(p=p, alpha=alpha))

    return nn.Sequential(*layers)

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_mixfeature=False, p=0.5, alpha=0.5):
        super(ResidualBlock1D, self).__init__()
        self.use_shortcut = in_channels != out_channels
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )

        self.mix = MixFeature(p=p, alpha=alpha) if use_mixfeature else None

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        if self.mix is not None:
            out = self.mix(out)  # 放在主分支中，避免扰动 shortcut
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

def residual_layer(channel, kernel, use_mixfeature=False, p=0.5, alpha=0.5):
    return ResidualBlock1D(channel[0], channel[1], kernel, use_mixfeature, p, alpha)

# ECA layer
class att_layer(nn.Module):
    def __init__(self, k=3):
        super(att_layer, self).__init__()
        self.att = ECA_Layer(kernel_size=k)

    def forward(self, x):
        out = self.att(x)
        return out + x


class SFFMT(nn.Module):
    def __init__(self):
        super(SFFMT, self).__init__()
        # initialise network parameters
        filters = [64, 64, 128]
        self.class_1 = 5
        self.class_2 = 3

        # define first layers of every encoder block
        self.encoder_block_1 = nn.ModuleList([residual_layer([1, 64], 7, use_mixfeature=True, p=0.3)])#, use_mixfeature=True, p=0.3
        self.encoder_block_1.append(residual_layer([64, 64], 5))
        self.encoder_block_1.append(residual_layer([64, 128], 3))

        # define second layers of every encoder block
        self.encoder_block_2 = nn.ModuleList([residual_layer([64, 64], 7, use_mixfeature=True, p=0.3)])
        self.encoder_block_2.append(residual_layer([64, 64], 5))
        self.encoder_block_2.append(residual_layer([128, 128], 3))

        # define task attention layers
        self.encoder_att = nn.ModuleList([nn.ModuleList([att_layer(3)])])
        for j in range(2):
            if j < 1:
                self.encoder_att.append(nn.ModuleList([att_layer(3)]))
            for i in range(2):
                self.encoder_att[j].append(att_layer(3))


        # define task conv layers
        self.encoder_att_conv = nn.ModuleList([conv_layer([64, 64], 1)])
        for i in range(2):
            if i == 0:
                self.encoder_att_conv.append(conv_layer([filters[i + 1], filters[i + 2]], 1))
            else:
                self.encoder_att_conv.append(conv_layer([filters[i + 1], 2 * filters[i + 1]], 1))

        # define pooling function
        self.maxpool = nn.MaxPool1d(kernel_size=4, stride=4)

        # define fc layers
        self.task1_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            nn.Linear(256, self.class_1)
        )
        self.task2_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Flatten(),
            # nn.Dropout(0.2),
            # nn.Linear(128, 32),
            # nn.ReLU(),
            nn.Linear(256, self.class_2)
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        g_encoder, g_maxpool = ([0] * 3 for _ in range(2))
        for i in range(3):
            g_encoder[i] = [0] * 2

        # def attention list for tasks
        atten_encoder = [0, 0]
        for j in range(2):
            atten_encoder[j] = [0] * 3
        for i in range(2):
            for k in range(3):
                atten_encoder[i][k] = [0] * 3

        # define global shared network
        for i in range(3):
            if i == 0:
                g_encoder[i][0] = self.encoder_block_1[i](x)
                g_encoder[i][1] = self.encoder_block_2[i](g_encoder[i][0])
                g_maxpool[i] = self.maxpool(g_encoder[i][1])
            elif i == 1:
                g_encoder[i][0] = self.encoder_block_1[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.encoder_block_2[i](g_encoder[i][0])
                g_maxpool[i] = self.maxpool(g_encoder[i][1])
            else:
                g_encoder[i][0] = self.encoder_block_1[i](g_maxpool[i - 1])
                g_encoder[i][1] = self.encoder_block_2[i](g_encoder[i][0])

        # define task dependent module
        for i in range(2):
            for j in range(3):
                if j == 0:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][1])
                    atten_encoder[i][j][1] = self.encoder_att_conv[j](atten_encoder[i][j][0])
                    atten_encoder[i][j][2] = F.max_pool1d(atten_encoder[i][j][1], kernel_size=4, stride=4)
                elif j == 1:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][1] + atten_encoder[i][j - 1][2])
                    atten_encoder[i][j][1] = self.encoder_att_conv[j](atten_encoder[i][j][0])
                    atten_encoder[i][j][2] = F.max_pool1d(atten_encoder[i][j][1], kernel_size=4, stride=4)
                else:
                    atten_encoder[i][j][0] = self.encoder_att[i][j](g_encoder[j][1] + atten_encoder[i][j - 1][2])
                    atten_encoder[i][j][1] = self.encoder_att_conv[j](atten_encoder[i][j][0])
                    atten_encoder[i][j][2] = F.max_pool1d(atten_encoder[i][j][1], kernel_size=2, stride=2)

        # define task prediction layers
        t1_pred = self.task1_fc(atten_encoder[0][-1][-1])
        t2_pred = self.task2_fc(atten_encoder[1][-1][-1])
        return t1_pred, t2_pred


if __name__ == '__main__':
    t = torch.randn(50, 1, 2048)
    Net = SFFMT()
    t = Net(t)
    print(Net)

