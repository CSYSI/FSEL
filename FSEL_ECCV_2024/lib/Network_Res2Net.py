
import timm
import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.FSEL_modules import DRP_1, DRP_2, DRP_3, JDPM, ETB
from lib.Res2Net_v1b import res2net50_v1b_26w_4s


'''
backbone: Res2Net50
'''


class Network(nn.Module):
    def __init__(self, channels=128):
        super(Network, self).__init__()
        self.shared_encoder = res2net50_v1b_26w_4s()

        pretrained_dict = torch.load('/opt/data/private/Syg/COD/pre_train_pth/res2net50_v1b_26w_4s-3cf99910.pth')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.shared_encoder.state_dict()}
        self.shared_encoder.load_state_dict(pretrained_dict)

        self.dePixelShuffle = torch.nn.PixelShuffle(2)

        self.up = nn.Sequential(
            nn.Conv2d(channels//4, channels, kernel_size=1),nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),nn.BatchNorm2d(channels),nn.ReLU(True)
        )

        self.ETB_5 = ETB(2048+channels, channels)
        self.ETB_4 = ETB(1024+channels, channels)
        self.ETB_3 = ETB(512+channels, channels)
        self.ETB_2 = ETB(256+channels, channels)

        self.JDPM = JDPM(2048, channels)

        self.DRP_1 = DRP_1(channels, channels)
        self.DRP_2 = DRP_2(channels, channels)
        self.DRP_3 = DRP_3(channels,channels)

    def forward(self, x):
        image = x

        x = self.shared_encoder.conv1(x)
        x = self.shared_encoder.bn1(x)
        x = self.shared_encoder.relu(x)
        x = self.shared_encoder.maxpool(x)  # bs, 64, 88, 88
        x1 = self.shared_encoder.layer1(x)  # bs, 256, 88, 88
        x2 = self.shared_encoder.layer2(x1)  # bs, 512, 44, 44
        x3 = self.shared_encoder.layer3(x2)  # bs, 1024, 22, 22
        x4 = self.shared_encoder.layer4(x3)


        p1 = self.JDPM(x4)
        x5_4 = p1
        x5_4_1 = x5_4.expand(-1, 128, -1, -1)

        x4   = self.ETB_5(torch.cat((x4,x5_4_1),1))
        x4_up = self.up(self.dePixelShuffle(x4))


        x3   = self.ETB_4(torch.cat((x3,x4_up),1))
        x3_up = self.up(self.dePixelShuffle(x3))


        x2   = self.ETB_3(torch.cat((x2,x3_up),1))
        x2_up = self.up(self.dePixelShuffle(x2))


        x1   = self.ETB_2(torch.cat((x1,x2_up),1))

        x4 = self.DRP_1(x4,x5_4)
        x3 = self.DRP_1(x3,x4)
        x2 = self.DRP_2(x2,x3,x4)
        x1 = self.DRP_3(x1,x2,x3,x4)

        p0 = F.interpolate(p1, size=image.size()[2:], mode='bilinear', align_corners=True)
        f4 = F.interpolate(x4, size=image.size()[2:], mode='bilinear', align_corners=True)
        f3 = F.interpolate(x3, size=image.size()[2:], mode='bilinear', align_corners=True)
        f2 = F.interpolate(x2, size=image.size()[2:], mode='bilinear', align_corners=True)
        f1 = F.interpolate(x1, size=image.size()[2:], mode='bilinear', align_corners=True)

        return p0, f4, f3, f2, f1


