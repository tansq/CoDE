import torch.nn as nn
import torch
from efficientnet_pytorch import EfficientNet



#-----EffinientNet Components-----#
class EfficientNet_Encoder(nn.Module):
    def __init__(self, efn, start, end):
        super(EfficientNet_Encoder, self).__init__()
        self.blocks = efn._blocks[start:end]

    def forward(self, x):
        for idx, block in enumerate(self.blocks):
            x = block(x)
        return x

class AC_CoDE(nn.Module):
    def __init__(self, encoder='efficientnet-b0', pretrained=True):
        super(AC_CoDE, self).__init__()
        #_/_/_/ Feature Extractor _/_/_/
        if pretrained:
            efn = EfficientNet.from_pretrained(encoder)
        else:
            efn = EfficientNet.from_name(encoder)

        efn_params = {
            'efficientnet-b0': {'filters': [32, 24, 40, 80, 192], 'ends': [3, 5, 8, 15]},
            'efficientnet-b1': {'filters': [32, 24, 40, 80, 192], 'ends': [5, 8, 12, 21]},
            'efficientnet-b2': {'filters': [32, 24, 48, 88, 208], 'ends': [5, 8, 12, 21]},
            'efficientnet-b3': {'filters': [40, 32, 48, 96, 232], 'ends': [5, 8, 13, 24]},
            'efficientnet-b4': {'filters': [48, 32, 56, 112, 272], 'ends': [6, 10, 16, 30]},
            'efficientnet-b5': {'filters': [48, 40, 64, 128, 304], 'ends': [8, 13, 20, 36]},
            'efficientnet-b6': {'filters': [56, 40, 72, 144, 344], 'ends': [9, 15, 23, 42]},
            'efficientnet-b7': {'filters': [64, 48, 80, 160, 384], 'ends': [11, 18, 28, 51]},
        }

        # feature extractor
        self.conv_stem = nn.Sequential(
            efn._conv_stem,
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
        )

        self.down1 = EfficientNet_Encoder(efn, start=0,
                                          end=efn_params[encoder]['ends'][0])
        self.down2 = EfficientNet_Encoder(efn, start=efn_params[encoder]['ends'][0],
                                          end=efn_params[encoder]['ends'][1])
        self.down3 = EfficientNet_Encoder(efn, start=efn_params[encoder]['ends'][1],
                                          end=efn_params[encoder]['ends'][2])
        self.down4 = EfficientNet_Encoder(efn, start=efn_params[encoder]['ends'][2],
                                          end=efn_params[encoder]['ends'][3])

        # policy network
        self.p_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][4], efn_params[encoder]['filters'][3], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][3]),
            nn.ReLU(inplace=True),
        )
        self.p_merge1 = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][3]*2, efn_params[encoder]['filters'][3], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][3], efn_params[encoder]['filters'][3], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][3]),
            nn.ReLU(inplace=True),
        )

        self.p_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][3], efn_params[encoder]['filters'][2], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][2]),
            nn.ReLU(inplace=True),
        )
        self.p_merge2 = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][2] * 2, efn_params[encoder]['filters'][2], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][2], efn_params[encoder]['filters'][2], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][2]),
            nn.ReLU(inplace=True),
        )

        self.p_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][2], efn_params[encoder]['filters'][1], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][1]),
            nn.ReLU(inplace=True),
        )
        self.p_merge3 = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][1] * 2, efn_params[encoder]['filters'][1], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][1], efn_params[encoder]['filters'][1], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][1]),
            nn.ReLU(inplace=True),
        )

        self.p_up4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][1], efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
        )
        self.p_merge4 = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][0] * 2, efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][0], efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
        )

        # policy mask conv
        self.p_mask_out = nn.Sequential(
            nn.Conv2d(1, efn_params[encoder]['filters'][0], 1, bias=False),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][0], efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][0], efn_params[encoder]['filters'][0], 3, 2, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
        )

        self.mu = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][0]*2, efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][0], efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][0], 1, 3, 1, 1),
            nn.Softsign(),
        )
        self.sigma = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][0] * 2, efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][0], efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][0], 1, 3, 1, 1),
            nn.Softplus(),
        )

        # value network
        self.v_up1 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][4], efn_params[encoder]['filters'][3], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][3]),
            nn.ReLU(inplace=True),
        )
        self.v_merge1 = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][3] * 2, efn_params[encoder]['filters'][3], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][3], efn_params[encoder]['filters'][3], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][3]),
            nn.ReLU(inplace=True),
        )

        self.v_up2 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][3], efn_params[encoder]['filters'][2], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][2]),
            nn.ReLU(inplace=True),
        )
        self.v_merge2 = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][2] * 2, efn_params[encoder]['filters'][2], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][2], efn_params[encoder]['filters'][2], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][2]),
            nn.ReLU(inplace=True),
        )

        self.v_up3 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][2], efn_params[encoder]['filters'][1], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][1]),
            nn.ReLU(inplace=True),
        )
        self.v_merge3 = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][1] * 2, efn_params[encoder]['filters'][1], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][1], efn_params[encoder]['filters'][1], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][1]),
            nn.ReLU(inplace=True),
        )

        self.v_up4 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][1], efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
        )
        self.v_merge4 = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][0] * 2, efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][0], efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
        )

        # policy mask conv
        self.v_mask_out = nn.Sequential(
            nn.Conv2d(1, efn_params[encoder]['filters'][0], 1, bias=False),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][0], efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(efn_params[encoder]['filters'][0], efn_params[encoder]['filters'][0], 3, 2, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
        )

        self.v_head = nn.Sequential(
            nn.Conv2d(efn_params[encoder]['filters'][0] * 2, efn_params[encoder]['filters'][0], 3, 1, 1),
            nn.BatchNorm2d(efn_params[encoder]['filters'][0]),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(efn_params[encoder]['filters'][0], 1, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forgery_forward(self, x):
        x1 = self.conv_stem(x[:, :-1, :, :])
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        p1 = self.p_up1(d4)
        p1 = self.p_merge1(torch.cat((p1, d3), dim=1))
        p2 = self.p_up2(p1)
        p2 = self.p_merge2(torch.cat((p2, d2), dim=1))
        p3 = self.p_up3(p2)
        p3 = self.p_merge3(torch.cat((p3, d1), dim=1))
        p4 = self.p_up4(p3)
        p4 = self.p_merge4(torch.cat((p4, x1), dim=1))

        v1 = self.v_up1(d4)
        v1 = self.v_merge1(torch.cat((v1, d3), dim=1))
        v2 = self.v_up2(v1)
        v2 = self.v_merge2(torch.cat((v2, d2), dim=1))
        v3 = self.v_up3(v2)
        v3 = self.v_merge3(torch.cat((v3, d1), dim=1))
        v4 = self.v_up4(v3)
        v4 = self.v_merge4(torch.cat((v4, x1), dim=1))
        return p4, v4

    def prob_forward(self, x, p, v):
        p_mask_out = self.p_mask_out(x[:, -1:, :, :])
        mu = self.mu(torch.cat((p, p_mask_out), dim=1))
        sigma = self.sigma(torch.cat((p, p_mask_out), dim=1))

        v_mask_out = self.v_mask_out(x[:, -1:, :, :])
        v_out = self.v_head(torch.cat((v, v_mask_out), dim=1))
        return mu, sigma, v_out

    def pi_and_v(self, x):
        x1 = self.conv_stem(x[:, :-1, :, :])
        d1 = self.down1(x1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)

        p1 = self.p_up1(d4)
        p1 = self.p_merge1(torch.cat((p1, d3), dim=1))
        p2 = self.p_up2(p1)
        p2 = self.p_merge2(torch.cat((p2, d2), dim=1))
        p3 = self.p_up3(p2)
        p3 = self.p_merge3(torch.cat((p3, d1), dim=1))
        p4 = self.p_up4(p3)
        p4 = self.p_merge4(torch.cat((p4, x1), dim=1))
        p_mask_out = self.p_mask_out(x[:, -1:, :, :])
        mu = self.mu(torch.cat((p4, p_mask_out), dim=1))
        sigma = self.sigma(torch.cat((p4, p_mask_out), dim=1))

        v1 = self.v_up1(d4)
        v1 = self.v_merge1(torch.cat((v1, d3), dim=1))
        v2 = self.v_up2(v1)
        v2 = self.v_merge2(torch.cat((v2, d2), dim=1))
        v3 = self.v_up3(v2)
        v3 = self.v_merge3(torch.cat((v3, d1), dim=1))
        v4 = self.v_up4(v3)
        v4 = self.v_merge4(torch.cat((v4, x1), dim=1))
        v_mask_out = self.v_mask_out(x[:, -1:, :, :])
        v_out = self.v_head(torch.cat((v4, v_mask_out), dim=1))

        return mu, sigma, v_out

