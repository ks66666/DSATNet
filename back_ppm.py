from module_v8 import *
from resnet_ import *

class Net(nn.Module):
    def __init__(self, num_classes=2, bins=(1, 2, 4, 6)):
        super(Net, self).__init__()
        filters = [64, 128, 256, 512, 1024, 2048]
        Filters = [96, 192, 384, 768, 1536, 3096]
        self.backbone = resnet34(pretrained=True)

        # PPM
        self.ppm = PPM__(512, 512, bins=bins)

        #transformer_attention
        # self.gau1 = GAU(channels=64, embed_size=128, head=32)
        # self.gau2 = GAU(channels=64, embed_size=64, head=16)
        # self.gau3 = GAU(channels=128, embed_size=32, head=8)
        # self.gau4 = GAU(channels=256, embed_size=16, head=4)
        # self.gau5 = GAU(channels=512, embed_size=8, head=2)

        #上采样
        self.up1 = up(64, 32)
        self.up2 = up(filters[0], filters[0])
        self.up3 = up(filters[1], filters[0])
        self.up4 = up(filters[2], filters[1])
        self.up5 = up(filters[3], filters[2])

        # 输出
        self.conv = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x, y):

        x1, x2, x3, x4, x5 = self.backbone(x)
        y1, y2, y3, y4, y5 = self.backbone(y)


        ppm = self. ppm(x5 , y5)

        up5 = self.up5(ppm + y5 + x5)
        up4 = self.up4(up5 + x4 + y4)
        up3 = self.up3(up4 + x3 + y3)
        up2 = self.up2(up3+x2 + y2)
        up1 = self.up1(up2+x1 + y1)

        out = self.conv(up1)
        return out


