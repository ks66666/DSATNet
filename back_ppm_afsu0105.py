import torch
from module_v8 import *
from resnet_ import *
from AFSU import *

class Net(nn.Module):
    def __init__(self, num_classes=2, bins=(1, 2, 4, 6)):
        super(Net, self).__init__()
        filters = [64, 128, 256, 512, 1024, 2048]
        Filters = [16, 32, 64, 128, 256]
        self.backbone = resnet34(pretrained=True)

        #HWD
        self.ppm = PPM__(512, 512, bins=bins)
        # self.DRFD2 = DRFD(filters[0], filters[0])
        # self.DRFD3 = DRFD(filters[0], filters[1])
        # self.DRFD4 = DRFD(filters[1], ·filters[2])
        # self.DRFD5 = DRFD(filters[2], filters[3])
        self.msfm1 = CloMSFM(dim=Filters[0], num_heads=Filters[0], in_cl=Filters[0], out_cl=Filters[0], strde=1, group_split=[Filters[0]//2, Filters[0]//2], kernel_sizes=[3], window_size=8)
        self.msfm2 = CloMSFM(dim=Filters[0], num_heads=Filters[0], in_cl=Filters[0], out_cl=Filters[0], strde=2, group_split=[Filters[0]//2, Filters[0]//2], kernel_sizes=[3], window_size=8)
        self.msfm3 = CloMSFM(dim=Filters[0], num_heads=Filters[0], in_cl=Filters[0], out_cl=Filters[1], strde=2, group_split=[Filters[0]//2, Filters[0]//2], kernel_sizes=[3], window_size=8)
        self.msfm4 = CloMSFM(dim=Filters[1], num_heads=Filters[1], in_cl=Filters[1], out_cl=Filters[2], strde=2, group_split=[Filters[1]//2, Filters[1]//2], kernel_sizes=[3], window_size=8)
        self.msfm5 = CloMSFM(dim=Filters[2], num_heads=Filters[2], in_cl=Filters[2], out_cl=Filters[3], strde=2, group_split=[Filters[2]//2, Filters[2]//2], kernel_sizes=[3], window_size=8)


        #transformer_attention
        self.gau1 = GAU(in_channels=80, channels=64, embed_size=128, head=32)
        self.gau2 = GAU(in_channels=80, channels=64, embed_size=64, head=16)
        self.gau3 = GAU(in_channels=160, channels=128, embed_size=32, head=8)
        self.gau4 = GAU(in_channels=320, channels=256, embed_size=16, head=4)
        self.gau5 = GAU(in_channels=640, channels=512, embed_size=8, head=2)

        #上采样
        self.up1 = up(64, 32)
        self.up2 = up(filters[0], filters[0])
        self.up3 = up(filters[1], filters[0])
        self.up4 = up(filters[2], filters[1])
        self.up5 = up(filters[3], filters[2])

        self.conv__1 = nn.Conv2d(80, 64, 1, 1, 0, bias=False)
        self.conv__2 = nn.Conv2d(80, 64, 1, 1, 0, bias=False)
        self.conv__3 = nn.Conv2d(160, 128, 1, 1, 0, bias=False)
        self.conv__4 = nn.Conv2d(320, 256, 1, 1, 0, bias=False)
        self.conv__5 = nn.Conv2d(640, 512, 1, 1, 0, bias=False)


        # 输出
        self.conv = nn.Sequential(
            nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0, bias=False),
        )
        #
        self.conv_ = nn.Conv2d(3, 16, 1, 2, 0, bias=False)

    def forward(self, x, y):

        x1, x2, x3, x4, x5 = self.backbone(x)
        y1, y2, y3, y4, y5 = self.backbone(y)
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())
        # print(x5.size())

        x_ = self.conv_(x)
        y_ = self.conv_(y)
        x1_ = self.msfm1(x_)
        x2_ = self.msfm2(x1_)
        x3_ = self.msfm3(x2_)
        x4_ = self.msfm4(x3_)
        x5_ = self.msfm5(x4_)
        y1_ = self.msfm1(y_)
        y2_ = self.msfm2(y1_)
        y3_ = self.msfm3(y2_)
        y4_ = self.msfm4(y3_)
        y5_ = self.msfm5(y4_)
        # x2_ = self.DRFD2(x1_)
        # x3_ = self.DRFD3(x2_)
        # x4_ = self.DRFD4(x3_)
        # x5_ = self.DRFD5(x4_)
        # y2_ = self.DRFD2(y1_)
        # y3_ = self.DRFD3(y2_)
        # y4_ = self.DRFD4(y3_)
        # y5_ = self.DRFD5(y4_)
        # print(x1_.size())
        # print(x1_.size())
        # print(x2_.size())
        # print(x3_.size())
        # print(x4_.size())
        # print(x5_.size())

        # gau1 = self.gau1((torch.cat((x1,x1_),1)), (torch.cat((y1,y1_),1)))
        # gau2 = self.gau2((torch.cat((x2,x2_),1)), (torch.cat((y2,y2_),1)))
        # gau3 = self.gau3((torch.cat((x3,x3_),1)), (torch.cat((y3,y3_),1)))
        # gau4 = self.gau4((torch.cat((x4,x4_),1)), (torch.cat((y4,y4_),1)))
        # gau5 = self.gau5((torch.cat((x5,x5_),1)), (torch.cat((y5,y5_),1)))

        gau1 = self.conv__1((torch.cat((x1, x1_), 1))+(torch.cat((y1, y1_),1)))
        gau2 = self.conv__2((torch.cat((x2, x2_), 1))+(torch.cat((y2, y2_),1)))
        gau3 = self.conv__3((torch.cat((x3, x3_), 1))+(torch.cat((y3, y3_),1)))
        gau4 = self.conv__4((torch.cat((x4, x4_), 1))+(torch.cat((y4, y4_),1)))
        gau5 = self.conv__5((torch.cat((x5, x5_), 1))+(torch.cat((y5, y5_),1)))

        ppm1 = self.conv__5((torch.cat((x5, x5_), 1)))
        ppm2 = self.conv__5((torch.cat((y5, y5_), 1)))

        ppm = self.ppm(ppm1, ppm2)
        # print(ppm.size())
        # print(gau5.size())
        # ppm = （x5+y5)
        up5 = self.up5(ppm + gau5)
        up4 = self.up4(up5 + gau4)
        up3 = self.up3(up4 + gau3)
        up2 = self.up2(up3 + gau2)
        up1 = self.up1(up2 + gau1)

        out = self.conv(up1)
        return out
if __name__ == '__main__':
# #     from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
# # #
# # #     # block = HWD(in_ch=32, out_ch=64)  # 输入通道数，输出通道数
# # #     # input = torch.rand(1, 32, 64, 64)
# # #     # output = block(input)
# # #     # print('input :', input.size())
# # #     # print('output :', output.size())
# # #     # model = Spatial_Transformer(64, embed_size=128)
    model = Net()
# # #     # # # model = FFU(64)
# # #     # # #
    x1 = torch.rand((2, 3, 256, 256))
    y1 = torch.rand((2, 3, 256, 256))
# # #     #
# # #     #
# # #     # model_eval = add_flops_counting_methods(model)
# # #     # model_eval.eval().start_flops_count()\
    out = model(x1, y1)
    print(out.size())
