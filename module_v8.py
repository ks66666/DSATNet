import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward
#卷积三件套
class ConvBlock_v1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock_v1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.init_weight()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return self.relu(x)

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)

#上采样
class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up, self).__init__()
        self.up = nn.Sequential(
            ConvBlock_v1(in_channels, out_channels),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        )

    def forward(self, x):
        return self.up(x)

#通道注意力
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.conv1(torch.cat((avg_out, max_out), 1))
        return self.sigmoid(x)

class LAU(nn.Module):
    def __init__(self, channels):
        super(LAU, self).__init__()
        self.conv_sub = ConvBlock_v1(in_channels=channels, out_channels=channels)
        self.cam = ChannelAttention(in_planes=channels)
        self.conv_x = ConvBlock_v1(in_channels=channels, out_channels=channels)
        self.conv_y = ConvBlock_v1(in_channels=channels, out_channels=channels)
        self.conv_cat = ConvBlock_v1(in_channels=channels * 2, out_channels=channels)
        self.sam = SpatialAttention()
        self.conv = ConvBlock_v1(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)

    def forward(self, x, y):
        sub = self.sam(self.conv_sub(torch.abs(x - y)))
        x = self.conv_x(x.mul(sub) + x)
        y = self.conv_y(y.mul(sub) + y)
        fusion = self.conv_cat(torch.cat([x, y], dim=1))
        fusion = self.cam(fusion)
        out = self.conv(sub + fusion)
        return out

class TFIM(nn.Module):
    def __init__(self, channels):
        super(TFIM, self).__init__()
        self.conv_sub = ConvBlock_v1(in_channels=channels, out_channels=channels)
        self.conv_x = ConvBlock_v1(in_channels=channels, out_channels=channels)
        self.conv_y = ConvBlock_v1(in_channels=channels, out_channels=channels)
        self.conv_cat = ConvBlock_v1(in_channels=channels * 2, out_channels=channels)
        self.conv = ConvBlock_v1(in_channels=channels, out_channels=channels,kernel_size=1, padding=0)

    def forward(self, x, y):
        sub = self.conv_sub(torch.abs(x - y))
        x = self.conv_x(x.mul(sub) + x)
        y = self.conv_y(y.mul(sub) + y)
        fusion = torch.cat([x, y], dim=1)
        fusion = self.conv_cat(fusion)
        out = self.conv(sub + fusion)
        return out

class PRU(nn.Module):
    def __init__(self, in_channels,channels):
        super(PRU, self).__init__()
        self.conv_sub = ConvBlock_v1(in_channels=in_channels, out_channels=in_channels)
        self.conv_x = ConvBlock_v1(in_channels=in_channels, out_channels=in_channels)
        self.conv_y = ConvBlock_v1(in_channels=in_channels, out_channels=in_channels)
        self.conv_cat = ConvBlock_v1(in_channels=in_channels * 2, out_channels=in_channels)
        self.conv = ConvBlock_v1(in_channels=in_channels, out_channels=channels,kernel_size=1, padding=0)

    def forward(self, x, y):
        sub = self.conv_sub(torch.abs(x - y))
        x = self.conv_x(x.mul(sub) + x)
        y = self.conv_y(y.mul(sub) + y)
        fusion = torch.cat([x, y], dim=1)
        fusion = self.conv_cat(fusion)
        out = self.conv(x + y + fusion)
        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (N, query_len, heads, heads_dim)
        # keys shape : (N, key_len, heads, heads_dim)
        # energy shape: (N, heads, query_len, key_len)

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql, nlhd->nqhd", [attention, values]).reshape(N, query_len, self.heads * self.head_dim)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        # (N, query_len, heads, head_dim)

        out = self.fc_out(out)
        return out

class Spatial_Transformer(nn.Module):
    def __init__(self, channels, embed_size, head=8, dropout=0.1):
        super(Spatial_Transformer, self).__init__()
        self.Avgpool1 = nn.AdaptiveAvgPool2d((embed_size, 1))
        self.Avgpool2 = nn.AdaptiveAvgPool2d((1, embed_size))
        self.SelfAttention1 = SelfAttention(embed_size=embed_size, heads=head)
        self.SelfAttention2 = SelfAttention(embed_size=embed_size, heads=head)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = ConvBlock_v1(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.Avgpool1(x).squeeze(dim=3)
        x2 = self.Avgpool2(x).squeeze(dim=2)
        attention1 = self.dropout(self.norm1(self.SelfAttention1(x1, x1, x1) + x1)).unsqueeze(dim=3)
        attention2 = self.dropout(self.norm2(self.SelfAttention1(x2, x2, x2) + x2)).unsqueeze(dim=2)

        attention = x * (attention1 + attention2)
        out = attention + self.feed_forward(attention)
        return out

class Channel_Transformer(nn.Module):
    def __init__(self, channels, embed_size, head=8, dropout=0.1):
        super(Channel_Transformer, self).__init__()
        self.gap =nn.AdaptiveAvgPool2d(1)
        self.SelfAttention = SelfAttention(embed_size=embed_size, heads=head)
        self.norm = nn.LayerNorm(embed_size)
        self.feed_forward = ConvBlock_v1(in_channels=channels, out_channels=channels, kernel_size=1, padding=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gap = self.gap(x).squeeze(dim=-1).permute(0, 2, 1)
        attention = self.dropout(self.norm(self.SelfAttention(gap, gap, gap) + gap)).permute(0, 2, 1).unsqueeze(dim=3)
        x = x * attention
        out = x + self.feed_forward(x)
        return out

class GAU(nn.Module):
    def __init__(self, in_channels, channels, embed_size, head=8):
        super(GAU, self).__init__()
        self.pru = PRU(in_channels, channels)
        self.spatial_transformer = Spatial_Transformer(channels, embed_size=embed_size, head=head)
        self.channel_transformer = Channel_Transformer(channels, embed_size=channels, head=head)

    def forward(self, x, y):
        tfim = self.pru(x, y)
        spatial_transformer = self.spatial_transformer(tfim)
        channel_transformer = self.channel_transformer(spatial_transformer)
        channel_transformer_ = channel_transformer + tfim

        return channel_transformer_

class GAU_backbone(nn.Module):
    def __init__(self):
        super(GAU_backbone, self).__init__()

    def forward(self, x, y):
        return torch.abs(x - y)

class GAU_pru(nn.Module):
    def __init__(self, channels, embed_size, head=8):
        super(GAU_pru, self).__init__()
        self.pru = PRU(channels)

    def forward(self, x, y):
        pru = self.pru(x, y)
        return pru

class GAU_sau(nn.Module):
    def __init__(self, channels, embed_size, head=8):
        super(GAU_sau, self).__init__()
        self.pru = PRU(channels)
        self.spatial_transformer = Spatial_Transformer(channels, embed_size=embed_size, head=head)

    def forward(self, x, y):
        tfim = self.pru(x, y)
        spatial_transformer = self.spatial_transformer(tfim)
        return spatial_transformer





class PPM(nn.Module):
    def __init__(self, in_channels, out_channels, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
            )
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return x + torch.cat(out, 1)

class PPM_(nn.Module):
    def __init__(self, in_channels, out_channels, bins):
        super(PPM_, self).__init__()
        self.features = []
        self.feature0 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(bins[0]),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
        self.feature1 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(bins[1]),
                    nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
        self.feature2 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(bins[2]),
                    nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )
        self.feature3 = nn.Sequential(
                    nn.AdaptiveAvgPool2d(bins[3]),
                    nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(True),
                )

    def forward(self, x):
        x_size = x.size()
        feature0 = F.interpolate(self.feature0(x), x_size[2:], mode='bilinear', align_corners=True)
        feature1 = F.interpolate(self.feature1(torch.cat((x, feature0), 1)), x_size[2:], mode='bilinear', align_corners=True)
        feature2 = F.interpolate(self.feature2(torch.cat((x, feature1), 1)), x_size[2:], mode='bilinear', align_corners=True)
        feature3 = F.interpolate(self.feature3(torch.cat((x, feature2), 1)), x_size[2:], mode='bilinear', align_corners=True)

        return torch.cat((feature0, feature1, feature2, feature3), 1)

class PPM__(nn.Module):
    def __init__(self, in_channels, out_channels, bins):
        super(PPM__, self).__init__()
        self.features = []

        self.feature0_ = nn.Sequential(
            nn.AdaptiveAvgPool2d(bins[0]),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        self.feature1_ = nn.Sequential(
            nn.AdaptiveAvgPool2d(bins[1]),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        self.feature2_ = nn.Sequential(
            nn.AdaptiveAvgPool2d(bins[2]),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )
        self.feature3_ = nn.Sequential(
            nn.AdaptiveAvgPool2d(bins[3]),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
        )


    def forward(self, x, y):
        x_size = x.size()
        y_size = y.size()
        #A(1x)
        feature0 = F.interpolate(self.feature0_(x), x_size[2:], mode='bilinear', align_corners=True)
        # A(1y)
        feature0_ = F.interpolate(self.feature0_(y), x_size[2:], mode='bilinear', align_corners=True)
        # A(2x)
        feature1 = F.interpolate(self.feature1_(x+feature0_), x_size[2:], mode='bilinear', align_corners=True)
        # A(2y)
        feature1_ = F.interpolate(self.feature1_(y+feature0), y_size[2:], mode='bilinear', align_corners=True)
        # A(3x)
        feature2 = F.interpolate(self.feature2_(x+feature1_), x_size[2:], mode='bilinear', align_corners=True)
        # A(3y)
        feature2_ = F.interpolate(self.feature2_(y+feature1), y_size[2:], mode='bilinear', align_corners=True)
        # A(4x)
        feature3 = F.interpolate(self.feature2_(x+feature2_), x_size[2:], mode='bilinear', align_corners=True)
        # A(4y)
        feature3_ = F.interpolate(self.feature2_(y+feature2), x_size[2:], mode='bilinear', align_corners=True)


        return feature0+feature1+feature2+feature3+feature0_+feature1_+feature3_+feature2_+x+y

class HWD(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(HWD, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x_size = x.size()
        x_ = F.interpolate(torch.cat([yL, y_HL, y_LH, y_HH], dim=1), x_size[2:], mode='bilinear',
                                 align_corners=True)
        x_ = self.conv_bn_relu(x_)
        return x+x_
class Down_wt(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down_wt, self).__init__()
        self.wt = DWTForward(J=1, mode='zero', wave='haar')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv_bn_relu_ = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x

class SRFD(nn.Module):
    def __init__(self, in_channels=3, out_channels=96):
        super().__init__()
        out_c14 = int(out_channels / 4)  # out_channels / 4
        out_c12 = int(out_channels / 2)  # out_channels / 2

        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        self.conv_init = nn.Conv2d(in_channels, out_c14, kernel_size=7, stride=1, padding=3)

        # original size to 2x downsampling layer
        self.conv_1 = nn.Conv2d(out_c14, out_c12, kernel_size=3, stride=1, padding=1, groups=out_c14)
        self.conv_x1 = nn.Conv2d(out_c12, out_c12, kernel_size=3, stride=2, padding=1, groups=out_c12)
        self.batch_norm_x1 = nn.BatchNorm2d(out_c12)
        self.cut_c = Cut(out_c14, out_c12)
        self.fusion1 = nn.Conv2d(out_channels, out_c12, kernel_size=1, stride=1)

        # 2x to 4x downsampling layer
        self.conv_2 = nn.Conv2d(out_c12, out_channels, kernel_size=3, stride=1, padding=1, groups=out_c12)
        self.conv_x2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.batch_norm_x2 = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.cut_r = Cut(out_c12, out_channels)
        self.fusion2 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 7x7 convolution with stride 1 for feature reinforcement, Channels from 3 to 1/4C.
        x = self.conv_init(x)  # x = [B, C/4, H, W]

    # original size to 2x downsampling layer
        c = x                   # c = [B, C/4, H, W]
        # CutD
        c = self.cut_c(c)       # c = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]
        # ConvD
        x = self.conv_1(x)      # x = [B, C/4, H, W] --> [B, C/2, H/2, W/2]
        x = self.conv_x1(x)     # x = [B, C/2, H/2, W/2]
        x = self.batch_norm_x1(x)
        # Concat + conv
        x = torch.cat([x, c], dim=1)    # x = [B, C, H/2, W/2]
        x = self.fusion1(x)     # x = [B, C, H/2, W/2] --> [B, C/2, H/2, W/2]

        # 2x to 4x downsampling layer
        r = x                   # r = [B, C/2, H/2, W/2]
        x = self.conv_2(x)      # x = [B, C/2, H/2, W/2] --> [B, C, H/2, W/2]
        m = x                   # m = [B, C, H/2, W/2]
        # ConvD
        x = self.conv_x2(x)     # x = [B, C, H/4, W/4]
        x = self.batch_norm_x2(x)
        # MaxD
        m = self.max_m(m)       # m = [B, C, H/4, W/4]
        m = self.batch_norm_m(m)
        # CutD
        r = self.cut_r(r)       # r = [B, C, H/4, W/4]
        # Concat + conv
        x = torch.cat([x, r, m], dim=1)  # x = [B, C*3, H/4, W/4]
        x = self.fusion2(x)     # x = [B, C*3, H/4, W/4] --> [B, C, H/4, W/4]
        return x


# CutD
class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x


# Deep feature downsampling
class DRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):       # input: x = [B, C, H, W]
        c = x                   # c = [B, C, H, W]
        x = self.conv(x)        # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x                   # m = [B, 2C, H, W]

        # CutD
        c = self.cut_c(c)       # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)      # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)       # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv
        x = torch.cat([c, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)      # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x                # x = [B, 2C, H/2, W/2]
# if __name__ == '__main__':
#     from flops_counter import add_flops_counting_methods, flops_to_string, get_model_parameters_number
#
#     # block = HWD(in_ch=32, out_ch=64)  # 输入通道数，输出通道数
#     # input = torch.rand(1, 32, 64, 64)
#     # output = block(input)
#     # print('input :', input.size())
#     # print('output :', output.size())
#     # model = Spatial_Transformer(64, embed_size=128)
#     model = HWD(in_channels=64, channels=64, embed_size=128, head=32)
# #     # # # model = FFU(64)
# #     # # #
#     x1 = torch.rand((2, 64, 128, 128))
#     y1 = torch.rand((2, 64, 128, 128))
# #     #
# #     #
# #     # model_eval = add_flops_counting_methods(model)
# #     # model_eval.eval().start_flops_count()\
#     out = model(x1, y1)
#     print(out.size())
# #     print(out.size())
#     # # out = model_eval(x1, x2)
#     # out = model_eval(x1)
#     #
#     print('Flops: {}'.format(flops_to_string(model.compute_average_flops_cost())))
#     print('Params: ' + get_model_parameters_number(model))
#     # print('Output shape: {}'.format(list(out.shape)))
#     total_paramters = sum(p.numel() for p in model.parameters())
#     print('Total paramters: {}'.format(total_paramters))
#     #
