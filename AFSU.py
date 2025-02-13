import torch
import torch.nn as nn
from typing import List


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class AttnMap(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.act_block = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0),
            MemoryEfficientSwish(),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        return self.norm(self.act_block(x))


class CloMSFM(nn.Module):
    def __init__(self, dim, in_cl, out_cl, strde, num_heads, group_split: List[int], kernel_sizes: List[int],
                 window_size=7, attn_drop=0., proj_drop=0., qkv_bias=True):
        super().__init__()
        assert sum(group_split) == num_heads
        assert len(kernel_sizes) + 1 == len(group_split)

        self.dim = dim
        self.num_heads = num_heads
        self.dim_head = dim // num_heads
        self.scalor = self.dim_head ** -0.5
        self.kernel_sizes = kernel_sizes
        self.window_size = window_size
        self.group_split = group_split

        convs = []
        act_blocks = []
        qkvs = []
        #创建高频注意力组件
        for i, (kernel_size, group_head) in enumerate(zip(kernel_sizes, group_split[:-1])):
            if group_head == 0:
                continue
            #深度可分离卷积
            convs.append(
                nn.Conv2d(
                    self.dim_head * group_head,
                    self.dim_head * group_head,
                    kernel_size, 1, kernel_size // 2,
                    groups=self.dim_head * group_head
                )
            )
            #添加注意力映射和投影层
            act_blocks.append(AttnMap(self.dim_head * group_head))
            qkvs.append(
                nn.Conv2d(dim, group_head * self.dim_head, 1, 1, 0, bias=qkv_bias)
            )

        if group_split[-1] != 0:
            self.global_q = nn.Conv2d(dim, group_split[-1] * self.dim_head, 1, 1, 0, bias=qkv_bias)
            self.global_kv = nn.Conv2d(dim, group_split[-1] * self.dim_head * 2, 1, 1, 0, bias=qkv_bias)
            self.avgpool = nn.AvgPool2d(window_size, window_size) if window_size != 1 else nn.Identity()

        self.convs = nn.ModuleList(convs)
        self.act_blocks = nn.ModuleList(act_blocks)
        self.qkvs = nn.ModuleList(qkvs)

        self.proj = nn.Conv2d(dim, dim, 1, 1, 0, bias=qkv_bias)
        self.norm = nn.BatchNorm2d(dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.conv = nn.Conv2d(in_cl, out_cl, 1, strde, 0, bias=False)

    def high_fre_attntion(self, x: torch.Tensor, to_qkv: nn.Module, mixer: nn.Module, attn_block: nn.Module):
        #特征投影
        qkv = to_qkv(x)
        #特征混合
        qkv = mixer(qkv)           #通过深度可分离卷积处理
        #注意力计算、激活和dropout
        attn = attn_block(qkv).mul(self.scalor)
        attn = self.attn_drop(torch.tanh(attn))
        #特征融合
        return attn.mul(qkv)

    def low_fre_attention(self, x: torch.Tensor, to_q: nn.Module, to_kv: nn.Module, avgpool: nn.Module):
        b, c, h, w = x.size()

        q = to_q(x).reshape(b, -1, self.dim_head, h * w).transpose(-1, -2).contiguous()
        kv = avgpool(x)
        kv = to_kv(kv).view(b, 2, -1, self.dim_head, (h * w) // (self.window_size ** 2)).permute(1, 0, 2, 4,
                                                                                                 3).contiguous()
        k, v = kv

        attn = self.scalor * q @ k.transpose(-1, -2)
        attn = self.attn_drop(attn.softmax(dim=-1))
        out = attn @ v

        return out.transpose(2, 3).reshape(b, -1, h, w).contiguous()

    def forward(self, x: torch.Tensor):
        res = []

        for i in range(len(self.kernel_sizes)):
            if self.group_split[i] == 0:
                continue
            res.append(
                self.high_fre_attntion(x, self.qkvs[i], self.convs[i], self.act_blocks[i])
            )

        if self.group_split[-1] != 0:
            res.append(
                self.low_fre_attention(x, self.global_q, self.global_kv, self.avgpool)
            )

        return self.conv(
            x + self.proj_drop(self.proj(torch.cat(res, dim=1)))
        )


if __name__ == '__main__':
    model = CloMSFM(
        dim=64,
        num_heads=64,
        in_cl=64,
        out_cl=128,
        strde=2,
        group_split=[32, 32],
        kernel_sizes=[3],
        window_size=32
    )

    input_tensor = torch.randn(1, 64, 64, 64)
    output = model(input_tensor)

    print('Input size:', input_tensor.size())
    print('Output size:', output.size())