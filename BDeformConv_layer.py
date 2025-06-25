import torch
import torch.nn as nn
from torchvision.ops import deform_conv2d
import thop


class BDeformConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, max_sscale=3, min_sscale=0.5, max_rscale=3):
        """
        use the Tanh activation function to restrict
            the sketch scalar : max_sscale, min_sscale
        and
            the re-scale scalar : max_rscale, min value fixed to 1
        make training more stable
        """
        super(BDeformConv, self).__init__()
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)
        if isinstance(padding, tuple):
            self.padding = padding
        else:
            self.padding = (padding, padding)
        if isinstance(stride, tuple):
            self.stride = stride
        else:
            self.stride = (stride, stride)
        base_points = [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]
        # (2, k)
        base_points = torch.tensor(base_points).permute(1, 0)
        # (1, 1, 1, 2, k)
        self.register_buffer('base_offset', base_points[None, None, None, :, :])

        # for rotation，learn \sin \theta, \cos \theta (following a normalization operation)
        self.conv_rotation = nn.Conv2d(inc, 2, 3, 1, 1)
        # learn a sketch scalar for h-dimension, which is restricted to [min_sscale, max_sscale]
        self.conv_stretch = nn.Conv2d(inc, 1, 3, 1, 1)
        # learn a scalar for the whole grid
        self.conv_whole = nn.Conv2d(inc, 1, 3, 1, 1)
        self.acti = nn.Tanh()

        self.max_sscale = max_sscale
        self.min_sscale = min_sscale
        self.a = (max_sscale - min_sscale) / 2
        self.b = (max_sscale + min_sscale) / 2

        self.max_rscale = max_rscale
        self.c = (max_rscale - 1) / 2
        self.d = (self.max_rscale + 1) / 2

        self.init_offset()

    def init_offset(self):
        ### set initial sketch scalar as 1
        nn.init.constant_(self.conv_stretch.weight, 0)
        # Calculate bias for r=1
        tanh_c1 = (2 - self.max_sscale - self.min_sscale) / (self.max_sscale - self.min_sscale)
        # Ensure tanh_c1 is in [-1, 1] to avoid nan
        tanh_c1 = torch.clamp(torch.tensor(tanh_c1), -0.9999, 0.9999)
        nn.init.constant_(self.conv_stretch.bias, float(torch.atanh(tanh_c1)))
        ### set initial whole grid scalar as 1
        nn.init.constant_(self.conv_whole.weight, 0)
        tanh_c2 = (1 - self.max_rscale) / (self.max_rscale - 1)
        # Ensure tanh_c2 is in [-1, 1] to avoid nan
        tanh_c2 = torch.clamp(torch.tensor(tanh_c2), -0.9999, 0.9999)
        nn.init.constant_(self.conv_whole.bias, float(torch.atanh(tanh_c2)))
        ### set initial rotation angle as 0
        nn.init.constant_(self.conv_rotation.weight, 0)
        nn.init.constant_(self.conv_rotation.bias[0], 0)
        nn.init.constant_(self.conv_rotation.bias[1], 1)

    def forward(self, x):
        b, c, h, w = x.shape
        rotation = self.conv_rotation(x)
        # (B, H, W)
        sin = rotation[:, 0, :, :]
        cos = rotation[:, 1, :, :]
        norm = torch.sqrt(sin ** 2 + cos ** 2 + 1e-6)
        sin = sin / norm
        cos = cos / norm
        # (B, H, W, 2, 2)
        rotation_matrix = torch.stack([
            torch.stack([cos, sin], dim=-1),  # (B, H, W, 2)
            torch.stack([-sin, cos], dim=-1)  # (B, H, W, 2)
        ], dim=-2)

        r = self.acti(self.conv_stretch(x)) * self.a + self.b
        # (b, H, W, 1)
        r = r.permute(0, 2, 3, 1)
        wr = self.acti(self.conv_whole(x)) * self.c + self.d
        wr = wr.permute(0, 2, 3, 1)
        # (1, 1, 1, 2, k)
        base_offset = self.base_offset.to(x.dtype)
        # (b, h, w, 2, k)
        base_offset = base_offset.expand(b, h, w, -1, -1).clone()
        banded_offset = base_offset.clone()
        # use the whole grid scalar for all pixels and the sketch scalar for h-dimension
        banded_offset[..., 0, :] *= (wr * r)
        banded_offset[..., 1, :] *= wr
        # (B, H, W, 2, k)
        banded_offset = torch.matmul(rotation_matrix, banded_offset) - base_offset
        banded_offset = banded_offset.permute(0, 4, 3, 1, 2).contiguous().view(b, -1, h, w)

        return deform_conv2d(x, offset=banded_offset, weight=self.conv.weight, bias=None, stride=self.stride,
                             padding=self.padding)


if __name__ == "__main__":
    channel = 16
    out_channel = 32
    height = width = 256
    model = BDeformConv(channel, out_channel)
    flops, params = thop.profile(model, inputs=(torch.randn(1, channel, height, width),), verbose=False)
    print(f"model FLOPs: {flops / (10 ** 9)}G")
    print(f"model Params: {params / (10 ** 6)}M")
