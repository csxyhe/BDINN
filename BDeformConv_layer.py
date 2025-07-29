import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
import thop
import math


class BDeformConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, stride=1, groups=1, bias=None, max_sscale=3, min_sscale=0.5, isReScale=True, init_angle=0):
        """
        args:
            groups:     (int) default to be 1, although `torchvision.ops.deform_conv2d` doesn't explicitly provide a `groups` parameter, 
                        it can automatically infer the number of groups from the shape of `weight`.
            max_sscale: (optional) default to be 3, can be set to `None`. Max threshold for the stretch scalar.
            min_sscale: (optional) default to be 0.5, can be set to `None`. Min threshold for the stretch scalar.
            isReScale: (bool) default to `True`,
                        whether re-scale the whole sampling grid.
            init_angle: default to `0`, usually can be set to [0, 30, 45, 90].
                        Initialize the rotation angle with the given value.                        

        descriptions:
            An efficient and stable alternative of standard deformable convolution.

            hyper-parameters setting:
                **optional** use the Tanh activation function to restrict the stretch scalar : [min_sscale,max_sscale]
                make training more stable
            and
                whether apply re-scale operation to the whole sampling grid : isReScale

        """
        super(BDeformConv, self).__init__()
        assert (min_sscale is None and max_sscale is None) or (min_sscale is not None and max_sscale is not None), \
            "Both max_sscale and min_sscale must be set simultaneously or both must be None"
        assert min_sscale < max_sscale, "max_sscale must be larger than min_sscale"

        if isinstance(stride, tuple):
            self.stride = stride
        else:
            self.stride = (stride, stride)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        assert isinstance(kernel_size, tuple), "kernel_size must be tuple or int type"
        padding = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
        self.padding = padding

        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias, groups=groups)

        base_points = self.generate_base_points(kernel_size)
        # (2, k)
        base_points = torch.tensor(base_points).permute(1, 0)
        # (1, 1, 1, 2, k)
        self.register_buffer('base_offset', base_points[None, None, None, :, :])

        # for rotation，learn \sin \theta, \cos \theta (following a normalization operation)
        self.conv_rotation = nn.Conv2d(inc, 2, 3, 1, 1)
        # learn a stretch scalar for h-dimension, which is restricted to [min_sscale, max_sscale]
        self.conv_stretch = nn.Conv2d(inc, 1, 3, 1, 1)

        self.acti = nn.Tanh()

        if max_sscale is not None and min_sscale is not None:
            self.max_sscale = max_sscale
            self.min_sscale = min_sscale
            self.a = (max_sscale - min_sscale) / 2
            self.b = (max_sscale + min_sscale) / 2

        self.isReScale = isReScale
        if self.isReScale:
            # learn a scalar for the whole grid
            self.conv_whole = nn.Conv2d(inc, 1, 3, 1, 1)

        self.init_angle = init_angle

        self.init_offset()

    def init_offset(self):
        # set initial stretch scalar as 1
        nn.init.constant_(self.conv_stretch.weight, 0)
        if self.max_sscale is not None and self.min_sscale is not None:
            # Calculate bias for r=1
            tanh_c1 = (2 - self.max_sscale - self.min_sscale) / (self.max_sscale - self.min_sscale)
            # Ensure tanh_c1 is in [-1, 1] to avoid nan
            tanh_c1 = torch.clamp(torch.tensor(tanh_c1), -0.9999, 0.9999)
            nn.init.constant_(self.conv_stretch.bias, float(torch.atanh(tanh_c1)))
        else:
            nn.init.constant_(self.conv_stretch.bias, 1)
        # set initial whole grid scalar as 1
        if self.isReScale:
            nn.init.constant_(self.conv_whole.weight, 0)
            nn.init.constant_(self.conv_whole.bias, 0)
        # set initial rotation angle as 0
        nn.init.constant_(self.conv_rotation.weight, 0)
        nn.init.constant_(self.conv_rotation.bias[0], math.sin(math.radians(self.init_angle)))
        nn.init.constant_(self.conv_rotation.bias[1], math.cos(math.radians(self.init_angle)))

    def generate_base_points(self, kernel_size):
        kh, kw = kernel_size
        base_points = []
        for i in range(kh):
            for j in range(kw):
                base_points.append([i - kh // 2, j - kw // 2])
        return base_points

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

        # (1, 1, 1, 2, k) -> (b, h, w, 2, k)
        base_offset = self.base_offset.to(x.dtype).expand(b, h, w, -1, -1)
        banded_offset = base_offset.clone()
        # apply the stretch scalar for h-dimension
        if self.max_sscale is not None and self.min_sscale is not None:
            r = self.acti(self.conv_stretch(x)) * self.a + self.b
        else:
            r = self.conv_stretch(x)
        # (b, H, W, 1)
        r = r.permute(0, 2, 3, 1)
        banded_offset[..., 0, :] *= r
        # apply the re-scale scalar for all points
        if self.isReScale:
            wr = 1 + F.relu(self.conv_whole(x))
            # (b, h, w, 2)
            wr = wr.permute(0, 2, 3, 1).unsqueeze(-1)
            banded_offset *= wr
        # (B, H, W, 2, k)
        banded_offset = torch.matmul(rotation_matrix, banded_offset) - base_offset
        banded_offset = banded_offset.permute(0, 4, 3, 1, 2).contiguous().view(b, -1, h, w)

        return deform_conv2d(x, offset=banded_offset, weight=self.conv.weight, bias=None, stride=self.stride,
                             padding=self.padding)


if __name__ == "__main__":
    channel = 16
    out_channel = 32
    height = width = 256
    model = BDeformConv(channel, out_channel, 3)
    flops, params = thop.profile(model, inputs=(torch.randn(1, channel, height, width),), verbose=False)
    print(f"model FLOPs: {flops / (10 ** 9)}G")
    print(f"model Params: {params / (10 ** 6)}M")
