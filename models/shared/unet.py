import torch
from torch import Tensor, nn
from torch.nn import functional as F


class INConvBlock(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        stride: int = 1,
        instance_norm: bool = True,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride, 1, bias=not instance_norm)
        if instance_norm:
            self.instance_norm = nn.InstanceNorm2d(nout, affine=True)
        else:
            self.instance_norm = None
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        return self.act(x)
    
class GNConvBlock(nn.Module):
    def __init__(
        self,
        nin: int,
        nout: int,
        stride: int = 1,
        group_norm: bool = True,
        num_groups: int = 4,
        act: nn.Module = nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, 3, stride, 1, bias=not group_norm)
        if group_norm:
            self.group_norm = nn.GroupNorm(num_groups, nout, affine=True)
        else:
            self.group_norm = None
        self.act = act

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        if self.group_norm is not None:
            x = self.group_norm(x)
        return self.act(x)


##TODO: Should add latent_height and width for new datasets of genesis v2 24jul24
class UNet(nn.Module):
    def __init__(self, input_channels: int, output_channels: int, num_blocks: int, 
                         filter_start: int = 32, norm: str = "instance", add_alpha: bool = False):
        super().__init__()
        c = filter_start
        if norm == "instance":
            ConvBlock = INConvBlock
        elif norm == "group":
            ConvBlock = GNConvBlock
        else:
            raise ValueError("no valid normalization methods found")

        if add_alpha:
            input_channels = input_channels + 1
        if num_blocks == 4:
            self.down = nn.ModuleList(
                [
                    ConvBlock(input_channels, c),
                    ConvBlock(c, 2 * c),
                    ConvBlock(2 * c, 2 * c),
                    ConvBlock(2 * c, 2 * c),  # no downsampling
                ]
            )
            self.up = nn.ModuleList(
                [
                    ConvBlock(4 * c, 2 * c),
                    ConvBlock(4 * c, 2 * c),
                    ConvBlock(4 * c, c),
                    ConvBlock(2 * c, c),
                ]
            )
        elif num_blocks == 5:
            self.down = nn.ModuleList(
                [
                    ConvBlock(input_channels, c),
                    ConvBlock(c, c),
                    ConvBlock(c, 2 * c),
                    ConvBlock(2 * c, 2 * c),
                    ConvBlock(2 * c, 2 * c),  # no downsampling
                ]
            )
            self.up = nn.ModuleList(
                [
                    ConvBlock(4 * c, 2 * c),
                    ConvBlock(4 * c, 2 * c),
                    ConvBlock(4 * c, c),
                    ConvBlock(2 * c, c),
                    ConvBlock(2 * c, c),
                ]
            )
        elif num_blocks == 6:
            self.down = nn.ModuleList(
                [
                    ConvBlock(input_channels, c),
                    ConvBlock(c, c),
                    ConvBlock(c, c),
                    ConvBlock(c, 2 * c),
                    ConvBlock(2 * c, 2 * c),
                    ConvBlock(2 * c, 2 * c),  # no downsampling
                ]
            )
            self.up = nn.ModuleList(
                [
                    ConvBlock(4 * c, 2 * c),
                    ConvBlock(4 * c, 2 * c),
                    ConvBlock(4 * c, c),
                    ConvBlock(2 * c, c),
                    ConvBlock(2 * c, c),
                    ConvBlock(2 * c, c),
                ]
            )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 4 * 2 * c, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4 * 4 * 2 * c),
            nn.ReLU(inplace=True),
        )
        self.final_conv = nn.Conv2d(c, output_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.size(0)
        x_down = [x]
        skip = []
        for i, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)
            if i < len(self.down) - 1:
                act = F.interpolate(
                    act, scale_factor=0.5, mode="nearest", recompute_scale_factor=True
                )
            x_down.append(act)
        x_up = self.mlp(x_down[-1]).view(batch_size, -1, 4, 4)
        for i, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - i]], dim=1)
            x_up = block(features)
            if i < len(self.up) - 1:
                x_up = F.interpolate(x_up, scale_factor=2.0, mode="nearest")
        return self.final_conv(x_up)
