import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint



class _ConvLayer(nn.Module):
    """
    Basic convolution-normalization-activation layer.
    """
    def __init__(self, in_channels, out_channels, kernel, stride=1, padding=0,
            dilation=1, groups=1, bias=True, actv=nn.LeakyReLU):
        super().__init__()

        self._conv = nn.Conv3d(in_channels, out_channels,
            kernel, stride, padding, dilation, groups, bias)
        self._norm = nn.InstanceNorm3d(out_channels, affine=True)
        if actv in (nn.ReLU, nn.LeakyReLU):
            self._actv = actv(inplace=True)
        elif actv is not None:
            self._actv = actv()
        else:
            self._actv = actv

    def forward(self, x):
        output = self._norm(self._conv(x))
        if self._actv is not None:
            output = self._actv(output)

        return output


class _DownBlock(nn.Module):
    """
    U-Net downsampling block.
    """

    def __init__(self, in_channels, out_channels, num_layers, down_stride=1):
        super().__init__()

        self._conv_layers = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self._conv_layers.add_module(f"_conv_layer_{i}",
                    _ConvLayer(in_channels, out_channels, 3, down_stride, 1))
            else:
                self._conv_layers.add_module(f"_conv_layer_{i}",
                    _ConvLayer(out_channels, out_channels, 3, padding=1))
    
    def forward(self, x):
        output = self._conv_layers(x)

        return output


class _UpBlock(nn.Module):
    """
    U-Net upsampling block.
    """

    def __init__(self, in_channels, out_channels, num_layers, up_stride=2):
        super().__init__()

        self._upsample = nn.ConvTranspose3d(in_channels, out_channels,
            up_stride, up_stride, bias=False)

        self._conv_layers = nn.Sequential()
        for i in range(num_layers):
            if i == 0:
                self._conv_layers.add_module(f"_conv_layer_{i}",
                    _ConvLayer(2 * out_channels, out_channels, 3, padding=1))
            else:
                self._conv_layers.add_module(f"_conv_layer_{i}",
                    _ConvLayer(out_channels, out_channels, 3, padding=1))

    def forward(self, up_feat, down_feat):
        up_feat = self._upsample(up_feat)
        feat = torch.cat((up_feat, down_feat), dim=1)
        output = self._conv_layers(feat)

        return output


class UNet(nn.Module):
    """
    nnU-Net version of U-Net.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels. The default value is 1.
    num_classes : int, optional
        Number of predicting classes. The default value is 3.
    num_layers : int, optional
        Number of convolutional layers in downsampling/upsampling blocks.
        The default value is 2.
    down_strides : tuple of int or tuple, optional
        Strides in downsampling blocks. The default value follows
        nnU-Net settings for one-stage 3D full resolution segmentation.
    up_strides : tuple of int or tuple, optional
        Strides in upsampling blocks. The default value follows
        nnU-Net settings for one-stage 3D full resolution segmentation.
    down_channels : tuple of int, optional
        Number of channels in downsampling blocks. The default value follows
        nnU-Net settings for one-stage 3D full resolution segmentation.
    up_channels : tuple of int, optional
        Number of channels in upsampling blocks. The default value follows
        nnU-Net settings for one-stage 3D full resolution segmentation.
    """

    def __init__(self,
            in_channels=1,
            num_classes=1,
            num_layers=2,
            down_strides=(1, 2, 2, 2, 2, (1, 2, 2)),
            up_strides=((1, 2, 2), 2, 2, 2, 2),
            down_channels=(32, 64, 128, 256, 320, 320),
            up_channels=(320, 256, 128, 64, 32)):
        super().__init__()

        # downsampling path
        self._down_path = nn.ModuleList()
        for i in range(len(down_channels)):
            self._down_path.append(_DownBlock(in_channels, down_channels[i],
                num_layers, down_strides[i]))
            in_channels = down_channels[i]

        # upsampling path
        self._up_path = nn.ModuleList()
        for i in range(len(up_channels)):
            self._up_path.append(_UpBlock(in_channels, up_channels[i], 2,
                up_strides[i]))
            in_channels = up_channels[i]

        # prediction layer
        self._output_layer = nn.Conv3d(up_channels[-1], num_classes, 1,
                bias=False)

    def forward(self, x):
        # downsampling path
        down_feats = {"down_0": x}
        for i in range(len(self._down_path)):
            down_feats[f"down_{i + 1}"] = self._down_path[i](
                down_feats[f"down_{i}"])

        # upsampling path
        up_feats = {"up_0": down_feats[f"down_{len(self._down_path)}"]}
        for i in range(len(self._up_path)):
            up_feats[f"up_{i + 1}"] = self._up_path[i](
                up_feats[f"up_{i}"],
                down_feats[f"down_{len(self._down_path) - i - 1}"])

        # prediction
        output = self._output_layer(up_feats[f"up_{len(self._up_path)}"])

        return output
