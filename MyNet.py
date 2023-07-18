
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg import utils
from paddleseg.cvlibs import manager
from paddleseg.models import layers


@manager.MODELS.add_component
class DCM(nn.Layer):
    """
    Dynamic Convolutional Module used in DMNet.

    Args:
        filter_size (int): The filter size of generated convolution kernel used in Dynamic Convolutional Module.
        fusion (bool): Add one conv to fuse DCM output feature.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
    """

    def __init__(self, filter_size, fusion, in_channels, channels):
        super().__init__()
        self.filter_size = filter_size  # 滤波器尺寸
        self.fusion = fusion  # 是否在DCM的输出特征图后面加上一个卷积层
        self.channels = channels  # 输出通道数

        pad = (self.filter_size - 1) // 2  # 计算pad
        if (self.filter_size - 1) % 2 == 0:
            self.pad = (pad, pad, pad, pad)
        else:
            self.pad = (pad + 1, pad, pad + 1, pad)

        self.avg_pool = nn.AdaptiveAvgPool2D(filter_size)  # 通过自适应池化产生filter的权重
        self.filter_gen_conv = nn.Conv2D(in_channels, channels, 1)
        self.input_redu_conv = layers.ConvBNReLU(in_channels, channels, 1)  # 1x1的卷积减少特征图通道

        self.norm = layers.SyncBatchNorm(channels)
        self.act = nn.ReLU()

        if self.fusion:
            self.fusion_conv = layers.ConvBNReLU(channels, channels, 1)

    def forward(self, x):
        generated_filter = self.filter_gen_conv(
            self.avg_pool(x))  # 分支二：生成滤波器权重
        x = self.input_redu_conv(x)  # 分支一：减少特征图通道
        b, c, h, w = x.shape
        x = x.reshape([1, b * c, h, w])
        generated_filter = generated_filter.reshape(
            [b * c, 1, self.filter_size, self.filter_size])

        x = F.pad(x, self.pad, mode='constant', value=0)
        output = F.conv2d(x, weight=generated_filter, groups=b * c)  # 深度可分离卷积
        output = output.reshape([b, self.channels, h, w])
        output = self.norm(output)
        output = self.act(output)
        if self.fusion:
            output = self.fusion_conv(output)
        return output


