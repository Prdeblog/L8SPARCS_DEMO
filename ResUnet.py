import paddle
from paddle import nn
from ASPP import ASPP
from SA import sa_layer

# 残差块和上采样部分
class ResidualConv(nn.Layer):
    def __init__(self, in_channels, out_channels, first_layer=False):
        super().__init__()
        # 第一层的模块
        self.first_layer = first_layer
        if self.first_layer:
            self.conv_block = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(out_channels),
                nn.ReLU(),
                nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            # 输出C为1，广播机制
            self.conv_skip = nn.Conv2D(in_channels, 1, kernel_size=1, stride=1)
        else:
            # two paths: 1. BN_ACT_Conv
            self.conv_block = nn.Sequential(
                nn.BatchNorm2D(in_channels),
                nn.ReLU(),
                nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(out_channels),
                nn.ReLU(),
                nn.Conv2D(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            )
            # 2. add, 相加
            self.conv_skip = nn.Sequential(
                nn.Conv2D(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2D(out_channels)
            )

    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Layer):
    def __init__(self, in_channels, out_channels, k, s, p="same"):
        super().__init__()
        # 使用反卷积上采样
        # 后续还要经过残差模块
        self.up = nn.Conv2DTranspose(
            in_channels, out_channels, kernel_size=k, stride=s, padding=p)

    def forward(self, x):
        return self.up(x)



class MyNet(nn.Layer):
    def __init__(self, in_channel, num_classes):
        super().__init__()

        self.down_res1 = ResidualConv(in_channel, 64, first_layer=True)

        # 先做下采样在进入残差块
        self.down_res2 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2),
            ResidualConv(64, 128))
        self.down_res3 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2),
            ResidualConv(128, 256),
            sa_layer(256)
        )

        self.down_res4 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2),
            ResidualConv(256, 512),
            sa_layer(512)
        )
        self.down_res5 = nn.Sequential(
            nn.MaxPool2D(kernel_size=2, stride=2),
            ResidualConv(512, 1024),
            sa_layer(1024)
        )

        self.ASPP = ASPP(in_channel=1024,out_channel=1024)

        # 上采样+残差块
        self.up1 = Upsample(1024, 512, 2, 2)
        self.up_res1 = ResidualConv(512 * 2, 512)

        self.up2 = Upsample(512, 256, 2, 2)
        self.up_res2 = ResidualConv(256 * 2, 256)

        self.up3 = Upsample(256, 128, 2, 2)
        self.up_res3 = ResidualConv(128 * 2, 128)

        self.up4 = Upsample(128, 64, 2, 2)
        self.up_res4 = ResidualConv(64 * 2, 64)

        self.output_layer = nn.Sequential(
            nn.Conv2D(64, num_classes, kernel_size=1, stride=1),
        )

    def forward(self, x):
        # 存放中间值
        x1 = self.down_res1(x)
        x2 = self.down_res2(x1)
        x3 = self.down_res3(x2)
        x4 = self.down_res4(x3)
        x5 = self.down_res5(x4)

        x6 = self.ASPP(x5)

        x_ = self.up_res1(
            paddle.concat([self.up1(x6), x4], axis=1))
        x_ = self.up_res2(
            paddle.concat([self.up2(x_), x3], axis=1))
        x_ = self.up_res3(
            paddle.concat([self.up3(x_), x2], axis=1))
        x_ = self.up_res4(
            paddle.concat([self.up4(x_), x1], axis=1))

        x_output = self.output_layer(x_)

        return [x_output]

