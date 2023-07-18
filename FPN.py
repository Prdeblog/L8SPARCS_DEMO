import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class ASPP(nn.Layer):
    def __init__(self, in_channel, out_channel):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2D([1, 1])
        self.conv = nn.Conv2D(in_channel, out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2D(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2D(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2D(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2D(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2D(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        out = self.conv_1x1_output(paddle.concat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], axis=1))
        return out

x = paddle.randn([32, 64, 112, 112])
aspp = ASPP(64,128)
y = aspp(x)
print(y)