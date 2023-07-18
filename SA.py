import paddle.nn as nn
import paddle

class sa_layer(nn.Layer):
    """Constructs a Channel Spatial Group module.
    Args:
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, groups=32):
        super(sa_layer, self).__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool2D(1)
        self.cweight = self.create_parameter(shape=[1, channel // (2 * groups), 1, 1],default_initializer=paddle.nn.initializer.Assign(paddle.zeros([1, channel // (2 * groups), 1, 1])))#paddle.nn.initializer.Assign=初始化参数
        self.cbias = self.create_parameter(shape=[1, channel // (2 * groups), 1, 1],default_initializer=paddle.nn.initializer.Assign(paddle.ones([1, channel // (2 * groups), 1, 1])))#paddle.zeros=全为0的tensor
        self.sweight = self.create_parameter(shape=[1, channel // (2 * groups), 1, 1],default_initializer=paddle.nn.initializer.Assign(paddle.zeros([1, channel // (2 * groups), 1, 1])))
        self.sbias = self.create_parameter(shape=[1, channel // (2 * groups), 1, 1],default_initializer=paddle.nn.initializer.Assign(paddle.ones([1, channel // (2 * groups), 1, 1])))

        self.sigmoid = nn.Sigmoid()
        self.gn = nn.GroupNorm(channel // (2 * groups), channel // (2 * groups))

    @staticmethod
    def channel_shuffle(x, groups):
        b, c, h, w = x.shape

        x = paddle.reshape(x, [b, groups, -1, h, w])#  x.shape=[1,2,3,4]    -1为x全部元素相乘然后除其他元素
        x = paddle.transpose(x, [0, 2, 1, 3, 4])

        # flatten
        x = paddle.reshape(x, [b, -1, h, w])

        return x

    def forward(self, x):
        b, c, h, w = x.shape

        x = paddle.reshape(x, [b * self.groups, -1, h, w])
        x_0, x_1 = paddle.chunk(x, 2, axis=1)

        # channel attention
        xn = self.avg_pool(x_0)
        xn = self.cweight * xn + self.cbias
        xn = x_0 * self.sigmoid(xn)

        # spatial attention
        xs = self.gn(x_1)
        xs = self.sweight * xs + self.sbias
        xs = x_1 * self.sigmoid(xs)

        # concatenate along channel axis
        out = paddle.concat([xn, xs], axis=1)
        out = paddle.reshape(out, [b, -1, h, w])

        out = self.channel_shuffle(out, 2)
        return out