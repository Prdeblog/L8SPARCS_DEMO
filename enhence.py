import paddle
import paddle.nn as nn

class SAM(nn.Layer):
    """
     i_ch: input channel
    """
    def __init__(self, i_ch):
        super(SAM, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2D(i_ch, i_ch, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_attention = self.conv(x)

        return x * x_attention


class srm_layer(nn.Layer):
    def __init__(self, channel):
        super(srm_layer, self).__init__()

        self.cfc = self.create_parameter(shape=[channel, 2],
                                         default_initializer=nn.initializer.Assign(paddle.zeros([channel, 2])))

        self.bn = nn.BatchNorm2D(channel)
        self.activation = nn.Sigmoid()

        setattr(self.cfc, 'srm_param', True)
        setattr(self.bn.weight, 'srm_param', True)
        setattr(self.bn.bias, 'srm_param', True)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.shape

        channel_mean = paddle.mean(paddle.reshape(x, [N, C, -1]), axis=2, keepdim=True)
        channel_var = paddle.var(paddle.reshape(x, [N, C, -1]), axis=2, keepdim=True) + eps
        channel_std = paddle.sqrt(channel_var)

        t = paddle.concat((channel_mean, channel_std), axis=2)
        return t

    def _style_integration(self, t):
        z = t * paddle.reshape(self.cfc, [-1, self.cfc.shape[0], self.cfc.shape[1]])
        tmp = paddle.sum(z, axis=2)
        z = paddle.reshape(tmp, [tmp.shape[0], tmp.shape[1], 1, 1])  # B x C x 1 x 1

        z_hat = self.bn(z)
        g = self.activation(z_hat)

        return g

    def forward(self, x):
        # B x C x 2
        t = self._style_pooling(x)

        # B x C x 1 x 1
        g = self._style_integration(t)

        return x * g

class switch(nn.Layer):
    def __init__(self):
        super(switch, self).__init__()
    def forward(self, x):
        x = x * paddle.fluid.layers.sigmoid(x)
        return x

import paddle
import math
import paddle.nn as nn
import paddle.nn.functional as F

class BasicConv(nn.Layer):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias_attr=bias)
        self.bn = nn.BatchNorm2D(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Layer):
    def forward(self, x):
        return x.reshape([x.shape[0], -1])

class ChannelGate(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.shape[2], x.shape[3]), stride=(x.shape[2], x.shape[3]))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Layer):
    def forward(self, x):
        return paddle.concat( (paddle.max(x,1).unsqueeze(1), paddle.mean(x,1).unsqueeze(1)), axis=1 )

class SpatialGate(nn.Layer):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Layer):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out