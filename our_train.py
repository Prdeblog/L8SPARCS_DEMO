import paddle
from upernet import UPerNet
from datasets import Dataset  # 修改自官方代码
import rs_aug as T
from paddleseg.models.losses import LovaszSoftmaxLoss,MixedLoss,CrossEntropyLoss
from paddleseg.core import train
from paddle.regularizer import L1Decay

unet = UPerNet(
    in_channel=12,
    num_classes=5,
    )
# 构建训练集
train_transforms = [
    T.NDVI(r_band=3, nir_band=4),
    T.NDWI(g_band=2, nir_band=4),
    T.RandomFlip(prob=0.5, direction='Both'),  # 随机翻转
    T.RandomRotate(prob=0.4, ig_pix=255),  # 随机旋转
    T.RandomEnlarge(prob=0.5, min_clip_rate=[0.7, 0.7]),  # 随机放大局部图像
    T.Resize(target_size=(512, 512), interp='NEAREST'),
    T.Normalize(mean=([0] * 10), std=([1] * 10), bit_num=16, band_num=10)
    # 这里不用计算NDBI，因为没有建筑，数据质量较好，也不用加入条带、拼接色差什么的
    ]
train_dataset = Dataset(
    transforms=train_transforms,
    dataset_root='L8SPARCS',
    num_classes=5,
    train_path='L8SPARCS/train.txt',
    separator=' ',
    mode='train'
    )

val_transforms = [
    T.NDVI(r_band=3, nir_band=4),
    T.NDWI(g_band=2, nir_band=4),
    T.Resize(target_size=(512, 512), interp='NEAREST'),
    T.Normalize(mean=([0] * 10), std=([1] * 10), bit_num=16, band_num=10)
    ]
## 构建验证集
val_dataset = Dataset(
    transforms=val_transforms,
    dataset_root='L8SPARCS',
    num_classes=5,
    val_path='L8SPARCS/val.txt',
    separator=' ',
    mode='val'
    )


# 设置学习率
base_lr = 0.01
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=20000, end_lr=0)
# 设置优化器
optimizer = paddle.optimizer.Momentum(lr, parameters=unet.parameters(), momentum=0.9, weight_decay=L1Decay(0.0001))
# 损失
losses = {}
losses['types'] = [CrossEntropyLoss()]
losses['coef'] = [1]


train(
    model=unet,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output',
    iters=20000,
    batch_size=1,
    save_interval=200,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)

