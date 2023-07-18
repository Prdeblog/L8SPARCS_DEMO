import paddle
from MyNet import MyUNet
from datasets import Dataset  # 修改自官方代码
import rs_aug as T
from paddleseg.models.losses import CrossEntropyLoss,MixedLoss,DiceLoss
from paddleseg.core import train
from paddle.regularizer import L2Decay

model = MyUNet(in_channels=12,num_classes=5)


train_transforms = [
    T.NDVI(r_band=3, nir_band=4),
    T.NDWI(g_band=2, nir_band=4),
    T.RandomFog(prob=0.1, fog_range=[0.03, 0.56], band_num=10),  # 以0.1的概率随机添加薄雾的效果
    T.RandomSharpening(prob=0.1, laplacian_mode='8-1', band_num=10),  # 以0.1的概率随机使用拉普拉斯算子进行图像锐化
    T.RandomBlur(prob=0.1, ksize=3, band_num=10),  # 以0.1的概率随机使用3x3大小的核进行高斯模糊
    T.RandomColor(prob=0.1, alpha_range=[0.8, 1.2], beta_range=[0, 100], band_num=10),
    T.RandomFlip(prob=0.5, direction='Both'),
    T.RandomRotate(prob=0.4, ig_pix=255),
    T.RandomNarrow(prob=0.2, min_size_rate=[0.5, 0.5], ig_pix=255),
    T.Resize(target_size=(512, 512), interp='NEAREST'),
    T.Normalize(mean=([0] * 10), std=([1] * 10), bit_num=16, band_num=10)
    # 这里不用计算NDBI，因为没有建筑，数据质量较好，也不用加入条带、拼接色差什么的
    ]
## 构建训练集
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
optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)
# 损失
losses = {}
losses['types'] = [MixedLoss([CrossEntropyLoss(), DiceLoss()], [0.8, 0.2])]
losses['coef'] = [1]


train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output_model',
    iters=20000,
    batch_size=1,
    save_interval=200,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)