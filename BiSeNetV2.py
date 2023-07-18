import paddle
import paddleseg.transforms as T
from paddleseg.datasets import Dataset
from paddleseg.models.losses import CrossEntropyLoss
from paddleseg.core import train
from paddleseg.models import BiSeNetV2
model = BiSeNetV2(num_classes=5,
                 lambd=0.25,
                 align_corners=False,
                 pretrained=None)

# 构建训练集
train_transforms = [
    T.RandomHorizontalFlip(),  # 水平翻转
    T.RandomVerticalFlip(),  # 垂直翻转
    T.RandomRotation(),  # 随机旋转
    T.RandomScaleAspect(),  # 随机缩放
    T.RandomDistort(),  # 随机扭曲
    T.Resize(target_size=(512, 512)), #不压缩图像以及增强图像训练结果更好
    T.Normalize()  # 归一化
]
train_dataset = Dataset(
    transforms=train_transforms,
    dataset_root='L8SPARCS',
    num_classes=5,
    mode='train',
    train_path='L8SPARCS/train_list.txt',
    separator=' ',
)
# 构建验证集
val_transforms = [
    T.Resize(target_size=(512, 512)),
    T.Normalize()
]
val_dataset = Dataset(
    transforms=val_transforms,
    dataset_root='L8SPARCS',
    num_classes=5,
    mode='val',
    val_path='L8SPARCS/val_list.txt',
    separator=' ',
)


base_lr = 0.01
lr = paddle.optimizer.lr.PolynomialDecay(base_lr, power=0.9, decay_steps=30000, end_lr=0)

optimizer = paddle.optimizer.Momentum(lr, parameters=model.parameters(), momentum=0.9, weight_decay=4.0e-5)

# 损失

losses = {}
losses['types'] = [CrossEntropyLoss()]*5
losses['coef'] = [1]*5
# 训练
train(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    optimizer=optimizer,
    save_dir='output_BiSeNetV2',
    iters=30000,
    batch_size=2,
    save_interval=500,
    log_iters=10,
    num_workers=0,
    losses=losses,
    use_vdl=True)