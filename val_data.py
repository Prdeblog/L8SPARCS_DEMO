# from paddleseg.models import BiSeNetV2,ResNet50_vd,PSPNet,GCNet,OCRNet,HRNet_W48,UNet,DeepLabV3P
from paddleseg.models import UNetPlusPlus
from ccnet import CCNet
import paddle
import paddleseg.transforms as T
from paddleseg.models import ResNet50_vd
from upernet import UPerNet
# import rs_aug as T
from paddleseg.datasets import Dataset
# from datasets import Dataset  # 修改自官方代码
from paddleseg.core import evaluate

# model = BiSeNetV2(num_classes=5,
#                   lambd=0.25,
#                   align_corners=False,
#                   pretrained=None)
# backbone = HRNet_W48(
#     pretrained='https://bj.bcebos.com/paddleseg/dygraph/hrnet_w48_ssld.tar.gz',
# )
# model = OCRNet(num_classes=5,backbone=backbone,backbone_indices=(0,))
# model = UNet(num_classes=5)
# model = DeepLabV3P(num_classes=5,backbone=ResNet50_vd(),backbone_indices=(0,1))

backbone = ResNet50_vd(
    pretrained='https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz',
)
model = CCNet(num_classes=5,backbone=backbone)

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

# unet_model = UNetPlusPlus(in_channels=3,num_classes=5)
# backbone = ResNet50_vd(
#     pretrained='https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz',
#     multi_grid=(1,2,4),
#     output_stride= 8
# )
#
# model = UPerNet(num_classes=5,backbone=backbone,backbone_indices=(0, 1, 2, 3))

model_path = 'output_CCNet/best_model/model.pdparams'	# 最优模型路径
if model_path:
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)            	# 加载模型参数
    print('Loaded trained params of model successfully')
else:
    raise ValueError('The model_path is wrong: {}'.format(model_path))

# val_transforms = [
#     T.NDVI(r_band=3, nir_band=4),
#     T.NDWI(g_band=2, nir_band=4),
#     T.Resize(target_size=(512, 512), interp='NEAREST'),
#     T.Normalize(mean=([0] * 10), std=([1] * 10), bit_num=16, band_num=10)
#     ]
# ## 构建验证集
# val_dataset = Dataset(
#     transforms=val_transforms,
#     dataset_root='L8SPARCS',
#     num_classes=5,
#     val_path='L8SPARCS/val.txt',
#     separator=' ',
#     mode='val'
#     )

evaluate(model,
         val_dataset,
         aug_eval=True,  	# 是否使用数据增强
         scales=[0.75, 1.0, 1.25],  # 缩放因子
         flip_horizontal=True)  	# 是否水平翻转


