import paddle
from paddleseg.models.backbones import SwinTransformer
from paddleseg.models import OCRNet
import rs_aug as T
from datasets import Dataset  # 修改自官方代码
from paddleseg.core import evaluate

backbone = SwinTransformer(pretrain_img_size=512,in_chans=12)
model = OCRNet(num_classes=5,backbone=backbone,backbone_indices=(0,))

model_path = 'output_OCRNET_CHANGE/best_model/model.pdparams'	# 最优模型路径
if model_path:
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)            	# 加载模型参数
    print('Loaded trained params of model successfully')
else:
    raise ValueError('The model_path is wrong: {}'.format(model_path))

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

evaluate(model,
         val_dataset,
         aug_eval=True,  	# 是否使用数据增强
         scales=[0.75, 1.0, 1.25],  # 缩放因子
         flip_horizontal=True)  	# 是否水平翻转


