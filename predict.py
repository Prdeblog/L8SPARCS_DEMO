import numpy as np
import cv2
import paddle
from PIL import Image
from paddleseg.core import infer
import rs_aug as T
import matplotlib.pyplot as plt
import matplotlib as mpl
from paddleseg.models.backbones import ShuffleNetV2_swish
from paddleseg.models.backbones import SwinTransformer
from paddleseg.models import OCRNet

backbone = ShuffleNetV2_swish()
model = OCRNet(num_classes=5,backbone=backbone,backbone_indices=(0,),
               pretrained='output_OCRNET_ShuffleNetV2/best_model/model.pdparams')
# model = OCRNet(num_classes=5,backbone=backbone,
#                backbone_indices=(0,),
#                pretrained='output_OCRNET_CHANGE/best_model/model.pdparams')

transforms = T.Compose([
    T.NDVI(r_band=3, nir_band=4),
    T.NDWI(g_band=2, nir_band=4),
    T.Resize(target_size=(512, 512), interp='NEAREST'),
    T.Normalize(mean=([0] * 10), std=([1] * 10), bit_num=16, band_num=10)
    ])

# vis_path = 'L8SPARCS/data_vis/LC80150242014146LGN00_23_photo.png'
# img_path = 'L8SPARCS/data_npy/LC80150242014146LGN00_23_data.npy'
# lab_path = 'L8SPARCS/mask/LC80050562014076LGN00_33_mask.png'
vis_path = 'L8SPARCS/data_vis/LC80050562014076LGN00_33_photo.png'
img_path = 'L8SPARCS/data_npy/LC80050562014076LGN00_33_data.npy'
lab_path = 'L8SPARCS/mask/LC80050562014076LGN00_33_mask.png'

img, _, lable = transforms(img=img_path, label=lab_path)
## 预测
img = paddle.to_tensor(img.transpose((2, 0, 1))[np.newaxis, :])
pre = infer.inference(model, img)
pred = paddle.argmax(pre, axis=1, dtype='int32').numpy().reshape((512, 512))
## 自定义colormap配色表
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#000000', '#00FEFE', '#0C00FB','#969696'], 256)
## 显示

vis_img = Image.open(vis_path)
lab_img = Image.open(lab_path)

plt.figure(figsize=(15, 10))
plt.subplot(131);plt.imshow(vis_img);plt.title('Image')
plt.subplot(132);plt.imshow(lab_img);plt.title('Label')
plt.subplot(133);plt.imshow(pred, cmap=colormap());plt.title('swin_OCRNet')
plt.imshow(pred, cmap=colormap());plt.title('ShuffleNetV2-OCRNet')
# plt.savefig('result_pre/image.jpg')
plt.show()

