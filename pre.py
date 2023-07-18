
import numpy as np
import paddle
from ccnet import CCNet
from upernet import UPerNet
from paddleseg.models import UNet,OCRNet,HRNet_W48,BiSeNetV2,GCNet,ResNet50_vd,PSPNet,DeepLabV3P,UNetPlusPlus,DNLNet,DANet
import paddleseg.transforms as T
from paddleseg.core import infer
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

def nn_infer(model, img_path, model_path):
    # 网络定义
    para_state_dict = paddle.load(model_path)
    model.set_dict(para_state_dict)
    # 预测结果
    transforms = T.Compose([
        T.Resize(target_size=(512, 512)),
        T.Normalize()
    ])
    img, _ = transforms(img_path)
    img = paddle.to_tensor(img[np.newaxis, :])
    pre = infer.inference(model, img)
    pred = paddle.argmax(pre, axis=1).numpy().reshape((512, 512))
    return pred.astype('uint8')

backbone = ResNet50_vd(
    pretrained='https://bj.bcebos.com/paddleseg/dygraph/resnet50_vd_ssld_v2.tar.gz',
    multi_grid=(1,2,4),
    output_stride= 8
)

model = UPerNet(num_classes=5,backbone=backbone,backbone_indices=(0, 1, 2, 3))


img_path = 'L8SPARCS/data_vis/LC80150242014146LGN00_23_photo.png'
lab_path = 'L8SPARCS/mask/LC80150242014146LGN00_23_mask.png'
unt_params = 'output_upernet/iter_20000/model.pdparams'

def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#000000', '#00FEFE', '#0C00FB','#969696'], 256)
## 显示

image = Image.open(img_path)
label = Image.open(lab_path)
unt_img = nn_infer(model, img_path, unt_params)
plt.figure(figsize=(15, 10))
plt.subplot(131);plt.imshow(image);plt.title('image')
plt.subplot(132);plt.imshow(label);plt.title('label')
plt.subplot(133);plt.imshow(unt_img, cmap=colormap());plt.title('predict')
# plt.savefig('result_pre/HRNet-OCRNet.jpg')
plt.show()
