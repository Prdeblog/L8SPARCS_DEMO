import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from rsutils import show_img
import rs_aug as rsg

# 数据路径
npy_path = 'L8SPARCS/data_npy'
vis_path = 'L8SPARCS/data_vis'
lab_path = 'L8SPARCS/mask'

# 查看
vis_names = os.listdir(vis_path)
random.shuffle(vis_names)  # 随机打乱一下

# 部分数据增强
transforms = rsg.Compose([
    rsg.NDVI(r_band=3, nir_band=4),  # 计算NDVI指数并叠在通道11
    rsg.NDWI(g_band=2, nir_band=4),  # 计算NDWI指数并叠在通道12
    rsg.NDBI(nir_band=4, mir_band=5),  # 计算NDBI指数并叠在通道13
    # 以下的非旋转、缩放等操作仅针对前10个通道
    rsg.RandomColor(prob=0.1, alpha_range=[0.8, 1.2], beta_range=[0, 100], band_num=10),  # 以0.1的概率随机改变图像的亮度、对比度
    rsg.RandomFog(prob=0.1, fog_range=[0.03, 0.56], band_num=10),  # 以0.1的概率随机添加薄雾的效果
    rsg.RandomSharpening(prob=0.1, laplacian_mode='8-1', band_num=10),  # 以0.1的概率随机使用拉普拉斯算子进行图像锐化
    rsg.RandomBlur(prob=0.1, ksize=3, band_num=10),  # 以0.1的概率随机使用3x3大小的核进行高斯模糊
    rsg.RandomSplicing(prob=0.5, direction='Vertical', band_num=10),  # 随机以0.5的概率添加竖直方向的拼接色差
    rsg.RandomStrip(prob=0.5, strip_rate=0.05, direction='Horizontal', band_num=10),  # 随机以0.5的概率添加水平方向的条带噪声
    # rsg.RandomEnlarge(prob=0.5, min_clip_rate=[0.7, 0.7]),  # 随机以0.5的概率放大图像局部到原大小
    # rsg.RandomNarrow(prob=0.2, min_size_rate=[0.5, 0.5], ig_pix=0),  # 随机以0.2的概率缩小全图并填充回原大小
    rsg.Normalize(mean=([0] * 10), std=([1] * 10), bit_num=16, band_num=10),  # 标准化
    rsg.Resize(target_size=1000, interp='NEAREST')  # 定义尺寸
    ])
# 自定义colormap配色表
def colormap():  # 与官方png相同的配色表
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#000000', '#00FEFE', '#0C00FB','#969696'], 256)
# 显示
plt.figure(figsize=(20, 8))
for i in range(10):
    vis_img = cv2.cvtColor(cv2.imread(os.path.join(vis_path, vis_names[i])), cv2.COLOR_BGR2RGB)
    npy_img_path = os.path.join(npy_path, vis_names[i].replace('png', 'npy').replace('photo', 'data'))
    lab_img_path = os.path.join(lab_path, vis_names[i].replace('photo', 'mask'))
    npy, _, lab = transforms(img=npy_img_path, label=lab_img_path)
    npy_RGB = cv2.merge([npy[:, :, 3], npy[:, :, 2], npy[:, :, 1]])  # 官方彩色图
    plt.subplot(4, 10, i+1);plt.imshow(vis_img);plt.xticks([]);plt.yticks([])  # 真彩色原图
    if i == 0: plt.ylabel('Image')
    plt.subplot(4, 10, i+11);plt.imshow(show_img(npy_RGB));plt.xticks([]);plt.yticks([])  # 变化图
    if i == 0: plt.ylabel('AUG_Image')
    plt.subplot(4, 10, i+21);plt.imshow(show_img(npy[:, :, -3])[:, :, 0], cmap=plt.cm.jet);plt.xticks([]);plt.yticks([])  # NDVI
    if i == 0: plt.ylabel('NDVI')
    plt.subplot(4, 10, i+31);plt.imshow(show_img(lab)[:, :, 0], cmap=colormap());plt.xticks([]);plt.yticks([])  # 标签图
    if i == 0: plt.ylabel('AUG_Label')
plt.show()