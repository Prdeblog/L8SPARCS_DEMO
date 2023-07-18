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
    # 以下的非旋转、缩放等操作仅针对前10个通道
    rsg.RandomSharpening(prob=0.1, laplacian_mode='8-1', band_num=10),  # 以0.1的概率随机使用拉普拉斯算子进行图像锐化
    rsg.RandomBlur(prob=0.1, ksize=3, band_num=10),  # 以0.1的概率随机使用3x3大小的核进行高斯模糊
    rsg.Normalize(mean=([0] * 10), std=([1] * 10), bit_num=16, band_num=10),  # 标准化
    rsg.Resize(target_size=1000, interp='NEAREST')  # 定义尺寸
    ])
# 自定义colormap配色表
def colormap():  # 与官方png相同的配色表
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#000000', '#00FEFE', '#0C00FB','#969696'], 256)
# 显示
plt.figure(figsize=(8, 5))
for i in range(5):
    vis_img = cv2.cvtColor(cv2.imread(os.path.join(vis_path, vis_names[i])), cv2.COLOR_BGR2RGB)
    npy_img_path = os.path.join(npy_path, vis_names[i].replace('png', 'npy').replace('photo', 'data'))
    lab_img_path = os.path.join(lab_path, vis_names[i].replace('photo', 'mask'))
    npy, _, lab = transforms(img=npy_img_path, label=lab_img_path)
    npy_RGB = cv2.merge([npy[:, :, 3], npy[:, :, 2], npy[:, :, 1]])  # 官方彩色图
    plt.subplot(3, 5, i+1);plt.imshow(vis_img);plt.xticks([]);plt.yticks([])  # 真彩色原图
    if i == 0: plt.ylabel('Image')
    plt.subplot(3, 5, i+6);plt.imshow(show_img(npy_RGB));plt.xticks([]);plt.yticks([])  # 变化图
    if i == 0: plt.ylabel('tans_Image')
    plt.subplot(3, 5, i+11);plt.imshow(show_img(npy[:, :, -3])[:, :, 0], cmap=plt.cm.jet);plt.xticks([]);plt.yticks([])  # NDVI
    if i == 0: plt.ylabel('add_channel')
    # plt.subplot(3, 5, i+10);plt.imshow(show_img(lab)[:, :, 0], cmap=colormap());plt.xticks([]);plt.yticks([])  # 标签图
    # if i == 0: plt.ylabel('AUG_Label')
plt.show()