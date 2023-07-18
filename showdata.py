import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt
import matplotlib as mpl


# 数据路径
npy_path = 'L8SPARCS/data_npy'
vis_path = 'L8SPARCS/data_vis'
lab_path = 'L8SPARCS/mask'
# 查看
vis_names = os.listdir(vis_path)
random.shuffle(vis_names)  # 随机打乱一下
plt.figure(figsize=(20, 4))
for i in range(10):
    vis_img = cv2.cvtColor(cv2.imread(os.path.join(vis_path, vis_names[i])), cv2.COLOR_BGR2RGB)
    lab_img = cv2.cvtColor(cv2.imread(os.path.join(lab_path, vis_names[i].replace('photo', 'mask'))), cv2.COLOR_BGR2RGB)
    plt.subplot(2, 10, i+1);plt.imshow(vis_img);plt.xticks([]);plt.yticks([])
    if i == 0: plt.ylabel('Image')
    plt.subplot(2, 10, i+11);plt.imshow(lab_img);plt.xticks([]);plt.yticks([])
    if i == 0: plt.ylabel('Label')
plt.show()