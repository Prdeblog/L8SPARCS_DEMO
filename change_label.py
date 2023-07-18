import os
from PIL import Image
from tqdm import trange
import numpy as np

relab_path = "L8SPARCS/re_label"
if not os.path.exists(relab_path):
    os.mkdir(relab_path)  # 创建新标签的文件夹

label_dir = 'L8SPARCS/mask'
label_dir_list = os.listdir(label_dir)
for d in trange(len(label_dir_list)):
    im = Image.open(os.path.join(label_dir,label_dir_list[d]))  # 打开图片
    width = im.size[0]  # 获取宽度
    height = im.size[1]  # 获取长度
    im = np.array(im)
    for x in range(width):
        for y in range(height):
            label_origin = im[x, y]  # 原坐标对应的标签
            if (label_origin <= 5):
                im[x, y] -= 1  # 1-6变为0-5
            else:
                print(label_origin)
    new_im = Image.fromarray(im.astype(np.uint8), mode='P')
    new_im.save(os.path.join(relab_path, label_dir_list[d]))