from PIL import Image, ImageFilter
import matplotlib as mpl
import matplotlib.pyplot as plt

def sharp_data(input_path):
    img = Image.open(input_path)
    print(img.size)
    # 两次锐化
    image = img.filter(ImageFilter.SHARPEN)
    image = image.filter(ImageFilter.SHARPEN)

    plt.figure(figsize=(15, 10))
    plt.subplot(121);
    plt.imshow(img);
    plt.title('image')
    plt.subplot(122);
    plt.imshow(image);
    plt.title('re_image')
    plt.show()
    #img.save(output_path)

sharp_data('L8SPARCS/data_vis/LC82320772014306LGN00_38_photo.png')