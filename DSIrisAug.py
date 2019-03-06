
# 图片 Augmentation
"""
  常用的API：
      (1) 高斯模糊
      (2) 垂直、水平翻转 Fliplr, Flipud(0.2)  20%的图片上下翻转
      (3) crop
      (4) 对比度ContrastNormalization
      (5) 仿射变换Affine
      (6) 颜色抖动
  其他：
  (1) 50%的概率进行0~0.5 随机sigmal的高斯模糊
      iaa.Sometimes(0.5,
           iaa.GaussianBlur(sigma=(0, 0.5))
        )
  (2) 每张图片执行0~5 个定义的操作
      iaa.SomeOf((0, 5),[...])
  (3) 图片显示： ia.imshow
      resize: ia.imresize_single_image(img,[500,500])
      画点: img.
      画线


"""
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import numpy as np
import Utils
import matplotlib.pyplot as plt


# 画点
def drawPoints(img,points):
    for point in points:
        cv2.line(img,point,point,255,7)
    return img

# 图片augment
# img 图片大小必须为[height,width,1] , points 要映射的点
def v4aug(img,points=None):
    seq = iaa.Sequential([
        #iaa.GaussianBlur(sigma=(0,3.0))
        #iaa.Fliplr(0.3)
        #iaa.ContrastNormalization((0.75,2.5))
        #iaa.Multiply((0.1, 0.6), per_channel=0)
        iaa.Affine(
            #scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
            #translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            #rotate=(-25, 25),
            shear=(-8, 8)
        )
    ])
    img_aug = seq.augment_image(img)
    return img_aug

###### TEST #########
if __name__ == "__main__":
    file = r"E:\虹膜数据集\S5004R04.jpg"
    img = plt.imread(file)
    img = v4aug(img)
    #
    points = [(150,150),(200,150),(200,200),(150,200)]
    img = drawPoints(img,points)
    Utils.showImage(img)