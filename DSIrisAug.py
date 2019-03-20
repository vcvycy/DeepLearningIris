
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
import random

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

# 画实心圆
def drawRandomCircle(img,radius_range,num_range,color_range):
    num = random.randint(num_range[0],num_range[1])
    for _ in range(num):
        h = random.randint(0,img.shape[0]-1)
        w = random.randint(0,img.shape[1]-1)
        r = random.randint(radius_range[0],radius_range[1])
        c = random.randint(color_range[0],color_range[1])
        cv2.circle(img, (w, h), r, c, thickness=-1)
    return

def drawRandomLine(img,thick_range,num_range, color_range):
    num = random.randint(num_range[0],num_range[1])
    for _ in range(num):
        h = random.randint(0,img.shape[0]-1)
        w = random.randint(0,img.shape[1]-1)
        h2 = random.randint(0,img.shape[0]-1)
        w2 = random.randint(0,img.shape[1]-1)
        thick = random.randint(thick_range[0], thick_range[1])

        color = random.randint(color_range[0],color_range[1])
        cv2.line(img,(w,h),(w2, h2),color,thickness=thick)
    return

# 虹膜定位图片增强
# (1) crop (2) 水平翻转 (3) 白块 (4) 黑色线条 (5) 高斯模糊
def v4locationAug(img , loc, crop_size = (24, 64), scale=2):
    # 虹膜/瞳孔位置
    iris = loc["iris"]
    pupil = loc["pupil"]
    # 白块(模拟反光)
    drawRandomCircle(img,(5,8),(0,10),color_range=(230,255))
    #drawRandomCircle(img,(20,50),(0,2),color_range=(180,255))

    # 黑色线条 (模拟眼镜)
    # drawRandomLine(img,(5,10), (0,3), color_range=(0,50))

    # 翻转+模糊+crop
    seq = iaa.Sequential([
        # iaa.SomeOf((0,1),   [iaa.Fliplr(0.5)]),        # 一半的概率左右翻转
        iaa.OneOf([
            iaa.GaussianBlur((0, 3.0)),
            iaa.AverageBlur(k=(2, 3)),
            iaa.MedianBlur(k=(3, 7)),
        ])
     ])
    img = seq.augment_image(img)

    # 左右翻转
    h, w =img.shape[0],img.shape[1]
    if  random.randint(0,1) == 0:
        img = cv2.flip(img,1)
        pupil["c"][0] = w - 1 - pupil["c"][0]
        iris["c"][0]  = w - 1 - iris["c"][0]

    # crop
    nh, nw = (h - crop_size[0], w-crop_size[1])
    h1 = iris["c"][1] - iris["r"]
    h2 = iris["c"][1] + iris["r"]
    w1 = iris["c"][0] - iris["r"]
    w2 = iris["c"][0] + iris["r"]
    list_h =[0]
    for i in range(0,h1):
        if i + nh -1 >= h2 and i + nh - 1 <= h-1:
            list_h.append(i)
    list_w = [0]
    for i in range(0,w1):
        if i + nw -1 >=w2 and i + nw -1 <= w-1:
            list_w.append(i)
    start_h = list_h[random.randint(0,len(list_h)-1)]
    start_w = list_w[random.randint(0,len(list_w)-1)]
    img = img[start_h : start_h + nh, start_w:start_w + nw ]
    ##
    pupil["c"][0] -= start_w
    pupil["c"][1] -= start_h
    iris["c"][0] -= start_w
    iris["c"][1] -= start_h

    # reshape
    if scale != 1:
        h,w = img.shape[0],img.shape[1]
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_CUBIC)
        iris["c"][0] //= scale
        iris["c"][1] //= scale
        iris["r"] //= scale
        pupil["c"][0] //= scale
        pupil["c"][1] //= scale
        pupil["r"] //= scale
    # 显示图片
    loc = {"iris": iris, "pupil": pupil}
    # Utils.drawIrisAndShow(img, loc)
    return img,loc

def v4MTCNNAug(img, loc):
    # 虹膜/瞳孔位置
    iris = loc["iris"]
    pupil = loc["pupil"]
    # 白块(模拟反光)
    drawRandomCircle(img,(5,8),(0,10),color_range=(230,255))
    #drawRandomCircle(img,(20,50),(0,1),color_range=(250,255))

    # 黑色线条 (模拟眼镜)
    # drawRandomLine(img,(5,10), (0,3), color_range=(0,50))

    # 翻转+模糊+crop
    seq = iaa.Sequential([
        # iaa.SomeOf((0,1),   [iaa.Fliplr(0.5)]),        # 一半的概率左右翻转
        iaa.OneOf([
            iaa.GaussianBlur((0, 3.0)),
            iaa.AverageBlur(k=(2, 3)),
            iaa.MedianBlur(k=(3, 7)),
        ])
     ])
    img = seq.augment_image(img)

    # 左右翻转
    h, w =img.shape[0],img.shape[1]
    if  random.randint(0,1) == 0:
        img = cv2.flip(img,1)
        pupil["c"][0] = w - 1 - pupil["c"][0]
        iris["c"][0]  = w - 1 - iris["c"][0]

    loc = {"iris": iris, "pupil": pupil}
    # Utils.drawIrisAndShow(img, loc)
    return img,loc

###### TEST #########
if __name__ == "__main__":
    while True:
        file = r"E:\虹膜数据集\S5004R04.jpg"
        loc = {"iris": {"c": [319, 216], "r": 99}, "pupil": {"c": [317, 218], "r": 54}}
        img = plt.imread(file)
        print(type(img))
        img = v4locationAug(img,loc)
        #Utils.showImage(img)
    #img = v4aug(img)
    #
    # points = [(150,150),(200,150),(200,200),(150,200)]
    # img = drawPoints(img,points)
    # Utils.showImage(img)