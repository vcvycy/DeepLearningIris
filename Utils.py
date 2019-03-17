import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import cv2
import os
import random

# 枚举root,返回所有jpg文件，文件名->路径的映射的dict
def getFile2Path(root,suffix="jpg"):
    filename2path = {}
    for r,dirs,files in os.walk(root):
        for file in files:
            if file.split(".")[-1] == suffix:
                filename2path[file] = os.path.join(r,file)
    return filename2path

def showImage(mat,name=None):
    if name==None:
        name = "%d x %d" %(mat.shape[0],mat.shape[1])
    cv2.namedWindow(name)
    cv2.imshow(name,mat)
    cv2.waitKey(0)
    #ia.imshow(mat)
    return

def drawAndShowBaseOnCNNOutput(img, o):
    print(o)
    h,w = img.shape[0] , img.shape[1]
    iris_c = int((o[1]+o[3])/2*w),int((o[0]+o[2])/2*h)
    iris_r = max( int((o[3]-o[1])/2*w) , int((o[2]-o[0])/2*h),0)

    pupil_c = int((o[5]+o[7])/2*w),int((o[4]+o[6])/2*h)
    pupil_r = max( int((o[7]-o[5])/2*w) , int((o[6]-o[4])/2*h),0)
    cv2.circle(img, iris_c, iris_r, (255, 0, 255), thickness=3)
    # cv2.circle(img, pupil_c, pupil_r, (255, 0, 255), thickness=3)
    showImage(img)
    return

# 在图中画虹膜、瞳孔位置，然后显示出来
def drawIrisAndShow(img, postion):
    iris = postion["iris"]
    pupil = postion["pupil"]
    cv2.circle(img, tuple(iris["c"]), iris["r"], (255, 0, 255),thickness=3)
    cv2.circle(img, tuple(pupil["c"]),pupil["r"], (255, 0, 255),thickness=3)
    showImage(img)
    return

# r1,r2= (x,y,x2,y2) 两个点矩阵,x纵轴坐标，y是横轴坐标
def getIOU(r1,r2,method = "Min"):
    area1 = (r1[2]-r1[0])*(r1[3]-r1[1])
    area2 = (r2[2]-r2[0])*(r2[3]-r2[1])
    x = max(r1[0],r2[0])
    y = max(r1[1],r2[1])
    x2 = min(r1[2],r2[2])
    y2 = min(r1[3],r2[3])
    h = max(x2-x,0)
    w = max(y2-y,0)
    # 相交的面积
    inner = w*h
    return inner / (area1+area2-inner)

# 获取随机正方形,min_size = 最短边长
def getRandomSquare(h,w,min_size ):
    # 左上角坐标
    x = random.randint(0,h-1-min_size)
    y = random.randint(0,w-1-min_size)
    # 边长
    size = random.randint(min_size,min(h -1 - x, w - y -1))
    return x,y,x+size , y+size

def bbr_calibrate(rect,bbr):
    h = rect[2]-rect[0]
    w = rect[3]-rect[1]
    print("{0}*{1}".format(h,bbr[0]))
    x = round(rect[0] + h * bbr[0])
    y = round(rect[1] + w * bbr[1])
    x2 = round(rect[2] + h * bbr[2])
    y2 = round(rect[3] + w * bbr[3])
    return x,y,x2,y2

def showImageWithBBR(image,rect,bbr=[0,0,0,0]):
    r = bbr_calibrate(rect,bbr)
    print("rect_calibrate {0}".format(r))
    img = image[r[0]:r[2]+1, r[1]:r[3]+1]
    showImage(img)
    return

def cropAndResize(img,rect,size):
    img = img[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1]
    img = resize(img,(size,size))
    img = np.reshape(img, (size, size, 1))
    return img

def getIrisRectFromPosition(pos):
    iris = pos["iris"]
    x = iris["c"][1]
    y = iris["c"][0]
    r = iris["r"]
    return x-r, y-r ,x+r, y+r

def resize(img,shape):
    return cv2.resize(img,shape, interpolation=cv2.INTER_CUBIC)
if __name__ == "__main__":
    r=(0,0,2,2)
    r2=(1,1,3,3)
    print(getIOU(r,r2))
    print(getRandomSquare(480,640))