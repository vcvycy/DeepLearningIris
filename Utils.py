import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import cv2
import os

# 枚举root,返回所有jpg文件，文件名->路径的映射的dict
def getFile2Path(root,suffix="jpg"):
    filename2path = {}
    for r,dirs,files in os.walk(root):
        for file in files:
            if file.split(".")[-1] == suffix:
                filename2path[file] = os.path.join(r,file)
    return filename2path

def showImage(mat):
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