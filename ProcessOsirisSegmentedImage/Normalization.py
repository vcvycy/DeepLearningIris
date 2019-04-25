# 归一化
import sys
import os
import numpy as np
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())
import Utils
import math

# 缩小瞳孔
# to_r : 半径为图宽比例
import math
import cv2
def dist(x,y, x2, y2):
    return math.sqrt((x-x2)*(x-x2)+ (y-y2)*(y-y2))

# 双线性插值获取某个实数位置像素值
def getPixelBiLinear(img, pos):
    def get(x,y):
        x = max(0,x)
        y = max(0,y)
        assert x<=img.shape[0] and y<=img.shape[1], "shape= %s x=%s y=%s" %(img.shape, x ,y)
        x = min(img.shape[0]-1, x)
        y = min(img.shape[1]-1, y)
        return img[x,y]
    x, y =pos
    x1 = int(x)
    x2 = x1+1
    y1 = int(y)
    y2 = y1+1

    r1 = (x2-x)*get(x1,y1) + (x-x1)* get(x2,y1)
    r2 = (x2-x)*get(x1,y2) + (x-x1)* get(x2,y2)

    value = (y2-y) * r1 + (y-y1)*r2
    return round(value)

# 与x轴夹角
def getAngleWithXAxis(x,y):
    if x==0 and y==0:
        return 0
    flag=False
    if x<0:
        x=-x
        y=-y
        flag=True
    q = y / math.sqrt(x * x + y * y)
    if flag:
        return math.acos(q) + 3.14159265
    else:
        return math.acos(q)

# px ,py,pr是中间要消去的圆形区域
def normalize(src,ix,iy,ir, px, py, pr,angle_start=1,angle_end=360+45,pace=360/512,height=64):  #
    # ix,iy,ir 为虹膜圆心和半径
    # px,py,pr 为瞳孔圆心和半径
    #
    img=src.copy();
    tmp = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2RGB)
    Utils.drawCircle(tmp,ix,iy,ir)
    Utils.drawCircle(tmp,px,py,pr)
    Utils.showImage(tmp)
    h,w = img.shape[:2]
    #
    nor_h = height
    nor_w = int((angle_end-angle_start+1)/pace)
    nor_img = np.zeros((nor_h,nor_w),np.uint8)
    # 生成归一化图像
    for y in range(nor_w):
        theta = ((angle_end-angle_start)*(y/nor_w)+angle_start)/180*3.14159265
        print(theta)
        p_inner = (
            px + pr * math.sin(theta),
            py + pr * math.cos(theta)
        )
        p_outer = (
            ix + ir * math.sin(theta),
            iy + ir * math.cos(theta)
        )
        if y % 20==0:
            cv2.line(src,(int(p_inner[0]),int(p_inner[1])),(int(p_outer[0]),int(p_outer[1])),(0,0,255),2)
            Utils.drawCircle(src,int(p_inner[0]),int(p_inner[1]),1)
            Utils.drawCircle(src,int(p_outer[0]),int(p_outer[1]),1)
        # 将内外点连线并映射
        for x in range(nor_h):
            pos = (
                p_inner[1] + (p_outer[1]-p_inner[1])*((x+1)/nor_h),
                p_inner[0] + (p_outer[0]-p_inner[0])*((x+1)/nor_h)
            )
            #nor_img[x,y] = img[int(pos[0]),int(pos[1])]#
            nor_img[x,y] =  getPixelBiLinear(img, pos)
    Utils.showImage(src)
    nor_img=cv2.equalizeHist(nor_img)
    nor_img = cv2.cvtColor(nor_img.copy(), cv2.COLOR_GRAY2RGB)
    #cv2.line(nor_img,(nor_w//2,0),(nor_w//2,height),(0,0,255),2)
    cv2.imwrite(r"e:\nor.jpg",nor_img)
    Utils.showImage(nor_img)
    return nor_img
import time
if __name__ == "__main__":
    path = r"E:\IrisDataset\S5009L00.jpg"
    #img= normalize(cv2.imread(r"E:\IrisDataset\S5009L01.jpg",cv2.IMREAD_GRAYSCALE),366,266,115,365,272,38)
    img = normalize(cv2.imread(r"E:\IrisDataset\S5009L00.jpg",cv2.IMREAD_GRAYSCALE),380,272,106,379,274,37)
    # path = r"E:\IrisDataset\S5001R00.jpg"
    # img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    # #img= normalize(img,380,268,113,379,274,35)
    # img= normalize(img,366,266,115,365,272,38)
    # #img= normalize(img,224,216,100,221,210,35)