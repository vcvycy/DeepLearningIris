# 缩小瞳孔
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
def shrink_pupil(src,ix,iy,ir, px, py, pr):  #
    img = src.copy()
    h,w = img.shape[:2]
    for x in range(h):
        for y in range(w):
            # 如果当前像素在虹膜区域，则重写
            dist2iris = dist(x,y,ix,iy)   # 与虹膜中心点距离
            if dist2iris < ir:
                theta = getAngleWithXAxis(x-ix, y-iy)            # 夹角
                k = dist2iris / ir                             # 距离比例
                # 映射到原图上的坐标点
                p_inner = (
                    px + pr * math.sin(theta),
                    py + pr * math.cos(theta)
                )
                p_outer = (
                    ix + ir * math.sin(theta),
                    iy + ir * math.cos(theta)
                )
                pos = (
                    p_inner[0] + (p_outer[0]-p_inner[0])*k,
                    p_inner[1] + (p_outer[1]-p_inner[1])*k,
                )
                img[x,y] = getPixelBiLinear(src, pos)
    return img

def shrink_pupil_in_irisimage(img,px,py,pr):
    h,w = img.shape[:2]
    ix = h//2
    iy = w//2
    ir = (h+w)//4
    return  shrink_pupil(img,ix,iy,ir, px, py, pr)

if __name__ == "__main__":
    path = r"E:\CASIA-V4-Location\train\S5000L00.jpg"
    # path = r"E:\iris_crop_onet_with_proposal\S5004L08.jpg"
    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    Utils.showImage(img)
    for x in range(0,40,10):
        img = shrink_pupil(img,253,262,91,258,264,x)
    # img = shrink_pupil_in_irisimage(img, 114,100,30)
        Utils.showImage(img)