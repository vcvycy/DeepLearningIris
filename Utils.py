import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import cv2
import os
import random

# 枚举root,返回所有jpg文件，文件名->路径的映射的dict
def getFile2Path(root,suffix="jpg|bmp"):
    suffix_set = set(suffix.split("|"))
    filename2path = {}
    for r,dirs,files in os.walk(root):
        for file in files:
            if file.split(".")[-1] in suffix_set:
                filename2path[file] = os.path.join(r,file)
    return filename2path
# 获取文件扩展名
def getExt(filename):
    return filename.split(".")[-1]

def showImage(mat,name=None):
    if name==None:
        name = "%d x %d" %(mat.shape[0],mat.shape[1])
    cv2.namedWindow(name)
    cv2.imshow(name,mat)
    cv2.waitKey(0)
    #ia.imshow(mat)
    return

def drawAndShowBaseOnCNNOutput(img, o):
    img=img.copy()
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
def drawIrisAndShow(img, postion,copy=True,show=True):
    if copy:
        img = img.copy()
    iris = postion["iris"]
    pupil = postion["pupil"]
    cv2.circle(img, tuple(iris["c"]), iris["r"], (255, 0, 255),thickness=3)
    cv2.circle(img, tuple(pupil["c"]),pupil["r"], (255, 0, 255),thickness=3)
    if show:
        showImage(img)
    return

# r1,r2= (x,y,x2,y2) 两个点矩阵,x纵轴坐标，y是横轴坐标
def getIOU(r1,r2,method = "Union"):
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
    if method == "Union":
        return inner / (area1+area2-inner)
    elif method == "Min":
        return inner / min(area1,area2)
    else:
        raise Exception("method")

# 获取随机正方形,

def getRandomLenSquare(h,w,min_len = 12):
    # 以下是先随机边长，后随机位置
    size = random.randint(min_len, min(h,w))      # 不同边长的矩形并非等概率分布，会导致边长越大的正方形，概率越高，看作等概率分布
    x = random.randint(0, h - size)
    y = random.randint(0, w - size)
    return x, y, x+size-1 , y+size-1

def bbr_calibrate(rect,bbr, shape=None):
    h = rect[2]-rect[0]
    w = rect[3]-rect[1]
    x = int(round(rect[0] + h * bbr[0]))
    y = int(round(rect[1] + w * bbr[1]))
    x2 = int(round(rect[2] + h * bbr[2]))
    y2 = int(round(rect[3] + w * bbr[3]))

    if shape == None:
        return x , y ,x2 ,y2
    else:
        h,w = shape[:2]
        return max(0,x), max(0,y), min(h-1, x2), min(w-1, y2)

def showImageWithBBR(image,rect,bbr=[0,0,0,0],copy=True,show=True):
    if copy:
        image = image.copy()
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # image = cv2.rectangle(image,(rect[1],rect[0]),(rect[3],rect[2]),color=(0,0,255),thickness=1)
    r = bbr_calibrate(rect,bbr)
    image = cv2.rectangle(image,(r[1],r[0]),(r[3],r[2]),color=(0,255,0),thickness=2)
    if show:
        showImage(image)
    return image

# 画出pupil，其是按百分比表示的
def drawPupilPercent(img, pupil):
    h,w = img.shape[:2]
    p = (
        int(h*pupil[0]),
        int(w*pupil[1]),
        int(h*pupil[2]),
        int(w*pupil[3])
    )
    drawRectsAndShow(img,p)
    return

# 在图中画多个矩形
def drawRectsAndShow(img,*rects):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    c=[
        (0,255,0),
        (0,0,255),
        (255,0,0)
    ]
    num = len(rects)
    for i in range(num):
        r = rects[i]
        img = cv2.rectangle(img, (r[1], r[0]), (r[3], r[2]), color=c[i], thickness=1)
    showImage(img)
    return

def rndColor():
    b = random.randint(0,255)
    g = random.randint(0,255)
    r = random.randint(0,255)
    return (b,g,r)

def shrinkRect(rect, share=0.08):
    h,w = rect[2]-rect[0], rect[3]-rect[1]
    pad = int(h* share)
    return (
        rect[0]+pad,
        rect[1]+pad,
        rect[2]-pad,
        rect[3]-pad
    )

def drawRectsListAndShow(img,rects):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    for r in rects:
        img = cv2.rectangle(img, (r[1], r[0]), (r[3], r[2]), color=rndColor(), thickness=2)
    showImage(img)
    return

def cropAndResize(img,rect,size=None):
    img = img[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1]
    if size!=None:
        img = resize(img,(size,size))
        img = np.reshape(img, (size, size, 1))
    return img

# 获取虹膜rect，如果shape!=None,则限制rect在shape范围内
def getIrisRectFromPosition(pos,shape=None):
    iris = pos["iris"]
    x = iris["c"][1]
    y = iris["c"][0]
    r = iris["r"]
    if shape== None:
        return x-r, y-r ,x+r, y+r
    else:
        h ,w = shape[:2]
        return max(x-r,0), max(y-r,0), min(x+r, h-1),min(y+r, w-1)
def getPupilRectFromPosition(pos, shape=None):
    pupil = pos["pupil"]
    x = pupil["c"][1]
    y = pupil["c"][0]
    r = pupil["r"]
    if shape== None:
        return x-r, y-r ,x+r, y+r
    else:
        h ,w = shape[:2]
        return max(x-r,0), max(y-r,0), min(x+r, h-1),min(y+r, w-1)


# Non - Maximum- Suppression , 返回一个列表，true表示保留下来，false表示被删除
def nms(rects, probs, iou_threshold, method):  # method : Union/Min
    is_remained = []
    n = len(rects)
    for i in range(n):
        remain = True
        r1 = rects[i]
        p1 = probs[i]
        for j in range(n):
            if i != j:
                r2 = rects[j]
                p2 = probs[j]
                iou = getIOU(r1,r2,method)
                if iou > iou_threshold and p1 < p2:
                    remain=False
                    break
        is_remained.append(remain)
    return is_remained

def resize(img,shape):
    return cv2.resize(img,shape, interpolation=cv2.INTER_CUBIC)

# 返回图像金字塔，已经其缩放倍数
def getImagePyramid(img,min_iris_size = 100,max_iris_size = 300):
    # 图像金字塔
    images = []
    iris_size = max_iris_size
    h,w = img.shape[:2]
    while iris_size > min_iris_size:
        scale = 12.0 / iris_size
        w2,h2 = (round(w*scale), round(h*scale))
        scaled_img = resize(img,(w2,h2))
        scaled_img = np.reshape(scaled_img,(h2,w2,1))
        # print(scaled_img.shape)
        # showImage(scaled_img)
        images.append((scaled_img,scale))
        iris_size *= 0.709
    return images


# 转为方形，切除超出方形的区域
def toSquareShape(rect):
    h = rect[2] - rect[0]
    w = rect[3] - rect[1]
    if w > h:
        r = rect[0], rect[1]+ (w-h)//2, rect[2] , rect[3]-(w-h+1)//2
    else:
        r = rect[0]+ (h-w)//2, rect[1], rect[2] - (h-w+1)//2, rect[3]
    assert r[2]-r[0] == r[3]-r[1] , " rect :{}".format(r)
    return r

def rect_pad_and_crop(img,rect,pad=0):
    # rect padding
    rect = list(rect)
    h,w = img.shape[:2]
    pad_top_left = min(pad, rect[0], rect[1])
    rect[0] -= pad_top_left
    rect[1] -= pad_top_left
    pad_right_bottom = min(pad, h-1-rect[2], w-1-rect[3])
    rect[2] += pad_right_bottom
    rect[3] += pad_right_bottom
    #
    print(rect)
    img = img[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1]
    return img

def getRecallAndPrecision(dist,label, threshold):
    n =len(label)
    need_recall_num = 0                  # 所有需要recall 的pair
    real_recall_num = 0             # 被recall 的个数
    recall_correct_num = 0         # 被recall，且正确的个数
    for i in range(n):
        for j in range(i+1,n):
            l1 = label[i]
            l2 = label[j]
            if l1 == l2:
                need_recall_num += 1
            if dist[i][j] <= threshold:
                real_recall_num += 1
                if l1 == l2:
                    recall_correct_num += 1
    if need_recall_num!=0:
        recall = recall_correct_num / need_recall_num
    else:
        recall =0
    if real_recall_num!=0:
        precision = recall_correct_num / real_recall_num
    else:
        precision = 0
    return recall, precision , (recall_correct_num , real_recall_num , need_recall_num)

def getBBR(predict,label):
    rh = predict[2] - predict[0]
    rw = predict[3] - predict[1]
    return (
        (label[0] - predict[0]) / rh,
        (label[1] - predict[1]) / rw,
        (label[2] - predict[2]) / rh,
        (label[3] - predict[3]) / rw
    )
if __name__ == "__main__":
    r=(0,0,2,2)
    r2=(1,1,3,3)
    print(getIOU(r,r2))