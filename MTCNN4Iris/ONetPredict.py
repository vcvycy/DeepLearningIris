import tensorflow as tf
import os
import Config
from MTCNN4Iris import DSV4MTCNN
from MTCNN4Iris.IrisONet import IrisONet
import cv2
import numpy as np
import Utils
import sys

# 将img 中的虹膜rect 缩放 ，然后返回一组image
def getScaledImagesAndRects(img, position):
    iris = position["iris"]
    _r = iris["r"]
    r = _r
    imgs = []
    rects= []
    while r > 70:
        iris["r"] = r
        rect = Utils.getIrisRectFromPosition(position,img.shape)
        rects.append(rect)
        cur_img = Utils.cropAndResize(img,rect,48)
        imgs.append(cur_img)
        r = round(r*0.9)
    iris["r"]=_r
    return imgs,rects

# 将img 中的虹膜rect 缩放 ，然后返回一组image
def getImagePyramidByPupil(img, position):
    pupil = position["pupil"]
    x = pupil["c"][1]
    y = pupil["c"][0]
    r = 280
    imgs = []
    rects= []
    h ,w = img.shape[:2]
    while r > 50:
        rect = max(x-r,0), max(y-r,0), min(x+r, h-1),min(y+r, w-1)
        rects.append(rect)
        cur_img = Utils.cropAndResize(img,rect,48)
        imgs.append(cur_img)
        r = round(r*0.9)
    return imgs,rects

if __name__ == "__main__":
    # 运行目录
    # experiment_dir = "experiments/onet_final"
    experiment_dir = "experiments/onet_neg_15"
    config = Config.Config(os.path.join(experiment_dir,"config.json"))
    config.show()
    # session
    sess = tf.Session()

    # 数据集
    test_data = DSV4MTCNN.DSV4MTCNN(sess, config.iris_position_json_file, r"E:\CASIA-V4-Location\tmp", config.pos_iou, config.neg_iou)
    # 网络
    onet = IrisONet(sess,config,experiment_dir,target="test")
    try:
        onet.restore(os.path.join(experiment_dir,"model"))
    except:
        print("[!] 无法restore")
    while True:
        filename, img, position = test_data.getOneImageForTest()
        print("[*] Image {}".format(filename))
        # 缩放rect 得到prob 最大的rect
        #imgs,rects = getScaledImagesAndRects(img, position)
        imgs,rects = getImagePyramidByPupil(img, position)
        prob,bbr=onet.predict(imgs)
        Utils.drawIrisAndShow(img,position,copy=False,show=False)

        max_prob =0
        max_idx = 0
        for i in range(len(prob)):
            if prob[i][1] > max_prob:
                max_prob = prob[i][1]
                max_idx = i
            #print("     [*] prob ={0} bbr={1}".format(prob[i],bbr[i]))
            #Utils.showImageWithBBR(img, rects[i], bbr[i])
        print("     [*] prob ={0} bbr={1}".format(prob[max_idx],bbr[max_idx]))
        Utils.showImageWithBBR(img, rects[max_idx], bbr[max_idx])

        # Onet跑2次进行调整两次
        # if True:
        #     calibrate_rect = Utils.bbr_calibrate(rects[max_idx], bbr[max_idx])
        #     img_2  = Utils.cropAndResize(img,calibrate_rect,48)
        #     p2 , bbr2 = onet.predict([img_2])
        #     print("     [*] sencond onet  prob ={0} bbr={1}".format(p2[0],bbr2[0]))
        #     ca2_rect = Utils.bbr_calibrate(calibrate_rect,bbr2[0])
        #     Utils.drawRectsAndShow(img,rects[max_idx],calibrate_rect,ca2_rect)