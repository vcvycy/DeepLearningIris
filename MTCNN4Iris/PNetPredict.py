import tensorflow as tf
import os
import Config
from MTCNN4Iris import DSV4MTCNN
from MTCNN4Iris.IrisPNet import IrisPNet
import cv2
import numpy as np
import Utils
import sys
class PNetPredictor():
    def __init__(self,sess,config,experiment_dir):
        self.experiment_dir = experiment_dir
        self.config = config
        # session
        self.sess = sess
        # 网络
        self.pnet = IrisPNet(sess, self.config, experiment_dir, target="test")
        try:
            self.pnet.restore(os.path.join(experiment_dir, "model"))
        except:
            print("[!] 无法restore")
            exit(0)

    # 处理一张图片经过Pnet后的输出，返回rect, prob,bbr  三元组列表
    def parsePNetOutput(self,prob,bbr, scale, pnet_threshold ):
        rects_prob_bbr = []
        h,w = prob.shape[:2]
        for x in range(h):
            for y in range(w):
                if prob[x][y][1] > pnet_threshold:
                    rect = (
                        round(x*2/scale),
                        round(y*2/scale),
                        round((x*2+11)/scale),
                        round((y*2+11)/scale)
                    )
                    rects_prob_bbr.append((rect,prob[x][y][1], bbr[x][y]))
        return rects_prob_bbr

    def predict(self,img ,threshold,min_size=0, nms_threshold =0.7):
        if len(img.shape) == 2:
            h, w = img.shape
            img = np.reshape(img, (h, w, 1))

        assert  img.shape[2] == 1
        # 图像金字塔，然后放入网络运行
        img_pyramid = Utils.getImagePyramid(img)
        tmp = []
        results = []                             # 保存所有图片的满足threshold> 0.7的方框。( rect , prob ,bbr) 三元组列表
        for s_img,scale in img_pyramid:
            prob,bbr = self.pnet.predict([s_img])
            rects_prob_bbr = self.parsePNetOutput(prob[0],bbr[0],scale,pnet_threshold = threshold)
            for item in rects_prob_bbr:
                r = item[0]
                tmp.append(r)
                if r[2]-r[0] > min_size:
                    results.append(item)
                # print(item[2])
        print(tmp[0])
        Utils.drawRectsListAndShow(img,tmp)
        # nms
        rects_bbr = [Utils.bbr_calibrate(item[0],item[2]) for item in results]
        probs = [item[1] for item in results]
        is_remained = Utils.nms(rects_bbr, probs, iou_threshold= nms_threshold, method= "Union")

        rst_nms = []
        for i in range(len(results)):
            if is_remained[i]:
                rst_nms.append(results[i])
        # 返回bbr 后的rects
        shape = img.shape
        calibrated_rects=[]
        for item in rst_nms:
            r = Utils.bbr_calibrate(item[0], item[2], shape)
            if r[2]-r[0] > min_size and r[3]-r[1] > min_size:
                calibrated_rects.append(r)
        for item in results:
            print(item[2])
        return calibrated_rects,[item[0] for item in results]


if __name__ == "__main__":
    # 运行目录
    # experiment_dir = "experiments/pnet1"
    experiment_dir = "experiments/pnet_expand_0.08"
    config = Config.Config(os.path.join(experiment_dir,"config.json"))
    config.show()
    sess = tf.Session()
    pnet_predictor = PNetPredictor(sess,config, experiment_dir)
    # 数据集
    test_dir = r"E:\CASIA-V4-Location\tmp"
    filename2path = Utils.getFile2Path(test_dir,suffix="jpg")
    print("[*] size in {0} : {1}".format(test_dir,len(filename2path)))
    for filename in filename2path:
        path = filename2path[filename]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        rects = pnet_predictor.predict(img,threshold= 0.3)
        Utils.drawRectsListAndShow(img,rects)