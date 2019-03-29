import tensorflow as tf
import os
import Config
from MTCNN4Iris import DSV4MTCNN
from MTCNN4Iris.IrisPNet import IrisPNet
from MTCNN4Iris.IrisONet import IrisONet
import cv2
import numpy as np
import Utils
import sys
from  MTCNN4Iris.PNetPredict import PNetPredictor
class Predictor:
    def __init__(self, pnet_dir , onet_dir):
        self.sess = tf.Session()
        # pnet
        config_pnet = Config.Config(os.path.join(pnet_dir, "config.json"))
        # config_pnet.show()
        self.pnet_predictor = PNetPredictor(self.sess, config_pnet, pnet_dir)

        # onet
        config_onet = Config.Config(os.path.join(onet_dir, "config.json"))
        self.onet = IrisONet(self.sess, config_onet, onet_dir, target="test")
        try:
            self.onet.restore(os.path.join(onet_dir, "model"))
        except:
            print("[!] onet无法restore")
            exit(0)

    def predict(self,img, pnet_threshold ):
        # Pnet
        rects = self.pnet_predictor.predict(img, threshold = pnet_threshold,min_size=120)
        if len(rects) == 0:
            rects =  self.pnet_predictor.predict(img, threshold = 0.2)
        if len(rects) == 0:
            raise Exception("PNet 找不到rect")
        Utils.drawRectsListAndShow(img, rects)
        # ONet
        final = (0, None, None)
        for rect in rects:
            rect = Utils.toSquareShape(rect)
            region = Utils.cropAndResize(img, rect, 48)
            prob, bbr = self.onet.predict([region])
            if prob[0][1] > final[0]:
                final = (prob[0][1], rect, bbr[0])
            #if prob[0][1] > 0.4:
                # print("[*]prob={0}".format(prob[0][1]))
                # Utils.showImageWithBBR(img,rect,bbr[0])
        # print("[*] final = {}".format(final))
        # Utils.showImageWithBBR(img, final[1], final[2])
        rect = Utils.bbr_calibrate(final[1],final[2])
        return img,rect

if __name__ == "__main__":
    # 运行目录
    #pnet_dir = "experiments/pnet1"
    pnet_dir = "experiments/pnet_expand_0.08"
    #onet_dir = "experiments/onet_neg_15_pupil"
    onet_dir = "experiments/onet_with_proposal"
    predictor = Predictor(pnet_dir, onet_dir)
    pnet_threshold = 0.3
    # 数据集
    # test_dir = r"E:\IrisDataset\CASIA-Iris-Thousand"
    # save_to  = r"e:\iris_crop"
    test_dir = r"E:\CASIA-V4-Location\tmp"
    save_to = None
    filename2path = Utils.getFile2Path(test_dir,suffix="jpg")
    print("[*] size in {0} : {1}".format(test_dir,len(filename2path)))
    for filename in filename2path:
        path = filename2path[filename]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        try:
            img,rect = predictor.predict(img,pnet_threshold= pnet_threshold)
            rect = Utils.toSquareShape(rect)
            Utils.drawRectsAndShow(img,rect)
            if save_to!=None:
                img_crop = Utils.rect_pad_and_crop(img,rect,10)
                cv2.imwrite(os.path.join(save_to, filename),img_crop)
        except Exception as e:
            print("[!] filename{0}  exception:{1}".format(filename, e))
            input("")