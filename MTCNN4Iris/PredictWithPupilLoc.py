import tensorflow as tf
import os
import Config
from MTCNN4Iris import DSV4MTCNN
from MTCNN4Iris.IrisPNet import IrisPNet
from MTCNN4Iris.IrisONet import IrisONet
import cv2
from ProcessOsirisSegmentedImage import PupilShrink
import numpy as np
import json
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
        self.onet = IrisONet(self.sess, config_onet, onet_dir, target="test", use_pupil_loc= True)
        try:
            self.onet.restore(os.path.join(onet_dir, "model"))
        except:
            print("[!] onet无法restore")
            exit(0)

    def predict(self,img, pnet_threshold):
        # Pnet
        rects = self.pnet_predictor.predict(img, threshold = pnet_threshold,min_size=150, nms_threshold= 0.75)
        if len(rects) == 0:
            rects =  self.pnet_predictor.predict(img, threshold = 0.2)
        if len(rects) == 0:
            raise Exception("PNet 找不到rect")
        # Utils.drawRectsListAndShow(img, rects)
        # ONet
        final = (0, None, None)
        for rect in rects:
            rect = Utils.toSquareShape(rect)
            region = Utils.cropAndResize(img, rect, 48)
            prob, bbr ,pupil= self.onet.predictWithPupilLoc([region])
            if prob[0][1] > final[0]:
                final = (prob[0][1], rect, bbr[0] ,pupil[0])
        rect = Utils.bbr_calibrate(final[1],final[2])
        p = final[3]
        p = (
             (p[0] - final[2][0]) / (1-final[2][0]+ final[2][2]),
             (p[1] - final[2][1]) / (1-final[2][1]+ final[2][3]),
             (p[2] - final[2][0]) / (1-final[2][0]+ final[2][2]),
             (p[3] - final[2][1]) / (1-final[2][1]+ final[2][3])
        )
        return rect ,p

if __name__ == "__main__":
    # 运行目录
    pnet_dir = "experiments/pnet_expand_0.08"
    # onet_dir = "experiments/onet_neg_15_pupil"
    onet_dir = "experiments/onet_with_proposal"
    predictor = Predictor(pnet_dir, onet_dir)
    pnet_threshold = 0.3
    # 数据集
    test_dir = r"E:\IrisDataset\CASIA-Iris-Thousand"
    save_to  = r"e:\iris_crop_onet_with_proposal"
    pupil_position_json ={}
    # test_dir = r"E:\CASIA-V4-Location\test"
    # save_to = None
    #########
    filename2path = Utils.getFile2Path(test_dir,suffix="jpg")
    print("[*] size in {0} : {1}".format(test_dir,len(filename2path)))
    for filename in filename2path:
        print(filename)
        path = filename2path[filename]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 预测得到 虹膜/瞳孔 rect
        iris_rect, pupil_percent = predictor.predict(img, pnet_threshold= pnet_threshold)
        iris_rect = Utils.toSquareShape(iris_rect)
        # 虹膜图片
        iris_image = Utils.cropAndResize(img, iris_rect)
        pupil_rect = (
            int(pupil_percent[0] * iris_image.shape[0]),
            int(pupil_percent[1] * iris_image.shape[1]),
            int(pupil_percent[2] * iris_image.shape[0]),
            int(pupil_percent[3] * iris_image.shape[1])
        )
        # 瞳孔缩小
        pupil_rect = Utils.toSquareShape(pupil_rect)
        pupil_shrink_pixels = max(0, (pupil_rect[2]-pupil_rect[0]-8)//2)
        px, py , pr = (pupil_rect[2]+pupil_rect[0])//2 , (pupil_rect[3]+pupil_rect[1])//2 , max(0,(pupil_rect[2]-pupil_rect[0])//2-15)

        if save_to!=None:
            pupil_position_json[filename] = {"iris":iris_rect, "pupil":pupil_rect}
            cv2.imwrite(os.path.join(save_to, filename), iris_image)
        else:
            # 在瞳孔中画虹膜
            Utils.drawPupilPercent(iris_image,pupil_percent)
            img_shrink = PupilShrink.shrink_pupil_in_irisimage(iris_image, px, py, pr)
            Utils.showImage(img_shrink)
        #break
    # 保存瞳孔信息
    f_pupil = open(os.path.join(save_to,"pupil.json"),"w")
    f_pupil.write(json.dumps(pupil_position_json))