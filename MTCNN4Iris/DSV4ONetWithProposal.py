# PNet的output作为Onet的input

import os
import numpy as np
import tensorflow as tf
import Utils
from DSIris import DSIris
import  DSIrisAug
from  ProcessOsirisSegmentedImage.Step3_GetV4LocationTrainingData import *
import cv2
import sys
import math
import random
from MTCNN4Iris import PNetPredict

# CASIA.v4 Location数据集读取
class DSV4ONetWithProposal(DSIris):
    def __init__(self,sess,dir,target="train"):
        super().__init__(sess)
        # (*) 获取所有文件名列表
        filename2path = Utils.getFile2Path(dir,"jpg")
        self.images_path = [filename2path[filename] for filename in filename2path]
        # (*) 获取文件对应的label值,和某个label对应的所有文件
        label_file_path = os.path.join(dir, "label.json")
        self.filename2label = json.loads(open(label_file_path,"r").read())
        # print("%s -> %s" %(len(self.filename2label), len(self.images_path)))
        # assert len(self.filename2label) == len(self.images_path)
        # (*) 生成Tensorflow队列
        if target== "train":
            super().createTFQueue(num_epochs= 30,shuffle= True)
        else:
            super().createTFQueue(num_epochs=1,shuffle=False)
        # (*) 显示数据集状态
        self.showDetail()
        return


    # (*) 显示数据集详情
    def showDetail(self):
        print("[*] V4 Location 数据集" )
        print("     [*] 数据集大小:%s" % (len(self.images_path)))
        return

    def getBatch(self,batch_size):
        raw_batch = super().getRawBatch(batch_size)
        batch = ([], [], [],[])
        for item in raw_batch:
            filename = os.path.basename(item[0])
            if filename not in self.filename2label:
                continue
            img = DSIrisAug.onet_with_proposal_aug(item[1])
            #Utils.showImage(img)
            batch[0].append(img)  # image
            prob = self.filename2label[filename]["prob"]

            if prob == 1:
                batch[1].append([0,1])
                batch[2].append(self.filename2label[filename]["bbr"])
                # print(self.filename2label[filename])
                batch[3].append(self.filename2label[filename]["pupil_rect"])
            else:
                batch[1].append([1,0])
                batch[2].append([0,0,0,0])
                batch[3].append([0,0,0,0])
        return batch

if __name__ == "__main__":
    dir=r"E:\CASIA-V4-Location\PNetProposal"
    sess = tf.Session()
    a = DSV4ONetWithProposal(sess, dir)
    b = a.getBatch(100)
    pos_cnt =0
    for x in b[1]:
        if x[1]==1:
            pos_cnt+=1
    print(pos_cnt)