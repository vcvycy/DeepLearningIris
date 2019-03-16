import os
import numpy as np
import tensorflow as tf
import Utils
from DSIris import DSIris
import  DSIrisAug
from  ProcessOsirisSegmentedImage.Step3_GetV4LocationTrainingData import *
import cv2
import sys
import random
# CASIA.v4 Location数据集读取
class DSV4Recog(DSIris):
    # (*) ds_root : 数据集根目录
    #       image_dir : 从哪个文件夹载入
    def __init__(self,sess,image_dir,steps = 10000):
        super().__init__(sess)
        self.batch_K = 5    # 每一个mini_batch，每一类有batch_K 张图片
        self.batch_P = 8    # 每一个mini_batch，有batch_P个类别

        # (*) 获取所有文件名列表.
        filename2path = Utils.getFile2Path(image_dir,"jpg")
        print("[*] 原始图片个数:{0}".format(len(filename2path)))
        label2files=[[] for i in range(2000)]
        for filename in filename2path:
            label = self.__getLabelFromFilename(filename)
            label2files[label].append(filename)

        # (*) 生成三元组
        self.images_path =[]
        for _ in range(steps*self.batch_P):
            label = random.randint(0,1999)
            assert len(label2files[label])==10 ,"error"
            idx = [i for i in range(10)]
            for i in range(1,10):
                x = random.randint(0,i-1)
                idx[i],idx[x] = idx[x],idx[i]
            for i in range(self.batch_K):
                filename = label2files[label][idx[i]]
                self.images_path.append(filename2path[filename])

        # (*) 生成Tensorflow队列
        super().createTFQueue(shuffle=False)
        # (*) 显示数据集状态
        self.showDetail()
        return

    # (*) 从filename转为其label的ID，ID值大小应该从0开始
    def __getLabelFromFilename(self,filename):
        eye_lr = filename[5]
        idx = int(filename[2:5])
        label = int(idx)*2
        if eye_lr == "R":
            label += 1
        return label

    # (*) 显示数据集详情
    def showDetail(self):
        print("[*] V4 Location 数据集" )
        print("     [*] 数据集大小:%s" % (len(self.images_path)))
        return

    # 对图片执行crop 操作
    def crop(self,img,pad_h,pad_w):
        return

    # (*) 获取一个batch
    def getBatch(self,batch_size):
        # (*) 获取 filename -> tensor(int) 格式的batch
        raw_batch = super().getRawBatch(batch_size)
        # (*) 获取batch
        batch = ([],[])
        for item in raw_batch:
            # 原始图片和位置
            filename = os.path.basename(item[0])
            label = self.__getLabelFromFilename(filename)
            img_origin = item[1]                          # h*w = 480 * 640
            img_origin = cv2.resize(img_origin,(240,240), interpolation=cv2.INTER_CUBIC)
            img_origin = np.reshape(img_origin, (240, 240, 1))
            batch[0].append(img_origin)
            batch[1].append(label)
        return batch


########### TEST ###############


if __name__ == "__main__":
    sess = tf.Session()
    dir = r"E:\IrisDataset\CASIA-Iris-Thousand"
    a = DSV4Recog(sess,dir,steps=1000)
    print("--------BATCH-------------")
    batch = a.getBatch(50)
    print(batch[1])
    print(batch[1][0])
    print(batch[0][0].shape);
    Utils.showImage(batch[0][0])