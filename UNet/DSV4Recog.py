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
    def __init__(self,sess,
                 image_dir,
                 steps,
                 input_size ,
                 crop_pixels ,
                 batch_img_num_each_class ,
                 batch_class_num,
                 training_classes ):
        super().__init__(sess)
        self.input_size = input_size
        self.crop_pixels = crop_pixels
        self.batch_K = batch_img_num_each_class    # 每一个mini_batch，每一类有batch_K 张图片
        self.batch_P = batch_class_num    # 每一个mini_batch，有batch_P个类别
        self.training_classes = training_classes
        # (*) 获取所有文件名列表.
        self.filename2path = Utils.getFile2Path(image_dir,"jpg|bmp")
        print("[*] 原始图片个数:{0}".format(len(self.filename2path)))
        label2files=[[] for i in range(training_classes)]
        for filename in self.filename2path:
            label = self.__getLabelFromFilename(filename)
            label2files[label].append(filename)

        # (*) 生成三元组： 一共有steps * batch_P * batch_K 个文件名.
        self.images_path =[]
        for _ in range(steps * self.batch_P):  # 一共有steps* batch_P 类次被选中
            label = random.randint(0,self.training_classes-1)
            # assert len(label2files[label])==10 ,"error"
            # 类label的10张图片中，随机选取batch_K 张( 生成0~9的 随机排列，然后取前batch_K张)
            image_num_in_cur_label = len(label2files[label])
            # 随机生成排列
            idx_choosed = [i for i in range(image_num_in_cur_label)]
            for i in range(1,image_num_in_cur_label):
                x = random.randint(0,i-1)
                idx_choosed[i],idx_choosed[x] = idx_choosed[x],idx_choosed[i]
            for i in range(min(self.batch_K, image_num_in_cur_label)):
                filename = label2files[label][idx_choosed[i]]
                self.images_path.append(self.filename2path[filename])

        # (*) 生成Tensorflow队列
        super().createTFQueue(shuffle=False,decode_image= False)
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
        print("     [*] 数据集大小:%s" % (len(self.filename2path)))
        return

    # def toOneHot(self,label , classes):
    #     l = [0 for _ in range(classes)]
    #     l[label] =1
    #     return l

    # (*) 获取一个batch
    def getBatch(self):
        # (*) 获取 filename -> tensor(int) 格式的batch
        batch_size = self.batch_K * self.batch_P
        raw_batch = super().getRawBatch(batch_size)
        # (*) 获取batch
        batch = ([],[])
        for item in raw_batch:
            # 原始图片和位置
            filename = os.path.basename(item[0])
            label = self.__getLabelFromFilename(filename)
            # img_origin = item[1]
            nparr = np.fromstring(item[1], np.uint8)
            img_origin = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            # 先缩放到 size* size 再在高和宽各减去self.crop_pixels个像素
            size = self.input_size + self.crop_pixels
            img_origin = cv2.resize(img_origin,(size, size), interpolation=cv2.INTER_CUBIC)
            img_origin = np.reshape(img_origin, (size, size, 1))
            #
            crop_start_h = random.randint(0,self.crop_pixels)
            crop_start_w = random.randint(0,self.crop_pixels)
            img = img_origin[crop_start_h: crop_start_h+ self.input_size, crop_start_w: self.input_size + crop_start_w]
            batch[0].append(img)
            batch[1].append(label)
        return batch


########### TEST ###############


if __name__ == "__main__":
    sess = tf.Session()
    dir = r"E:\iris_crop"
    a = DSV4Recog(sess,dir,steps=1000, input_size= 200, crop_pixels=0, batch_img_num_each_class=5, batch_class_num= 8)
    print("--------BATCH-------------")
    batch = a.getBatch()
    print(batch[0][0].shape)
    print(len(batch[0]))
    Utils.showImage(batch[0][0])