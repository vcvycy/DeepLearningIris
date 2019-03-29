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
class DSV4MTCNN(DSIris):
    # (*) ds_root : 数据集根目录
    #       json_file : 保存虹膜定位信息的文件
    #       image_dir : 从哪个文件夹载入
    def __init__(self,sess,json_file,image_dir,pos_iou, neg_iou, target="train"):
        self.pos_iou = pos_iou
        self.neg_iou = neg_iou
        super().__init__(sess)
        # (*) 获取所有文件名列表.
        filename2path, filename2position = main_location(json_file, image_dir,show=False)
        self.images_path = [filename2path[filename] for filename in filename2path]
        # print(self.images_path[0])
        # (*) 获取文件对应的label值,和某个label对应的所有文件
        self.filename2label = filename2position

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

    def cropForONet(self,img,rect):
        img = img[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1]
        img = Utils.resize(img,(48,48))
        img = np.reshape(img, (48, 48, 1))
        return img

    def cropForPNet(self,img,rect):
        img = img[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1]
        img = Utils.resize(img,(12,12))
        img = np.reshape(img, (12, 12, 1))
        return img

    def getOneImageForTest(self):
        raw_batch = super().getRawBatch(1)
        filename = os.path.basename(raw_batch[0][0])
        img = raw_batch[0][1]
        label = self.filename2label[filename]
        return filename,img,label

        # (*) 获取一个batch  image , prob, bbr 三元组

    def getBatchForONet(self, batch_size, pos_region_each_image, neg_region_each_image):  # batch_size :4的倍数

        # 正负样本
        # (*) 获取 filename -> tensor(int) 格式的batch
        fetch_image_size = math.ceil(batch_size / (pos_region_each_image + neg_region_each_image))
        raw_batch = super().getRawBatch(fetch_image_size)

        # (*) 获取batch,
        batch = ([], [], [])
        for item in raw_batch:
            # 原始图片和位置
            filename = os.path.basename(item[0])

            # Image Augmentation
            img, label_dict = DSIrisAug.v4MTCNNAug(item[1], self.filename2label[filename])
            h, w = img.shape[0], img.shape[1]

            # 虹膜Ground True Box
            rect_iris = Utils.getIrisRectFromPosition(label_dict, img.shape)
            # Utils.showImageWithBBR(img,rect_iris,[0,0,0,0],copy=True)
            # 获取正样本
            for i in range(pos_region_each_image):
                cnt = 0
                while True:  # 拒绝采样法
                    # rect = random_square_generator.generate()
                    rect = Utils.getRandomLenSquare(h, w)
                    iou = Utils.getIOU(rect, rect_iris)
                    if iou > self.pos_iou:
                        # 找到正样本
                        region = self.cropForONet(img, rect)
                        batch[0].append(region)
                        batch[1].append([0, 1])

                        # 区域的bbr
                        rh = rect[2] - rect[0]
                        rw = rect[3] - rect[1]
                        bbr_label = (
                            (rect_iris[0] - rect[0]) / rh,
                            (rect_iris[1] - rect[1]) / rw,
                            (rect_iris[2] - rect[2]) / rh,
                            (rect_iris[3] - rect[3]) / rw
                        )
                        batch[2].append(bbr_label)
                        # print("iou={0} rect={1} rect_iris={2}".format(iou,rect,rect_iris))
                        # Utils.showImageWithBBR(img,rect,bbr_label)
                        # Utils.showImageWithBBR(item[1],rect_iris,[0,0,0,0])
                        break
            # 获取负样本
            for _ in range(neg_region_each_image):
                while True:  # 拒绝采样法
                    # rect = Utils.getRandomSquare(h,w)
                    # rect = random_square_generator.generate()
                    rect = Utils.getRandomLenSquare(h, w)
                    iou = Utils.getIOU(rect, rect_iris)
                    if iou < self.neg_iou:
                        # 找到负
                        region = self.cropForONet(img, rect)
                        batch[0].append(region)
                        batch[1].append([1, 0])
                        batch[2].append([0, 0, 0, 0])
                        # Utils.showImage(region)
                        break

        return batch
        # (*) 获取一个batch  image , prob, bbr 三元组

    def getBatchForONetWithPupil(self, batch_size, pos_region_each_image, neg_region_each_image):  # batch_size :4的倍数

        # 正负样本
        # (*) 获取 filename -> tensor(int) 格式的batch
        fetch_image_size = math.ceil(batch_size / (pos_region_each_image + neg_region_each_image))
        raw_batch = super().getRawBatch(fetch_image_size)

        # (*) 获取batch,
        batch = ([], [], [], [])
        for item in raw_batch:
            # 原始图片和位置
            filename = os.path.basename(item[0])

            # Image Augmentation
            img, label_dict = DSIrisAug.v4MTCNNAug(item[1], self.filename2label[filename])
            h, w = img.shape[0], img.shape[1]

            # 虹膜Ground True Box
            rect_iris = Utils.getIrisRectFromPosition(label_dict, img.shape)
            rect_pupil = Utils.getPupilRectFromPosition(label_dict, img.shape)
            # Utils.showImageWithBBR(img,rect_iris,[0,0,0,0],copy=True)
            # 获取正样本
            for i in range(pos_region_each_image):
                cnt = 0
                while True:  # 拒绝采样法
                    rect = Utils.getRandomLenSquare(h, w)
                    iou = Utils.getIOU(rect, rect_iris)
                    if iou > self.pos_iou:
                        # 找到正样本
                        region = self.cropForONet(img, rect)
                        batch[0].append(region)
                        batch[1].append([0, 1])

                        # 区域的bbr
                        rh = rect[2] - rect[0]
                        rw = rect[3] - rect[1]
                        bbr_label = (
                            (rect_iris[0] - rect[0]) / rh,
                            (rect_iris[1] - rect[1]) / rw,
                            (rect_iris[2] - rect[2]) / rh,
                            (rect_iris[3] - rect[3]) / rw
                        )
                        batch[2].append(bbr_label)
                        rect_len = rect[2]-rect[0]
                        pupil_pos = (
                            (rect_pupil[0] - rect[0])/rect_len,
                            (rect_pupil[1] - rect[1])/rect_len,
                            (rect_pupil[2] - rect[0])/rect_len,
                            (rect_pupil[3] - rect[1])/rect_len
                        )

                        batch[3].append(pupil_pos)
                        # print("iou={0} rect={1} rect_iris={2}".format(iou,rect,rect_iris))
                        # Utils.showImageWithBBR(img,rect,bbr_label)
                        # Utils.showImageWithBBR(item[1],rect_iris,[0,0,0,0])
                        break
            # 获取负样本
            for _ in range(neg_region_each_image):
                while True:  # 拒绝采样法
                    # rect = Utils.getRandomSquare(h,w)
                    # rect = random_square_generator.generate()
                    rect = Utils.getRandomLenSquare(h, w)
                    iou = Utils.getIOU(rect, rect_iris)
                    if iou < self.neg_iou:
                        # 找到负
                        region = self.cropForONet(img, rect)
                        batch[0].append(region)
                        batch[1].append([1, 0])
                        batch[2].append([0, 0, 0, 0])
                        batch[3].append([0, 0, 0, 0])
                        # Utils.showImage(region)
                        break

        return batch

    # (*) 获取一个batch给PNet训练  image , prob, bbr 三元组
    def getBatchForPNet(self,batch_size,pos_region_each_image,neg_region_each_image):  # batch_size :4的倍数
        # 正负样本
        # (*) 获取 filename -> tensor(int) 格式的batch. 每张图片会取出(neg_xx + pos_xx) 的区域用于训练
        fetch_image_size = (batch_size+ neg_region_each_image+pos_region_each_image-1)//(neg_region_each_image+pos_region_each_image)
        raw_batch = super().getRawBatch(fetch_image_size)
        # (*) 获取batch,
        batch = ([],[],[])
        self.last_files = []
        for item in raw_batch:
            # 原始图片和位置
            filename = os.path.basename(item[0])
            self.last_files.append(filename)
            # Image Augmentation
            img,label_dict = DSIrisAug.v4MTCNNAug(item[1], self.filename2label[filename].copy())
            h,w = img.shape[:2]

            # 虹膜Ground True Box
            rect_iris = Utils.getIrisRectFromPosition(label_dict, shape=(h,w))
            # 获取正样本
            for i in range(pos_region_each_image):
                while True:  # 拒绝采样法
                    rect = Utils.getRandomLenSquare(h,w)
                    # rect = random_square_generator.generate()
                    iou = Utils.getIOU(rect, rect_iris)
                    if iou > self.pos_iou:
                        # 找到正样本
                        region = self.cropForPNet(img,rect)
                        batch[0].append(region)
                        batch[1].append([0,1])

                        # 区域的bbr
                        rh = rect[2]-rect[0]
                        rw = rect[3]-rect[1]
                        bbr_label = (
                                (rect_iris[0] - rect[0])/rh,
                                (rect_iris[1] - rect[1])/rw,
                                (rect_iris[2] - rect[2])/rh,
                                (rect_iris[3] - rect[3])/rw
                        )
                        batch[2].append(bbr_label)
                        break
            # 获取负样本
            for _ in range(neg_region_each_image):
                while True:  # 拒绝采样法
                    rect = Utils.getRandomLenSquare(h,w)
                    # rect = random_square_generator.generate()
                    iou = Utils.getIOU(rect,rect_iris)
                    if iou < self.neg_iou:
                        # 找到负
                        region = self.cropForPNet(img,rect)
                        batch[0].append(region)
                        batch[1].append([1,0])
                        batch[2].append([0,0,0,0])
                        break

        return batch


########### TEST ###############
import Config

if __name__ == "__main__":
    sess = tf.Session()
    json_file = r"E:\CASIA-V4-Location\Iris_Pupil_Position.json"   # 格式为 V4_ROOT/000/L/SXXX.jpg
    location_data_root = r"E:\CASIA-V4-Location\train"
    a = DSV4MTCNN(sess,json_file,location_data_root,0.7, 0.3)
    print("--------BATCH-------------")

    while True:
        b1 = a.getBatchForONetWithPupil(1,1,1)
        img = Utils.resize(b1[0][0],(300,300))
        Utils.drawPupilPercent(img, b1[3][0])
        Utils.showImage(img)
    # for x in range(50):
    #     img = batch[0][x]
    #     prob = batch[1][x]
    #     bbr  = batch[1][x]
    #     print("prob = {0}".format(prob))
    #     print(img.shape)
    #     Utils.showImage(img )