import os
import numpy as np
import tensorflow as tf
import Utils
from DSIris import DSIris
import  DSIrisAug
from  ProcessOsirisSegmentedImage.Step3_GetV4LocationTrainingData import *
import cv2
# CASIA.v4 Location数据集读取
class DSV4Location(DSIris):
    # (*) ds_root : 数据集根目录
    #       json_file : 保存虹膜定位信息的文件
    #       image_dir : 从哪个文件夹载入
    def __init__(self,sess,json_file,image_dir):
        super().__init__(sess)
        # (*) 获取所有文件名列表.
        filename2path,filename2position = main_location(json_file, image_dir,show=False)
        self.images_path = [filename2path[filename] for filename in filename2path]

        # (*) 获取文件对应的label值,和某个label对应的所有文件
        self.filename2label = filename2position

        # (*) 生成Tensorflow队列
        super().createTFQueue()
        # (*) 显示数据集状态
        self.showDetail()
        return

    # (*) 从filename转为其label的ID，ID值大小应该从0开始
    def __getLabelFromFilename(self,filename):
        parts = filename.split(os.path.sep)
        label = int(parts[-3])*2
        if parts[-2] == "L":
            label+=1
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
            img_origin = item[1]                          # h*w = 480 * 640
            label_dict = self.filename2label[filename]

            #
            # 对图片进行resize，缩小2倍
            scale = 1
            img,label_dict = DSIrisAug.v4locationAug(img_origin,label_dict,scale= scale)
            h,w = img.shape[0],img.shape[1]
            img = np.reshape(img,(h,w,1))                # 灰度图，channel设为1
            # Utils.drawIrisAndShow(img,label_dict)

            # 坐标映射到0-1区间
            pupil = label_dict["pupil"]
            iris  = label_dict["iris"]
            h,w = img.shape[0],img.shape[1]
            label = [
                 (iris["c"][1] - iris["r"])/h,
                 (iris["c"][0] - iris["r"])/w,
                 (iris["c"][1] + iris["r"])/h,
                 (iris["c"][0] + iris["r"])/w,
                 (pupil["c"][1] - pupil["r"])/h,
                 (pupil["c"][0] - pupil["r"])/w,
                 (pupil["c"][1] + pupil["r"])/h,
                 (pupil["c"][0] + pupil["r"])/w,
            ]
            # 防止超出 [0,1]范围
            for i in range(len(label)):
                label[i] = min(1,label[i])
                label[i] = max(0,label[i])
            #
            batch[0].append(img)
            batch[1].append(label)
        return batch


########### TEST ###############


if __name__ == "__main__":
    sess = tf.Session()
    json_file = r"E:\CASIA-V4-Location\Iris_Pupil_Position.json"   # 格式为 V4_ROOT/000/L/SXXX.jpg
    location_data_root = r"E:\CASIA-V4-Location"
    a = DSV4Location(sess,json_file,location_data_root)
    print("--------BATCH-------------")
    batch = a.getBatch(10)
    print(batch[1][0])
    print(batch[0][0].shape);
    Utils.showImage(batch[0][0])