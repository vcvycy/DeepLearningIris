import os
import numpy as np
import tensorflow as tf
import Utils
from DSIris import DSIris


# CASIA.v4 数据集读取
class DSV4(DSIris):

    # (*) ds_root : 数据集根目录
    def __init__(self,sess,ds_root):
        super().__init__(sess)
        # (*) 获取所有文件名列表
        self.filenames = super().getFilenamesFromDir(ds_root, "jpg")

        # (*) 获取文件对应的label值,和某个label对应的所有文件
        self.filename2label = {}
        self.label2filenames = {}
        for file in self.filenames:
            label = self.__getLabelFromFilename(file)
            self.filename2label[file] = label
            if label not in self.label2filenames:
                self.label2filenames[label] = []
            self.label2filenames[label].append(file)

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
        print("[*] CASIA v4数据集")
        print("  [*] 训练集大小:%s" %(len(self.filenames)))
        print("  [*] 一共有%s类" %(len(self.label2filenames)))
        return

    # (*) 获取一个batch
    def getBatch(self,batch_size):
        # (*) 获取 filename -> tensor(int) 格式的batch
        raw_batch = super().getRawBatch(batch_size)
        # (*) 获取batch
        batch = []
        for item in raw_batch:
            filename = item[0]
            img_int  = item[1]
            batch.append((filename,img_int))
        return batch


########### TEST ###############


if __name__ == "__main__":
    sess = tf.Session()
    a = DSV4(sess,r"E:\虹膜数据集\CASIA-Iris-Thousand\000")
    batch = a.getBatch(2)
    Utils.showImage(batch[0][1])