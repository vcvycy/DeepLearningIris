import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 数据集通用类
class DSIris:
    def __init__(self,sess):               # sess Tensorflow Session
        # 训练数据集
        self.images_path = []                # 训练数据文件名
        self.filename2label = {}           # 文件名 -> label
        self.label2filename = {}

        # Tensorflow 队列
        self.sess = sess
        self.queue = None
        self.reader = None
        self.queue_item = None
        return

    # (*) 从文件夹dir中递归读取所有符合accept_suffix 后缀名(用|分割)的文件完整路径
    def getFilenamesFromDir(self, dir, accept_suffix):  # 文件名list
        suffix = set(accept_suffix.split("|"))
        filename_list = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.strip().split(".")[-1] in suffix:
                    path = os.path.join(root, file)
                    filename_list.append(path)
        return filename_list

    # (*) 创建Tensorflow 队列, 文件为self.filenames
    def createTFQueue(self,num_epochs=50):
        print("[*]正在创建TF 队列...")
        # (*) 定义节点
        self.queue = tf.train.string_input_producer(self.images_path, num_epochs = num_epochs, shuffle=True)
        self.reader = tf.WholeFileReader()
        item_filename, item_binary = self.reader.read(self.queue)
        self.queue_item = (item_filename, tf.image.decode_jpeg(item_binary))
        print("     [*] 正在启动线程...")
        # (*) 初始化 + 启动线程
        sess = self.sess
        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("     [*] 图片队列创建成功！")
        return

    # (*) 从tensorflow queue 读取batch, 格式filename -> tensor(int类型)
    def getRawBatch(self,batch_size):
        # 由子类重写
        raw_batch = []
        for i in range(batch_size):
            filename, img = self.sess.run(self.queue_item)
            filename = filename.decode(encoding = "utf-8")
            item = (filename,img)
            raw_batch.append(item)
        return raw_batch

