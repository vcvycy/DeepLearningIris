import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())
import tensorflow as tf
from Recognition import ResNetFCN
from Recognition import  DSV4Recog
from Config import Config
import argparse
import cv2
import  numpy as np
import Utils
import math
def getLabelFromFilename( filename):
    eye_lr = filename[5]
    idx = int(filename[2:5])
    label = int(idx) * 2
    if eye_lr == "R":
        label += 1
    return label

# 获取两两之间的距离。所有旋转中，距离最近的那个
def getMinDistWithRoatate(resnet, paths, rotate, sz, dim):
    embeds = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = Utils.resize(img,(200,200))
        cur_embeds=[]
        # print(p)
        for theta in rotate:
            mat = cv2.getRotationMatrix2D((sz//2, sz//2), theta, 1.08)
            img_rotate = cv2.warpAffine(img, mat, (sz, sz))
            # print(img_rotate.shape)
            # Utils.showImage(img_rotate)
            img_rotate = np.reshape(img_rotate,(sz,sz,1))
            embedding = resnet.forward(batch_input= [img_rotate])[0]
            cur_embeds.append(embedding)
        embeds.append(np.array(cur_embeds))

    # 获取距离
    n = len(embeds)
    m = len(rotate)
    dist = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i+1,n):
            ei = np.reshape(embeds[i],(m,1,dim))
            ej = np.reshape(embeds[j],(1,m,dim))
            d = np.sqrt(np.sum((ei - ej) * (ei - ej), 2))  # 两个图片不同rotate下的距离
            d = np.min(d)                                  # 两个虹膜的距离
            dist[i][j]=dist[j][i] = d
    return dist

# 当FAR=1e-3时，FRR的值
def getFRRWhenFARat(dist, labels, far_below=1e-3):
    tl =0.1
    tr =2
    while tr-tl> 1e-4:
        threshold = (tr+tl)/2
        far,frr,_ = getFarFrrByDist(dist, labels, threshold)
        if far> far_below:
            tr = threshold
        else:
            tl = threshold
    threshold = tl
    return getFarFrrByDist(dist, labels, threshold), threshold

def getEqualErrorRate(dist, labels):
    tl = 0.1
    tr = 2
    while tr-tl> 1e-4:
        threshold = (tr+tl)/2
        far,frr,_ = getFarFrrByDist(dist, labels, threshold)
        if far> frr:
            tr = threshold
        else:
            tl = threshold
    threshold = (tl+tr)/2
    return getFarFrrByDist(dist,labels,threshold), threshold

def getFarFrrByDist(dist,labels, threshold):
    false_accept = 0
    false_reject = 0
    accept_num = 0
    need_accept = 0
    n=len(dist)
    for i in range(n):
        for j in range(i + 1, n):
            labeli = labels[i]
            labelj = labels[j]
            if labelj == labeli:
                need_accept += 1
                if dist[i][j] >= threshold:
                    false_reject += 1
            if dist[i][j] < threshold:
                accept_num += 1
                if labeli != labelj:
                    false_accept += 1

    FAR = false_accept / accept_num
    FRR = false_reject / need_accept
    return FAR,FRR, (false_accept, accept_num , false_reject, need_accept)

def getModelFARFRR(resnet, test_dir, config):
    # 读取所有文件和文件对应的label值
    filename2path = Utils.getFile2Path(test_dir)
    path_list = [filename2path[f] for f in filename2path]
    labels = [getLabelFromFilename(f) for f in filename2path]
    n = len(path_list)
    print("[*] 测试文件个数:%s" % (n))
    dist = getMinDistWithRoatate(resnet, path_list, [0], sz=config.input_size, dim=config.dims)

    # FAR = 1e-3
    frr_3, frr_3threshold = getFRRWhenFARat(dist, labels, far_below=1e-3)
    print("[*] 当FAR = 0.1%%， FRR=%s threshold=%s" % (frr_3, frr_3threshold))

    # Equal Error Rate
    data, threshold = getEqualErrorRate(dist, labels)
    print("Equal Error Rate %s(far) %s(frr) threshold=%s" % (data[0], data[1], threshold))
    return {
        "eer" :  data[0],
        "err_threshold" : threshold,
        "frr_3" : frr_3,
        "frr_3threshold" : frr_3threshold
    }

if __name__ == "__main__":
    # 读取训练存放目录；已经目录中的配置文件
    parser = argparse.ArgumentParser()
    # parser.add_argument("--training_dir",default="TripletSelection")
    parser.add_argument("--training_dir",default="64-dims")
    parser.add_argument("--test_image_dir", default=r"E:\iris_recog_test")
    # parser.add_argument("--test_image_dir", default=r"E:\tmp")
    cmd_args = parser.parse_args()

    # 模型地址
    train_on_dir = os.path.join("./experiments",cmd_args.training_dir)
    # 配置文件
    config_path = os.path.join(train_on_dir,"config.json")
    print(config_path)
    config = Config(config_path)
    config.show()


    sess = tf.Session()
    threshold = 0.8
    #网络
    resnet=ResNetFCN.ResNet(sess, config, os.path.join(train_on_dir, "tboard"))
    print("[*]网络参数%d" %(resnet.param_num))
    # restore
    resnet.restore_embedding(os.path.join(os.path.join(train_on_dir, "model")))
    print(getModelFARFRR(resnet, cmd_args.test_image_dir,config))