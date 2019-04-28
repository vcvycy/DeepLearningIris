import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())
import tensorflow as tf
from Recognition import ResNetSemi
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


# (*) 返回embeddings 经过eweight加权后，两两之间的距离
def getNpDist(embeddings, eweight):
    # 参数说明：
    # (1) embeddings 为特征向量(已经从二维展开为一维)
    # (2) eweight 为特征向量的权值()
    eweight0 = np.expand_dims(eweight, 0)
    eweight1 = np.expand_dims(eweight, 1)
    eweight_3d = eweight0 * eweight1
    embed0 = np.expand_dims(embeddings, 0)
    embed1 = np.expand_dims(embeddings, 1)
    embed_weighted = embed1 * eweight_3d

    embed_l2 = embed_weighted / np.expand_dims(np.linalg.norm(embed_weighted, axis=2), 2)
    embed_l2_transpose = np.transpose(embed_l2, [1, 0, 2])
    dist = np.sum(embed_l2 * embed_l2 + embed_l2_transpose * embed_l2_transpose - 2 * embed_l2 * embed_l2_transpose, 2)
    return dist

def getDist(resnet,paths,h,w, debug):
    embeddings =[]
    eweight=[]
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = Utils.resize(img,(h,w))
        img =np.reshape(img, (h, w, 1))
        output=resnet.forward([img])
        embeddings.append(output[0][0])
        eweight.append(output[1][0])
        if debug:
            visualizeEWeight(output[1][0], img)
    embeddings = np.array(embeddings)
    eweight = np.array(eweight)
    return getNpDist(embeddings,eweight)

def visualizeEWeight(eweight, origin_image):
    h,w=origin_image.shape[0], origin_image.shape[1]
    eh,ew= h//4, w//4
    eweight=np.reshape(eweight,(eh,ew))
    # 权值矩阵
    img = np.zeros((h,w),np.uint8)
    # 加权后的origin_image
    img_weighted =  np.zeros((h,w,3),np.uint8)
    for i in range(h):
        for j in range(w):
            img[i,j] = int(eweight[i//4,j//4]*255)
            if eweight[i//4,j//4]>0.1:
                img_weighted[i,j] = (origin_image[i,j], origin_image[i,j], origin_image[i,j])
            else:
                img_weighted[i, j] = (0,255,0)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    origin_image =cv2.cvtColor(origin_image, cv2.COLOR_GRAY2RGB)

    origin_image = np.reshape(origin_image,(h,w,3))
    # 红线隔开
    pad = np.zeros((5,w,3),np.uint8)
    for i in range(5):
        for j in range(w):
            pad[i,j,2]=255
    #
    img= np.concatenate((origin_image,pad,img,pad,img_weighted),axis=0)
    Utils.showImage(img)
    return img

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

    FAR = false_accept / max(accept_num,1)
    FRR = false_reject / max(need_accept,1)
    return FAR,FRR, (false_accept, accept_num , false_reject, need_accept)

def getModelFARFRR(resnet, test_dir, config,debug=False):
    # 读取所有文件和文件对应的label值
    filename2path = Utils.getFile2Path(test_dir)
    path_list = [filename2path[f] for f in filename2path]
    labels = [getLabelFromFilename(f) for f in filename2path]
    n = len(path_list)
    print("[*] 测试文件个数:%s" % (n))
    # dist = getMinDistWithRoatate(resnet, path_list, [0], sz=config.input_size, dim=config.dims)
    dist = getDist(resnet,path_list,config.input_height,config.input_width, debug)

    # FAR = 1e-3
    frr_3, frr_3threshold = getFRRWhenFARat(dist, labels, far_below=1e-3)
    print("[*] 当FAR = 0.1%%， FRR=%s threshold=%s" % (frr_3, frr_3threshold))

    # Equal Error Rate
    data, threshold = getEqualErrorRate(dist, labels)
    print("Equal Error Rate %s(far) %s(frr) threshold=%s" % (data[0], data[1], threshold))
    return {
        "eer" :  data[0],
        "err_threshold" : threshold,
        "frr_when far=1e-3" : frr_3,
        "frr_when far=1e-3 threshold" : frr_3threshold
    }

def getDistNor(resnet,paths,h,w):
    embeddings =[]
    eweight=[]
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = Utils.resize(img,(w,h))
        img =np.reshape(img, (h,w, 1))
        # Utils.showImage(img)
        output=resnet.forward([img])
        embeddings.append(output[0][0])
        eweight.append(output[1][0])
    embeddings = np.array(embeddings)
    eweight = np.array(eweight)
    return getNpDist(embeddings,eweight)

def getModelFARFRRNor(resnet, test_dir, config):
    # 读取所有文件和文件对应的label值
    filename2path = Utils.getFile2Path(test_dir)
    path_list = [filename2path[f] for f in filename2path]
    labels = [getLabelFromFilename(f) for f in filename2path]
    n = len(path_list)
    print("[*] 测试文件个数:%s" % (n))
    # dist = getMinDistWithRoatate(resnet, path_list, [0], sz=config.input_size, dim=config.dims)
    dist = getDistNor(resnet,path_list,config.input_height,config.input_width)

    # FAR = 1e-3
    frr_3, frr_3threshold = getFRRWhenFARat(dist, labels, far_below=1e-3)
    print("[*] 当FAR = 0.1%%， FRR=%s threshold=%s" % (frr_3, frr_3threshold))

    # Equal Error Rate
    data, threshold = getEqualErrorRate(dist, labels)
    print("Equal Error Rate %s(far) %s(frr) threshold=%s" % (data[0], data[1], threshold))
    return {
        "eer" :  data[0],
        "err_threshold" : threshold,
        "frr_when far=1e-3" : frr_3,
        "frr_when far=1e-3 threshold" : frr_3threshold
    }
if __name__ == "__main__":
    # 读取训练存放目录；已经目录中的配置文件
    parser = argparse.ArgumentParser()
    # parser.add_argument("--training_dir",default="TripletSelection")
    parser.add_argument("--training_dir",default="semi")
    # parser.add_argument("--test_image_dir", default=r"E:\iris_recog_test")
    parser.add_argument("--test_image_dir", default=r"E:\tmp2")
    cmd_args = parser.parse_args()

    # 模型地址
    train_on_dir = os.path.join("./experiments",cmd_args.training_dir)
    # 配置文件
    config_path = os.path.join(train_on_dir,"config.json")
    print(config_path)
    config = Config(config_path)
    config.show()


    sess = tf.Session()
    #网络
    resnet=ResNetSemi.ResNet(sess, config, os.path.join(train_on_dir, "tboard"))
    print("[*]网络参数%d" %(resnet.param_num))
    # restore
    resnet.restore_embedding(os.path.join(os.path.join(train_on_dir, "model")))
    print(getModelFARFRR(resnet, cmd_args.test_image_dir,config,debug=True))