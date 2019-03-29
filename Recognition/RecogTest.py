import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())
import tensorflow as tf
from Recognition import ResNet
from Recognition import  DSV4Recog
from Config import Config
import argparse
import cv2
import  numpy as np
import Utils
def getLabelFromFilename( filename):
    eye_lr = filename[5]
    idx = int(filename[2:5])
    label = int(idx) * 2
    if eye_lr == "R":
        label += 1
    return label
if __name__ == "__main__":
    # 读取训练存放目录；已经目录中的配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir",default="with_classfication")
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
    resnet=ResNet.ResNet(sess, config,os.path.join(train_on_dir,"tboard"))
    print("[*]网络参数%d" %(resnet.param_num))
    # restore
    resnet.restore_embedding(os.path.join(os.path.join(train_on_dir, "model")))
    # 读取所有文件和文件对应的label值
    filename2path = Utils.getFile2Path(r"E:\iris_recog_test")
    path_list = [filename2path[f] for f in filename2path]
    labels = [getLabelFromFilename(f) for f in filename2path]
    n = len(path_list)
    print("[*] 测试文件个数:%s" %(n))
    dim = 128
    embeds = []
    for i in range(n):
        path = path_list[i]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = Utils.resize(img,(200,200))
        img = np.reshape(img,(200,200,1))
        filename = os.path.basename(path)
        embedding = resnet.forward(batch_input= [img])[0]
        embeds.append(embedding)
        # print("[*] {0} label {1}".format(filename,getLabelFromFilename(filename)))
    embeds = np.array(embeds)

    # 两两之间的距离
    e1 = np.reshape(embeds,(n,1,dim))
    e2 = np.reshape(embeds,(1,n,dim))
    dist = np.sqrt(np.sum((e1-e2)*(e1-e2), 2))

    #
    threshold = 0.42
    recall, precision, x = Utils.getRecallAndPrecision(dist,labels, threshold)
    print("[*] recall {0} ({1}/{2})".format(recall, x[0],x[2]))
    print("[*] precision {0} ({1}/{2})".format(precision, x[0],x[1]))
    FAR =0
    FRR =0
    total = 0
    for i in range(n):
        for j in range(i+1, n):
            labeli = labels[i]
            labelj = labels[j]
            total += 1
            # FRR 网络认为不是同一个人, 但是是同一个
            if dist[i][j] > threshold and labeli == labelj:
                FRR += 1
            if dist[i][j] <= threshold and labeli!= labelj:
                FAR += 1
            # print(" label %s -> %s dist=%s" %(labeli, labelj, dist[i][j]))
            # input("")

    print("[*]FAR ={0}/{1}  {2}".format(FAR, total, FAR / total))
    print("[*]FRR ={0}/{1}  {2}".format(FRR, total, FRR / total))
    # for i in range(n):
    #     for j in range(n):
    #         li = labels[i]
    #         lj = labels[j]
    #         if li == lj:
    #             s="Y"
    #         else:
    #             s="N"
    #         print("%.2f%s" %(dist[i,j], s), end="   ")
    #     print("")
