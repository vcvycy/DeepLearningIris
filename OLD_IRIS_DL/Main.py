import sys
import os

sys.path.append(os.path.join(os.getcwd(), ".."))
sys.path.append(os.getcwd())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from OLD_IRIS_DL import IrisReader
from OLD_IRIS_DL import ResNet

# 参数
learning_rate = 1e-3
resnet_stack_n = 2
batch_sz = 50
epochs = 100
classes = 2000
img_w = 192
# 网络
resnet = ResNet.ResNet([None, img_w, img_w, 1], [None, classes], resnet_stack_n)
print("[*]网络参数%d" % (resnet.param_num))
# restore
# resnet.restore_except_softmax_layer("./model/")
resnet.restore("./model/")
# resnet.save("./model/xxx",0)
# Reader
dir = "/home/jack/iris_dataset/egmented_iris"
dir = r"E:\IrisDataset\segmented_iris"
reader = IrisReader.IrisReader(resnet.sess, dir, img_w, classes)
train_img_num = reader.cntTrainImg()
print("[*]训练图片总数:%d" % (train_img_num))
# 开始训练

for epoch in range(epochs):
    for i in range(int(train_img_num / batch_sz)):
        batch = reader.getTrainBatch(batch_sz)
        rst = resnet.train(batch[0], batch[1], learning_rate=learning_rate)
        print("[*%d]%s" % (i, rst))
    resnet.save("model/epoch", epoch)
