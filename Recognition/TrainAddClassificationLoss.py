import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())
from Recognition.TripletSelection import TripletSelection
from Recognition import RecogTest
import tensorflow as tf
from Recognition import ResNetFCN
from Recognition import  DSV4Recog
from Config import Config
import argparse

# import signal
# def signal_handler(signal, frame):
#     print("[!] 保存模型后程序即退出")
#     resnet.save(os.path.join(train_on_dir, "model/at_step"), cur_step)
#     sys.exit(0)
# signal.signal(signal.SIGINT, signal_handler)
def toOneHot(label, classes):
    one_hot_label = []
    for v in label:
        l = [0 for _ in range(classes)]
        l[v] = 1
        one_hot_label.append(l)
    return one_hot_label
if __name__ == "__main__":
    # 读取训练存放目录；已经目录中的配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir",default="FCN")
    parser.add_argument("--gpu")
    cmd_args = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = cmd_args.gpu
    # 模型地址
    train_on_dir = os.path.join("./experiments",cmd_args.training_dir)
    # 配置文件
    config_path = os.path.join(train_on_dir,"config.json")
    config = Config(config_path)
    config.show()

    sess = tf.Session()
    #网络
    resnet=ResNetFCN.ResNet(sess, config, os.path.join(train_on_dir, "tboard"))
    print("[*]网络参数%d" %(resnet.param_num))
    # restore
    try:
        resnet.restore_embedding(os.path.join(os.path.join(train_on_dir, "model")))
    except:
        print("[*] restore失败")

    # 数据集
    ds = TripletSelection(config.iris_training_images_dir, config.training_classes)

    # 开始训练
    while True:
        batch = ds.getBatch(config.input_size, config.batch_class_num, config.batch_img_num_each_class)
        loss,cur_step = resnet.trainWithClassification(batch_input = batch[0],
                                                       batch_output = batch[1],
                                                       batch_output_ont_hot= toOneHot(batch[1],config.training_classes),
                                                       learning_rate= config.learning_rate,
                                                       cls_weight=  config.classification_loss_weight,
                                                       batch_all_weight= config.batch_all_weight,
                                                       batch_hard_weight = config.batch_hard_weight
                                                        )
        print("[*] Step[{0}] loss{1} ".format(cur_step,loss))

        # 每隔 save_every_steps ，保存一次模型
        if cur_step % config.save_every_steps == 0:
            resnet.save_embedding(os.path.join(train_on_dir,"model/at_step"), cur_step)
            print("[*] 测试精度:%s" %(config.test_dir))
            print("[*] %s" %(RecogTest.getModelFARFRR(resnet,config.test_dir, config)))

        # resnet.save_embedding(os.path.join(train_on_dir,"model/at_step"), cur_step)
        # 更新配置
        config.update(config_path)

        # 训练完 total_training_steps 后，退出程序
        #if cur_step > config.total_training_steps:
        #    break