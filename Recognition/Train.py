import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())
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
if __name__ == "__main__":
    # 读取训练存放目录；已经目录中的配置文件
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_dir",default="test")
    cmd_args = parser.parse_args()

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
    resnet.restore_embedding(os.path.join(os.path.join(train_on_dir, "model")))

    # 数据集
    ds = DSV4Recog.DSV4Recog(sess,
                             config.iris_training_images_dir,
                             steps= min(201000,config.total_training_steps),
                             input_size= config.input_size,
                             crop_pixels= config.crop_each_side,
                             batch_img_num_each_class= config.batch_img_num_each_class,
                             batch_class_num= config.batch_class_num,
                             training_classes= config.training_classes
                             )



    # 开始训练
    while True:
        batch = ds.getBatch()
        loss,cur_step = resnet.train(batch[0], batch[1], learning_rate= config.learning_rate)
        print("[*] Step[{0}] loss{1} ".format(cur_step,loss))

        # 每隔 save_every_steps ，保存一次模型
        if cur_step % config.save_every_steps == 0:
            resnet.save_embedding(os.path.join(train_on_dir,"model/at_step"), cur_step)

        resnet.save_embedding(os.path.join(train_on_dir,"model/at_step"), cur_step)
        # 更新配置
        config.update(config_path)
        # 训练完 total_training_steps 后，退出程序
        #if cur_step > config.total_training_steps:
        #    break