import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())
from UNet.TripletSelection import TripletSelection
from UNet import RecogTestSemi
import tensorflow as tf
from UNet import unet,DSV4Recog
from Config import Config
import argparse
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
    myunet=unet.Unet(sess, config, os.path.join(train_on_dir, "tboard"))
    print("[*]网络参数%d" % (myunet.param_num))
    # restore
    try:
        myunet.restore_embedding(os.path.join(os.path.join(train_on_dir, "model")))
        print("[*] 测试精度:%s" %(config.test_dir))
        print("[*] %s" % (RecogTestSemi.getModelFARFRRNor(myunet, config.test_dir, config)))
    except:
        print("[*] restore失败")
    # 数据集
    ds = TripletSelection(config.iris_training_images_dir)

    # 开始训练
    while True:
        batch = ds.getBatchNor(config.input_height, config.input_width, config.batch_class_num, config.batch_img_num_each_class)
        loss,cur_step = myunet.train(  batch_input = batch[0],
                                       batch_output = batch[1],
                                       learning_rate= config.learning_rate,
                                       batch_all_weight= config.batch_all_weight,
                                       batch_hard_weight = config.batch_hard_weight
                                       )
        print("[*] Step[{0}] loss{1} ".format(cur_step,loss))

        # print("[*] 测试精度:%s" % (config.test_dir))
        # print("[*] %s" % (RecogTestSemi.getModelFARFRRNor(resnet, config.test_dir, config)))
        # 每隔 save_every_steps ，保存一次模型
        if cur_step % config.save_every_steps == 0:
            myunet.save_embedding(os.path.join(train_on_dir, "model/at_step"), cur_step)
            print("[*] 测试精度:%s" %(config.test_dir))
            print("[*] %s" % (RecogTestSemi.getModelFARFRRNor(myunet, config.test_dir, config)))
        # resnet.save_embedding(os.path.join(train_on_dir,"model/at_step"), cur_step)
        # 更新配置
        config.update(config_path)

        # 训练完 total_training_steps 后，退出程序
        #if cur_step > config.total_training_steps:
        #    break