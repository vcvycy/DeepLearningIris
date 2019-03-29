import tensorflow as tf
import os
import Config
from MTCNN4Iris import DSV4MTCNN
from MTCNN4Iris.IrisPNet import IrisPNet

if __name__ == "__main__":
    # 运行目录
    experiment_dir = "experiments/pnet_expand_0.08"
    config = Config.Config("{0}/config.json".format(experiment_dir))
    config.show()
    # session
    sess = tf.Session()
    # 数据集
    train_data = DSV4MTCNN.DSV4MTCNN(sess, config.iris_position_json_file, config.iris_images_dir, config.pos_iou, config.neg_iou)
    # 网络
    pnet = IrisPNet(sess,config,experiment_dir)
    # restore
    try:
        pnet.restore(os.path.join(os.path.join(experiment_dir,"model")))
    except:
        print("[!]无法restore")

    #开始训练
    cur_step = 0
    learning_rate = config.learning_rate
    while cur_step< config.total_training_steps:
        if cur_step == 1500:
            learning_rate *=0.1
        input , label_prob, label_bbr = train_data.getBatchForPNet(config.batch_size,config.pos_region_each_image, config.neg_region_each_image)

        loss,cur_step = pnet.train(input , label_prob, label_bbr,learning_rate)
        print("[*] Step[{0}] loss{1} ".format(cur_step,loss))
        # 每隔 save_every_steps ，保存一次模型
        if cur_step % config.save_every_steps == 0:
            pnet.save(os.path.join(experiment_dir,"model/at_step"),cur_step)