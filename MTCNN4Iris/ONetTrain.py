import tensorflow as tf
import os
import Config
from MTCNN4Iris import DSV4MTCNN
from MTCNN4Iris.IrisONet import IrisONet

if __name__ == "__main__":
    # 运行目录
    experiment_dir = "experiments/onet_neg_15"
    config = Config.Config("{0}/config.json".format(experiment_dir))
    config.show()
    # session
    sess = tf.Session()
    # 数据集
    train_data = DSV4MTCNN.DSV4MTCNN(sess, config.iris_position_json_file, config.iris_images_dir, config.pos_iou, config.neg_iou)
    # 网络
    onet = IrisONet(sess,config,experiment_dir)
    # restore
    try:
        onet.restore(os.path.join(os.path.join(experiment_dir,"model")))
    except:
        print("[!]无法restore")
    cur_step = 0
    learning_rate = config.learning_rate
    while cur_step< config.total_training_steps:
        if cur_step == 2000:
            learning_rate *= 0.1
        batch = train_data.getBatchForONet(config.batch_size,config.pos_region_each_image, config.neg_region_each_image)
        assert len(batch[0]) == config.batch_size ," {0} != {1}".format(len(batch[0]), config.batch_size)
        loss,cur_step = onet.train(batch[0],batch[1],batch[2],learning_rate)

        print("[*] Step[{0}] loss{1} ".format(cur_step,loss))
        # 每隔 save_every_steps ，保存一次模型
        if cur_step % config.save_every_steps == 0:
            onet.save(os.path.join(experiment_dir,"model/step"),cur_step)