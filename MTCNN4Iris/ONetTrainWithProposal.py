# Pnet 的输出作为ONet的输入
import tensorflow as tf
import os
import Config
from MTCNN4Iris import DSV4MTCNN
from MTCNN4Iris.IrisONet import IrisONet
from MTCNN4Iris.DSV4ONetWithProposal import DSV4ONetWithProposal
if __name__ == "__main__":
    # 运行目录
    experiment_dir = "experiments/onet_neg_15_pupil"
    pnet_output_images_dir = r"E:\CASIA-V4-Location\PNetProposalWithRejectSample"
    config = Config.Config("{0}/config.json".format(experiment_dir))
    config.show()
    # session
    sess = tf.Session()
    # 数据集
    train_data = DSV4ONetWithProposal(sess, pnet_output_images_dir)
    # 网络
    onet = IrisONet(sess,config,experiment_dir, use_pupil_loc=True)
    # restore
    try:
        onet.restore(os.path.join(os.path.join(experiment_dir,"model")))
    except:
        print("[!]无法restore")
    cur_step = 0
    learning_rate = config.learning_rate
    while cur_step< config.total_training_steps:
        if cur_step >2000:
            learning_rate = 1e-5
        batch = train_data.getBatch(config.batch_size )
        loss,cur_step = onet.trainWithPupilLoc(batch[0],batch[1],batch[2],batch[3],learning_rate)
        print("[*] Step[{0}] loss{1} ".format(cur_step,loss))

        # 每隔 save_every_steps ，保存一次模型
        if cur_step % config.save_every_steps == 0:
            onet.save(os.path.join(experiment_dir,"model/step"),cur_step)