import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())

import Config
import tensorflow as tf
from MTCNN4Iris.MTCNN import *
from MTCNN4Iris import DSV4MTCNN

class IrisPNet():
    def __init__(self,sess,config,experiment_dir, target="train"):
        self.config = config
        self.experiment_dir = experiment_dir         # 保存模型用的
        with tf.variable_scope('pnet'):
            self.sess = sess
            # 定义输入输出和hyper parameters
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.input = tf.placeholder(tf.float32, (None, None, None, 1), 'input')

            # 定义PNet
            self.pnet = PNet({'input': self.input},config.l2_lambda)

            # 输出节点： 概率、BBR
            self.prob = self.pnet.layers["prob1"]                                   # shape 为[None,None,None,2]
            self.label_prob = tf.placeholder(tf.float32,(None,2),"label_prob")
            self.bbr = self.pnet.layers["conv4-2"]                                  # shape 为[None,None,None,2]
            self.label_bbr = tf.placeholder(tf.float32,(None,4),"label_bbr")

            if target == "train":
                # Loss 训练的图像只能是12*12
                with tf.variable_scope("Loss"):
                    with tf.variable_scope("loss_prob"):
                        prob_flat = tf.reshape(self.prob,shape=(-1,2))
                        self.loss_prob = probFocalLoss(self.label_prob, prob_flat, self.config.focal_loss_lambda)
                    with tf.variable_scope("loss_bbr"):
                        pos_example = self.label_prob[:,1]                    # [None], 训练样本是正样本为1，负样本为0
                        bbr_flat = tf.reshape(self.bbr, shape=(-1,4))
                        self.loss_bbr  = bbrLoss(pos_example,self.label_bbr, bbr_flat)
                    # for t in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                    #    print(t.name)
                    self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
                    self.loss = self.loss_prob   + self.loss_bbr * self.config.bbr_loss_weight + self.l2_loss

                # optimizer
                self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.config.momemtum).minimize(
                    self.loss, global_step=self.global_step)

                # global var init
                self.sess.run(tf.global_variables_initializer())
                # summary - tf board - init
                tf.summary.scalar("Loss",self.loss)
                tf.summary.scalar("Learning Rate", self.learning_rate)
                self.all_summary = tf.summary.merge_all()
                self.writer = tf.summary.FileWriter(os.path.join(self.experiment_dir,"tboard"), self.sess.graph)
            return
    # 训练
    def train(self, input, o_prob, o_bbr, learning_rate):
        _, loss,summary_val , global_step,p  = self.sess.run(
                                           [self.optimizer, self.loss,self.all_summary,self.global_step,self.prob],
                                           feed_dict = {  self.input: input,
                                                          self.label_prob: o_prob,
                                                          self.label_bbr : o_bbr,
                                                          self.learning_rate: learning_rate}
                                            )
        p = np.squeeze(p)
        print(p[:10])
        self.writer.add_summary(summary_val, global_step)
        return loss, global_step

    def predict(self,input):
        prob,bbr = self.sess.run([self.prob,self.bbr],
                                 feed_dict={
                                     self.input:input
                                 })
        return prob,bbr
        #return prob.tolist(),bbr.tolist()

    def save(self, save_path, steps):
        saver = tf.train.Saver(max_to_keep=5)
        saver.save(self.sess, save_path, global_step=steps)
        print("[*]save success to {}".format(save_path))

    def restore(self, restore_path):
        path = tf.train.latest_checkpoint(restore_path)
        if path == None:
            return False
        pnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pnet')
        saver = tf.train.Saver(pnet_vars)
        saver.restore(self.sess, path)
        print("[*]Restore from %s success" % (path))
        return True
