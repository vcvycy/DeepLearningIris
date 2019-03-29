import sys
import os
sys.path.append(os.path.join(os.getcwd(),".."))
sys.path.append(os.getcwd())

import Config
import tensorflow as tf
from MTCNN4Iris.MTCNN import *
from MTCNN4Iris import DSV4MTCNN

class IrisONet():
    def __init__(self, sess, config, experiment_dir, target="train", use_pupil_loc=False):
        self.use_pupil_loc = use_pupil_loc
        self.config = config
        self.experiment_dir = experiment_dir         # 保存模型用的
        with tf.variable_scope('onet'):
            self.sess = sess
            # 定义输入输出和hyper parameters
            self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
            self.global_step = tf.Variable(0, name="global_step", trainable=False)
            self.input = tf.placeholder(tf.float32, (None, 48, 48, 1), 'input')
            # 定义ONet
            self.onet = ONet({'input': self.input},l2_lambda= config.l2_lambda,use_pupil_loc= use_pupil_loc)

            # 输出节点： 概率、BBR
            self.prob = self.onet.layers["prob1"]
            self.label_prob = tf.placeholder(tf.float32,(None,2),"label_prob")
            self.bbr  = self.onet.layers["conv6-2"]
            if self.use_pupil_loc:
                self.pupil  = self.onet.layers["conv6-3"]
                self.label_pupil = tf.placeholder(tf.float32,(None,4),"pupil_loc")
            self.label_bbr  = tf.placeholder(tf.float32,(None,4),"label_bbr")

            if target == "train":
                # Loss
                with tf.variable_scope("Loss"):
                    # Classification Loss
                    with tf.variable_scope("loss_prob"):
                        self.loss_prob = probFocalLoss(self.label_prob,self.prob,self.config.focal_loss_lambda)
                        tf.summary.scalar("loss_prob",self.loss_prob)
                    # Bounding Box Loss
                    with tf.variable_scope("loss_bbr"):
                        pos_example = self.label_prob[:,1]                    # [None], 训练样本是正样本为1，负样本为0
                        self.loss_bbr  = bbrLoss(pos_example,self.label_bbr,self.bbr)
                        tf.summary.scalar("loss_bbr",self.loss_bbr)
                    self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="l2_loss")
                    if self.use_pupil_loc:
                        with tf.variable_scope("loss_pupil"):
                            self.loss_pupil = pupilLoss(pos_example, self.label_pupil, self.pupil)
                            tf.summary.scalar("Loss Pupil", self.loss_pupil)
                        self.loss = self.loss_prob   + self.loss_bbr * self.config.bbr_loss_weight + self.l2_loss + self.loss_pupil *self.config.pupil_loss_weight
                    else:
                        self.loss = self.loss_prob + self.loss_bbr * self.config.bbr_loss_weight + self.l2_loss

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
                                          feed_dict={self.input: input,
                                                     self.label_prob: o_prob,
                                                     self.label_bbr : o_bbr,
                                                     self.learning_rate: learning_rate}
                                            )
        print(p[:17])
        self.writer.add_summary(summary_val,global_step)
        return loss,global_step

    # 加上
    def trainWithPupilLoc(self,input, o_prob, o_bbr, o_pupil, learning_rate):
        _, loss,summary_val , global_step,p  = self.sess.run(
                                           [self.optimizer, self.loss,self.all_summary,self.global_step,self.prob],
                                          feed_dict={self.input: input,
                                                     self.label_prob: o_prob,
                                                     self.label_bbr : o_bbr,
                                                     self.label_pupil : o_pupil,
                                                     self.learning_rate: learning_rate}
                                            )
        self.writer.add_summary(summary_val,global_step)
        return loss,global_step

    def predict(self,input):
        prob,bbr = self.sess.run([self.prob,self.bbr],
                                 feed_dict={
                                     self.input:input
                                 })
        return prob.tolist(),bbr.tolist()

    def predictWithPupilLoc(self,input):
        prob,bbr,pupil = self.sess.run([self.prob,self.bbr, self.pupil],
                                 feed_dict={
                                     self.input:input
                                 })
        return prob.tolist(),bbr.tolist(),pupil.tolist()

    def save(self, save_path, steps):
        saver = tf.train.Saver(max_to_keep=5)
        saver.save(self.sess, save_path, global_step=steps)
        print("[*]save success to {}".format(save_path))

    def restore(self, restore_path):
        path = tf.train.latest_checkpoint(restore_path)
        if path == None:
            return False
        onet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='onet')
        saver = tf.train.Saver(onet_vars)
        saver.restore(self.sess, path)
        print("[*]Restore from %s success" % (path))
        return True
