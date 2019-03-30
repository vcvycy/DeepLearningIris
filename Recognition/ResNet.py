import tensorflow as tf
from Recognition import TripletLoss
"""
(1)构造函数__init__参数
  input_sz：  输入层placeholder的4-D shape，如mnist是[None,28,28,1] 
(2)train函数：训练一步
   batch_input： 输入的batch
   batch_output: label
   learning_rate:学习率
   返回：正确率和loss值(float)   格式：{"accuracy":accuracy,"loss":loss}
(3)forward：训练后用于测试
(4)save(save_path,steps)保存模型
(5)restore(path):从文件夹中读取最后一个模型
(6)loss函数使用cross-entrop one-hot版本:y*log(y_net)
(7)optimizer使用adamoptimier
"""


class ResNet:
    param_num = 0  # 参数个数

    CONV_PADDING = "SAME"
    MAX_POOL_PADDING = "SAME"
    CONV_WEIGHT_INITAILIZER = tf.keras.initializers.he_normal()  # tf.truncated_normal_initializer(stddev=0.1)
    CONV_BIAS_INITAILIZER = tf.constant_initializer(value=0.0)
    FC_WEIGHT_INITAILIZER = tf.keras.initializers.he_normal()  # tf.truncated_normal_initializer(stddev=0.1)
    FC_BIAS_INITAILIZER = tf.constant_initializer(value=0.0)

    def train(self, batch_input, batch_output, learning_rate):
        _, loss,summary_val , global_step = self.sess.run([self.optimizer, self.loss,self.all_summary,self.global_step],
                                          feed_dict={self.input: batch_input,
                                                     self.desired_out: batch_output,
                                                     self.learning_rate: learning_rate})
        self.writer.add_summary(summary_val,global_step)
        return loss, global_step

    def  trainWithClassification(self, batch_input, batch_output,batch_output_ont_hot, learning_rate, cls_weight,
                                 batch_all_weight=1,
                                 batch_hard_weight=0):
        _, cls_accu,loss,summary_val , global_step, l1,l2,l3,l4 = self.sess.run([self.optimizer,
                                                           self.accuracy,
                                                           self.loss,
                                                           self.all_summary,
                                                           self.global_step,
                                                           self.l2_loss,
                                                           self.classfication_loss,
                                                           self.trilet_loss,
                                                           self.loss],
                                                          feed_dict={self.input: batch_input,
                                                                     self.desired_out: batch_output,
                                                                     self.one_hot_label : batch_output_ont_hot,
                                                                     self.learning_rate: learning_rate,
                                                                     self.classfication_loss_weight : cls_weight,
                                                                     self.batch_all_loss_weight : batch_all_weight,
                                                                     self.batch_hard_loss_weight : batch_hard_weight
                                                                     })
        print("[*]accu = %s loss = %s  %s  %s  %s " %(cls_accu,l1, l2, l3, l4))
        self.writer.add_summary(summary_val,global_step)
        return loss, global_step

    def forward(self, batch_input):
        return self.sess.run(self.embed, feed_dict={self.input: batch_input})

    def getSaverCollection(self):
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='EmbeddingLayers')
        vars.append(self.global_step)
        return vars

    def save(self, save_path, steps):
        saver = tf.train.Saver()
        saver.save(self.sess, save_path, global_step=steps)
        print("[*]save success")

    def save_embedding(self,save_path, steps):
        saver = tf.train.Saver(self.getSaverCollection())
        saver.save(self.sess, save_path, global_step=steps)



    def restore(self, restore_path):
        path = tf.train.latest_checkpoint(restore_path)
        if path == None:
            return False
        saver = tf.train.Saver(max_to_keep=5)
        saver.restore(self.sess, path)
        print("[*]Restore from %s success" % (path))
        return True

    def restore_embedding(self, restore_path):
        path = tf.train.latest_checkpoint(restore_path)
        if path == None:
            return False
        saver = tf.train.Saver(var_list= self.getSaverCollection())
        saver.restore(self.sess, path)
        print("[*]restore_except_softmax_layer %s success" % (path))
        return True

    def bn(self, x, name="bn"):
        with tf.variable_scope(name):
            # return x
            axes = [d for d in range(len(x.get_shape()))]
            beta = self._get_variable("beta", shape=[], initializer=tf.constant_initializer(0.0))
            gamma = self._get_variable("gamma", shape=[], initializer=tf.constant_initializer(1.0))
            x_mean, x_variance = tf.nn.moments(x, axes)
            y = tf.nn.batch_normalization(x, x_mean, x_variance, beta, gamma, 1e-10)
            return y

    def get_optimizer(self):
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.config.momemtum).minimize(
            self.loss, global_step=self.global_step)

    # 对x执行一次卷积操作+Relu
    def conv(self, x, name, channels, ksize=[3, 3], strides=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            x_shape = x.get_shape()
            x_channels = x_shape[3].value
            weight_shape = [ksize[0], ksize[1], x_channels, channels]
            bias_shape = [channels]
            weight = self._get_variable("weight", weight_shape, initializer=self.CONV_WEIGHT_INITAILIZER)
            bias = self._get_variable("bias", bias_shape, initializer=self.CONV_BIAS_INITAILIZER)
            y = tf.nn.conv2d(x, weight, strides=strides, padding=self.CONV_PADDING)
            y = tf.add(y, bias)
            return y

    def max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.MAX_POOL_PADDING, name=name)

    # 定义_get_variable方便进行l2_regularization以及其他一些操作
    def _get_variable(self, name, shape, initializer):
        param = 1
        for i in range(0, len(shape)):
            param *= shape[i]
        self.param_num += param
        if self.config.l2_lambda > 0:
            regularizer = tf.contrib.layers.l2_regularizer(self.config.l2_lambda)
        else:
            regularizer = None
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               regularizer=regularizer)

    def fc(self, x, num, name):
        x_num = x.get_shape()[1].value
        weight_shape = [x_num, num]
        bias_shape = [num]
        weight = self._get_variable("weight", shape=weight_shape, initializer=self.FC_WEIGHT_INITAILIZER)
        bias = self._get_variable("bias", shape=bias_shape, initializer=self.FC_BIAS_INITAILIZER)
        y = tf.add(tf.matmul(x, weight), bias, name=name)
        return y

    def res_block(self, x, channels, name, increase=False):
        with tf.variable_scope(name):
            if increase:
                strides = [1, 2, 2, 1]
            else:
                strides = [1, 1, 1, 1]
            # 1
            y = self.bn(x, "bn_a")
            y = self.ACTIVATE(y)
            y = self.conv(y, "conv_a", channels, [3, 3], strides)
            # 2
            y = self.bn(y, "bn_b")
            y = self.ACTIVATE(y)
            y = self.conv(y, "conv_b", channels)
            if increase:
                projection = self.conv(x, "conv_proj", channels, [3, 3], [1, 2, 2, 1])
                y = tf.add(projection, y)
            else:
                y = tf.add(x, y)
            return y

    def __init__(self,sess, config, tboard_dir):  #
        self.config = config
        input_shape = [None,config.input_size,config.input_size,1]
        output_shape = [None]
        stack_n = config.resnet_stack_n
        self.sess = sess
        layers = []
        # (1)placeholder定义(输入、输出、learning_rate)
        # input
        self.input = tf.placeholder(tf.float32, input_shape, name="input")
        layers.append(self.input)
        #
        with tf.variable_scope("EmbeddingLayers"):
            layers.append(self.bn(layers[-1]))
            # if True:
            self.ACTIVATE = tf.nn.relu
            self.param_num = 0  # 返回参数个数
            # (2)插入卷积层+池化层
            x = layers[-1]
            y = self.conv(x, "first_conv", 16, ksize=[3, 3])
            y = self.max_pool(y, "first_pool")
            layers.append(y)
            with tf.variable_scope("Residual_Blocks"):
                with tf.variable_scope("Residual_Blocks_STACK_0"):
                    for id in range(stack_n):
                        x = layers[-1]
                        b = self.res_block(x, 16, "block_%d" % (id))
                        layers.append(b)
                with tf.variable_scope("Residual_Blocks_STACK_1"):
                    x = layers[-1]
                    b = self.res_block(x, 24, "block_0", True)
                    layers.append(b)
                    for id in range(1, stack_n):
                        x = layers[-1]
                        b = self.res_block(x, 24, "block_%d" % (id))
                        layers.append(b)
                with tf.variable_scope("Residual_Blocks_STACK_2"):
                    x = layers[-1]
                    b = self.res_block(x, 32, "block_0", True)
                    layers.append(b)
                    for id in range(1, stack_n):
                        x = layers[-1]
                        b = self.res_block(x, 32, "block_%d" % (id))
                        layers.append(b)
                with tf.variable_scope("Residual_Blocks_STACK_3"):
                    x = layers[-1]
                    b = self.res_block(x, 48, "block_0", True)
                    layers.append(b)
                    for id in range(1, stack_n):
                        x = layers[-1]
                        b = self.res_block(x, 48, "block_%d" % (id))
                        layers.append(b)
                        # maxpool
            """
            x=layers[-1]
            y=self.max_pool(x,"maxpool_after_resblocks")
            layers.append(y)
            """
            x = self.ACTIVATE(self.bn(layers[-1],name = "bn_before_compress"))

            y = self.conv(x, "compress", 4, ksize=[3, 3])
            layers.append(y)
            # (3)卷积层flatten
            with tf.variable_scope("Flatten"):
                last_layer = layers[-1]
                last_shape = last_layer.get_shape()
                neu_num = 1
                for dim in range(1, len(last_shape)):
                    neu_num *= last_shape[dim].value
                flat_layer = tf.reshape(last_layer, [-1, neu_num], name="flatten")
                layers.append(flat_layer)

            # (4)embedding 层
            with tf.variable_scope("Embedding"):
                x = layers[-1]
                y = self.bn(x)
                y = self.fc(y, 128, "embedding")
                self.embed = tf.nn.l2_normalize(y,1)
                layers.append(self.embed)

        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.desired_out = tf.placeholder(tf.float32, output_shape, name="desired_out")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.batch_all_loss_weight = tf.placeholder(tf.float32, name="batch_all_weight")
        self.batch_hard_loss_weight = tf.placeholder(tf.float32, name="batch_hard_weight")

        self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="l2_loss")
        with tf.variable_scope("TripletLoss"):
            if config.use_triplet_loss:
                self.batch_all_loss = self.batch_all_loss_weight * TripletLoss.batch_all_triplet_loss(self.desired_out, self.embed, config.triplet_loss_margin)[0]
                self.batch_hard_loss = self.batch_hard_loss_weight * TripletLoss.batch_hard_triplet_loss(self.desired_out, self.embed, config.triplet_loss_margin)
                self.trilet_loss = self.batch_all_loss + self.batch_hard_loss
            else:
                self.trilet_loss = tf.constant(0.0, dtype=tf.float32)

        ####### 分类loss
        with tf.variable_scope("ClassificationLayers"):
            self.one_hot_output = self.fc(self.embed, config.training_classes, "one_hot_output")
            self.one_hot_label = tf.placeholder(tf.float32, [None, config.training_classes], name ="one_hot_label")
            self.classfication_loss_weight = tf.placeholder(tf.float32,name =  "classfication_loss_weight")
            # self.classfication_loss = self.classfication_loss_weight * (-tf.reduce_mean(self.one_hot_label * tf.log(tf.clip_by_value(self.one_hot_output,1e-10,1.0))))
            # 准确率

            self.iscorrect = tf.equal(tf.argmax(self.one_hot_label, 1), tf.argmax(self.one_hot_output, 1), name="iscorrect")
            self.accuracy = tf.reduce_mean(tf.cast(self.iscorrect, dtype=tf.float32), name="accuracy")
            self.classfication_loss = self.classfication_loss_weight * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= self.one_hot_output, labels=self.one_hot_label))

        self.loss = tf.add_n([self.l2_loss, self.trilet_loss,self.classfication_loss], name = "WeightedLoss")

        tf.summary.scalar("Classfication Accuracy", self.accuracy)
        tf.summary.scalar("Classfication Loss", self.classfication_loss)
        tf.summary.scalar("L2 Loss",self.l2_loss)
        tf.summary.scalar("Batch All Loss", self.batch_all_loss)
        tf.summary.scalar("Batch Hard Loss", self.batch_hard_loss)
        tf.summary.scalar("Triplet Loss",self.trilet_loss)
        tf.summary.scalar("Loss", self.loss)
        tf.summary.scalar("Learning Rate", self.learning_rate)

        self.all_summary = tf.summary.merge_all()
        # output
        # loss函数
        # (7)优化器和 variables初始化
        self.get_optimizer()
        self.sess.run(tf.global_variables_initializer())
        self.writer = tf.summary.FileWriter(tboard_dir, self.sess.graph)

    def __del__(self):
        self.sess.close()
