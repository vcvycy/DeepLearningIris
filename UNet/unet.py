import tensorflow as tf
from UNet import TripletLoss
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


class Unet:
    param_num = 0  # 参数个数

    CONV_PADDING = "SAME"
    MAX_POOL_PADDING = "SAME"
    CONV_WEIGHT_INITAILIZER = tf.keras.initializers.he_normal()  # tf.truncated_normal_initializer(stddev=0.1)
    CONV_BIAS_INITAILIZER = tf.constant_initializer(value=0.0)
    FC_WEIGHT_INITAILIZER = tf.keras.initializers.he_normal()  # tf.truncated_normal_initializer(stddev=0.1)
    FC_BIAS_INITAILIZER = tf.constant_initializer(value=0.0)

    def train(self, batch_input, batch_output, learning_rate,batch_all_weight,batch_hard_weight):
        _, loss,summary_val , global_step = self.sess.run([self.optimizer, self.loss,self.all_summary,self.global_step],
                                          feed_dict={self.input: batch_input,
                                                     self.desired_out: batch_output,
                                                     self.learning_rate: learning_rate,
                                                     self.batch_all_loss_weight : batch_all_weight,
                                                     self.batch_hard_loss_weight : batch_hard_weight})
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

    # 对x执行一次bn+卷积操作+Relu
    def conv(self, x, name, channels, ksize=[3, 3], strides=[1, 1, 1, 1]):
        with tf.variable_scope(name):
            x=self.bn(x,"bn")
            x_shape = x.get_shape()
            x_channels = x_shape[3].value
            weight_shape = [ksize[0], ksize[1], x_channels, channels]
            bias_shape = [channels]
            weight = self._get_variable("weight", weight_shape, initializer=self.CONV_WEIGHT_INITAILIZER)
            bias = self._get_variable("bias", bias_shape, initializer=self.CONV_BIAS_INITAILIZER)
            y = tf.nn.conv2d(x, weight, strides=strides, padding=self.CONV_PADDING)
            y = tf.add(y, bias)
            y = tf.nn.relu(y)
            return y

    # 反卷积、特征图扩大两倍。 bn+deconv+relu
    def deconv(self,x,name,out_channel):
        with tf.name_scope(name):
            x=self.bn(x,"bn")
            x_shape = x.get_shape()
            in_channel = x_shape[3].value
            weight_shape = [3,3,out_channel,in_channel]
            weight = self._get_variable("weight", weight_shape, initializer=self.CONV_WEIGHT_INITAILIZER)
            # print([x_shape[0].value, x_shape[1].value * 2, x_shape[2].value * 2, out_channel])
            output_shape = tf.stack([self.batch_size, x_shape[1].value * 2, x_shape[2].value * 2, out_channel])
            return tf.nn.relu(tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, 2, 2, 1], padding='SAME',
                                              name="conv2d_transpose"))
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

    def __init__(self,sess, config, tboard_dir):  #
        self.config = config
        self.batch_size= config.batch_img_num_each_class * config.batch_class_num
        input_shape = [None,config.input_height,config.input_width,1]
        output_shape = [None]
        self.sess = sess
        layers = dict()
        self.layers=layers
        # (1)placeholder定义(输入、输出、learning_rate)
        # input
        layers["input"]= tf.placeholder(tf.float32, input_shape, name="input")
        self.input=layers["input"]
        channels = [16,24,32,48]        # 四次下采样
        self.param_num = 0  # 返回参数个数

        with tf.variable_scope("EmbeddingLayers"):
            last_layer = layers["input"]
            # 下采样层
            for i in range(len(channels)):
                out_channel = channels[i]
                # print("down_conv_{}".format(out_channel))
                with tf.variable_scope("down_conv_{}".format(out_channel)):
                    conv1 = self.conv(last_layer, "conv1",out_channel,ksize=[3,3])  # bn + conv+ relu
                    conv2 = self.conv(conv1, "conv2",out_channel,ksize=[3,3])
                    if out_channel != len(channels)-1:
                        last_layer = self.max_pool(conv2,"max_pool")
                    else:
                        last_layer = conv2
                    layers["down_conv_{}".format(out_channel)]=last_layer
            #上采样
            for i in range(len(channels)-2,-1,-1):
                out_channel = channels[i]
                #print("up_conv_{}".format(out_channel))
                with tf.variable_scope("up_conv_{}".format(out_channel)):
                    deconv = self.deconv(last_layer,"deconv",out_channel)
                    #print(deconv)
                    #print(layers["down_conv_{}".format(out_channel)])
                    concated = tf.concat([layers["down_conv_{}".format(out_channel)],deconv],3)
                    conv1 = self.conv(concated,"conv1",out_channel)
                    conv2 = self.conv(conv1,"conv2",out_channel)
                    last_layer = conv2

            # (4)embedding 层
            with tf.variable_scope("Embedding"):
                x = self.bn(last_layer)
                y = self.conv(x, "mat_embedding", 1, ksize=[1, 1])
                sz = y.shape[1]*y.shape[2]
                y = tf.reshape(y,[-1,sz],name="flatten")
                # eweight
                self.eweight = tf.nn.sigmoid(tf.nn.relu(tf.reshape(self.conv(x,"mat_weight",1,ksize=[1,1]),[-1,sz])))
                self.embed = y # tf.nn.l2_normalize(y,1)
                layers["embed"] = self.embed
                layers["eweight"]=self.eweight
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")
        self.desired_out = tf.placeholder(tf.float32, output_shape, name="desired_out")
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.batch_all_loss_weight = tf.placeholder(tf.float32, name="batch_all_weight")
        self.batch_hard_loss_weight = tf.placeholder(tf.float32, name="batch_hard_weight")

        self.l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),name="l2_loss")
        with tf.variable_scope("TripletLoss"):
            self.batch_all_loss = TripletLoss.batch_all_triplet_loss_semi(self.desired_out, self.embed, self.eweight,config.triplet_loss_margin)[0]
            self.batch_hard_loss = TripletLoss.batch_hard_triplet_loss_semi(self.desired_out, self.embed, self.eweight,config.triplet_loss_margin)
            self.trilet_loss = self.batch_all_loss_weight * self.batch_all_loss + self.batch_hard_loss_weight * self.batch_hard_loss

        self.loss = tf.add_n([self.l2_loss, self.trilet_loss], name = "WeightedLoss")

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
