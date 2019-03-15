import tensorflow as tf
import sys
"""
(1)构造函数__init__参数
  input_sz：  输入层placeholder的4-D shape，如mnist是[None,28,28,1]
  fc_layers： 全连接层每一层大小，接在卷积层后面。如mnist可以为[128,84,10],[10]
  conv_info： 卷积层、池化层。
    如vgg16可以这样写：[(2,64),(2,128),(3,256),(3,512),(3,512)]，表示2+2+3+3+3=13个卷积层，4个池化层，以及channels
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
from Recognition import TripleLoss

class VGG:  # VGG分类器
    sess = None
    # Tensor
    input = None
    output = None
    desired_out = None
    loss = None
    iscorrect = None
    accuracy = None
    optimizer = None
    param_num = 0  # 参数个数
    # 参数
    learning_rate = None
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4  # L2 REGULARIZATION
    ACTIVATE = None
    CONV_PADDING = "SAME"
    MAX_POOL_PADDING = "SAME"
    CONV_WEIGHT_INITAILIZER = tf.keras.initializers.he_normal() #tf.truncated_normal_initializer(stddev=0.1)
    CONV_BIAS_INITAILIZER   = tf.constant_initializer(value=0.0)
    FC_WEIGHT_INITAILIZER   = tf.keras.initializers.he_normal() #tf.truncated_normal_initializer(stddev=0.1)
    FC_BIAS_INITAILIZER     = tf.constant_initializer(value=0.0)

    def train(self, batch_input, batch_output, learning_rate):
        # print("learning_rate: %s" %(learning_rate))
        _,loss,o,do,summary_value = self.sess.run([self.optimizer,self.loss,self.output,self.desired_out,self.all_summary],feed_dict={
                                        self.input : batch_input,
                                        self.desired_out : batch_output,
                                        self.learning_rate : learning_rate
                                    })
        # print(o[0])
        # print(do[0])
        self.writer.add_summary(summary_value,self.train_step)
        self.train_step+=1
        return loss

    def forward(self, batch_input):
        return self.sess.run(self.output, feed_dict={self.input: batch_input})

    def save(self, save_path, steps):
        saver = tf.train.Saver(max_to_keep=5)
        saver.save(self.sess, save_path, global_step=steps)
        return

    def restore(self, restore_path):
        path = tf.train.latest_checkpoint(restore_path)
        print("[*]Restore from %s" % (path))
        if path == None:
            print("[*]失败！")
            sys.exit(0)
        saver = tf.train.Saver(max_to_keep=5)
        saver.restore(self.sess, path)
        return

    def bn(self, x, name="bn"):
        # return x
        axes = [d for d in range(len(x.get_shape()))]
        beta = self._get_variable("beta", shape=[], initializer=tf.constant_initializer(0.0))
        gamma = self._get_variable("gamma", shape=[], initializer=tf.constant_initializer(1.0))
        x_mean, x_variance = tf.nn.moments(x, axes)
        y = tf.nn.batch_normalization(x, x_mean, x_variance, beta, gamma, 1e-10, name)
        return y

    def get_optimizer(self):  #
        # Optimizer
        # sself.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        # self.optimizer =tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss) #1300 steps后达到误差范围。
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.MOMENTUM).minimize(
            self.loss)  # 9000 steps后达到误差范围。

    # 对x执行一次卷积操作+Relu
    def conv(self, x, name, channels, ksize=3,strides=[1,1,1,1]):
        x_shape = x.get_shape()
        x_channels = x_shape[3].value
        weight_shape = [ksize, ksize, x_channels, channels]
        bias_shape = [channels]
        weight = self._get_variable("weight", weight_shape, initializer=self.CONV_WEIGHT_INITAILIZER)
        bias = self._get_variable("bias", bias_shape, initializer=self.CONV_BIAS_INITAILIZER)
        y = tf.nn.conv2d(x, weight, strides=strides, padding=self.CONV_PADDING, name= name)
        y = tf.add(y, bias, name="%s-bias" %name)
        return y

    def max_pool(self, x, name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=self.MAX_POOL_PADDING, name=name)

    # 定义_get_variable方便进行l2_regularization以及其他一些操作
    def _get_variable(self, name, shape, initializer,need_l2=True):
        param = 1
        for i in range(0, len(shape)):
            param *= shape[i]
        self.param_num += param
        if need_l2:
            l2_reg = self.l2_reg
        else:
            l2_reg = None
        return tf.get_variable(name,
                               shape=shape,
                               initializer=initializer,
                               regularizer=l2_reg)

    def fc(self, x, num, name):
        x_num = x.get_shape()[1].value
        weight_shape = [x_num, num]
        bias_shape = [num]
        weight = self._get_variable("weight", shape=weight_shape, initializer=self.FC_WEIGHT_INITAILIZER)
        bias = self._get_variable("bias", shape=bias_shape, initializer=self.FC_BIAS_INITAILIZER)
        y = tf.add(tf.matmul(x, weight), bias, name=name)
        return y

    def square_loss(self):
        loss = (self.desired_out-self.output)*(self.desired_out-self.output)
        loss = tf.reduce_sum(loss,0) * [1,1,1,1,0,0,0,0]      # 只处理虹膜的Loss

        l2_vars = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.add_n(l2_vars)

        self.loss = tf.reduce_mean(loss) + l2_loss
        return self.loss

    def __init__(self, input_sz, fc_layers, conv_info=[],  activate_fun=tf.nn.relu):  #
        self.ACTIVATE = activate_fun
        self.param_num = 0  # 返回参数个数
        self.sess = tf.Session()
        self.train_step =0

        if self.WEIGHT_DECAY > 0:
            self.l2_reg = tf.contrib.layers.l2_regularizer(self.WEIGHT_DECAY)
        else:
            self.l2_reg = None
        layers = []
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # (1)placeholder定义(输入、输出、learning_rate)
        # input
        self.input = tf.placeholder(tf.float32, input_sz, name="input")
        self.input_nor = self.bn(self.input)
        self.input_conv = self.conv(self.input_nor,"first_conv",16,7,[1,2,2,1]);
        layers.append(self.input_conv)
        # output
        output_sz = [None, fc_layers[-1]]
        self.desired_out = tf.placeholder(tf.float32, output_sz, name="desired_out")

        # (2)插入卷积层+池化层
        with tf.variable_scope("convolution"):
            conv_block_id = 0
            for cur_layers in conv_info:
                # 添加卷积层block
                with tf.variable_scope("conv_block_%d" % (conv_block_id)) as scope:
                    cur_conv_num = cur_layers[0]  # cur_conv_num个卷积层叠放
                    cur_channels = cur_layers[1]  # 每个卷积层的通道
                    # cur_conv_num个卷积层叠加
                    for conv_id in range(0, cur_conv_num):
                        with tf.variable_scope("conv_%d" % (conv_id)):
                            # 添加卷积层
                            x = layers[-1]
                            #"""
                            #顺序一：x->bn->weight->relu
                            x2=self.bn(x) 
                            x3=self.conv(x2,channels=cur_channels,name="conv")
                            x4=self.ACTIVATE(x3)
                            #"""

                            """
                            # 顺序二: x->bn->relu->weight
                            x2 = self.bn(x)
                            x3 = self.ACTIVATE(x2)
                            x4 = self.conv(x3, channels=cur_channels, name="conv")
                             """

                            """
                            #顺序三：x->weight->bn->relu
                            x2=self.conv(x,channels=cur_channels,name="conv")
                            x3=self.bn(x2)
                            x4=self.ACTIVATE(x3)
                            """
                            layers.append(x4)
                            # 每个卷积块后是pool层
                    last_layer = layers[-1]
                    pool = self.max_pool(last_layer, "max_pool")
                    layers.append(pool)
                    conv_block_id += 1

        # (3)卷积层flatten
        last_layer = layers[-1]
        last_shape = last_layer.get_shape()
        neu_num = 1
        for dim in range(1, len(last_shape)):
            neu_num *= last_shape[dim].value
        flat_layer = tf.reshape(last_layer, [-1, neu_num], name="flatten")
        layers.append(flat_layer)

        embed = self.fc(self.bn(flat_layer),64,"embeddings")

        self.output = self.fc(embed,2000)
        # loss函数
        self.square_loss()
        # summary scalar
        tf.summary.scalar("Loss", self.loss)
        self.all_summary = tf.summary.merge_all()
        # (7)优化器和 variables初始化
        self.get_optimizer()
        self.sess.run(tf.global_variables_initializer())
        # tf.gfile.DeleteRecursively('./tboard/')
        self.writer = tf.summary.FileWriter("./tboard/", self.sess.graph)

        print("[*]参数个数 :%s" %(self.param_num))

    def __del__(self):
        self.sess.close()
        self.writer.close()