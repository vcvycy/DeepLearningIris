import tensorflow as tf
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
  sess=None
  #Tensor
  input=None 
  output=None
  desired_out=None
  loss=None
  iscorrect=None
  accuracy=None
  optimizer=None
  param_num=0             #参数个数
  #参数
  learning_rate=None 
  MOMENTUM         = 0.9
  WEIGHT_DECAY     = 1e-4       #L2 REGULARIZATION
  ACTIVATE         = None
  CONV_PADDING     = "SAME"
  MAX_POOL_PADDING = "SAME"
  CONV_WEIGHT_INITAILIZER = tf.keras.initializers.he_normal()#tf.truncated_normal_initializer(stddev=0.1)
  CONV_BIAS_INITAILIZER   = tf.constant_initializer(value=0.0)
  FC_WEIGHT_INITAILIZER   = tf.keras.initializers.he_normal()#tf.truncated_normal_initializer(stddev=0.1)
  FC_BIAS_INITAILIZER     = tf.constant_initializer(value=0.0)
  
  
  def train(self,batch_input,batch_output,learning_rate):  
    _,accuracy,loss=self.sess.run([self.optimizer,self.accuracy,self.loss],
       feed_dict={self.input:batch_input,self.desired_out:batch_output,self.learning_rate:learning_rate})
    return {"accuracy":accuracy,"loss":loss}
    
  def forward(self,batch_input):
    return self.sess.run(self.output,feed_dict={self.input:batch_input})
  
  def save(self,save_path,steps):
    saver=tf.train.Saver(max_to_keep=5)
    saver.save(self.sess,save_path,global_step=steps)
    print("[*]save success")
    
  def restore(self,restore_path):
    path=tf.train.latest_checkpoint(restore_path)
    if path==None:
      return False
    saver=tf.train.Saver(max_to_keep=5)
    saver.restore(self.sess,path)
    print("[*]Restore from %s success" %(path))
    return True
  
  def restore_except_softmax_layer(self,restore_path):
    path=tf.train.latest_checkpoint(restore_path)
    if path==None:
      return False
    var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="LAYERS_EXCEPT_SOFTMAX")
    """
    var_dict={}
    for var in var_list:
      if var.name[0]=="L":
        new_name=var.name[22:]
        new_name=new_name[0:len(new_name)-2]
        if new_name[0]=="F":
          continue
        var_dict[new_name]=var   
        print("[%d]%s" %(len(var_dict),new_name))
    """
    saver=tf.train.Saver(var_list=var_list)
    saver.restore(self.sess,path)
    print("[*]restore_except_softmax_layer %s success" %(path)) 
    return True
  
  
  def bn(self,x,name="bn"):
    with tf.variable_scope(name):
      #return x
      axes = [d for d in range(len(x.get_shape()))]
      beta = self._get_variable("beta", shape=[],initializer=tf.constant_initializer(0.0))
      gamma= self._get_variable("gamma",shape=[],initializer=tf.constant_initializer(1.0))
      x_mean,x_variance=tf.nn.moments(x,axes)  
      y=tf.nn.batch_normalization(x,x_mean,x_variance,beta,gamma,1e-10)
      return y
    
  def get_optimizer(self): 
    self.optimizer =tf.train.MomentumOptimizer(self.learning_rate,self.MOMENTUM).minimize(self.loss)            #9000 steps后达到误差范围。  
  
  #对x执行一次卷积操作+Relu
  def conv(self,x,name,channels,ksize=[3,3],strides=[1,1,1,1]):
    with tf.variable_scope(name):
      x_shape=x.get_shape()
      x_channels=x_shape[3].value
      weight_shape=[ksize[0],ksize[1],x_channels,channels]
      bias_shape=[channels]
      weight = self._get_variable("weight",weight_shape,initializer=self.CONV_WEIGHT_INITAILIZER)
      bias   = self._get_variable("bias",bias_shape,initializer=self.CONV_BIAS_INITAILIZER) 
      y=tf.nn.conv2d(x,weight,strides=strides,padding=self.CONV_PADDING)
      y=tf.add(y,bias)
      return y
  
  def max_pool(self,x,name):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding=self.MAX_POOL_PADDING,name=name)
    
  #定义_get_variable方便进行l2_regularization以及其他一些操作
  def _get_variable(self,name,shape,initializer):
    param=1
    for i in range(0,len(shape)):
      param*=shape[i]
    self.param_num+=param
    
    if self.WEIGHT_DECAY>0:
      regularizer=tf.contrib.layers.l2_regularizer(self.WEIGHT_DECAY)
    else:
      regularizer=None  
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           regularizer=regularizer)
                           
  def fc(self,x,num,name):
    x_num=x.get_shape()[1].value
    weight_shape=[x_num,num]
    bias_shape  =[num]
    weight=self._get_variable("weight",shape=weight_shape,initializer=self.FC_WEIGHT_INITAILIZER)
    bias  =self._get_variable("bias",shape=bias_shape,initializer=self.FC_BIAS_INITAILIZER)
    y=tf.add(tf.matmul(x,weight),bias,name=name)
    return y 
  def _loss(self): 
    cross_entropy=-tf.reduce_sum(self.desired_out*tf.log(tf.clip_by_value(self.output,1e-10,1.0)))
    regularization_losses=tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    self.loss = tf.add_n([cross_entropy]+regularization_losses)
    #tf.scalar_summary('loss', loss_)
    return self.loss
  
  def res_block(self,x,channels,name,increase=False): 
    with tf.variable_scope(name):
      if increase:
        strides=[1,2,2,1]
      else:
        strides=[1,1,1,1]
      #1
      y=self.bn(x,"bn_a")
      y=self.ACTIVATE(y)
      y=self.conv(y,"conv_a",channels,[3,3],strides)
      #2
      y=self.bn(y,"bn_b")
      y=self.ACTIVATE(y)
      y=self.conv(y,"conv_b",channels)
      if increase:
        projection=self.conv(x,"conv_proj",channels,[3,3],[1,2,2,1])
        y=tf.add(projection,y)
      else:
        y=tf.add(x,y)
      return y
    
  def __init__(self,input_sz,output_sz,stack_n): #
    with tf.variable_scope("LAYERS_EXCEPT_SOFTMAX"):
    #if True:
        self.ACTIVATE=tf.nn.relu
        self.param_num=0  #返回参数个数
        self.sess=tf.Session()
        layers=[]
        #(1)placeholder定义(输入、输出、learning_rate)
        #input
        self.input=tf.placeholder(tf.float32,input_sz,name="input") 
        layers.append(self.input)
        #
        layers.append(self.bn(layers[-1])) 
        #(2)插入卷积层+池化层
        x=layers[-1]
        y=self.conv(x,"first_conv",16,ksize=[7,7])
        y=self.max_pool(y,"first_pool")
        layers.append(y) 
        with tf.variable_scope("Residual_Blocks"):
          with tf.variable_scope("Residual_Blocks_STACK_0"):
            for id in range(stack_n):
              x=layers[-1]
              b=self.res_block(x,16,"block_%d" %(id))
              layers.append(b)
          with tf.variable_scope("Residual_Blocks_STACK_1"):
            x=layers[-1]
            b=self.res_block(x,32,"block_0",True)
            layers.append(b)
            for id in range(1,stack_n):
              x=layers[-1]
              b=self.res_block(x,32,"block_%d" %(id))
              layers.append(b)
          with tf.variable_scope("Residual_Blocks_STACK_2"):
            x=layers[-1]
            b=self.res_block(x,64,"block_0",True)
            layers.append(b)
            for id in range(1,stack_n):
              x=layers[-1]
              b=self.res_block(x,64,"block_%d" %(id))
              layers.append(b)      
          with tf.variable_scope("Residual_Blocks_STACK_3"):
            x=layers[-1]
            b=self.res_block(x,64,"block_0",True)
            layers.append(b)
            for id in range(1,stack_n):
              x=layers[-1]
              b=self.res_block(x,64,"block_%d" %(id))
              layers.append(b) 
        #maxpool
        """
        x=layers[-1]
        y=self.max_pool(x,"maxpool_after_resblocks")
        layers.append(y)
        """
        #(3)卷积层flatten
        with tf.variable_scope("Flatten"):
          last_layer=layers[-1]
          last_shape=last_layer.get_shape()
          neu_num=1
          for dim in range(1,len(last_shape)): 
           neu_num*= last_shape[dim].value
          flat_layer=tf.reshape(last_layer,[-1,neu_num],name="flatten")
          layers.append(flat_layer) 
        #(4)全连接层 #!!!!!!!!!最后一层不要加上relu!!!!!!
        with tf.variable_scope("Feature"):
          x=layers[-1]
          y=self.bn(x)
          y=self.ACTIVATE(y)
          y=self.fc(y,128,"Feature")
          layers.append(y)
    with tf.variable_scope("softmax_layer"):  #
      x=layers[-1]
      y=self.bn(x)
      y=self.fc(y,output_sz[1],"fc")
      layers.append(y) 
      #(5)softmax和loss函数
      self.output=tf.nn.softmax(layers[-1])
    
    #output 
    self.desired_out=tf.placeholder(tf.float32,output_sz,name="desired_out")
    self.learning_rate=tf.placeholder(tf.float32,name="learning_rate")
    #loss函数
    self._loss()
    #(6)辅助信息：正确率
    self.iscorrect=tf.equal(tf.argmax(self.desired_out,1),tf.argmax(self.output,1),name="iscorrect")
    self.accuracy=tf.reduce_mean(tf.cast(self.iscorrect,dtype=tf.float32),name="accuracy")
    #(7)优化器和 variables初始化
    self.get_optimizer()
    self.sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter("./tboard/",self.sess.graph)  
  def __del__(self):
    self.sess.close()
