import os
import sys
import tensorflow as tf
#只负责对读取图片二进制数据，文件名二进制
class MyBinReader: 
  #tensor
  train_input=None
  train_lable=None 
  #
  train_cnt=None
  def getFileNameFrom(self,dir,classes,accept_suffix="jpg|bmp"): #文件名list
    suffix=set(accept_suffix.split("|"))
    filename_list=[]
    for root,dirs,files in os.walk(dir):
      for file in files:
        if file.strip().split(".")[-1] in suffix: 
          path="%s/%s" %(root,file)
          iris_id=int(file[2:5])
          if iris_id*2+2<=classes:
#          print("path%s idx=%s" %(file,file[2:5]))
            filename_list.append(path) 
    self.train_cnt=len(filename_list)
    return filename_list 
    
  def cntTrainImg(self):
    return self.train_cnt
    
  def __init__(self,sess,train_dir,classes):
    #(1)文件名list
    train_filename_list=self.getFileNameFrom(train_dir,classes)
    print("[*]从%s读取训练数据，一共有%d张图" %(train_dir,len(train_filename_list)))
    #(2)文件名队列
    train_filename_queue=tf.train.string_input_producer(train_filename_list,shuffle=True)
    #(3)reader
    train_reader=tf.WholeFileReader()
    #(4)生成
    key,val=train_reader.read(train_filename_queue)
    self.train_input = val
    self.train_label = key
    #(5)初始化 
    sess.run(tf.local_variables_initializer()) 
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
  def getTrainBatch(self,sess,batch_sz):
    data=([],[])
    for _ in range(batch_sz):
      input,label=sess.run([self.train_input,self.train_label])
      data[0].append(input) 
      data[1].append(label)
    return data
    
