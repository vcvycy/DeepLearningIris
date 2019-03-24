import BinReader
import image_augument
import tensorflow as tf
import sys
class IrisReader:  
  def __init__(self,sess,dir,img_w,classes):
    self.classes=classes
    self.sess=sess
    self.BinReader=BinReader.MyBinReader(sess,dir,classes) 
    self.img_aug=image_augument.IRIS_AUG(img_w) 
    
  def int2onehot(self,val):
    if val>=self.classes:
      print("[!]err val=%d classes=%d" %(val,self.classes))
      sys.exit(0)
    one_hot=[]
    for _ in range(self.classes):
      one_hot.append(0)
    one_hot[val]=1
    return one_hot
    
  def getLabelFromBin(self,label_bin):
    str=label_bin.decode("utf-8")  
    if str.find("L")!=-1:
      lr = "L"
    elif str.find("R")!=-1:
      lr = "R"
    else:
      error()
    s5idx = str.find("S5")
    label=int(str[s5idx+2:s5idx+5])*2
    if lr=="R":
      label+=1
    return label
  def cntTrainImg(self):
    return self.BinReader.cntTrainImg()
    
  def getTrainBatch(self,batch_sz): 
    ret=([],[]) 
    input_bin,label_bin=self.BinReader.getTrainBatch(self.sess,batch_sz) 
    for i in range(batch_sz):
      ret[0].append(self.img_aug.loadFromBin(input_bin[i],label_bin[i]))
      ret[1].append(self.int2onehot(self.getLabelFromBin(label_bin[i]))) 
    return ret
    
"""sess=tf.Session()
reader=IrisReader(sess,"D:/Dataset/segmented_iris",192)
batch=reader.getTrainBatch(1)
print(batch[0][0][0])
print(batch[1][0])
"""
