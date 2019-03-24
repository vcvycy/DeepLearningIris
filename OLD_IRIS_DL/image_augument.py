import cv2
import os
import sys
import numpy as np
import math
import random
class IRIS_AUG:   
  #输出大小统一：200px * 200px
  nor_w=None
  nor_h=None
  #0代表虹膜像素，1代表瞳孔内，2代表虹膜外
  is_iris=None
  rot_mat=None
  def __init__(self,img_w): 
    IRIS_AUG.nor_w=img_w+10
    IRIS_AUG.nor_h=img_w+10
    #瞳孔坐标
    self.pupil_x=0
    self.pupil_y=0
    self.pupil_r=0
    self.img=None 
    self.is_iris=[[0 for _ in range(self.nor_w)] for _ in range(self.nor_h)]
    #初始化旋转矩阵
    if self.rot_mat==None:
      self.rot_mat=[]
      for degree in range(0,360):
        self.rot_mat.append(cv2.getRotationMatrix2D((IRIS_AUG.nor_w/2,IRIS_AUG.nor_h/2),degree,1.0))
        
  #######算法###########
  #(1)影响图片大小算法(先执行此操作后，resize到(nor_w,nor_h)，然后执行(2)中的算法
  #(2)不影响图片大小算法
  #图片rotation 
  def rotate(self,angle):   
    #print("[*]旋转%d度" %(angle))
    if angle<0:
      angle+=360 
    self.img = cv2.warpAffine(self.img,self.rot_mat[angle],(IRIS_AUG.nor_w,IRIS_AUG.nor_h)) 
  #眼睑遮挡，遮挡占比percent
  def overlay_top(self):
    #(1)上眼睑
    #print("[*]眼睑覆盖");
    top_pixel=random.randint(0,int(self.nor_w/2))
    size=self.img.shape
    #print(top_pixel)
    for i in range(top_pixel): 
        for j in range(size[1]): 
          #if self.is_iris[i][j]<2: 
            self.img[i,j]=0
    #(2)下眼睑
    btm_pixel=random.randint(0,int(self.nor_w/8))
    for i in range(self.nor_h-btm_pixel,self.nor_h): 
        for j in range(size[1]): 
          #if self.is_iris[i][j]<2: 
            self.img[i,j]=0
  #######debug##########
  def crop(self):
    x=random.randint(0,10)
    y=random.randint(0,10)
    self.img=self.img[x:x+self.nor_w-10,y:y+self.nor_h-10]
    #print("[*]crop %d %d" %(x,y))
  def show(self):
    win_name="%d*%d" %(self.img.shape[0],self.img.shape[1])
    cv2.imshow(win_name,self.img)
    cv2.waitKey(0)
  def init_is_iris(self):
    for x in range(self.nor_w):
      for y in range(self.nor_h):
        if (x-100)*(x-100)+(y-100)*(y-100)>10000:
          #self.img[x,y]/=2 #虹膜外
          self.is_iris[x][y]=2
        
        else:
          if (x-self.pupil_x)*(x-self.pupil_x)+(y-self.pupil_y)*(y-self.pupil_y) < self.pupil_r*self.pupil_r:
            #self.img[x,y]=100 #瞳孔内
            self.is_iris[x][y]=1
          else:
            #self.img[x,y]/=10 #虹膜像素
            self.is_iris[x][y]=0
  def img_transform(self):
    self.resize()
    
    #(1)随机旋转-45~45°
    theta=random.randint(-30,30)
    self.rotate(theta)
    #(2) 计算每个像素点属于什么(虹膜/瞳孔/其他)  (图片旋转后其他形变操作后需要重新调用)
    self.init_is_iris()
    #self.show()
    #(3)模拟上下眼睑遮挡(颜色重置为模糊)
    self.overlay_top();
    theta=random.randint(-30,30)
    self.rotate(theta)
    #(4) crop
    self.crop()  
  def resize(self): #resize后需要重新定位Pupil
    sz=self.img.shape
    self.pupil_x=int(self.pupil_x/sz[0]*self.nor_w)
    self.pupil_y=int(self.pupil_y/sz[1]*self.nor_h)
    self.pupil_r=int(self.pupil_r/sz[0]*self.nor_w) 
    self.img=cv2.resize(self.img,(self.nor_w,self.nor_h),interpolation=cv2.INTER_CUBIC) 
  def process(self,path):
    #从路径种读取参数
    par=path.split(".")
    self.pupil_y=int(par[-4])
    self.pupil_x=int(par[-3])
    self.pupil_r=int(par[-2]) 
    #
    self.img=cv2.imread(path)
    self.img_transform()
  def loadFromBin(self,imgdata,path):
    nparr = np.fromstring(imgdata, np.uint8)
    self.img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_COLOR in OpenCV 3.1
    path=path.decode("utf-8")
    par=path.split("/")[-1].split(".")
    self.pupil_y=int(par[-4])
    self.pupil_x=int(par[-3])
    self.pupil_r=int(par[-2]) 
    self.img_transform() 
    mat=[[[0] for _ in range(self.img.shape[0])] for _ in range(self.img.shape[1])]
    for x in range(self.img.shape[0]):
      for y in range(self.img.shape[1]):
        mat[x][y][0]=self.img[x,y]
    #self.show()
    return mat
if __name__=="__main__":
  iris=IRIS_AUG(192)    
  iris.process("S5037L09_segm.bmppupil.161.182.43.bmp")
  iris.show()
