import os
import sys

def get_list_1(file):   #文件中的jpg
  s=set()
  f=open(file,"r") 
  for line in f:
    line=line.strip().split('/')[-1]
    if (len(line)>4): 
      s.add(line)
  return s
def get_list_2():   #目录中的
  s=set()
  for root,dirs,files in os.walk("."):
    for file in files:
      if file.strip().split(".")[-1]=="jpg":
        s.add(file)
  return s
def get_list_3(dir):    #在segmented 中出现的
  s=set()
  for root,dirs,files in os.walk(dir):
    for file in files:
      if len(file.strip().split("."))>2:
        x=file.strip().split("_")[0]+".jpg"
        s.add(x)
  return s
#list_1=get_list_1("process_CASIA-IrisV2.txt")
list_2=get_list_2()
list_3=get_list_3("../segmented_iris")
print("[*] v4中的数据集大小:%s" %(len(list_2)))
print("[*] segmented大小:%s" %(len(list_3)))
cnt=0
for file in list_2:
  if not(file in list_3):
    cnt+=1
    print("[%d]%s" %(cnt,file))
    #print(".././../../CASIA-Iris-Thousand/%s/%s/%s" %(file[2:5],file[5:6],file))#S5306R02.jpg
#print("一共少了%d(%d)个" %(cnt,len(list_2)-len(list_3)))