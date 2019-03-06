# 生成数据集的文件名列表(用于给Osiris执行segmentation操作)
import os
def output_v4(filepath):
    id = filepath[2:5]
    left_right_eye = filepath[5:6]
    print("%s/%s/%s" %(id,left_right_eye,filepath))
    return
def search(root,cb_fun,suffix="jpg"):
    for root,dirs,files in os.walk(root):
        for file in files:
            if file.split(".")[-1] == suffix:
                cb_fun(file)
def GenerateV4FileList(root):
    search(dir,output_v4,"jpg")
    return
if __name__ == "__main__":
    dir = r"E:\虹膜数据集\CASIA-Iris-Thousand"
    GenerateV4FileList(dir)