import numpy as np
import matplotlib.pyplot as plt
import imgaug as ia
import cv2
import os

# 枚举root,返回所有jpg文件，文件名->路径的映射的dict
def getFile2Path(root,suffix="jpg"):
    filename2path = {}
    for r,dirs,files in os.walk(root):
        for file in files:
            if file.split(".")[-1] == suffix:
                filename2path[file] = os.path.join(r,file)
    return filename2path

def showImage(mat):
    cv2.namedWindow("show")
    cv2.imshow("show",mat)
    cv2.waitKey(0)
    #ia.imshow(mat)
    return