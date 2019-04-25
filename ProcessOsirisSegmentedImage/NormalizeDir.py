# import Normalization
import os
import json
import sys
import Utils
import cv2
import numpy as np
from ProcessOsirisSegmentedImage import Normalization
def getFilenamesFromDir(dir, accept_suffix):  # 文件名list
    suffix = set(accept_suffix.split("|"))
    filename_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if file.strip().split(".")[-1] in suffix:
                path = os.path.join(root, file)
                filename_list.append(path)
    return filename_list
# 枚举root,返回所有jpg文件，文件名->路径的映射的dict
def getFileSet(root,suffix="jpg"):
    filename2path = {}
    for r,dirs,files in os.walk(root):
        for file in files:
            if file.split(".")[-1] == suffix:
                filename2path[file] = os.path.join(r,file)
    return filename2path

def load_iris_position(in_file):
    print("[*] 载入%s 文件(保存文件对应的虹膜/瞳孔位置)" %(in_file))
    f = open(in_file,"r")
    json_obj = json.loads(f.read())
    f.close()
    return json_obj
def main(image_root,json_file,save_dir=None,debug=True):
    # 读取所有文件列表
    filename2path =getFileSet(image_root)
    print("[*] 目录%s文件个数:%d" %(image_root,len(filename2path)))

    # 读取虹膜位置信息
    relpath2position = load_iris_position(json_file)
    print("[*] json文件中的图片个数:%d" %(len(relpath2position)))
    # 将iris_position 中 000/L/S555XXX.jpg 格式转为 S55XXX.jpg格式
    filename2position = {}
    for x in relpath2position:
        filename2position[os.path.basename(x)]=relpath2position[x]

    # 取二者的交集
    files_iou = set()
    for filename in filename2path:
        if filename in filename2position:
            files_iou.add(filename)
    print("     [*]json存在的图片和目录中的图片取交集后的大小:%d" %(len(files_iou)))
    # 只保留filename2path, filename2position 相交的部分
    del_files = set()
    for filename in filename2path:
        if filename not in files_iou:
            del_files.add(filename)
    for f in del_files:
        filename2path.pop(f)
    del_files = set()
    for filename in filename2position:
        if filename not in filename2position:
            del_files.add(filename)
    for f in del_files:
        filename2position.pop(f)
    # 上面二者融合
    idx=0
    for filename in files_iou:
        idx+=1
        try:
            ## 以下画圆，保存或者显示
            path = filename2path[filename]
            position = filename2position[filename]
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            nor_img = Normalization.normalize(img,position["iris"]["c"][0],
                                              position["iris"]["c"][1],
                                              position["iris"]["r"],
                                              position["pupil"]["c"][0],
                                              position["pupil"]["c"][1],
                                              position["pupil"]["r"],
                                              )
            if debug:
                Utils.showImage(img)
                Utils.showImage(nor_img)
            else:
                filename=os.path.basename(path)
                save_path = os.path.join(save_dir, filename)
                print("[%d/%d] %s" %(idx,len(files_iou),save_path))
                # Utils.showImage(nor_img)
                if not cv2.imwrite(save_path,nor_img):
                    print("Failed")
                    input("")
        except:
            print("[!]%s Failed" %(filename))
    return filename2path,filename2position
if __name__ == "__main__":
    json_file=r"E:\CASIA-V4-Location\Iris_Pupil_Position.json"
    src_dir=r"E:\CASIA-V4-Location\train"
    target_dir=r"E:\IrisNormalizedImage"
    main(src_dir,json_file,target_dir,debug=False)