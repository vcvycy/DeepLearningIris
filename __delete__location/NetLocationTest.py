from location.VGG import VGG
import Utils
import numpy as np
import tensorflow as tf
import sys
from location.DSV4Location import DSV4Location
import cv2
import DSIrisAug

def main():
    # Restore 网络
    crop_h = 456
    crop_w = 576
    loc_net = VGG([None,crop_h , crop_w, 1], [8], [(1, 16), (1, 32),(1,32),(1,16)])
    loc_net.restore("./model")
    # 枚举
    test_image_dir = r"E:\CASIA-V4-Location\train"
    #test_image_dir = r"E:\CASIA-V4-Location-Wrong"
    file2path = Utils.getFile2Path(test_image_dir)
    print("[*]文件数:%s" %(len(file2path)))
    for file in file2path:
        print(file)
        img_origin = cv2.imread(file2path[file],cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img_origin, (crop_w,crop_h), interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (crop_h, crop_w, 1))
        output = loc_net.forward([img])
        print(output)
        Utils.drawAndShowBaseOnCNNOutput(img_origin, output[0])
    return

if __name__ == "__main__":
    main()