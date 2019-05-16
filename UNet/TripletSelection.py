import Utils
import numpy as np
import cv2
import os
import json
import random
import sys
class TripletSelection:
    def __init__(self, image_dir):
        self.file2path = Utils.getFile2Path(image_dir)
        self.label2path={}
        self.classIdx=[]
        for file in self.file2path:
            label = self.__getLabelFromFilename(file)
            path  = self.file2path[file]
            if label not in self.label2path:
                self.label2path[label]=[]
                self.classIdx.append(label)
            self.label2path[label].append(path)
        #print(self.classIdx)
        self.show()
        return

    def show(self):
        self.total = 0
        for label in self.label2path:
            self.total += len(self.label2path[label])
        print("[*]总的数据集大小;%s" %(self.total))
        return

    def getRandomPermutation(self,n):
        x = np.arange(n)
        np.random.shuffle(x)
        return x

    def getTripletName(self, classes_each_batch, images_each_class):
        batch_size = classes_each_batch * images_each_class
        class_indices = self.getRandomPermutation(len(self.classIdx))
        images_path = []
        i = 0
        while len(images_path) < batch_size:
            class_idx = self.classIdx[class_indices[i]]
            # print(class_idx)
            nrof_images_in_class = len(self.label2path[class_idx])
            image_indices = self.getRandomPermutation(nrof_images_in_class)
            nrof_images_from_class = min(nrof_images_in_class, images_each_class, batch_size - len(images_path))
            idx = image_indices[0:nrof_images_from_class]
            image_paths_for_class = [self.label2path[class_idx][j] for j in idx]
            images_path += image_paths_for_class
            i+=1
        return images_path

    # 读取+图片增强
    def readImageAndAug(self,path,input_size):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 先缩放到 size* size 再在高和宽各减去self.crop_pixels个像素
        crop_pad = int(input_size*0.05)
        size = input_size +crop_pad
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
        img = np.reshape(img, (size, size, 1))
        #
        crop_start_h = random.randint(0,crop_pad-1)
        crop_start_w = random.randint(0,crop_pad-1)
        img = img[crop_start_h: crop_start_h + input_size, crop_start_w: input_size + crop_start_w]
        return img

    def getBatch(self, input_size, classes_each_batch=16, images_each_class = 5):
        paths = self.getTripletName(classes_each_batch, images_each_class)
        print(paths)
        batch = [[],[],[]]
        for p in paths:
            img = self.readImageAndAug(p, input_size)
            label = self.__getLabelFromFilename(os.path.basename(p))
            batch[0].append(img)
            batch[1].append(label)
        return batch
        # 读取+图片增强

    def readImageAndAugNor(self, path, height,width):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        # 先缩放到 size* size 再在高和宽各减去self.crop_pixels个像素
        #print(img.shape)
        img=cv2.equalizeHist(img)
        #Utils.showImage(img)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        #print(img.shape)
        img = np.reshape(img, (height, width, 1))
        #Utils.showImage(img)
        return img

    def getBatchNor(self, height,width, classes_each_batch=16, images_each_class = 5):
        paths = self.getTripletName(classes_each_batch, images_each_class)
        batch = [[],[],[]]
        for p in paths:
            img = self.readImageAndAugNor(p, height,width)
            label = self.__getLabelFromFilename(os.path.basename(p))
            batch[0].append(img)
            batch[1].append(label)
        return batch

    def __getLabelFromFilename(self,filename):
        eye_lr = filename[5]
        idx = int(filename[2:5])
        label = int(idx)*2
        if eye_lr == "R":
            label += 1
        return label

if __name__ == "__main__":
    dir =r"E:/IrisNormalizedImage"
    ds = TripletSelection(dir, 1900)
    ds.getBatchNor(64,512)