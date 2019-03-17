import Config
import tensorflow as tf
from MTCNN4Iris.MTCNN import *

class IrisONet(ONet):
    def dd(self):
        print("yes")

if __name__ == "__main__":
    onet = IrisONet()