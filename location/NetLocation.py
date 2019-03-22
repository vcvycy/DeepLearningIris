from location.VGG import VGG
import tensorflow as tf
import sys
import  Utils
from location.DSV4Location import DSV4Location

def main():
    # 载入网络
    crop_h = 456
    crop_w = 576
    loc_net = VGG([None,crop_h , crop_w, 1], [8], [(1, 16), (1, 32),(1,32),(1,16)])
    # 载入数据
    sess = loc_net.sess
    json_file = r"E:\CASIA-V4-Location\Iris_Pupil_Position.json"   # 格式为 V4_ROOT/000/L/SXXX.jpg
    location_data_root = r"E:\CASIA-V4-Location\train"
    data = DSV4Location(sess, json_file, location_data_root)
    # 开始训练
    batch_size = 64
    learning_rate = 1e-4
    for epoch in range(0,5):
        if epoch == 4 :
            learning_rate = 1e-5
        print("[*] Epoch %d" %(epoch))
        left_image = len(data.images_path)
        for i in range(0,left_image,batch_size):
            batch = data.getBatch(batch_size)
            # Utils.drawAndShowBaseOnCNNOutput(batch[0][0],batch[1][0])
            loss = loc_net.train(batch[0],batch[1],learning_rate)
            print("     [*] loss=%s lr=%s" %(loss,learning_rate))
        loc_net.save("model/epoch-", epoch)
    return

if __name__ == "__main__":
    main()