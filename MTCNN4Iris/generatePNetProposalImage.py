# 产生所有Pnet生成的Proposal，供ONet训练
from MTCNN4Iris import PNetPredict
import Config
import os
import Utils
import cv2
import json
import tensorflow as tf
from ProcessOsirisSegmentedImage import Step3_GetV4LocationTrainingData
# 除了PNet的样例，再用拒绝采样法生成一些方框
def reject_sample(iris_rect_label,shape, pos_num=3, neg_num=3,pos_iou=0.7,neg_iou=0.3):
    rects = []
    h,w = shape
    for i in range(pos_num):
        cnt = 0
        while True:  # 拒绝采样法
            # rect = random_square_generator.generate()
            rect = Utils.getRandomLenSquare(h, w)
            iou = Utils.getIOU(rect, iris_rect_label)
            if iou > pos_iou:
                rects.append(rect)
                break
        # 获取负样本
    for _ in range(neg_num):
        while True:  # 拒绝采样法
            rect = Utils.getRandomLenSquare(h, w)
            iou = Utils.getIOU(rect, iris_rect_label)
            if iou < neg_iou:
                rects.append(rect)
                break
    return rects

if __name__ == "__main__":
    # Pnet模型地址和图片保存地址
    model_dir = "experiments/pnet_expand_0.08"
    config = Config.Config(os.path.join(model_dir, "config.json"))
    config.show()

    # 读取训练的图片，和要保存的文件夹
    src_image_dir = config.iris_images_dir                  # 原图片
    target_image_dir = r"E:\CASIA-V4-Location\PNetProposalWithRejectSample"  # proposal图片
    target_label_file = os.path.join(target_image_dir,"label.json")
    osiris_json_label = config.iris_position_json_file
    filename2path, filename2position = Step3_GetV4LocationTrainingData.main_location(osiris_json_label, src_image_dir, show=False)
    target_image_shape = (48,48)
    pnet_threshold = 0.25
    pos_iou = 0.7        # 大于0.7的是正例
    neg_iou = 0.3        # 小于0.3的是反例，中间的不训练
    #
    #
    sess = tf.Session()
    pnet_predictor = PNetPredict.PNetPredictor(sess, config, model_dir)
    #
    pnet_recall = 0             # recall
    target_label = {}
    pos_image_num =0            # 正例图片数
    neg_image_num =0            # 反例图片数
    for filename in filename2path:
        path = filename2path[filename]
        pos  = filename2position[filename]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        rects = pnet_predictor.predict(img,threshold = pnet_threshold, nms_threshold=0.7)
        iris_rect_label = Utils.getIrisRectFromPosition(pos, img.shape)
        pupil_rect_label = Utils.getPupilRectFromPosition(pos, img.shape)
        # 记录
        recall_cur_image = False
        # 拒绝采样法再加一些图片
        rects += reject_sample(iris_rect_label,img.shape,pos_num=3,neg_num=3)
        for rect in rects:
            iou = Utils.getIOU(iris_rect_label, rect)
            if iou > pos_iou:
                bbr = Utils.getBBR(rect, iris_rect_label)
                # 要保存的图片名字
                save_name =  "pos_%d_%s" %(pos_image_num, filename)
                save_path = os.path.join(target_image_dir,save_name)
                region = Utils.cropAndResize(img, rect, target_image_shape[0])
                cv2.imwrite(save_path,region)

                # 瞳孔位置相对于虹膜rect 的比例
                pupil_percent_in_region = (
                    (pupil_rect_label[0]-rect[0]) / (rect[2]-rect[0]),
                    (pupil_rect_label[1]-rect[1]) / (rect[3]-rect[1]),
                    (pupil_rect_label[2]-rect[0]) / (rect[2]-rect[0]),
                    (pupil_rect_label[3]-rect[1]) / (rect[3]-rect[1])
                )
                target_label[save_name]= {"prob":1, "bbr": bbr , "pupil_rect": pupil_percent_in_region}
                pos_image_num += 1
                recall_cur_image = True
            elif iou < neg_iou:
                save_name = "neg_%d_%s" %(neg_image_num, filename)
                save_path = os.path.join(target_image_dir, save_name)
                region = Utils.cropAndResize(img, rect, target_image_shape[0])
                cv2.imwrite(save_path,region)
                target_label[save_name]= {"prob":0}
                neg_image_num += 1
        if recall_cur_image:
            pnet_recall += 1
        print(filename)
    # label 写入文件中
    target_label_f = open(target_label_file,"w")
    target_label_f.write(json.dumps(target_label))
    target_label_f.close()
    print("[*]PNet Recall: %s/%s  Threshold=%s" %(pnet_recall, len(filename2path),pnet_threshold))