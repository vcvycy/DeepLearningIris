# 输入：iris/pupil 虹膜信息，
# 输出：将Iris的半径扩大10%
import json
import Utils
import cv2
import os
def showPosition():
    return
if __name__ == "__main__":
    expand = 0.08  # 半径扩大10%
    in_file  = "v4-position-raw.txt.parsed.json"
    out_file = "pupil_iris_location_expand_%s.json" %(expand)
    image_dir = r"E:\CASIA-V4-Location"
    ########
    in_f = open(in_file)
    param = json.load(in_f)
    for file in param:
        param[file]["iris"]["r"] = int(param[file]["iris"]["r"] * (1+expand))

    out_f = open(out_file, "w")
    out_f.write(json.dumps(param))
    ### 显示图片
    file2path = Utils.getFile2Path(image_dir)
    for file in param:
        f = os.path.basename(file)
        img = cv2.imread(file2path[f], cv2.IMREAD_GRAYSCALE)
        Utils.drawIrisAndShow(img,param[file])