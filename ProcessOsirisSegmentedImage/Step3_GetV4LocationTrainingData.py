# 载入虹膜定位信息，载入某个目录中所有图片
# 二者取交，然后返回
import json
import os
import cv2

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

# 在图中画虹膜、瞳孔位置，然后显示出来
def draw_circle(path, postion):
    iris = postion["iris"]
    pupil = postion["pupil"]
    img = cv2.imread(path)
    cv2.circle(img, tuple(iris["c"]), iris["r"], (255, 0, 255),thickness=3)
    cv2.circle(img, tuple(pupil["c"]),pupil["r"], (255, 0, 255),thickness=3)
    return img

# 读取Json位置信息和图片根目录
# 将其保存到save_dir中
def main_location(json_file,image_root,save_dir=None,show=False):
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
    for filename in files_iou:
        ## 以下画圆，保存或者显示
        path = filename2path[filename]
        position = filename2position[filename]
        if save_dir!= None or show:
            img = draw_circle(path,position)
        if save_dir != None:
            save_path = os.path.join(save_dir,filename)
            cv2.imwrite(save_path,img)
            print(save_path)
        if show:
            print(filename)
            print(position)
            cv2.namedWindow("cjf")
            cv2.imshow("cjf",img)
            cv2.waitKey(0)
    return filename2path,filename2position

if __name__ == "__main__":
    json_file = r"E:\CASIA-V4-Location\Iris_Pupil_Position.json"   # 格式为 V4_ROOT/000/L/SXXX.jpg
    location_data_root = r"E:\CASIA-V4-Location"
    save_dir = None  # 为None时，不保存，仅仅显示出来
    # save_dir = r"E:\CASIA-V4-Location-View"
    f2p,f2pos= main_location(json_file,location_data_root,None,False)
    print(len(f2p))