"""
Osiris 的输出为： 提取出其中的瞳孔、虹膜坐标
================
Start processing
================

1 / 12
Process 001/L/S5001L04.jpg
[*]瞳孔中心和半径(103,104,37),Iris中心和半径(99,99,198)
2 / 12
Process 001/L/S5001L05.jpg
[*]瞳孔中心和半径(101,98,36),Iris中心和半径(97,97,194)
3 / 12
Process 001/L/S5001L06.jpg
[*]瞳孔中心和半径(96,91,27),Iris中心和半径(94,94,188)
4 / 12
Process 001/L/S5001L07.jpg
[*]瞳孔中心和半径(96,95,27),Iris中心和半径(93,93,186)
5 / 12
Process 001/L/S5001L08.jpg
[*]瞳孔中心和半径(96,91,25),Iris中心和半径(92,92,184)
6 / 12
Process 001/L/S5001L09.jpg
[*]瞳孔中心和半径(97,89,25),Iris中心和半径(92,92,184)
7 / 12
Process 001/R/S5001R00.jpg
[*]瞳孔中心和半径(97,94,31),Iris中心和半径(100,100,200)
8 / 12
Process 001/R/S5001R01.jpg
[*]瞳孔中心和半径(98,92,30),Iris中心和半径(99,99,198)
9 / 12
Process 001/R/S5001R02.jpg
[*]瞳孔中心和半径(95,88,35),Iris中心和半径(96,96,192)
10 / 12
Process 001/R/S5001R03.jpg
[*]瞳孔中心和半径(95,99,36),Iris中心和半径(98,98,196)
11 / 12
Process 001/R/S5001R04.jpg
[*]瞳孔中心和半径(95,97,36),Iris中心和半径(97,97,194)
12 / 12
Process 001/R/S5001R05.jpg
[*]瞳孔中心和半径(91,86,25),Iris中心和半径(91,91,182)

==============
End processing
==============


"""
import os
import json
def run(in_path,out_path):
    # 存储所有信息
    all_data = {}
    # 读取文件
    image_path = ""
    f = open(in_path,"rb")
    for line in f.readlines():
        line = str(line,encoding="utf-8")
        # 提取文件名
        if line.startswith("Process"):
            image_path = line.split(" ")[-1][:-2]
        # 提取坐标
        if line.startswith("[*]"):
            pupil = line.split("),")[0].split("(")[-1].split(",")
            iris  = line.split("),")[1].split("(")[-1].split(")")[0].split(",")
            pupil = [int(pupil[i]) for i in range(3)]
            iris  = [int(iris[i]) for i in range(3)]
            all_data[image_path] = {"iris":{"c":iris[:2],"r":iris[2]},"pupil":{"c":pupil[:2],"r":pupil[2]}}

    f.close()
    fout = open(out_path,"w")
    fout.write(json.dumps(all_data))
    fout.close()
    print("[*]图片个数:%s" %(len(all_data)))
    print("[*]Osiris 原始输出文件:%s\n" % (os.path.abspath(in_path)))
    print("[*]解析后的文件路径[json格式]: %s" %(os.path.abspath(out_path)))
    return

if __name__ == "__main__":
    in_file = "v4-position-raw.txt"
    out_file = in_file+".parsed.json"
    run(in_file,out_file)
