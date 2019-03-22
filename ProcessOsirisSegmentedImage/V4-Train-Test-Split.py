# 手工将错误的定位放到 文件夹 E:\CASIA-V4-Location-Wrong 中
# 现在需要根据在 E:\CASIA-V4-Location-Wrong 出现的文件[名] ，将E:\CASIA-V4-Location train中对应文件移到 test中
import Utils
import shutil
import os
if __name__ == "__main__":
    train_dir = r"E:\CASIA-V4-Location\train"
    test_dir  = r"E:\CASIA-V4-Location\test"
    wrong_labeled_dir = r"E:\CASIA-V4-Location-Wrong"
    # 所有错误定位的图片
    wrong_files = Utils.getFile2Path(wrong_labeled_dir)
    print("[*] 要移动的图片数: %d" %(len(wrong_files)))
    #
    print("[*] 正在处理....")
    for file in wrong_files:
        path_in_train = os.path.join(train_dir,file)
        if os.path.exists(path_in_train):
            shutil.move(path_in_train,test_dir)
        else:
            print("No! %s" %(file))