# 从一个h*w 的矩阵中，等概率生成随机长度至少为min_len的子正方形
# 等概率生成子正方形，由于边长小的子正方形多，所以生成的正方形，边长越小，概率越大
import random
class RandomSquare:
    def __init__(self,h,w,min_len =1):
        self.h = h
        self.w = w
        # 使得 h <= w，生成子正方形后，再tranpose回去
        if h > w:
            self.transpose = True
            self.h,self.w = w,h
        else:
            self.transpose = False
        assert  self.h >= min_len
        # 边长不同的子正方形个数
        self.total_num = [0 for _ in range(h+1)]
        self.square_num_with_len_k = [0 for _ in range(h+1)]  #边长为0没有子正方形
        for l in range(min_len,h+1):
            num = (h-l+1) * (w-l+1)
            self.square_num_with_len_k[l] = num
            self.total_num[l] = self.total_num[l-1] + num
        # print(self.square_num_with_len_k)
        # print(self.total_num)
        return

    # 生成随机子正方形
    def generate(self):
        idx = random.randint(1,self.total_num[-1])
        # 二分边长, 看看idx落在哪个区间内,找出self.total_num 中，第一个>=idx的数
        l = 1
        r = self.h
        while l<r :
            mid = (l+r)//2
            if self.total_num[mid] >= idx:
                r = mid
            else:
                l = mid+1
        square_length = r
        # 生成左上角坐标
        x = random.randint(0, self.h - square_length)
        y = random.randint(0, self.w - square_length)
        if self.transpose:
            x,y = y,x
        return x,y, x+square_length-1 ,y+square_length-1
if __name__ == "__main__":
    a = RandomSquare(2,3,1)
    for _ in range(10):
        print(a.generate())