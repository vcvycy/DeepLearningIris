def decorator(fun):
    def run(self,*args,**kwargs):
        print("执行函数:{0}\n 参数:{1}".format(fun.__name__,args))
        return fun(*args,**kwargs)
