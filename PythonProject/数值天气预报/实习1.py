from pylab import *
from grads import GrADS #导入库
ga=GrADS() #使用GrADS客户端程序，要求GrADS版本大于2.0.0
fh = ga.open("http://monsoondata.org:9090/dods/model") #读取文件
ga("display ps")   #显示图形

