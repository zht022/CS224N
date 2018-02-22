import numpy as np
import random
a = np.array([[2,4,6]])
b = np.array([[1,2,3],[2,4,5],[4,3,7]])
print(random.randint(0, 4))

idx = [0,2]
print(b)
print(b[idx])
c = np.sum(np.dot(b, a.T))+2
print(c.shape)
print(np.matmul(b, a.T))

a = np.array([1,2,3])
print(a.shape)
print(np.diag(a))
b = np.array([[1], [2]])
print(np.diag(b.reshape([2,])))

c = np.zeros([3,3])
print(c[0].shape)
c[1] += np.array([1,2,3])
print(c)

'''d = np.array([[2],[1],[3]])
print(c[0].shape)
d = d.reshape(c[0].shape)
print(d)
c[1] = d
print(c)

import os.path as op
a = op.splitext(op.basename("/media/deasd_ssdw_3.txt"))
print(a)
print(int(a[0].split("_")[2]))'''

'''import glob
for f in glob.glob("saved_params_*.npy"):  # 获取指定目录下的所有类似文件
    # basename: 返回路径最后一个'\'后面的名字
    # splitext: 分解文件名和扩展名，返回字符串(2段)
    # .split("_"): 以'_'分割，[2]表示第二段'_'后面的字符串；在这里就表示'*'的内容
    iter = int(op.splitext(op.basename(f))[0].split("_")[2])
    if (iter > st):
        st = iter
'''


print(max('ah', 'bf', key=lambda x: x[0]))

print(np.log10(np.logspace(-4, 2, num=10, base=10)))