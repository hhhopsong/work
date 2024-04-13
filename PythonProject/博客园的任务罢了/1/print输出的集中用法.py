
# print输出的集中用法

# 用法一：用于输出单个字符串或单个变量
print('hey,u')

# 用法2： 用于输出多个数据项，用逗号分开
print('hey', ' u')
x = 1
y = 2
z = 3
print(x, y, z)

# 用法3：用户混合字符串和变量值
print('x = %d,y = %d,z = %d' % (x, y, z))       # 方法1
print('x = {},y = {},z = {}'.format(x, y, z))   # 方法2
print(f'x = {x},y = {y},z = {z}')               # 方法3

# 其他:默认分行与不分行
print(x)
print(y)
print(z)

print(x, end=' ')
print(y, end=' ')
print(z, end=' ')
