import math
check_code=[1,0,'X',9,8,7,6,5,4,3,2] # 余数对应的校验码
factor=[7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2] # 身份证号对应的前17位加权因子
print('Please input the first 17 numbers of your ID card:')
c_id=input() #身份证号前17位
sum=0
for i in range(len(c_id)):
    sum=sum+int(c_id[i])*factor[i] # 身份证号每一位与对应加权因子相乘并求和
print(sum)
remainder=divmod(sum,11)[1] # 所得和对11取余数
print("The last number is",check_code[remainder],'^_^')