import math
check_code=[1,0,'X',9,8,7,6,5,4,3,2] # 余数对应的校验码
factor=[7,9,10,5,8,4,2,1,6,3,7,9,10,5,8,4,2] # 身份证号对应的前17位加权因子
print('Please input the first 14 numbers of your ID card:')
c_id=input() #身份证号前14位
end = 2
maybe = []
for i in range(1000):
    sum=0
    if i%2 == 0:
        continue
    iii = f'{str(i):0>3}'
    for i in range(len(c_id)):
        sum=sum+int(c_id[i]+iii)*factor[i] # 身份证号每一位与对应加权因子相乘并求和
    remainder=divmod(sum,11)[1] # 所得和对11取余数
    if remainder == end:
        maybe.append(f'{c_id}{iii}{check_code[remainder]}')
print("The maybe number is")
for i in maybe:
    print(i)