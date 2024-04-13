# 综合

"""
编程实现：百钱买百鸡
公鸡5文钱一只，母鸡3文钱一只，小鸡3只一文钱，
用100文钱买一百只鸡,其中公鸡，母鸡，小鸡都必须要有
请打印输出全部方案
"""

money = 100
# ********** Begin *********#
male = -1


for i in range(101):
    male += 1
    female = -1
    for ii in range(101):
        female += 1
        chick = -1 
        for iii in range(101):
            chick += 1
            if male * 15 + female * 9 + chick * 1 == 300 and male + female + chick == 100 and male * female * chick != 0:
                print(f'{male} {female} {chick}')
# ********** End **********#
