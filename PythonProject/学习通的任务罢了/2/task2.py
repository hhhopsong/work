'''
info = input('^*_*^输入你的身份证：')
year = info[6:10]
month = info[10:12]
day = info[12:14]
list1 = (year, month, day)
list1 = ','.join(list1)
list2 = [[list1][0].replace("'","")]
print(list2)
'''
info = input('^*_*^输入你的身份证：')
year = (eval(info[6:14])) // 10000
month = ((eval(info[6:14]))-year * 10000) // 100
day = eval(info[6:14])-year * 10000-month * 100
list1 = [year,month,day]
print(list1)
