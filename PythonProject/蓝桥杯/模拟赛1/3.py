import datetime

start = datetime.date(1900, 1, 1)
end = datetime.date(9999, 12, 31)
D = datetime.timedelta(days=1)
res = 0
while start <= end:
    y = sum([eval(i) for i in str(start.year)])
    m = sum([eval(i) for i in str(start.month)])
    d = sum([eval(i) for i in str(start.day)])
    if y == m + d:
        res += 1
    try:
        start += D
    except:
        break
    print(start)
print(res)