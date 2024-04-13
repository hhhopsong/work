import datetime

t1 = eval(datetime.datetime.now().strftime("%S"))
while eval(datetime.datetime.now().strftime('%S')) + 90 < t1:
    print(f'还有{90 - (eval(datetime.datetime.now().strftime("%S")) - t1)}s发言时间')

