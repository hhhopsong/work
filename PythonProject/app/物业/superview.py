from tkinter import *
from tkinter.messagebox import *
from windnd import *


super_name = '宋风杰'
super_pwd = '123456'

class InputFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 创建Frame
        self.page.pack()
        hook_dropfiles(self, func=self.dragged_address)
        Label(self, text='将文件拖入下方浅绿色框中').grid(row=4, pady=10)  # xlsx、xls文件识别
        self.listbox = Listbox(self, bg='lightgreen', height=25, width=130, selectmode='SINGLE')
        self.listbox.grid(row=5)

    def dragged_address(self, files):
        msg = (i.decode('gbk') for i in files)
        tk = Toplevel()
        tk.geometry('300x130')
        tk.title('验证信息')
        tk.resizable(False, False)
        self.actionName = StringVar()
        self.password = StringVar()
        Label(tk, text='操作人账号: ').grid(row=2, pady=10)
        Entry(tk, textvariable=self.actionName).grid(row=2, column=1)
        Label(tk, text='操作人密码: ').grid(row=3, pady=10)
        Entry(tk, textvariable=self.password).grid(row=3, column=1)
        Button(tk, text='录入', bg='skyblue', width=20, command=self.impot_file).grid(row=4, column=1, pady=10)

    def impot_file(self):
        import datetime
        self.Name = self.actionName.get()
        self.pwd = self.password.get()
        self.user = []
        self.passwords = []
        with open('data/codes.csv', 'r', encoding='utf-8') as users:  # 需加密解密
            for i in users.readlines():
                self.user.append(i.strip().split(',')[0])
                self.passwords.append(i.strip().split(',')[1])
        if self.Name == super_name and self.pwd == super_pwd:  # 底层绝对管理员


        elif name in self.user:

            if secret == passwords[No]:

            else:
                showinfo(title='错误', message='账号不存在或密码错误！')
        else:
            showinfo(title='错误', message='账号不存在或密码错误！')



class SuperQueryFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.address = StringVar()
        self.name = StringVar()
        self.counter = StringVar()
        self.yr = StringVar()
        self.mn = StringVar()
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 创建Frame
        self.page.pack()
        Label(self, text='住户地址: ').grid(row=2, stick=W, pady=10)
        Entry(self, textvariable=self.address).grid(row=2, column=1, stick=E)
        Label(self, text='住户姓名: ').grid(row=3, stick=W, pady=10)
        Entry(self, textvariable=self.name).grid(row=3, column=1, stick=E)
        Label(self, text='收款人: ').grid(row=4, stick=W, pady=10)
        Entry(self, textvariable=self.counter).grid(row=4, column=1, stick=E)
        Label(self, text='年份: ').grid(row=5, stick=W, pady=10)
        Entry(self, textvariable=self.yr).grid(row=5, column=1, stick=E)
        Label(self, text='月份: ').grid(row=6, stick=W, pady=10)
        Entry(self, textvariable=self.mn).grid(row=6, column=1, stick=E)
        Button(self, text='查询', command=self.SearchCheck, bg='skyblue').grid(row=7, column=1, pady=10)

    def SearchCheck(self):
        add_q = self.address.get()
        name_q = self.name.get()
        counter_q = self.counter.get()
        year_q = self.yr.get()
        month_q = self.mn.get()
        userno = []
        times = []
        years = []
        months = []
        adds = []
        moneys = []
        ways = []
        counters = []
        var1s = []
        var2s = []
        var3s = []
        i = 0
        while True:
            try:
                with open(f'data1/物业统计表{i}.csv', 'r+', encoding='utf-8') as users:  # 需加密解密
                    datas = users.readlines()[1:]
                user = [i.strip().split(',')[6] for i in datas]
                year = [i.strip().split(',')[2] for i in datas]
                month = [i.strip().split(',')[4] for i in datas]
                add = [i.strip().split(',')[17] for i in datas]
                con = [i.strip().split(',')[18] for i in datas]
                way = [i.strip().split(',')[7] for i in datas]
                var1 = [i.strip().split(',')[19] for i in datas]
                var2 = [i.strip().split(',')[20] for i in datas]
                var3 = [i.strip().split(',')[21] for i in datas]
                money = [i.strip().split(',')[22] for i in datas]
                for ii in range(len(user)):
                    times.append(f'{year[ii]:*<4}{"-"}{month[ii]:0>2}')
                    years.append(year[ii])
                    months.append(month[ii])
                    adds.append(add[ii])
                    moneys.append(money[ii])
                    counters.append(con[ii])
                    ways.append(way[ii])
                    userno.append(user[ii])
                    var1s.append(var1[ii])
                    var2s.append(var2[ii])
                    var3s.append(var3[ii])
                i += 1
            except:
                break
        if add_q == '' and name_q in userno and counter_q == '' and year_q == '' and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if name_q == userno[i]:
                    user.append(name_q)
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q in adds and name_q == '' and counter_q == '' and year_q == '' and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if add_q == adds[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q == '' and name_q == '' and counter_q in counters and counter_q != '' and year_q == '' and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q in adds and name_q in userno and counter_q == '' and year_q == '' and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if add_q == adds[i] and name_q == userno[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q in adds and name_q == '' and counter_q in counters and counter_q != '' and year_q == '' and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if add_q == adds[i] and counter_q == counters[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q == '' and name_q in userno and counter_q in counters and counter_q != '' and year_q == '' and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if name_q == userno[i] and counter_q == counters[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q == '' and name_q == '' and counter_q == '' and year_q == '' and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q in adds and name_q == '' and counter_q in counters and year_q in years and month_q in months and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i] and add_q == adds[i] and year_q == years[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q == '' and name_q in userno and counter_q in counters and year_q in years and month_q in months and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i] and name_q == userno[i] and year_q == years[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q in adds and name_q in userno and counter_q == '' and year_q in years and month_q in months:
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if add_q == adds[i] and name_q == userno[i] and year_q == years[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q in adds and name_q in userno and counter_q in counters and year_q in years and month_q == '' and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i] and name_q == userno[i] and year_q == years[i] and add_q == adds[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q in adds and name_q in userno and counter_q in counters and year_q == '' and month_q in months and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i] and name_q == userno[i] and add_q == adds[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q == '' and name_q == '' and counter_q in counters and year_q in years and month_q in months and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i] and year_q == years[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q in adds and name_q == '' and counter_q == '' and year_q in years and month_q in months:
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if add_q == adds[i] and year_q == years[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q == '' and name_q in userno and counter_q == '' and year_q in years and month_q in months:
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if name_q == userno[i] and year_q == years[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q in adds and name_q in userno and counter_q == '' and year_q in years and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if add_q == adds[i] and name_q == userno[i] and year_q == years[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q in adds and name_q == '' and counter_q in counters and year_q in years and month_q == '' and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if add_q == adds[i] and counter_q == counters[i] and year_q == years[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q == '' and name_q in userno and counter_q in counters and year_q in years and month_q == '' and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if name_q == userno[i] and counter_q == counters[i] and year_q == years[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q in adds and name_q in userno and counter_q == '' and year_q == '' and month_q in months:
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if add_q == adds[i] and name_q == userno[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q in adds and name_q == '' and counter_q in counters and year_q == '' and month_q in months and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if add_q == adds[i] and counter_q == counters[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q == '' and name_q in userno and counter_q in counters and year_q == '' and month_q in months and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if name_q == userno[i] and counter_q == counters[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q == '' and name_q == '' and counter_q == '' and year_q in years and month_q in months:
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if month_q == months[i] and year_q == years[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q in adds and name_q == '' and counter_q == '' and year_q in years and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if add_q == adds[i] and year_q == years[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q == '' and name_q in userno and counter_q == '' and year_q in years and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if name_q == userno[i] and year_q == years[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q == '' and name_q == '' and counter_q in counters and year_q in years and month_q == '' and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i] and year_q == years[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q in adds and name_q == '' and counter_q == '' and year_q == '' and month_q in months:
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if add_q == adds[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q == '' and name_q in userno and counter_q == '' and year_q == '' and month_q in months:
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if name_q == userno[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q == '' and name_q == '' and counter_q in counters and year_q == '' and month_q in months and counter_q != '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            for i in range(len(userno)):
                if counter_q == counters[i] and month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
            get_pay = sum(money)
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, get_pay)
        elif add_q == '' and name_q == '' and counter_q == '' and year_q in years and month_q == '':
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if year_q == years[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3, unpay)
        elif add_q == '' and name_q == '' and counter_q == '' and year_q == '' and month_q in months:
            user = []
            money = []
            time = []
            add = []
            counter = []
            way = []
            var1 = []
            var2 = []
            var3 = []
            No_index = []  # 未缴费时次索引
            for i in range(len(userno)):
                if month_q == months[i]:
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
                    counter.append(counters[i])
                    way.append(ways[i])
                    var1.append(var1s[i])
                    var2.append(var2s[i])
                    var3.append(var3s[i])
                    if counters[i] == '':
                        No_index.append(i)
            unpay = 0
            for i in No_index:
                unpay += eval(moneys[i])
            # moneyframe
            self.moneypage = SuperMoneyFrame(self.root, user, add, time, money, counter, way, var1, var2, var3,
                                             unpay)  #
        else:
            showinfo(title='错误', message='查询信息错误或不支持此类查询！')


class SuperMoneyFrame(Frame):
    def __init__(self, master=None, nam=None, add=None, time=None, money=None, counter=None, way=None, var1=None,
                 var2=None, var3=None, unpay=0):
        Frame.__init__(self, master)
        self.root = Toplevel()  # 定义内部变量root
        self.root.resizable(False, False)
        self.root.title('查询结果')
        self.name = nam
        self.add = add
        self.time = time
        self.money = [f'{i:.2f}' for i in money]
        self.counter = counter
        self.way = way
        self.var1 = var1
        self.var2 = var2
        self.var3 = var3
        self.unpay = unpay
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 创建Frame
        self.page.pack()
        ###
        s = Scrollbar(self.page)  # 创建滚动条
        s.pack(side=RIGHT, fill=Y)  # 设置垂直滚动条显示的位置，使得滚动条，靠右侧；通过 fill 沿着 Y 轴填充
        listbox = Listbox(self.page, width=125, height=31, yscrollcommand=s.set)  # 为Listbox控件添加滚动条
        data = [
            f"{'住户姓名'}|{'地址':{chr(12288)}^26}|{'应缴时间':{chr(12288)}^6}|{'应缴金额':{chr(12288)}^7}|{'收款人员'}|{'支付方式'}|{'实耗①':{chr(12288)}^5}|{'实耗②':{chr(12288)}^5}|{'实耗③':{chr(12288)}^5}"]
        for i in range(len(self.add)):
            data.append(
                f'{self.name[i]:{chr(12288)}<4}|{self.add[i]:^74}|{self.time[i]:^13}|￥{self.money[i]:^15}|{self.counter[i]:{chr(12288)}^4}|{self.way[i]:{chr(12288)}>4}|{self.var1[i]:>15}|{self.var2[i]:>15}|{self.var3[i]:>15}')
        for i in range(len(data)):
            item = data[i]
            listbox.insert(i, item)
        listbox.pack()
        s.config(command=listbox.yview)  # 设置滚动条，使用 yview使其在垂直方向上滚动 Listbox 组件的内容
        ###
        Label(self.page, text=f'已找到{len(self.money)}个对象\t\t已缴/未缴 合计金额:{self.unpay:.2f}元').pack(anchor='e')
        self.page.mainloop()


class Action_recordsFrame(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.createPage()

    def createPage(self):
        self.page = Frame(self)
        self.page.pack()
        ###
        s = Scrollbar(self.page)  # 创建滚动条
        s.pack(side=RIGHT, fill=Y)  # 设置垂直滚动条显示的位置，使得滚动条，靠右侧；通过 fill 沿着 Y 轴填充
        self.listbox = Listbox(self.page, width=125, height=27, yscrollcommand=s.set,
                               selectmode='SINGLE')  # 为Listbox控件添加滚动条
        data = [f"|{'用户姓名'}|{'用户类型'}|{'最后登录时间':{chr(12288)}^12}|"]
        with open('data/codes.csv', 'r', encoding='utf-8') as user_data:  # 加密
            self.user_name = [i.strip().split(',')[0] for i in user_data.readlines()]
        with open('data/login_time.csv', 'r', encoding='utf-8') as user_data1:  # 加密
            self.user_time = [i.strip() for i in user_data1.readlines()]
        for i in self.user_name:
            lastest_time = []
            for ii in self.user_time:
                if i in ii:
                    lastest_time.append(ii.split(',')[1])
            try:
                if lastest_time:
                    data.append(f"|{i:{chr(12288)}<4}|普通用户|{max(lastest_time):^25}|")
                elif not lastest_time:
                    data.append(f"|{i:{chr(12288)}<4}|普通用户|{'':^25}|")
            except:
                data.append(f"|{'':{chr(12288)}<4}|{'':{chr(12288)}<4}|{'':^19}|")
        for i in range(len(data)):
            item = data[i]
            self.listbox.insert(i, item)
        self.listbox.pack()
        s.config(command=self.listbox.yview)  # 设置滚动条，使用 yview使其在垂直方向上滚动 Listbox 组件的内容


class Unit_priceFrame(Frame):
    pass


class MenuFrame(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.createPage()

    def createPage(self):
        self.page = Frame(self)
        self.page.pack()
        Button(self.page, text='删除', bg='red', command=self.p).pack(fill='both')
        ###
        s = Scrollbar(self.page)  # 创建滚动条
        s.pack(side=RIGHT, fill=Y)  # 设置垂直滚动条显示的位置，使得滚动条，靠右侧；通过 fill 沿着 Y 轴填充
        self.listbox = Listbox(self.page, width=125, height=27, yscrollcommand=s.set,
                               selectmode='SINGLE')  # 为Listbox控件添加滚动条
        data = [f"|{'用户姓名'}|{'用户类型'}|{'最后登录时间':{chr(12288)}^12}|"]
        with open('data/codes.csv', 'r', encoding='utf-8') as user_data:  # 加密
            self.user_name = [i.strip().split(',')[0] for i in user_data.readlines()]
        with open('data/login_time.csv', 'r', encoding='utf-8') as user_data1:  # 加密
            self.user_time = [i.strip() for i in user_data1.readlines()]
        for i in self.user_name:
            lastest_time = []
            for ii in self.user_time:
                if i in ii:
                    lastest_time.append(ii.split(',')[1])
            try:
                if lastest_time:
                    data.append(f"|{i:{chr(12288)}<4}|普通用户|{max(lastest_time):^25}|")
                elif not lastest_time:
                    data.append(f"|{i:{chr(12288)}<4}|普通用户|{'':^25}|")
            except:
                data.append(f"|{'':{chr(12288)}<4}|{'':{chr(12288)}<4}|{'':^19}|")
        for i in range(len(data)):
            item = data[i]
            self.listbox.insert(i, item)
        self.listbox.pack()
        s.config(command=self.listbox.yview)  # 设置滚动条，使用 yview使其在垂直方向上滚动 Listbox 组件的内容

    def p(self):
        value = ''
        try:
            for i in self.listbox.selection_get():
                value += i
            if value.split('|')[1].strip(f'{chr(12288)}') != '用户姓名':
                with open('data/codes.csv', 'r', encoding='utf-8') as user_data:  # 加密
                    self.user_old = user_data.readlines()
                self.user_old_name = [i.strip().split(',')[0] for i in self.user_old]
                self.No = self.user_old_name.index(value.split('|')[1].strip(f'{chr(12288)}'))
                del self.user_old[self.No]
                with open('data/codes.csv', 'w', encoding='utf-8') as user_data1:  # 加密
                    for i in self.user_old:
                        user_data1.write(i)
                self.listbox.delete(ACTIVE)
        except TclError:
            pass


class SetFrame(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.userName = StringVar()
        self.password1 = StringVar()
        self.password2 = StringVar()
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)
        self.page.pack()
        Label(self, text='新建用户名: ').grid(row=2, stick=W, pady=10)
        Entry(self, textvariable=self.userName).grid(row=2, column=1, stick=E)
        Label(self, text='密码: ').grid(row=3, stick=W, pady=10)
        Entry(self, textvariable=self.password1, show='*').grid(row=3, column=1, stick=E)
        Label(self, text='确认密码: ').grid(row=4, stick=W, pady=10)
        Entry(self, textvariable=self.password2, show='*').grid(row=4, column=1, stick=E)
        Button(self, text='新建用户', bg='skyblue', command=self.setup).grid(row=6, column=1, stick=E, pady=10)

    def setup(self):
        self.name = self.userName.get()
        self.pwd = self.password1.get()
        self.pwd1 = self.password2.get()
        with open('data/codes.csv', 'a+', encoding='utf-8') as data:
            user_name = [i.strip().split(',')[0] for i in data.readlines()]
        if self.name in user_name:
            showinfo(title='错误', message='用户名已存在！')
        elif self.name == '宋风杰':
            showinfo(title='错误', message='用户名已存在！')
        elif self.name == '':
            showinfo(title='错误', message='用户名不能为空！')
        elif self.pwd != self.pwd1:
            showinfo(title='错误', message='两次输入密码不一致！')
        elif self.pwd == self.pwd1 and self.pwd == '':
            showinfo(title='错误', message='密码不能为空！')
        elif self.pwd == self.pwd1:
            with open('data/codes.csv', 'a+', encoding='utf-8') as datas:
                datas.write(f"{self.name},{self.pwd}\n")
            showinfo(title='提示', message='重新启动软件以完成创建！')
        else:
            showinfo(title='未知错误', message='注册失败！')


class HistoryFrame(Frame):
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.createPage()

    def createPage(self):
        self.page = Frame(self)
        self.page.pack()
        ###
        s = Scrollbar(self.page)  # 创建滚动条
        s.pack(side=RIGHT, fill=Y)  # 设置垂直滚动条显示的位置，使得滚动条，靠右侧；通过 fill 沿着 Y 轴填充
        listbox = Listbox(self.page, width=125, height=25, yscrollcommand=s.set)  # 为Listbox控件添加滚动条
        data = [f"|{'用户姓名'}|{'用户类型'}|{'登录时间':{chr(12288)}^12}|"]
        with open('data/login_time.csv', 'r', encoding='utf-8') as user_data1:  # 加密
            self.user_history = [i.strip() for i in user_data1.readlines()]
        for i in self.user_history:
            if i.split(',')[2] != 'VIP':
                try:
                    data.append(f"|{i.split(',')[0]:{chr(12288)}<4}|普通用户|{i.split(',')[1]:^25}|")
                except:
                    data.append(f"|{'':{chr(12288)}<4}|{'':{chr(12288)}<4}|{'':^25}|")
        for i in range(len(data)):
            item = data[i]
            listbox.insert(i, item)
        listbox.pack()
        s.config(command=listbox.yview)  # 设置滚动条，使用 yview使其在垂直方向上滚动 Listbox 组件的内容
