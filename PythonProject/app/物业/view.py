from tkinter import *
from tkinter.messagebox import *
import pandas as pd
from SuperPage import *
from MainPage import *


class inputFrame(Frame):  # 继承Frame类
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
        actionName = StringVar()
        password = StringVar()
        Label(tk, text='操作人账号: ').grid(row=2, pady=10)
        Entry(tk, textvariable=actionName).grid(row=2, column=1)
        Label(tk, text='操作人密码: ').grid(row=3, pady=10)
        Entry(tk, textvariable=password).grid(row=3, column=1)
        Button(tk, text='录入', bg='skyblue',width=20).grid(row=4, column=1, pady=10)


class queryFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.address = StringVar()
        self.name = StringVar()
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 创建Frame
        self.page.pack()
        Label(self, text='住户地址: ').grid(row=2, stick=W, pady=10)
        Entry(self, textvariable=self.address).grid(row=2, column=1, stick=E)
        Label(self, text='住户姓名: ').grid(row=3, stick=W, pady=10)
        Entry(self, textvariable=self.name).grid(row=3, column=1, stick=E)
        Button(self, text='查询', command=self.SearchCheck, bg='skyblue').grid(row=5, column=1, stick=E, pady=10)

    def SearchCheck(self):
        add_q = self.address.get()
        name_q = self.name.get()
        userno = []
        times = []
        adds = []
        moneys = []
        counters = []
        i = 0
        while True:
            try:
                with open(f'data1/物业统计表{i}.csv', 'r+', encoding='utf-8') as us:  # 需加密解密
                    datas = us.readlines()[1:]
                user = [i.strip().split(',')[6] for i in datas]
                year = [i.strip().split(',')[2] for i in datas]
                month = [i.strip().split(',')[4] for i in datas]
                add = [i.strip().split(',')[17] for i in datas]
                con = [i.strip().split(',')[18] for i in datas]
                money = [i.strip().split(',')[22] for i in datas]
                for ii in range(len(user)):
                    times.append(f'{year[ii]:*<4}{"-"}{month[ii]:0>2}')
                    adds.append(add[ii])
                    moneys.append(money[ii])
                    userno.append(user[ii])
                    counters.append(con[ii])
                i += 1
            except:
                break
        if add_q == '' and name_q in userno:
            user = []
            money = []
            time = []
            add = []
            for i in range(len(userno)):
                if name_q == userno[i] and counters[i] == '':
                    user.append(userno[i])
                    money.append(eval(moneys[i]))
                    add.append(adds[i])
                    time.append(times[i])
            # moneyframe
            self.moneypage = MoneyFrame(self.root, user, add, time, money)
        elif add_q in adds and name_q == '':
            money = []
            time = []
            user = []
            add = []
            for i in range(len(userno)):
                if add_q == adds[i] and counters[i] == '':
                    add.append(adds[i])
                    money.append(eval(moneys[i]))
                    user.append(userno[i])
                    time.append(times[i])
            # moneyframe
            self.moneypage = MoneyFrame(self.root, user, add, time, money)
        elif add_q in adds and name_q in userno:
            user = []
            add = []
            money = []
            time = []
            for i in range(len(userno)):
                if name_q == userno[i] and add_q == adds[i] and counters[i] == '':
                    user.append(userno[i])
                    add.append(adds[i])
                    money.append(eval(moneys[i]))
                    time.append(times[i])
            # moneyframe
            self.moneypage = MoneyFrame(self.root, user, add, time, money)
        elif add_q == '' and name_q == '':
            user = []
            add = []
            money = []
            time = []
            for i in range(len(userno)):
                if counters[i] == '':
                    user.append(userno[i])
                    add.append(adds[i])
                    money.append(eval(moneys[i]))
                    time.append(times[i])
            # moneyframe
            self.moneypage = MoneyFrame(self.root, user, add, time, money)
        else:
            showinfo(title='错误', message='查询信息不匹配或不支持！')


class MoneyFrame(Frame):
    def __init__(self, master=None, nam=None, add=None, time=None, money=None):
        Frame.__init__(self, master)
        self.root = Toplevel()  # 定义内部变量root
        self.root.resizable(False, False)
        self.root.title('查询结果')
        self.name = nam
        self.add = add
        self.time = time
        self.money = [eval(f'{i:.2f}') for i in money]
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 创建Frame
        self.page.pack()
        ###
        s = Scrollbar(self.page)  # 创建滚动条
        s.pack(side=RIGHT, fill=Y)  # 设置垂直滚动条显示的位置，使得滚动条，靠右侧；通过 fill 沿着 Y 轴填充
        listbox = Listbox(self.page, width=80,height=31, yscrollcommand=s.set)  # 为Listbox控件添加滚动条
        data = [f"{'住户姓名'}|{'住户地址':{chr(12288)}^26}|{'应缴时间':{chr(12288)}^6}|{'欠缴金额':{chr(12288)}^7}"]
        for i in range(len(self.add)):
            data.append(f'{self.name[i]:{chr(12288)}<4}|{self.add[i]:^74}|{self.time[i]:^13}|￥{self.money[i]:^15}')
        for i in range(len(data)):
            item = data[i]
            listbox.insert(i, item)
        listbox.pack()
        s.config(command=listbox.yview)  # 设置滚动条，使用 yview使其在垂直方向上滚动 Listbox 组件的内容
        ###
        Label(self.page, text=f'已找到{len(self.money)} 个对象\t\t共欠金额:{sum(self.money):.2f}元').pack(anchor='e')


class CountFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.createPage()

    def createPage(self):
        Label(self, text='...').pack()
