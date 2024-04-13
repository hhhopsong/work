from tkinter import *
from tkinter.messagebox import *


class InputFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.itemName = StringVar()
        self.sellPrice = StringVar()
        self.deductPrice = StringVar()
        self.createPage()

    def createPage(self):
        def get_in():
            with open('data.txt', 'a+') as lines:
                lines.writelines(f'{self.itemName.get()}◇{self.sellPrice.get()}◇{self.deductPrice.get()}')
        Label(self).grid(row=0, stick=W, pady=10)
        Label(self, text='软件名称: ').grid(row=2, stick=W, pady=10)
        Entry(self, textvariable=self.itemName).grid(row=2, column=1, stick=E)
        Label(self, text='账号: ').grid(row=3, stick=W, pady=10)
        Entry(self, textvariable=self.sellPrice).grid(row=3, column=1, stick=E)
        Label(self, text='密码: ').grid(row=4, stick=W, pady=10)
        Entry(self, textvariable=self.deductPrice).grid(row=4, column=1, stick=E)
        Button(self, text='录入', command=get_in).grid(row=6, column=1, stick=E, pady=10)



class QueryFrame(Frame):  # 继承Frame类
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.root = master  # 定义内部变量root
        self.itemName = StringVar()
        self.sellPrice = StringVar()
        self.deductPrice = StringVar()
        self.createPage()

    def createPage(self):
        def search_in():
            key = self.itemName.get()
            with open('data.txt', 'r') as lines:
                datas = lines.readlines()
            for i in datas:
                i = i.split('◇')
                if key == i[0]:
                    t1.insert('end', i[1])
                    t2.insert('end', i[2])
                    break
        Label(self).grid(row=0, stick=W, pady=10)
        Label(self, text='软件名称: ').grid(row=2, stick=W, pady=10)
        Entry(self, textvariable=self.itemName).grid(row=2, column=1, stick=E)
        Label(self, text='账号: ').grid(row=3, stick=W, pady=10)
        t1 = Text(self, height=1.25, width=20)
        t1.grid(row=3, column=1, stick=E, pady=10)
        Label(self, text='密码: ').grid(row=4, stick=W, pady=10)
        t2 = Text(self, height=1.25, width=20)
        t2.grid(row=4, column=1, stick=E, pady=10)
        Button(self, text='查询', command=search_in).grid(row=6, column=1, stick=E, pady=10)
