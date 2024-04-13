from tkinter import *
from view import *  # 菜单栏对应的各个子页面


class MainPage(object):
    def __init__(self, master=None, num=1):
        self.root = master  # 定义内部变量root
        self.num = num
        self.h = self.root.winfo_screenheight()
        self.w = self.root.winfo_screenwidth()
        self.root.geometry('%dx%d' % (self.w / 1.5, self.h / 1.5))  # 设置窗口大小
        self.createPage()

    def createPage(self):
        self.inputPage = inputFrame(self.root)  # 创建不同Frame
        self.queryPage = queryFrame(self.root)
        if self.num == 1:
            self.inputPage.pack()  # 默认显示数据录入界面
        elif self.num == 2:
            self.queryPage.pack()
        menubar = Menu(self.root)
        menubar.add_command(label='数据录入', command=self.inputData)
        menubar.add_command(label='未缴查询', command=self.queryData)
        self.root['menu'] = menubar  # 设置菜单栏

    def inputData(self):    #拖动录入数据，需加密
        try:
            self.inputPage.pack()
            self.queryPage.pack_forget()
        except:
            self.inputPage.destroy()
            self.queryPage.destroy()
            MainPage(self.root)

    def queryData(self):
        try:
            self.queryPage.pack()
            self.inputPage.pack_forget()
        except:
            self.queryPage.destroy()
            self.inputPage.destroy()
            MainPage(self.root, 2)

    def destory(self):
        self.inputPage.destroy()
        self.queryPage.destroy()
