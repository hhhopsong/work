from tkinter import *

from app.keys_box.view import InputFrame
from app.物业.superview import SuperQueryFrame
from superview import *  # 菜单栏对应的各个子页面


class SuperPage(object):
    def __init__(self, master=None, num=1):
        self.root = master  # 定义内部变量root
        self.num = num
        self.h = self.root.winfo_screenheight()
        self.w = self.root.winfo_screenwidth()
        self.root.geometry('%dx%d' % (self.w / 1.5, self.h / 1.5))  # 设置窗口大小
        self.createPage()

    def createPage(self):
        self.inputPage = InputFrame(self.root)  # 创建不同Frame
        self.queryPage = SuperQueryFrame(self.root)
        self.action_recordsPage = Action_recordsFrame(self.root)
        self.unit_pricePage = Unit_priceFrame(self.root)
        self.menuPage = MenuFrame(self.root)  # 创建不同Frame
        self.setPage = SetFrame(self.root)
        self.hitoryPage = HistoryFrame(self.root)

        menubar = Menu(self.root)
        self.root['menu'] = menubar  # 设置菜单栏
        counter = Menu(menubar, tearoff=False)
        counter.add_command(label='用户信息', command=self.menuData)   # 设置子菜单栏
        counter.add_command(label='新建用户', command=self.setData)
        counter.add_command(label='使用记录', command=self.historyData)
        menubar.add_command(label='数据录入', command=self.inputData)   # 设置主菜单栏
        menubar.add_command(label='操作记录', command=self.action_recordData)   # 打印凭据,导入数据(人员,时次,操作类型)
        menubar.add_command(label='单价更改', command=self.unit_priceData)   # 单价更改
        menubar.add_command(label='缴费查询', command=self.queryData)
        menubar.add_cascade(label='用户管理', menu=counter)

        if self.num == 1:
            self.inputPage.pack()  # 默认显示数据录入界面
        elif self.num == 2:
            self.queryPage.pack()
        elif self.num == 3:
            self.menuPage.pack()
        elif self.num == 4:
            self.action_recordsPage.pack()
        elif self.num == 5:
            self.unit_pricePage.pack()

    def inputData(self):    #拖动录入数据，需加密
        try:
            self.inputPage.pack()
            self.queryPage.pack_forget()
            ##
            self.menuPage.pack_forget()
            self.setPage.pack_forget()
            self.hitoryPage.pack_forget()
        except:
            self.inputPage.destroy()
            self.queryPage.destroy()
            self.action_recordsPage.destory()
            self.unit_pricePage.destory()
            ##
            self.hitoryPage.destroy()
            self.menuPage.destroy()
            self.setPage.destroy()
            SuperPage(self.root)

    def queryData(self):    #查询电费,缴费状况（列表滑动，记录条数显示）,额外设置补打（No增加）,额外设置查询收款人、时间、金额
        try:
            self.queryPage.pack()
            self.inputPage.pack_forget()
            self.action_recordsPage.pack_forget()
            self.unit_pricePage.pack_forget()
            ##
            self.menuPage.pack_forget()
            self.setPage.pack_forget()
            self.hitoryPage.pack_forget()
        except:
            self.queryPage.destroy()
            self.inputPage.destroy()
            self.action_recordsPage.destory()
            self.unit_pricePage.destory()
            ##
            self.hitoryPage.destroy()
            self.menuPage.destroy()
            self.setPage.destroy()
            SuperPage(self.root, 2)

    def action_recordData(self):
        try:
            self.queryPage.pack_forget()
            self.inputPage.pack_forget()
            self.action_recordsPage.pack()
            self.unit_pricePage.pack_forget()
            ##
            self.menuPage.pack_forget()
            self.setPage.pack_forget()
            self.hitoryPage.pack_forget()
        except:
            self.queryPage.destroy()
            self.inputPage.destroy()
            self.action_recordsPage.destory()
            self.unit_pricePage.destory()
            self.action_recordsPage.mainloop()
            ##
            self.hitoryPage.destroy()
            self.menuPage.destroy()
            self.setPage.destroy()
            SuperPage(self.root, 4)

    def unit_priceData(self):
        try:
            self.queryPage.pack_forget()
            self.inputPage.pack_forget()
            self.action_recordsPage.pack_forget()
            self.unit_pricePage.pack()
            ##
            self.menuPage.pack_forget()
            self.setPage.pack_forget()
            self.hitoryPage.pack_forget()
        except:
            self.queryPage.destroy()
            self.inputPage.destroy()
            self.action_recordsPage.destory()
            self.unit_pricePage.destory()
            ##
            self.hitoryPage.destroy()
            self.menuPage.destroy()
            self.setPage.destroy()
            SuperPage(self.root, 5)

    def destory(self):
        self.inputPage.destroy()
        self.queryPage.destroy()

    def menuData(self):
        try:
            self.inputPage.pack_forget()
            self.queryPage.pack_forget()
            self.action_recordsPage.pack_forget()
            self.unit_pricePage.pack_forget()
            ##
            self.menuPage.pack()
            self.setPage.pack_forget()
            self.hitoryPage.pack_forget()
            self.menuPage.mainloop()
        except:
            self.inputPage.destroy()
            self.queryPage.destroy()
            self.action_recordsPage.destory()
            self.unit_pricePage.destory()
            ##
            self.menuPage.destroy()
            self.hitoryPage.destroy()
            self.setPage.destroy()
            SuperPage(self.root, 3)

    def setData(self):
        try:
            self.inputPage.pack_forget()
            self.queryPage.pack_forget()
            self.action_recordsPage.pack_forget()
            self.unit_pricePage.pack_forget()
            ##
            self.setPage.pack()
            self.menuPage.pack_forget()
            self.hitoryPage.pack_forget()
        except:
            self.inputPage.destroy()
            self.queryPage.destroy()
            self.action_recordsPage.destory()
            self.unit_pricePage.destory()
            ##
            self.setPage.destroy()
            self.hitoryPage.destroy()
            self.menuPage.destroy()
            SuperPage(self.root, 3)

    def historyData(self):
        try:
            self.inputPage.pack_forget()
            self.queryPage.pack_forget()
            self.action_recordsPage.pack_forget()
            self.unit_pricePage.pack_forget()
            ##
            self.hitoryPage.pack()
            self.menuPage.pack_forget()
            self.setPage.pack_forget()
            self.hitoryPage.mainloop()
        except:
            self.inputPage.destroy()
            self.queryPage.destroy()
            self.action_recordsPage.destory()
            self.unit_pricePage.destory()
            ##
            self.hitoryPage.destroy()
            self.menuPage.destroy()
            self.setPage.destroy()
            SuperPage(self.root, 3)
