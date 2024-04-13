from tkinter import *
from tkinter.messagebox import *
import datetime
from SuperPage import *
from MainPage import *

super_name = '宋风杰'
super_pwd = '123456'

class LoginPage(object):
    def __init__(self, master=None):
        self.root = master  # 定义内部变量root
        # self.root.iconbitmap('C:/Users/Administrator/Desktop/favicon.ico') logo图标
        self.h = self.root.winfo_screenheight()
        self.w = self.root.winfo_screenwidth()
        self.root.geometry('%dx%d' % (self.w/1.5, self.h/1.5))  # 设置窗口大小
        self.username = StringVar()
        self.password = StringVar()
        self.createPage()

    def createPage(self):
        self.page = Frame(self.root)  # 创建Frame
        self.page.pack(side=TOP, fill=BOTH, expand=1)
        Label(self.page, text='账户: ').place(rely=0.4, relx=0.45, anchor=E)
        Entry(self.page, textvariable=self.username).place(rely=0.4, relx=0.45, anchor=W)
        Label(self.page, text='密码: ').place(rely=0.5, relx=0.45, anchor=E)
        Entry(self.page, textvariable=self.password, show='*').place(rely=0.5, relx=0.45, anchor=W)
        Button(self.page, text='登陆', command=self.loginCheck, bg='skyblue').place(rely=0.6, relx=0.5, anchor=E)
        Button(self.page, text='退出', command=self.page.quit, bg='red').place(rely=0.6, relx=0.53, anchor=W)

    def loginCheck(self):
        global super_name
        global super_pwd
        name = self.username.get()
        secret = self.password.get()
        user = []
        passwords = []
        with open('data/codes.csv', 'r', encoding='utf-8') as users:  # 需加密解密
            for i in users.readlines():
                user.append(i.strip().split(',')[0])
                passwords.append(i.strip().split(',')[1])
        if name == super_name and secret == super_pwd:    # 底层绝对管理员
            with open('data/login_time.csv', 'a', encoding='utf-8') as lsT: # 需加密解密
                lsT.write(f"{name},{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')},VIP\n")
            self.page.destroy()
            SuperPage(self.root)
        elif name in user:
            No = user.index(name)
            if secret == passwords[No]:
                with open('data/login_time.csv', 'a', encoding='utf-8') as lsT:  # 需加密解密
                    lsT.write(f"{name},{datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')},N/A\n")
                self.page.destroy()
                MainPage(self.root)
            else:
                showinfo(title='错误', message='账号不存在或密码错误！')
        else:
            showinfo(title='错误', message='账号不存在或密码错误！')
