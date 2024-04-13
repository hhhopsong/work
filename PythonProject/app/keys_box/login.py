import tkinter as tk
from tkinter import messagebox
from main_window import *


def open_():
    info_pwd = var_pwd.get()
    with open('key.txt', 'r') as line:
        pwd = line.read().split(':')[1]
        log = line.read().split(':')[0]
        if log == '0':
            tk.messagebox.askokcancel(title='提示', message='您尚未注册！请先行注册后使用。')
        else:
            if str(info_pwd) != pwd:
                tk.messagebox.askokcancel(title='提示', message='密码错误！')
            else:
                window.destroy()
                main_window()


def sign_():
    def login_():
        pwd1 = entry_1.get()
        pwd2 = entry_2.get()
        if pwd1 == pwd2 and pwd1 != '':
            tk.messagebox.askokcancel(title='注册成功！', message='注册成功！注册后密码不可更改。')
            with open('key.txt', 'w+') as line1:
                line1.write(f'1:{pwd1}')
            window_sign.destroy()
        else:
            tk.messagebox.showwarning(title='注意', message='密码输入错误。')

    with open('key.txt', 'r+') as line:
        if line.read()[0] == '0':
            window_sign = tk.Toplevel()
            window_sign.title('新用户注册')
            window_sign.geometry('270x140')
            tk.Label(window_sign, text='初始化密码:').place(x=30, y=45, anchor='w')
            tk.Label(window_sign, text='确认密码:').place(x=30, y=75, anchor='w')
            tk.Entry(window_sign, textvariable=entry_1, show=None, width=13).place(x=100, y=45, anchor='w')
            tk.Entry(window_sign, textvariable=entry_2, show=None, width=13).place(x=100, y=75, anchor='w')
            tk.Button(window_sign, text='注册', command=login_).place(x=140, y=105, anchor='c')
        else:
            tk.messagebox.showwarning(title='注意', message='请勿重复注册！')


# user-info
window = tk.Tk()
window.title('keys_box')
window.geometry('480x270')
tk.Label(window, text='密码保险箱', font=('微软雅黑', 20)).place(x=240, y=80, anchor='c')
tk.Label(window, text='🔑:').place(x=125, y=125)
var_pwd = tk.StringVar()
entry_1 = tk.StringVar()
entry_2 = tk.StringVar()
entry_pwd = tk.Entry(window, textvariable=var_pwd, show='*')
entry_pwd.place(x=240, y=137, anchor='c')

# login and sign_up button
tk.Button(window, text='新用户注册', command=sign_).place(x=470, y=260, anchor='se')
tk.Button(window, text='OPEN', command=open_).place(x=240, y=160, anchor='n')

window.mainloop()
