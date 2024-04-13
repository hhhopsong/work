import tkinter as tk
from view import *  # 菜单栏对应的各个子页面


def main_window():
    def info_():
        info_page.pack()
        search_page.pack_forget()

    def search_():
        info_page.pack_forget()
        search_page.pack()

    root = tk.Tk()  # 定义内部变量root
    root.geometry('600x300')  # 设置窗口大小
    info_page = InputFrame(root)  # 创建不同Frame
    search_page = QueryFrame(root)
    info_page.pack()  # 默认显示数据录入界面
    menubar = tk.Menu(root)
    menubar.add_command(label='录入', command=info_)
    menubar.add_command(label='搜索', command=search_)
    root['menu'] = menubar  # 设置菜单栏