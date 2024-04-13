import uiautomation as auto
qq_win = auto.WindowControl(searchDepth=1, ClassName='TXGuiFoundation', Name='申富明')
input_edit = qq_win.EditControl()
msg_list = qq_win.ListControl() #找到 list
items = msg_list.GetChildren()
for one_item in items:      #遍历所有的 Children
    print(one_item.Name)    #打印消息