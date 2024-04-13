from pymouse import PyMouse
import time
import pyautogui

'''
m = PyMouse()
print(m.position())
'''

loc_x, loc_y = 864, 146
news_x, news_y = (722, 425)
dx = 471 - 368 + 10
Quit_x, Quit_y = 625, 77


m = PyMouse()

time.sleep(2)
m.click(1774, 19)   # 最小化python
time.sleep(2)
m.click(1826, 15)   # 对学习强国app进行缩放
time.sleep(2)
m.click(1068, 206)
time.sleep(2)

for i in range(2):  # 分享任务（2）
    m.click(loc_x, loc_y)   # 点击地方频道
    time.sleep(2)
    m.click(news_x, news_y + i * dx)   # 点击新闻
    time.sleep(5)
    m.click(1292, 994)  # 点击分享
    time.sleep(1)
    m.click(680, 670)   # 点击学习强国好友
    time.sleep(2)
    m.click(Quit_x, Quit_y)    # 退出
    time.sleep(1)
    m.click(Quit_x, Quit_y)   # 退出
    time.sleep(2)
    
m.click(loc_x, loc_y)  # 点击地方频道
time.sleep(2)

for i in range(10):  # 阅读任务（10）
    m.click(news_x, news_y)   # 点击新闻
    time.sleep(70)   # reading time
    m.click(Quit_x, Quit_y)    # 退出
    time.sleep(2)
    m.click(loc_x, loc_y)  # 点击地方频道
    time.sleep(2)

m.click(676, 282)
time.sleep(2)
m.click(Quit_x, Quit_y)    # 退出
