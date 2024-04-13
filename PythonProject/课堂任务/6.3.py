from turtle import *
tracer(False)

setup(600, 400)
bgcolor('yellow')

screensize(600, 400, 'pink')

hideturtle()
update()
done()

from turtle import *
from random import *

'''setup(400, 400)

for i in range(10):
    penup()
    loc = tuple(random() * 400 for i in range(2))
    rgb = tuple(random() for i in range(3))
    goto(loc)
    pendown()
    pencolor(rgb)
    color(rgb)
    begin_fill()
    circle(20, 360)
    penup()
    end_fill()


done()'''

'''hideturtle()
rad = 0
speed(0)
for i in range(100):
    rad += 15
    seth(rad)
    circle(80, 90)
    seth(rad + 180)
    circle(80, 90)
done()'''

'''rad = 0
hideturtle()
tracer(False)
for i in range(24):
    rad += 15
    n = 0
    for ii in range(4):
        seth(rad + n)
        fd(100)
        n += 90
update()
done()'''

'''pensize(5)

write('你好！', font=('楷体', 18, 'normal'))
done()'''
'''
x = 0
for i in range(3):
    setx(x)
    n = 0
    pendown()
    color('pink')
    begin_fill()
    for ii in range(4):
        seth(n)
        fd(40)
        n += 90
    end_fill()
    penup()
    x += 100'''

done()
