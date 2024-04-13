import turtle
'''
turtle.setup(480, 320, 100, 50)
turtle.bgcolor(1, 0.5, 0.25)

turtle.shape('turtle')
turtle.pensize(3)
turtle.circle(50)
while True:
    turtle.circle(-50, steps=2)
'''
'''
turtle.shape('turtle')
turtle.shapesize(5)

turtle.backward(100)
turtle.done()
'''

'''
shape('turtle')
pensize(4)

penup()
goto(100, 100)

color('orange')
pendown()
begin_fill()
circle(50)
end_fill()
done()
'''
from turtle import *
shape('turtle')
color('orange')

n = 90
step = 0
speed(0)
while True:
    seth(n)
    forward(step)
    n += 1
    step += 0.001
