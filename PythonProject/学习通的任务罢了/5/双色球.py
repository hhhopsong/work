
import random
red = set()
blue = tuple(('0', '{:0>2}'.format(random.randint(1, 16))))
while len(red) < 6:
    rand = random.randint(1,33)
    red.add(rand)
key = list(red)
red = set()
for i in key:
    red.add('{:0>2}'.format(i))
key = '-'.join(tuple(red) + blue[1:])
print(f'双色球号码是：{key}')

