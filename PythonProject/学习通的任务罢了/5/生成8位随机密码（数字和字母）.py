
import random
x = set()
while len(x) < 8:
    rand = random.randint(0, 123)
    if 0 <= rand <= 9:
        x.add(rand)
    elif 65 <= rand <= 90 or 97 <= rand <= 122:
        x.add(chr(rand))
for i in x:
    print(i, end='')
