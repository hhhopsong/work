a = 0
b = 1
list1 = []
for i in range(20):
    c = a + b
    a = b
    b = c
    list1.append(c)

print(f'{list1}')
