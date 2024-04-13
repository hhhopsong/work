
scores = [[88, 92, 55, 60, 71], [72, 66, 95, 81], [93, 89, 79, 86, 99, 85, 60]]
list1 = []
for i in scores:
    for o in i:
        list1.append(o)
print('课程最高分为:{}'.format(max(list1)))

