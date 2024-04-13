n, m = input().split()
n = int(n)
m = int(m)
square = [[1 for i in range(m)] for j in range(n)]
t = int(input())
for i in range(t):
    cl = [int(i) for i in input().split()]
    for j in range(cl[0]-1, cl[2]):
        for k in range(cl[1]-1, cl[3]):
            square[j][k] = 0
s = 0
for i in range(n):
    for j in range(m):
        s += square[i][j]
print(s)