W, H, n, R = input().split()
W, H, n, R = int(W), int(H), int(n), int(R)
loc = []
num = 0
for i in range(n):
    info = input().split()
    info = [int(info[0]), int(info[1])]
    loc.append(info)
for i in range(W+1):
    for j in range(H+1):
        for k in loc:
            if ((i - k[0])**2 + (j - k[1])**2)**0.5 <= R:
                num += 1
                break
print(num)
