n = int(input())
a = [int(x) for x in input().split()]
k = int(input())
r = []
for i in range(k, n - k):
    r.append(min(a[i-k:i+k+1]))
result = [r[0] * k]
for i in r:
    result.append(i)
result.append(r[-1] * k)
for i in result:
    print(i, end=' ')