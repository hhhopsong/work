n = int(input('输入样本数:'))
x = [0 for i in range(n)]
for i in range(n):
    x[i] = eval(input(f'样本值{i+1}='))
miu = float(sum(x)/n)
for i in range(n):
    x[i] = x[i] ** 2
print(f'\nRESULT:\nμ = {miu:.2f}\nS={pow((sum(x) - n*pow(miu, 2))/(n-1),0.5):.2f}\nS^2 = {(sum(x) - n*pow(miu, 2))/(n-1):.2f}')
print(f'\n方差:{sum(x)/n:.2f}\n标准差:{pow(sum(x)/n,0.5):.2f}')
