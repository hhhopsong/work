'''
母星数据:
    赤道转速:158m/s
    半径: R = 1274.2 km
    质量: M = 2.383561897 * 10^23 kg

luna数据:
     半径: R = 354 km
     质量: M = 3.00 * 10^21 kg

Cylero数据:
    半径: R = 678 km
    质量: M = 2.554717 * 10^22 kg

常量:
    G = 6.67 * 10^-11
'''
G = 6.67 * 10**(-11)
pi = 3.141592654
M_e = 2.383561897 * 10**23
R_e = 1274200

print('输入轨道远地点(km):')
Ap = eval(input()) * 1000
print('输入轨道近地点(km):')
Pe = eval(input()) * 1000

a = (Ap + Pe + 2*R_e)*0.5
T = 2 * pi * (((a**3)/(G*M_e))**0.5)

print('目标在前(1) or 在后(0)?')
loc = eval(input())
print('输入相差时间(eg:16s or 15.8m):')
t1 = input()
print('输入变轨位置:(km)')

if t1[-1] == 'm':
    t1 = eval(t1[:-1]) * 60
else:
    t1 = eval(t1[:-1])

Pe_s = eval(input()) * 1000
if loc == 0:
    delta_t = T - t1
else:
    delta_t = -t1

i = 0
while True:
    i += 1
    target_T = T + delta_t/i
    target_a = ((target_T ** 2 * G * M_e) / ((2 * pi) ** 2)) ** (1 / 3)
    Ap_s = target_a * 2 - Pe_s - 2*R_e
    if Ap_s >= 65000:
        break
print('**********Result**********')

if loc == 0:
    print(f'加速到远地点高度为:{Ap_s/1000:.1f} km')
    print(f'预计耦合圈数:{i}')
else:
    print(f'减速到远地点高度为:{Ap_s / 1000:.1f} km')
    print(f'预计耦合圈数:{i}')
