import numpy as np


T = 18.4       # 单位:℃
RH = 97         # 单位:%
P = 1003.5     # 单位:hPa
z = 35.2     # 单位:m
a = 17.27
b = 35.86
# 求露点温度
Td = T + 273.16      # 单位:K
e = 6.1078 * np.exp(a * (Td - 273.16) / (Td - b)) * RH / 100
while True:
    es = 6.1078 * np.exp(a * (Td - 273.16) / (Td - b))
    if e < es:
        Td -= 0.05
    else:
        break
print(f'该站的露点温度为: {Td - 273.16:.2f} ℃')
# 求凝结高度
Tdz = Td
Tz = T + 273.16
Pz = P
ez = 6.1078 * np.exp(a * (Tdz - 273.16) / (Tdz - b))
θz = Tz * (1000 / Pz) ** 0.286
qz = 0.622 * ez / (Pz - 0.378 * ez) * 1000  # 单位:g/kg
Cp = 1005   # 单位:J/...
PL = 300
while True:
    TL = θz * (PL / 1000) ** 0.286
    eL = 6.1078 * np.exp(a * (TL - 273.16) / (TL - b))
    qL = 0.622 * eL / (PL - 0.378 * eL) * 1000
    if qL - qz < 0:
        PL += 1
    else:
        ZL = (9.8 * z + Cp * Tz - Cp * TL) / 9.8
        break
print(f'凝结高度: {ZL:.2f} m\n凝结温度: {TL - 273.16:.2f} ℃\n该层的气压: {PL} hPa')
