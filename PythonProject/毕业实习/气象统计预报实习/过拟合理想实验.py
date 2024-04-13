import numpy as np


# 样本生成
Nt = 1000  # 设置样本量
Nx = 6  # 设置预报因子个数
'''Xsamples = [np.random.randn(Nt) for i in range(Nx)]  # 生成标准正态分布预报因子样本,共有Nx 6个,均值为0，标准差为1
np.save('Xsamples.npy', Xsamples)'''
Xsamples = np.load('Xsamples.npy')
Bgiven = [2., 1., -1., .5, -.1, .1] # 设置预报因子权重
Ygiven = np.sum([Bgiven[i] * Xsamples[i] for i in range(Nx)], axis=0)  # 生成预报量样本
Fgiven = .5  # 设置预报量可解释部分
Yrandom = np.random.normal(loc=0, scale=np.sqrt(np.var(Ygiven)/Fgiven*(1-Fgiven)), size=(Nt,))  # 预报量不可解释部分
np.save('Yrandom.npy', Yrandom)
Ysamples = Ygiven + Yrandom  # 生成预报量样本
YsamplesVar = np.var(Ysamples)  # 样本方差
YgivenVar = np.var(Ygiven)  # 预报量可解释部分方差
YrandomVar = np.var(Yrandom)  # 预报量不可解释部分方差
corr_YgYs = np.corrcoef(Ygiven, Ysamples)  # 预报量可解释部分与预测量的相关系数
Ylow, Yhigh = np.percentile(Ysamples, [25, 75]) # 返回YsamplesVar第10分位和第90分位的值
Ygivenlow , Ygivenhigh = np.percentile(Ygiven, [25, 75]) # 返回YgivenVar第10分位和第90分位的值
Yhighrank = [np.sum([1 for i in Ysamples[np.where(Ygiven >= Ygivenhigh)] if i >= Yhigh]),
             np.sum([1 for i in Ysamples[np.where(Ygiven >= Ygivenhigh)] if (Ylow < i < Yhigh)]),
             np.sum([1 for i in Ysamples[np.where(Ygiven >= Ygivenhigh)] if i <= Ylow])]
Ylowrank = [np.sum([1 for i in Ysamples[np.where(Ygiven <= Ygivenlow)] if i >= Yhigh]),
            np.sum([1 for i in Ysamples[np.where(Ygiven <= Ygivenlow)] if (Ylow < i < Yhigh)]),
            np.sum([1 for i in Ysamples[np.where(Ygiven <= Ygivenlow)] if i <= Ylow])]

print(f'总样本量: {Nt}')
print(f'Syy: {YsamplesVar:.3f}\tU: {YgivenVar:.3f}\tQ: {YrandomVar:.3f}\t'
      f'U/Syy: {YgivenVar / YsamplesVar:.3f}\tCorr Ys&Yg: {corr_YgYs[0, 1]:.4f}')
print(f'Y--X*\t权重\t\t 相关系数\t\t可解释方差')
print(f'Y--X1\t{Bgiven[0]:>6.3f}\t{np.corrcoef(Ygiven, Xsamples[0])[0, 1]:>8.5f}\t'
      f'{np.var(Bgiven[0] * Xsamples[0])/YsamplesVar:>8.5f}')
print(f'Y--X2\t{Bgiven[1]:>6.3f}\t{np.corrcoef(Ygiven, Xsamples[1])[0, 1]:>8.5f}\t'
      f'{np.var(Bgiven[1] * Xsamples[1])/YsamplesVar:>8.5f}')
print(f'Y--X3\t{Bgiven[2]:>6.3f}\t{np.corrcoef(Ygiven, Xsamples[2])[0, 1]:>8.5f}\t'
      f'{np.var(Bgiven[2] * Xsamples[2])/YsamplesVar:>8.5f}')
print(f'Y--X4\t{Bgiven[3]:>6.3f}\t{np.corrcoef(Ygiven, Xsamples[3])[0, 1]:>8.5f}\t'
      f'{np.var(Bgiven[3] * Xsamples[3])/YsamplesVar:>8.5f}')
print(f'Y--X5\t{Bgiven[4]:>6.3f}\t{np.corrcoef(Ygiven, Xsamples[4])[0, 1]:>8.5f}\t'
      f'{np.var(Bgiven[4] * Xsamples[4])/YsamplesVar:>8.5f}')
print(f'Y--X6\t{Bgiven[5]:>6.3f}\t{np.corrcoef(Ygiven, Xsamples[5])[0, 1]:>8.5f}\t'
      f'{np.var(Bgiven[5] * Xsamples[5])/YsamplesVar:>8.5f}')
print(f'\t\t总解释方差占比: \t\t{YgivenVar/YsamplesVar:>8.5f}')
print(f'样本高低值阈值: {Ygivenhigh:.8f}  {Ygivenlow:.8f}')
print(f'对应样本Y属类:\t低值占比\t中值占比\t高值占比')
print(f'预报因子X高值时\t{Yhighrank[2]/np.sum(Ygiven >= Ygivenhigh)*100:.0f}%\t'
      f'\t{Yhighrank[1]/np.sum(Ygiven >= Ygivenhigh)*100:.0f}%\t'
      f'\t{Yhighrank[0]/np.sum(Ygiven >= Ygivenhigh)*100:.0f}%')
print(f'预报因子X低值时\t{Ylowrank[2]/np.sum(Ygiven <= Ygivenlow)*100:.0f}%\t'
      f'\t{Ylowrank[1]/np.sum(Ygiven <= Ygivenlow)*100:.0f}%\t'
      f'\t{Ylowrank[0]/np.sum(Ygiven <= Ygivenlow)*100:.0f}%')
# 留一法交叉验证，得出最大相关系数对应Np
from sklearn.linear_model import LinearRegression
def loo(X, Y, N=10):
    t = X[0].size
    Np = []
    for i in range(N):
        Y_test = []
        for j in range(t):
            X_t = [[ii[iii] for iii in range(t) if iii != j] for ii in X[:i+1]]
            Y_t = [Y[ii] for ii in range(t) if ii != j]
            reg =LinearRegression().fit(np.array(X_t).T, Y_t)
            Y_test.append(eval(str(reg.predict(np.array([[ii[j]] for ii in X[:i+1]]).T)).strip('[]')))
        Np.append([i + 1, np.corrcoef(Y, Y_test)[0, 1]])
        Y_test.clear()
    return Np

# 选取50组变量进行拟合测试
N = 50      # 设置样本数
T = 1000    # 设置测试次数
Nptest = 6
forecasts = [[] for i in range(11)]
forecasts[10].append(0)
forecasts[10][0] = 0
for i in range(T):
    randomid = np.random.randint(0, Nt, size=N)
    X_test = [i[randomid] for i in Xsamples]
    Y_test = Ysamples[randomid]
    Np = sorted(loo(X_test, Y_test, N=Nptest), key=lambda x: x[1], reverse=True)[0][0]
    Npid = sorted([[i, np.corrcoef(X_test[i], Y_test)[0, 1]] for i in range(len(X_test))], key=lambda x: x[1], reverse=True)[:Np]
    reg = LinearRegression().fit(np.array([X_test[i[0]] for i in Npid]).T, Y_test)
    Y_test = reg.predict(np.array([Xsamples[i[0]] for i in Npid]).T)
    Y_testVar = np.var(Y_test)
    if 1 <= Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.1:
        forecasts[0].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.1 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.2:
        forecasts[1].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.2 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.3:
        forecasts[2].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.3 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.4:
        forecasts[3].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.4 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.5:
        forecasts[4].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.5 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.6:
        forecasts[5].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.6 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.7:
        forecasts[6].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.7 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.8:
        forecasts[7].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.8 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 1.9:
        forecasts[8].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
    elif 1.9 < Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar) <= 2:
        forecasts[9].append(Y_testVar/YsamplesVar/(YgivenVar / YsamplesVar))
        forecasts[10][0] += 1
F = forecasts[10][0]
forecasts = np.array(forecasts)
print(f'过拟合测试（X可解释Y变化的{Fgiven*100:.0f}%）')
print(f'1.00 - 1.10\t{len(forecasts[0])/F*100:.1f}%')
print(f'1.10 - 1.20\t{len(forecasts[1])/F*100:.1f}%')
print(f'1.20 - 1.30\t{len(forecasts[2])/F*100:.1f}%')
print(f'1.30 - 1.40\t{len(forecasts[3])/F*100:.1f}%')
print(f'1.40 - 1.50\t{len(forecasts[4])/F*100:.1f}%')
print(f'1.50 - 1.60\t{len(forecasts[5])/F*100:.1f}%')
print(f'1.60 - 1.70\t{len(forecasts[6])/F*100:.1f}%')
print(f'1.70 - 1.80\t{len(forecasts[7])/F*100:.1f}%')
print(f'1.80 - 1.90\t{len(forecasts[8])/F*100:.1f}%')
print(f'1.90 - 2.00\t{len(forecasts[9])/F*100:.1f}%')

