import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.distributions import chi2

def specx_anal(x,m,alpha1,alpha2):
    '''
    功率谱分析
    :param x：需要分析的时间序列(原始序列，未标准化或距平处理)
    :param m：最大滞后相关长度，m取值范围最好在(n/10)~(n/3)之间，n为样本数，可以多次调整m获得最佳效果，通常取m=n/3
    :param alpha1：红噪音检验信度
    :param alpha2：白噪音检验信度
    :return:
    l：功率谱图的X坐标，对应的周期为2m/l，使用时自己调整tick labels
    Sl：功率谱估计值
    Sr：红噪音
    Sw：白噪音
    r1：落后一个时刻的自相关函数，用于查看使用哪种噪音检验
    '''
    n = x.shape[0]
    x = (x - np.mean(x))/np.std(x)
    r1 = np.zeros((n-6))
    r2 = np.zeros((n-7))
    for i in np.arange(0,n-6):
        r1[i]=np.sum(x[:n-i]*x[i:])/x[:n-i].shape[0]
    for i in np.arange(1,n-6):
        r2[i-1]=np.sum(x[:n-i]*x[i:])/x[:n-i].shape[0]
    r2 = r2[::-1]
    r = np.hstack((r2,r1))
    l = np.arange(0,m+1,1)
    tao = np.arange(1,m,1)
    Sl  = np.zeros((m+1))
    Tl  = np.zeros((m+1))
    S0l = np.zeros((m+1))
    a = np.array((r.shape[0]+1)/2).astype('int32')
    r = r[a-1:a+m]
    a=r[1:-1]*(1+np.cos(np.pi*tao/m))
    for i in np.arange(2,m+1,1):
        Sl[i-1]=(r[0]+np.sum(a*np.cos(l[i-1]*np.pi*tao/m)))/m
    Sl[0]=(r[0]+np.sum(a*np.cos(l[0]*np.pi*tao/m)))/(2*m)
    Sl[-1]=(r[0]+np.sum(a*np.cos(l[-1]*np.pi*tao/m)))/(2*m)
    for i in range(l.shape[0]):
        Tl[i]=2*m/l[i]
    f=(2*n-m/2)/m
    S=np.mean(Sl)
    for i in range(l.shape[0]):
        S0l[i]=S*(1-r[1]*r[1])/(1+r[1]*r[1]-2*r[1]*np.cos(l[i]*np.pi/m))
    x2r = chi2.ppf(1-alpha1,df = f)
    Sr=S0l*x2r/f
    x2w = chi2.ppf(1-alpha2,df = f)
    Sw=S*x2w/f;
    r1=r[1]
    return l,Sl,Sr,Sw,r1

x = np.load("D:\PyFile\paper1\OLS35.npy")
l,Sl,Sr,Sw,r1 = specx_anal(x,x.shape[0]//3,0.1,0.1)
plt.plot(l,Sl,'-b',label='Real')
plt.plot(l,Sr,'--r',label='red noise')
plt.plot(l,np.linspace(Sw,Sw,l.shape[0]),'--m',label='white noise')
plt.legend()
plt.show()
print(r1)