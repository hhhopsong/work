# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:09:52 2023

@author: pc
"""
import numpy as np
import xarray as xr
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

######################
ob_pr1=xr.open_dataset(r'G:\reanalysis\observe_pr\interp_1979_789_invertlat_timmean\CMAP_mean_pr.nc')   #1979-2023
ob_pr2=xr.open_dataset(r'G:\reanalysis\observe_pr\interp_1979_789_invertlat_timmean\GPCP_mean_pr.nc')   #1979-2023

######################
ob_pr1=ob_pr1['precip']  
ob_pr2=ob_pr2['precip']  

######################
lon=ob_pr1['lon'];  
lat=ob_pr1['lat'] 

tt=ob_pr1.loc[:,20:0,100:190] 

Lat=tt['lat']
Lon=tt['lon']

######################
ob_pr1=np.mean(ob_pr1,axis=(0));
ob_pr2=np.mean(ob_pr2,axis=(0));

######################
ob_pr1=ob_pr1
ob_pr2=ob_pr2

######################################################Historical 降水
path1= r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/his/pr"  
files = os.listdir(path1)

pr1=np.zeros((35,180,360))
m=0
for file in files:
  f=xr.open_dataset(path1 + "\\" + file)   
  t=f['pr'] 
  t=t.loc[:,90:-90,0:360] 
  t=np.mean(t,axis=(0))
  pr1[m,:,:]=t;m=m+1

pr1=pr1*24*3600

##############################################Historical 850hpa纬向风
path3 = r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/his/850hpa_ua"  
files = os.listdir(path3)

u1=np.zeros((35,180,360))
m=0
for file in files:
  f=xr.open_dataset(path3 + "\\" + file)   
  ua=f['ua'] 
  ua=ua.loc[:,:,90:-90,0:360] 
  ua=np.mean(ua,axis=(0,1))
  u1[m,:,:]=ua;m=m+1

################################################SSP585 850hpa经向风
path4 = r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/ssp585/850hpa_ua"  
files = os.listdir(path4)

u2=np.zeros((35,180,360))
m=0
for file in files:
  f=xr.open_dataset(path4 + "\\" + file)   
  ua=f['ua'] 
  ua=ua.loc[:,:,90:-90,0:360] 
  ua=np.mean(ua,axis=(0,1))
  u2[m,:,:]=ua;m=m+1
  
######################
uu1=u1[:,69:89,100:190]
uu2=u2[:,69:89,100:190]

zlon1= np.full((35,90), np.nan);zlon2= np.full((35,90), np.nan)

for t in range(35): 
    for i in range(1,20,1):
        for j in range(0,90,1):
            if(uu1[t,i-1,j]*uu1[t,i,j]<0):
                zlon1[t,j]=(Lon[j])      

for t in range(35): 
    for i in range(1,20,1):
        for j in range(0,90,1):
            if(uu2[t,i-1,j]*uu2[t,i,j]<0):
                zlon2[t,j]=(Lon[j])   

######################     
zero_lon1=np.array(zlon1);zero_lon2=np.array(zlon2)      

######################
maxlon1=np.zeros(35)
maxlon2=np.zeros(35)

for m in range(35):
    maxlon1[m]=np.nanmax(zero_lon1[m,:])
    maxlon2[m]=np.nanmax(zero_lon2[m,:])

######################
deltaylon=maxlon2-maxlon1    
deltaylon[26]=-4

######################
tropical_influence=np.nanmean(pr1[:,59:119,:],axis=(1,2))
reduce=np.zeros((35,180,360)) 

for p in range(180):
    for q in range(360):
        reduce[:,p,q]=pr1[:,p,q]-tropical_influence[:]

######################
reduce_ob1=ob_pr1-np.nanmean(ob_pr1[59:119,:])
reduce_ob2=ob_pr2-np.nanmean(ob_pr2[59:119,:])

######################
from scipy import stats
regT=np.zeros((180,360))

for m in range(180):
    for n in range(360):
        regT[m,n], b, r, p, std = stats.linregress(deltaylon,reduce[:,m,n])

Pr_his=reduce[:,59:119,90:280] #20°S-30°N,160-210 印度洋-中太平洋区域

Pr_ob1=reduce_ob1[59:119,90:280]
Pr_ob2=reduce_ob2[59:119,90:280]

Pr_ob1=np.array(Pr_ob1)
Pr_ob2=np.array(Pr_ob2)

######################
Pr_MT=regT[59:119,90:280]

######################
Pr_his[np.isnan(Pr_his)]=0 
Pr_MT[np.isnan(Pr_MT)]=0  

Pr_ob1[np.isnan(Pr_ob1)]=0 
Pr_ob2[np.isnan(Pr_ob2)]=0 

######################
Pr_his=np.reshape(Pr_his,(35,11400)) 
Pr_MT =np.reshape(Pr_MT,(11400))     

Pr_ob1=np.reshape(Pr_ob1,(11400))
Pr_ob2=np.reshape(Pr_ob2,(11400))

######################
T=np.dot(Pr_his[:,:],Pr_MT[:])

T_ob1=np.dot(Pr_ob1[:],Pr_MT[:])
T_ob2=np.dot(Pr_ob2[:],Pr_MT[:])

MMET=np.mean(T)
MMEMT=np.mean(deltaylon)

######################
k, b, r, p, std = stats.linregress(T,deltaylon)

y=k*T+b;

print(k, b, r, p)

##########################
X=T;
Y=deltaylon

Xmean=MMET
Ymean=MMEMT

#############################
σX=np.std(T)
σY=np.std(deltaylon)

#############################
from scipy.stats import pearsonr

ρ,p = pearsonr(T, deltaylon) 

r=(σY/σX)*ρ 

#############################
Xo_mean=(T_ob1+T_ob2)/2 #

#############################
σO2=((T_ob1-Xo_mean)**2+(T_ob2-Xo_mean)**2)/2

SNR=(σX**2)/σO2

#############################
Ycon_mean=Ymean+(1+SNR**-1)**(-1)*r*(Xo_mean-Xmean)

σY2=σY**2 

σY2_con=(1-(ρ**2)/(1+SNR**-1))*(σY**2) 

y=Ymean+r*(X-Xmean) 

#############################
reducevar=1-σY2_con/σY2 

#############################
u1 = Ymean; 
u2 = Ycon_mean;

#############################
σ1 = σY 
σ2 = np.sqrt(σY2_con) 

#############################
def pdf(x,u,sigma): 
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-u)**2/(2*sigma**2))
    
#############################
x=np.linspace(-20,30)

#############################
y11=pdf(x,u1,σ1)  
y12=pdf(x,u2,σ2)  

#############################
fig = plt.figure(figsize=(11,5),dpi=400)

#########################################回归
ax1 = fig.add_subplot(121)
plt.subplots_adjust(wspace=0.2, hspace=0.1)
  
plt.scatter(T,deltaylon,color='cyan',marker='o',s=9) #NCEP
plt.scatter(Xmean,Ymean,color='k',marker='o',s=55) #NCEP

plt.plot(T, y, color='gray',linewidth=2,label='R = 0.69, p < 0.001')

plt.axvline(x=T_ob1, color='red', linestyle='--',linewidth=1.3, label='CMAP')#NCEP T1
plt.axvline(x=T_ob2, color='orange', linestyle='--',linewidth=1.3, label='GPCP')#NCEP T1

font = {'family' : 'Times New Roman','weight' : 'normal','size' : 11}

plt.rcParams.update({'font.size':9})#图例大小
plt.legend(loc='upper left',ncol=1)   #指定图例字体

plt.rcParams['font.sans-serif']=['SimHei'] #显示中文

# #图例字体风格
plt.legend(prop=font) #图例字体
plt.rcParams.update({'font.family':'Times New Roman'})#图例大小

##########################################散点加标签
model_num= np.arange(1,36,1)

for i in range(len(T)):
    ax1.text(T[i]*1, deltaylon[i]*1, model_num[i], fontsize=9, color = "gray", 
              style = "italic", weight = "light", verticalalignment='bottom', 
              horizontalalignment='center',rotation=0) #给散点加标签

##########################################95%置信区间
import seaborn as sns
sns.regplot(x=T, y=deltaylon, ci=95,color='gray',marker='None')

##########################################标注文字

font2 = {'family' : 'Times New Roman','weight' : 'bold','size' : 14}#设置横纵坐标的名称以及对应字体格式family'
ax1.set_xlabel('Pr index ', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
ax1.set_ylabel('NPMT change ', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来

##########################################最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=5,width=1,color='k')#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=0,width=0.3,color='k')#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

###########################################边框设置
bwith = 1.3 #边框宽度设置为2
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)

##########################################设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=12)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

####################################################概率密度曲线
ax2 = fig.add_subplot(122)
xxx=np.zeros(35) 
 
plt.scatter(deltaylon,xxx,color='gray',marker='o',s=6) #NCEP

plt.plot(x, y11, color='gray',linewidth=1.5,label='Original  (mean=4.11,σ=7.28)')
plt.plot(x, y12, color='red',linewidth=1.5,label='Constrain(mean=6.17,σ=5.24)')

plt.rcParams.update({'font.size':9})#图例大小

plt.legend(prop=font) #图例字体
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文
plt.legend(loc='upper left',ncol=1)  

ax2.set_xlabel('NPMT change ', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
ax2.set_ylabel('PDF ', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来

#最大刻度、最小刻度的刻度线长短，粗细设置
ax2.tick_params(which='major', length=5,width=1,color='k')#最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=0,width=0.3,color='k')#最小刻度长度，宽度设置
ax2.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=12)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

###########################################边框设置
ax2.spines['top'].set_linewidth(bwith)
ax2.spines['bottom'].set_linewidth(bwith)
ax2.spines['left'].set_linewidth(bwith)
ax2.spines['right'].set_linewidth(bwith)

#模式名称
f=open('F:\code\modelname(35).txt', encoding='gbk')

name=[]
for line in f:
    name.append(line.strip())
# print(name)

font3 = {'family' : 'Times New Roman','weight' : 'normal','size' : 9}
font4 = {'family' : 'Times New Roman','weight' : 'normal','size' : 14}

for i in range(0,18,1): #相对于图的横坐标0来的
    ax2.text(35, 0.0777-0.0048*i, name[i+0], 
              color = "k", style = "italic", weight = "light", 
            verticalalignment='center', horizontalalignment='left',rotation=0,fontdict=font3) #给散点加标签

for i in range(0,17,1): #相对于图的横坐标0来的
    ax2.text(53, 0.0777-0.0048*i, name[i+18], 
              color = "k", style = "italic", weight = "light", 
            verticalalignment='center', horizontalalignment='left',rotation=0,fontdict=font3) #给散点加标签

ax1.text(-400, -17, "y = 0.013* x - 0.39",color='k',ha="center", va="center",rotation=0,fontdict=font4)


plt.savefig('F:\code\date\submit\Fig8.jpg',bbox_inches = 'tight')
