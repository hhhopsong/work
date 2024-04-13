# -*- coding: utf-8 -*-
"""
Created on Mon May 22 15:09:52 2023

@author: pc
"""
import numpy as np
import xarray as xr
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter#专门提供经纬度的
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#####################################
ob_sst1=xr.open_dataset(r'G:\reanalysis\NCEP\interp_1964_789_invertlat\sst.mon.mean.nc')
ob_sst1=ob_sst1['sst']

#####################################
lon=ob_sst1['lon'];
lat=ob_sst1['lat'] 

tt=ob_sst1.loc[:,20:0,100:190] 

Lat=tt['lat']
Lon=tt['lon']

#####################################
path1 = r"G:/interp_historical/interp_1964_789_timmean_invertlat/remapdis_tos"  
files = os.listdir(path1)

tos1=np.zeros((47,180,360))
m=0
for file in files:
  f=xr.open_dataset(path1 + "\\" + file)   
  t=f['tos'] 
  t=t.loc[:,90:-90,0:360] 
  t=np.mean(t,axis=(0))
  tos1[m,:,:]=t;m=m+1

######################################################Historical 降水
path2 = r"G:/interp_historical/interp_1964_789_timmean_invertlat/pr"   
files = os.listdir(path2)

pr1=np.zeros((47,180,360))
m=0
for file in files:
  f=xr.open_dataset(path2 + "\\" + file)   
  t=f['pr'] 
  t=t.loc[:,90:-90,0:360] 
  pr1[m,:,:]=t;m=m+1

##############################################Historical 850hpa纬向风
path3 = r"G:/interp_historical/interp_1964_789_timmean_invertlat/850hpa_ua"  
files = os.listdir(path3)

u1=np.zeros((47,180,360))
m=0
for file in files:
  f=xr.open_dataset(path3 + "\\" + file)   
  ua=f['ua'] 
  ua=ua.loc[:,:,90:-90,0:360] 
  ua=np.mean(ua,axis=(0,1))
  u1[m,:,:]=ua;m=m+1

#####################################
uu1=u1[:,69:89,100:190]

zlon1= np.full((47,90), np.nan);

for t in range(47): 
    for i in range(1,20,1):
        for j in range(0,90,1):
            if(uu1[t,i-1,j]*uu1[t,i,j]<0):
                zlon1[t,j]=(Lon[j])    

#####################################     
zero_lon1=np.array(zlon1);      

#####################################
maxlon=np.zeros(47)

for t in range(47):
    maxlon[t]=np.nanmax(zero_lon1[t,:])

bais=maxlon-148

MME1=np.mean(maxlon,axis=0)
MME2=np.mean(bais,axis=0)

#####################################
tos1=np.nanmean(tos1[:,72:79,180:210],axis=(1,2))
pr1=np.nanmean(pr1[:,69:79,180:210],axis=(1,2))

#####################################
pr1=pr1*24*3600

#####################################
from scipy import stats

for n in range(35):
   k1, b1, r1, p1, std1 = stats.linregress(tos1,bais)

for n in range(35):
   k2, b2, r2, p2, std2 = stats.linregress(pr1,bais)

y1=tos1*k1+b1;
y2=pr1*k2+b2;

 
font3 = {'family' : 'Times New Roman','weight' : 'normal','size' : 9}

##############################################画图部分
fig = plt.figure(figsize=(14,12),dpi=400)

#############################################子图距离
plt.subplots_adjust(wspace=0.2, hspace=0.15)#wspace、hspace左右、上下的间距

#######################并列子图空白区域
spec=fig.add_gridspec(nrows=2,ncols=2,width_ratios=[1,1],height_ratios=[1,1])

#######################第一个子图
ax1=fig.add_subplot(spec[:1,0:2]) #选中了一二行和二三列
 
x = np.arange(1,48,1)

ax1.set_xlim([0,48])
ax1.set_ylim([-36,36])


b1=ax1.bar(x, bais ,color='crimson')

for bar,height in zip(b1,bais):
    if height<0:
        bar.set(color='dodgerblue')

###############################显示数据标签
for a, b in zip(x, bais):
  plt.text(a, b, '%.0f' % b, color='k',ha='center', va='bottom', fontsize=14)
        
############横坐标刻度标签
ax1.set_xticks(np.arange(1, 48, 5))

font = {'family' : 'Times New Roman','weight' : 'bold','size' :20,'color':'k'}

ax1.set_ylabel('NPMT bais ', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来

#################################设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=18)

labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#################################刻度标签
ymajorLocator = MultipleLocator(10)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
yminorLocator = MultipleLocator(2)
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
           
#################################最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=9,width=1,color='k')#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=0,width=0.8,color='k')#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

###########################################边框设置
bwith = 1.7 #边框宽度设置为2
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)

#######################第二个子图
ax2 = fig.add_subplot(223)
 
plt.scatter(tos1,bais,color='darkorange',marker='o',s=33) #NCEP

plt.plot(tos1, y1, color='gray',linewidth=2)

##########################################散点加标签
model_num= np.arange(1,48,1)

for i in range(len(bais)):
    ax2.text(tos1[i]*1, bais[i]*1, model_num[i], fontsize=14, color = "gray", 
              style = "italic", weight = "light", verticalalignment='bottom', 
              horizontalalignment='center',rotation=0) #给散点加标签

##########################################95%置信区间
import seaborn as sns
sns.regplot(x=tos1, y=bais, ci=95,color='gray',marker='None')

##########################################标注文字
ax2.set_xlabel('Hist SST', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
ax2.set_ylabel('NPMT bais ', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来

plt.xlabel("Hist SST",labelpad=12)
# plt.ylabel("MT bais",labelpad=12)

##########################################最大刻度、最小刻度的刻度线长短，粗细设置
ax2.tick_params(which='major', length=9,width=1,color='k')#最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=0,width=0.8,color='k')#最小刻度长度，宽度设置
ax2.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

###########################################边框设置
ax2.spines['top'].set_linewidth(bwith)
ax2.spines['bottom'].set_linewidth(bwith)
ax2.spines['left'].set_linewidth(bwith)
ax2.spines['right'].set_linewidth(bwith)

##########################################设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=18)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#######################第三个子图
ax3 = fig.add_subplot(224)
 
plt.scatter(pr1,bais,color='darkorange',marker='o',s=33) #NCEP

plt.plot(pr1, y2, color='gray',linewidth=2)

##########################################散点加标签
model_num= np.arange(1,48,1)

for i in range(len(bais)):
    ax3.text(pr1[i]*1, bais[i]*1, model_num[i], fontsize=14, color = "gray", 
              style = "italic", weight = "light", verticalalignment='bottom', 
              horizontalalignment='center',rotation=0) #给散点加标签

##########################################95%置信区间
import seaborn as sns
sns.regplot(x=pr1, y=bais, ci=95,color='gray',marker='None')

##########################################标注文字
ax3.set_xlabel('Hist Pr ', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
ax3.set_ylabel('NPMT bais ', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
plt.xlabel("Hist Pr",labelpad=12)

# #################设置y轴刻度标签保留两位小数
# ax3.set_xticks(np.arange(0,10*1e-05, 2*1e-05))

# ##########################################最大刻度、最小刻度的刻度线长短，粗细设置
ax3.tick_params(which='major', length=9,width=1,color='k')#最大刻度长度，宽度设置，
ax3.tick_params(which='minor', length=0,width=0.8,color='k')#最小刻度长度，宽度设置
ax3.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

##########################################设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=18)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

###########################################边框设置
ax3.spines['top'].set_linewidth(bwith)
ax3.spines['bottom'].set_linewidth(bwith)
ax3.spines['left'].set_linewidth(bwith)
ax3.spines['right'].set_linewidth(bwith)

####模式名称
f=open('F:\code\modelname(47).txt', encoding='gbk')

name=[]
for line in f:
    name.append(line.strip())

font3 = {'family' : 'Times New Roman','weight' : 'normal','size' : 14}
tt = {'family' : 'Times New Roman','weight' : 'bold','size' : 21}

###############################输出为一列
for i in range(0,24,1): #相对于图的横坐标0来的
    ax1.text(49.7, 34-6.61*i, name[i+0], 
            font=font3,color = "k", style = "italic", weight = "light", 
            verticalalignment='center', horizontalalignment='left',rotation=0,fontdict=font3) #给散点加标签

###############################输出为一列
for i in range(0,23):
    ax1.text(58.7, 34-6.61*i, name[i+24], 
            font=font3, color = "k", style = "italic", weight = "light", 
            verticalalignment='center', horizontalalignment='left',rotation=0) #给散点加标签

ax3.text(-11.2, 120, "a)",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)
ax3.text(-11.2, 30,  "b)",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)
ax3.text(0.2,   30,  "c)",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)

ax3.text(-3.5, -31,  "R = 0.62",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)
ax3.text(8,  -31,  "R = 0.76",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)

plt.savefig('F:\code\date\submit\Fig3.jpg',bbox_inches = 'tight')
