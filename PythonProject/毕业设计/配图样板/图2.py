# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 20:02:33 2023

@author: pc
"""
import numpy as np
import xarray as xr

from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.pyplot as plt
from xarray import DataArray as da

#############################
tf = np.loadtxt('F:\PC\TC\cma.txt',dtype='float')

#############################
select1 = ((1979<=tf[:,1])&(tf[:,1]<=2020)) & ((7<=tf[:,2])&(tf[:,2]<=9)) & ((105.0<=tf[:,-3])&(tf[:,-3]<=180.0)) & ((0.0<=tf[:,-4])&(tf[:,-4]<=30.0))

#############################
tf = tf[select1,:] 

#############################
tf1 = tf[tf[:,-1] == 1,:] 
tf2 = tf[tf[:,-1] == 2,:] 

#############################
nt = 42
time = np.arange(42) + 1979    

#############################
fre1= np.zeros((nt),dtype=int)
fre2= np.zeros((nt),dtype=int)

for k in range(len(tf1)):
    t = int(tf1[k,1]-time[0])
    fre1[t] = fre1[t]+1   

for j in range(len(tf2)):
    t = int(tf2[j,1]-time[0])
    fre2[t] = fre2[t]+1    

#############################
sumlon = np.zeros(nt)
sumlat = np.zeros(nt)

for m in range(len(tf1)):
    t = int(tf1[m,1]-time[0])
    sumlat[t]+=tf1[m,5] 
    sumlon[t]+=tf1[m,6] 

TClon=sumlon/fre1 
TClat=sumlat/fre1 

#############################
pdi= np.zeros((nt))

for k in range(len(tf)):
    v=tf[k,-2]**3 
    t = int(tf[k,1]-time[0])
    pdi[t]+=v
    
#############################
ob_uv=xr.open_dataset(r'G:\reanalysis\ERA5\interp_1964_789_invertlat_yearmean\1940-2023uv.nc')#Dimensions:  (level: 17, lat: 73, lon: 144, time: 876),1948-01-01 1948-02-01 ... 2020-12-01
ob_u2=xr.open_dataset(r'G:\reanalysis\NCEP\interp_1964_789_invertlat_yearmean\uwnd.mon.mean.nc')#Dimensions:  (level: 17, lat: 73, lon: 144, time: 876),1948-01-01 1948-02-01 ... 2020-12-01
ob_v2=xr.open_dataset(r'G:\reanalysis\NCEP\interp_1964_789_invertlat_yearmean\vwnd.mon.mean.nc')#Dimensions:  (level: 17, lat: 73, lon: 144, time: 876),1948-01-01 1948-02-01 ... 2020-12-01

#############################
ob_uwnd1=ob_uv['u'] 
ob_vwnd1=ob_uv['v'] 

ob_uwnd2=ob_u2['uwnd'] 
ob_vwnd2=ob_v2['vwnd'] 

#############################
tt=ob_uwnd1.loc[:,20:0,100:190] 

ob_uwnd1=ob_uwnd1.loc['1979-08-01':'2020-08-28':1]
ob_uwnd2=ob_uwnd2.loc['1979-08-01':'2020-08-28':1].loc[:,850,90:-90,0:360]

Lat=tt['lat'] 
Lon=tt['lon'] 

lat=ob_uwnd1['lat']
lon=ob_uwnd1['lon']

ob_uwnd1=np.mean(ob_uwnd1,axis=1)

ob_uwnd1=np.array(ob_uwnd1);      
ob_uwnd2=np.array(ob_uwnd2); 
     
#############################
nt=42

uu1=ob_uwnd1[:,69:89,100:190]

zlon1= np.full((42,90), np.nan);

for t in range(nt): 
    for i in range(1,20,1):
        for j in range(0,90,1):
            if(uu1[t,i-1,j]*uu1[t,i,j]<0):
                zlon1[t,j]=(Lon[j])  

#############################    
zero_lon=np.array(zlon1)      

#############################
maxlon=np.zeros(nt)
for m in range(nt):
    maxlon[m]=np.nanmax(zero_lon[m,:])

#############################
from scipy.stats import pearsonr

r1,p1=pearsonr(TClon,maxlon)
r2,p2=pearsonr(fre2,maxlon)
r3,p3=pearsonr(pdi,maxlon)

print(r1,r2,r3)
print(p1,p2,p3)

###################定义标准化公式
def standzation(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

maxlon=standzation(maxlon)

TClon=standzation(TClon)
fre2=standzation(fre2)
pdi=standzation(pdi)

##################### #画图部分
fig = plt.figure(figsize=(16,18),dpi=400)
plt.subplots_adjust(wspace=0.1, hspace=0.3)#wspace、hspace左右、上下的间距

#######################第一个子图
ax1=fig.add_subplot(311) 

x = np.arange(1979,2021,1)
ax1.set_ylim([-3,3])

l1=ax1.plot(x,TClon, color='blue',linewidth=3,linestyle='--',marker='o',markersize=6,label='TC Longitude')

#####################网格线
plt.grid(b='true',which='major',axis='both',color='lightgray', linestyle='-.',linewidth=1.5)

font = {'family' : 'Times New Roman','weight' : 'bold','size' :24,'color':'k'}

ax1.text(2018, 3.5, "R = 0.65",color='k',ha="center", va="center",rotation=0.4,fontdict=font)

ax1.text(1979, 2.5, "a)",color='k',ha="center", va="center",rotation=0.4,fontdict=font)

#####################纵坐标
font1 = {'family' : 'Times New Roman','weight' : 'bold','size' :21,'color':'blue'}

plt.ylabel("TC  Longitude",labelpad=8)
plt.ylabel("TC  Longitude",font1)

#####################设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=21)

labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

######################刻度标签
xmajorLocator = MultipleLocator(10)
xminorLocator = MultipleLocator(2)
ymajorLocator = MultipleLocator(1)
yminorLocator = MultipleLocator(1)

ax1.xaxis.set_major_locator(xmajorLocator)#y轴最大刻度
ax1.xaxis.set_minor_locator(xminorLocator)#y轴最小刻度
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度

######################边框粗细
ax1.spines['top'].set_linewidth('2.0')#设置边框线宽为2.0 
ax1.spines['bottom'].set_linewidth('2.0')#设置边框线宽为2.0
ax1.spines['left'].set_linewidth('2.0')#设置边框线宽为2.0
ax1.spines['right'].set_linewidth('2.0')#设置边框线宽为2.0                

######################最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=13,width=2.3,color='k')#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=7,width=1.8,color='k')#最小刻度长度，宽度设置

ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#######################第二个纵坐标
ax2=ax1.twinx()
ax2.set_ylim([-3,3]) #y轴范围
l2=plt.plot(x, maxlon, color='green',linewidth=3,linestyle='--',marker='s',markersize=6,label='MT Longitude')

######################设置坐标刻度值的大小以及刻度值的字体
labels1 = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]

plt.tick_params(labelsize=21)

font2 = {'family' : 'Times New Roman','weight' : 'bold','size' :21,'color':'green'}

ax2.set_ylabel('NPMT  Longitude ', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
plt.ylabel("NPMT  Longitude",labelpad=8)

######################刻度标签
ax2.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax2.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
        
######################最大刻度、最小刻度的刻度线长短，粗细设置
ax2.tick_params(which='major', length=13,width=2,color='k')#最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=6,width=1.8,color='k')#最小刻度长度，宽度设置

ax2.tick_params(which='both',bottom=True,top=False,left=False,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

######################图例字体大小
plt.rcParams.update({'font.size':18}) 

#######################第二个子图
ax3=fig.add_subplot(312) 

ax3.set_ylim([-3,3])

l1=ax3.plot(x,fre2, color='blue',linewidth=3,linestyle='--',marker='o',markersize=6,label='TC frequency')

#####################网格线
plt.grid(b='true',which='major',axis='both',color='lightgray', linestyle='-.',linewidth=1.5)

ax3.text(2018, 3.5, "R = 0.73",color='k',ha="center", va="center",rotation=0.4,fontdict=font)
ax3.text(1979, 2.5, "b)",color='k',ha="center", va="center",rotation=0.4,fontdict=font)

#####################纵坐标
plt.ylabel("TC frequency",labelpad=8)
plt.ylabel("TC frequency",font1)

#####################设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=21)

labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

######################刻度标签
xmajorLocator = MultipleLocator(10)
xminorLocator = MultipleLocator(2)
ymajorLocator = MultipleLocator(1)
yminorLocator = MultipleLocator(1)

ax3.xaxis.set_major_locator(xmajorLocator)#y轴最大刻度
ax3.xaxis.set_minor_locator(xminorLocator)#y轴最小刻度
ax3.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax3.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度

######################边框粗细
ax3.spines['top'].set_linewidth('2.0')#设置边框线宽为2.0 
ax3.spines['bottom'].set_linewidth('2.0')#设置边框线宽为2.0
ax3.spines['left'].set_linewidth('2.0')#设置边框线宽为2.0
ax3.spines['right'].set_linewidth('2.0')#设置边框线宽为2.0                
                
######################最大刻度、最小刻度的刻度线长短，粗细设置
ax3.tick_params(which='major', length=13,width=2.3,color='k')#最大刻度长度，宽度设置，
ax3.tick_params(which='minor', length=7,width=1.8,color='k')#最小刻度长度，宽度设置

ax3.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#######################第二个纵坐标
ax4=ax3.twinx()
ax4.set_ylim([-3,3]) #y轴范围
l2=plt.plot(x, maxlon, color='green',linewidth=3,linestyle='--',marker='s',markersize=6,label='MT Longitude')

######################设置坐标刻度值的大小以及刻度值的字体
labels1 = ax4.get_xticklabels() + ax4.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]

plt.tick_params(labelsize=21)

ax4.set_ylabel('NPMT  Longitude ', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
plt.ylabel("NPMT  Longitude",labelpad=8)

######################刻度标签
ax4.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax4.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
        
######################最大刻度、最小刻度的刻度线长短，粗细设置
ax4.tick_params(which='major', length=13,width=2,color='k')#最大刻度长度，宽度设置，
ax4.tick_params(which='minor', length=6,width=1.8,color='k')#最小刻度长度，宽度设置

ax4.tick_params(which='both',bottom=True,top=False,left=False,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外
 
######################图例字体大小
plt.rcParams.update({'font.size':18}) 

#######################第三个子图
ax5=fig.add_subplot(313) #用在循环里面就是47张单独的图

ax5.set_ylim([-3,3])

l1=ax5.plot(x,pdi, color='blue',linewidth=3,linestyle='--',marker='o',markersize=6,label='TC PDI')

#####################网格线
plt.grid(b='true',which='major',axis='both',color='lightgray', linestyle='-.',linewidth=1.5)

ax5.text(2018, 3.5, "R = 0.76",color='k',ha="center", va="center",rotation=0.4,fontdict=font)
ax5.text(1979, 2.5, "c)",color='k',ha="center", va="center",rotation=0.4,fontdict=font)

#####################纵坐标
plt.ylabel("TC  PDI",labelpad=8)
plt.ylabel("TC  PDI",font1)

#####################设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=21)

labels = ax5.get_xticklabels() + ax5.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

######################刻度标签
xmajorLocator = MultipleLocator(10)
xminorLocator = MultipleLocator(2)
ymajorLocator = MultipleLocator(1)
yminorLocator = MultipleLocator(1)

ax5.xaxis.set_major_locator(xmajorLocator)#y轴最大刻度
ax5.xaxis.set_minor_locator(xminorLocator)#y轴最小刻度
ax5.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax5.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度

######################边框粗细
ax5.spines['top'].set_linewidth('2.0')#设置边框线宽为2.0 
ax5.spines['bottom'].set_linewidth('2.0')#设置边框线宽为2.0
ax5.spines['left'].set_linewidth('2.0')#设置边框线宽为2.0
ax5.spines['right'].set_linewidth('2.0')#设置边框线宽为2.0                
                
######################最大刻度、最小刻度的刻度线长短，粗细设置
ax5.tick_params(which='major', length=13,width=2.3,color='k')#最大刻度长度，宽度设置，
ax5.tick_params(which='minor', length=7,width=1.8,color='k')#最小刻度长度，宽度设置

ax5.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#######################第二个纵坐标
ax6=ax5.twinx()
ax6.set_ylim([-3,3]) #y轴范围
l2=plt.plot(x, maxlon, color='green',linewidth=3,linestyle='--',marker='s',markersize=6,label='MT Longitude')

######################设置坐标刻度值的大小以及刻度值的字体
labels1 = ax6.get_xticklabels() + ax6.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels1]

plt.tick_params(labelsize=21)

ax6.set_ylabel('NPMT  Longitude ', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
plt.ylabel("NPMT  Longitude",labelpad=8)

######################刻度标签
ax6.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax6.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
        
######################最大刻度、最小刻度的刻度线长短，粗细设置
ax6.tick_params(which='major', length=13,width=2,color='k')#最大刻度长度，宽度设置，
ax6.tick_params(which='minor', length=6,width=1.8,color='k')#最小刻度长度，宽度设置

ax6.tick_params(which='both',bottom=True,top=False,left=False,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

######################图例字体大小
plt.rcParams.update({'font.size':18}) 
plt.rcParams.update({'font.family':'Times New Roman'})#图例大小

plt.savefig('F:\code\date\submit\Fig2.jpg',bbox_inches = 'tight')
