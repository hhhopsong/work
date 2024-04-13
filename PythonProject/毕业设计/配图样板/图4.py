# -*- coding: utf-8 -*-
"""
Created on Wed Mar  8 20:07:52 2023

@author: pc
"""
import numpy as np
import xarray as xr
import matplotlib as mpl
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter#专门提供经纬度的
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from xarray import DataArray as da
import os
import cmaps

#####################
xr_uwnd1=xr.open_dataset(r'G:\reanalysis\NCEP\interp_1964_789_invertlat_yearmean\uwnd.mon.mean.nc')#Dimensions:  (level: 17, lat: 73, lon: 144, time: 876),1948-01-01 1948-02-01 ... 2020-12-01

uwnd1=xr_uwnd1['uwnd'].loc[:,850,:,:]
uwnd2=xr_uwnd1['uwnd'].loc[:,850,20:0,100:190]

lat=uwnd1['lat']
lon=uwnd1['lon']

######################################################Historical 海温
path1 = r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/his/tos"  
files = os.listdir(path1)

tos1=np.zeros((35,180,360))
i=0
for file in files:
  f=xr.open_dataset(path1 + "\\" + file)   
  t=f['tos'] 
  t=t.loc[:,90:-90,0:360] 
  t=np.mean(t,axis=(0))
  tos1[i,:,:]=t;i=i+1

#########################################################SSP585 海温
path2 = r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/ssp585/tos"  
files = os.listdir(path2)

tos2=np.zeros((35,180,360))
l=0
for file in files:
  f=xr.open_dataset(path2 + "\\" + file)   
  t=f['tos'] 
  t=t.loc[:,90:-90,0:360] 
  t=np.mean(t,axis=(0))
  tos2[l,:,:]=t;l=l+1

##############################################Historical 850hpa纬向风
path3 = r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/his/850hpa_ua"  
files = os.listdir(path3)

u1=np.zeros((35,180,360))
j=0
for file in files:
  f=xr.open_dataset(path3 + "\\" + file)   
  ua=f['ua'] 
  ua=ua.loc[:,:,90:-90,0:360] 
  ua=np.mean(ua,axis=(0,1))
  u1[j,:,:]=ua;j=j+1

##############################################Historical 850hpa纬向风
path4 = r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/ssp585/850hpa_ua"  
files = os.listdir(path4)

u2=np.zeros((35,180,360))
j=0
for file in files:
  f=xr.open_dataset(path4 + "\\" + file)   
  ua=f['ua'] 
  ua=ua.loc[:,:,90:-90,0:360] 
  ua=np.mean(ua,axis=(0,1))
  u2[j,:,:]=ua;j=j+1

##############################################Historical 850hpa纬向风
path7 = r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/his/pr"  
files = os.listdir(path7)

pr1=np.zeros((35,180,360))
i=0
for file in files:
  f=xr.open_dataset(path7 + "\\" + file)   
  t=f['pr'] 
  t=t.loc[:,90:-90,0:360] 
  t=np.mean(t,axis=(0))
  pr1[i,:,:]=t;i=i+1

######################################################Historical 海温
path8 = r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/ssp585/pr"  
files = os.listdir(path8)

pr2=np.zeros((35,180,360))
i=0
for file in files:
  f=xr.open_dataset(path8 + "\\" + file)   
  t=f['pr'] 
  t=t.loc[:,90:-90,0:360] 
  t=np.mean(t,axis=(0))
  pr2[i,:,:]=t;i=i+1

######################################
deltaylon=np.loadtxt('F:\code\MT\deltaylon(35).txt', delimiter = ',')

regT=np.loadtxt('F:\code\MT\hgT(deltaylon_deltayT_35).txt', delimiter = ',')
regp=np.loadtxt('F:\code\MT\hgp(deltaylon_deltayp_35).txt', delimiter = ',')
regu=np.loadtxt('F:\code\MT\hgu(deltaylon_deltayu_35).txt', delimiter = ',')
regv=np.loadtxt('F:\code\MT\hgv(deltaylon_deltayv_35).txt', delimiter = ',')

deltayT=tos2-tos1
deltayp=pr2-pr1 
deltayu=u2-u1 

#####################定义标准化公式
def standzation(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

deltaylon=standzation(deltaylon)

#################################
from scipy import stats

TT=regT*1
PP=regp*1

#################################
deltayT1=deltayT[:,64:94,150:210]
deltayT2=deltayT[:,64:94,90:140] 
    
deltayp1=deltayp[:,64:94,150:210] 
deltayp2=deltayp[:,64:94,90:140] 
 
deltayT1=np.nanmean(deltayT1,axis=(1,2))
deltayT2=np.nanmean(deltayT2,axis=(1,2))

deltayp1=np.nanmean(deltayp1,axis=(1,2))
deltayp2=np.nanmean(deltayp2,axis=(1,2))

grdT=deltayT1-deltayT2 
grdp=deltayp1-deltayp2
grdp=grdp*24*3600

k1, b1, r1, p1, std = stats.linregress(grdT,deltaylon)
k2, b2, r2, p2, std = stats.linregress(grdp,deltaylon)

y1=grdT*k1+b1;
y2=grdp*k2+b2;


###################定义mask函数
from global_land_mask import globe

def mask(a, lat_south, lat_north, lon_west, lon_east):
    latitude = np.linspace(lat_north, lat_south, a.shape[0])
    longitude = np.linspace(lon_west, lon_east, a.shape[1])
    for i in range(0, len(latitude), 1):
        for j in range(0, len(longitude), 1):
            if longitude[j] > 180:
                longitude[j] = longitude[j] - 360
            if globe.is_land(latitude[i], longitude[j]):
                a[i,j] = np.nan
    return a

TT=mask(TT, -90, 90, 0, 359) #带入参数     
PP=mask(PP, -90, 90, 0, 359) #带入参数     
regu=mask(regu, -90, 90, 0, 359) #带入参数     
regv=mask(regv, -90, 90, 0, 359) #带入参数     

################################## #以下为地图投影以及坐标轴的设置
fig = plt.figure(figsize=(11,13),dpi=400)

plt.subplots_adjust(wspace=0.4, hspace=0.35)#wspace、hspace左右、上下的间距

#######################并列子图空白区域
spec=fig.add_gridspec(nrows=3,ncols=2,width_ratios=[1,1],height_ratios=[1,1,1])

ax1=fig.add_subplot(spec[:1,0:2],projection=ccrs.PlateCarree(central_longitude=180)) #选中了一二行和二三列
 
proj = ccrs.PlateCarree(central_longitude=180)##设置一个圆柱投影坐标，中心经度180°E
extent=[30,330,-30,60]#设置地图边界范围

xticks=np.arange(extent[0],extent[1]+1,30)
yticks=np.arange(extent[2],extent[3]+1,15)

plt.rcParams['font.sans-serif']=['Times New Roman']

#地图相关设置，包括边界，河流，海岸线，坐标
ax1.set_extent(extent, crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'),color='gray')
ax1.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')#添加陆地并且陆地部分全部填充成浅灰色
ax1.add_feature(cfeature.LAKES, alpha=0.9)

#横纵坐标设置
ax1.set_xticks(xticks, crs=ccrs.PlateCarree())
ax1.set_yticks(yticks, crs=ccrs.PlateCarree())

lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()

ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)

#  相关系数
fig1 =ax1.contourf(lon,lat,TT,np.linspace(-0.5,0.5,11), zorder=0, extend = 'both', transform=ccrs.PlateCarree(),cmap=cmaps.BlueWhiteOrangeRed)

########################################根据经纬度绘制矩形框
from matplotlib.patches import Rectangle

###########################序号矩形框
start_point = (-328, 43)#矩形框起点
RE1=Rectangle(start_point,18, 15,linewidth=3,linestyle='-' ,zorder=2,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax1.add_patch(RE1)  #显示矩形框

font = {'family' : 'Times New Roman','weight' : 'bold','size' :20,'color':'k'}
tt = {'family' : 'Times New Roman','weight' : 'normal','size' :14,'color':'k'}

#箭头矩形框
start_point = (296, -28.8)#矩形框起点
RE2=Rectangle(start_point,32, 17.8,linewidth=2,linestyle='-' ,zorder=1,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax1.add_patch(RE2)  #显示矩形框

#中太平洋梯度矩形框
start_point = (150, -5)#矩形框起点
RE2=Rectangle(start_point,60, 30,linewidth=2.1,linestyle='-' ,zorder=1,\
              edgecolor='k',facecolor='None', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax1.add_patch(RE2)  #显示矩形框

##海洋性大陆梯度矩形框
start_point = (90, -5)#矩形框起点
RE2=Rectangle(start_point,50, 30,linewidth=2.1,linestyle='-' ,zorder=1,\
              edgecolor='k',facecolor='None', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax1.add_patch(RE2)  #显示矩形框

# #################求相关区域
# start_point = (170, 2)#矩形框起点
# RE2=Rectangle(start_point,40, 8,linewidth=2,linestyle='-' ,zorder=1,\
#               edgecolor='gray',facecolor='None', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
# ax1.add_patch(RE2)  #显示矩形框

ax1.text(-139, 52, "a)",color='k',ha="center", va="center",rotation=0.4,fontdict=font)
# ax1.text(80, -52, "1e-1",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)

font2 = {'family' : 'serif', 'weight' : 'normal', 'size'   : 14}   

###画回归的风场
n=7
Q1=ax1.quiver(lon[::n],lat[::n],regu[::n,::n],regv[::n,::n],
              color='gray',pivot='mid',width=0.0042,
              scale=10,headwidth=4,headlength=6,headaxislength=4,transform=ccrs.PlateCarree())

ax1.quiverkey(Q1,  X=0.94, Y=0.07, U=0.5,angle = 0,  
              label='0.5m/s', labelpos='N', edgecolor='white', facecolor='g',
              labelcolor = 'k', fontproperties = font2,linewidth=0.1)#linewidth=1为箭头的大小

#####################箭头边框
import matplotlib.patheffects as path_effects

Q1.set_path_effects([path_effects.PathPatchEffect
                      (edgecolor='white', facecolor='g', 
                        linewidth= 1.4 )])

ax1.set_xticks(xticks, crs=ccrs.PlateCarree())
ax1.set_yticks(yticks, crs=ccrs.PlateCarree())

ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)

################################隐藏刻度标签
# ax1.axes.xaxis.set_ticklabels([]) 

#刻度线设置
xmajorLocator = MultipleLocator(30)#先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
xminorLocator = MultipleLocator(10)
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ymajorLocator = MultipleLocator(30)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
yminorLocator = MultipleLocator(10)
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度

#最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=8,width=1,color='k')#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=3,width=0.8,color='k')#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=16)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

###########################################边框设置
bwith = 1.2 #边框宽度设置为2
ax1.spines['top'].set_linewidth(bwith)
ax1.spines['bottom'].set_linewidth(bwith)
ax1.spines['left'].set_linewidth(bwith)
ax1.spines['right'].set_linewidth(bwith)

##################################第二个子图
ax2=fig.add_subplot(spec[1:2,0:2],projection=ccrs.PlateCarree(central_longitude=180)) #选中了一二行和二三列
 
proj = ccrs.PlateCarree(central_longitude=180)##设置一个圆柱投影坐标，中心经度180°E

plt.rcParams['font.sans-serif']=['Times New Roman']

#地图相关设置，包括边界，河流，海岸线，坐标
ax2.set_extent(extent, crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),color='gray')
ax2.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')#添加陆地并且陆地部分全部填充成浅灰色
ax2.add_feature(cfeature.LAKES, alpha=0.9)

#横纵坐标设置
ax2.set_xticks(xticks, crs=ccrs.PlateCarree())
ax2.set_yticks(yticks, crs=ccrs.PlateCarree())

ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)

#############相关系数
fig2 =ax2.contourf(lon,lat,PP,np.linspace(-1,1,11), zorder=0, extend = 'both', transform=ccrs.PlateCarree(),cmap=cmaps.MPL_BrBG)

###########################序号矩形框
start_point = (-328, 43)#矩形框起点
RE1=Rectangle(start_point,18, 15,linewidth=3,linestyle='-' ,zorder=2,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax2.add_patch(RE1)  #显示矩形框

#箭头矩形框
start_point = (296, -28.8)#矩形框起点
RE2=Rectangle(start_point,32, 17.8,linewidth=2,linestyle='-' ,zorder=1,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax2.add_patch(RE2)  #显示矩形框

ax2.text(-139, 52, "b)",color='k',ha="center", va="center",rotation=0.4,fontdict=font)
# ax2.text(80, -52, "1e-6",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)

#中太平洋梯度矩形框
start_point = (150, -5)#矩形框起点
RE2=Rectangle(start_point,60, 30,linewidth=2.1,linestyle='-' ,zorder=1,\
              edgecolor='k',facecolor='None', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax2.add_patch(RE2)  #显示矩形框

#海洋性大陆梯度矩形框
start_point = (90, -5)#矩形框起点
RE2=Rectangle(start_point,50, 30,linewidth=2.1,linestyle='-' ,zorder=1,\
              edgecolor='k',facecolor='None', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax2.add_patch(RE2)  #显示矩形框

###画回归的风场
n=7
Q2=ax2.quiver(lon[::n],lat[::n],regu[::n,::n],regv[::n,::n],
              color='gray',pivot='mid',width=0.0042,
              scale=10,headwidth=4,headlength=6,headaxislength=4,transform=ccrs.PlateCarree())

ax2.quiverkey(Q2,  X=0.94, Y=0.07, U=0.5,angle = 0,  
              label='0.5m/s', labelpos='N', edgecolor='white', facecolor='g',
              labelcolor = 'k', fontproperties = font2,linewidth=0.1)#linewidth=1为箭头的大小

import matplotlib.patheffects as path_effects

Q2.set_path_effects([path_effects.PathPatchEffect
                      (edgecolor='white', facecolor='m', 
                        linewidth= 1.3 )])

ax2.set_xticks(xticks, crs=ccrs.PlateCarree())
ax2.set_yticks(yticks, crs=ccrs.PlateCarree())

ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)

#刻度线设置
ax2.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
ax2.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ax2.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax2.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度

#最大刻度、最小刻度的刻度线长短，粗细设置
ax2.tick_params(which='major', length=8,width=1,color='k')#最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=3,width=0.8,color='k')#最小刻度长度，宽度设置
ax2.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=16)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

###########################################边框设置
ax2.spines['top'].set_linewidth(bwith)
ax2.spines['bottom'].set_linewidth(bwith)
ax2.spines['left'].set_linewidth(bwith)
ax2.spines['right'].set_linewidth(bwith)

##################################第三个子图
ax3=fig.add_subplot(spec[2:3,0:1]) #选中了一二行和二三列
plt.subplots_adjust(wspace=0.2, hspace=0.35)#wspace、hspace左右、上下的间距
   
plt.scatter(grdT,deltaylon,color='red',marker='o',s=33) #NCEP
plt.plot(grdT, y1, color='gray',linewidth=2)

plt.rcParams.update({'font.size':18})#图例大小

##########################################散点加标签
model_num= np.arange(1,36,1)

for i in range(len(deltaylon)):
    ax3.text(grdT[i]*1, deltaylon[i]*1, model_num[i], fontsize=12, color = "gray", 
              style = "italic", weight = "light", verticalalignment='bottom', 
              horizontalalignment='center',rotation=0) #给散点加标签

##########################################95%置信区间
import seaborn as sns
sns.regplot(x=grdT, y=deltaylon, ci=95,color='gray',marker='None')

##########################################标注文字
ax3.set_xlabel('tos change gradient ', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
ax3.set_ylabel('NPMT change ', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来

##########################################最大刻度、最小刻度的刻度线长短，粗细设置
ax3.tick_params(which='major', length=9,width=0.4,color='k')#最大刻度长度，宽度设置，
ax3.tick_params(which='minor', length=0,width=0.3,color='k')#最小刻度长度，宽度设置
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

##################################第四个子图
ax4=fig.add_subplot(spec[2:3,1:2]) #选中了一二行和二三列
 
plt.scatter(grdp,deltaylon,color='lawngreen',marker='o',s=33) #NCEP

plt.plot(grdp, y2, color='gray',linewidth=2)

##########################################散点加标签
model_num= np.arange(1,36,1)

for i in range(len(grdp)):
    ax4.text(grdp[i]*1, deltaylon[i]*1, model_num[i], fontsize=12, color = "gray", 
              style = "italic", weight = "light", verticalalignment='bottom', 
              horizontalalignment='center',rotation=0) #给散点加标签

##########################################95%置信区间
import seaborn as sns
sns.regplot(x=grdp, y=deltaylon, ci=95,color='gray',marker='None')

##########################################标注文字
ax4.set_xlabel('pr change gradient', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来
ax4.set_ylabel('NPMT change ', font )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来

##########################################最大刻度、最小刻度的刻度线长短，粗细设置
ax4.tick_params(which='major', length=9,width=0.4,color='k')#最大刻度长度，宽度设置，
ax4.tick_params(which='minor', length=0,width=0.3,color='k')#最小刻度长度，宽度设置
ax4.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

##########################################设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=18)
labels = ax4.get_xticklabels() + ax4.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

###########################################边框设置
ax4.spines['top'].set_linewidth(bwith)
ax4.spines['bottom'].set_linewidth(bwith)
ax4.spines['left'].set_linewidth(bwith)
ax4.spines['right'].set_linewidth(bwith)

tt = {'family' : 'Times New Roman','weight' : 'bold','size' : 18}

ax3.text(-0.3, 2.4,    "c)",color='k',ha="center", va="center",rotation=0.4,fontdict=font)
ax4.text(-0.2, 1.8, "d)",color='k',ha="center", va="center",rotation=0.4,fontdict=font)

ax3.text(0.6, -2,    "R = 0.70 ",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)
ax4.text(1.5, -2,    "R = 0.68 ",color='k',ha="center", va="center",rotation=0.4,fontdict=tt)

########################################垂直共用colorbar
position = fig.add_axes([0.322, 0.625, 0.38, 0.0099 ])#位置[左,下,右,上]
cb1 = plt.colorbar(fig1, cax=position,orientation='horizontal')
cb1.ax.tick_params(length=1, labelsize=12,color='lightgray')#length为刻度线的长度

position = fig.add_axes([0.322, 0.35, 0.38, 0.0099 ])#位置[左,下,右,上]
cb2 = plt.colorbar(fig2, cax=position,orientation='horizontal')
cb2.ax.tick_params(length=1, labelsize=12,color='lightgray')#length为刻度线的长度


cb1.set_ticks([-0.5, -0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4, 0.5])

cb1.set_ticklabels(['-0.5', '-0.4', '-0.3','-0.2','-0.1', '0','0.1','0.2','0.3','0.4','0.5'])

cb1.update_ticks()

#模式名称
f=open('F:\code\modelname(35).txt', encoding='gbk')

name=[]
for line in f:
    name.append(line.strip())

font3 = {'family' : 'Times New Roman','weight' : 'normal','size' : 12}

# ##############################输出为一列
for i in range(0,35,1): #相对于图的横坐标0来的
    ax4.text(2.2, 14.6-0.498*i, name[i], 
            font=font3,color = "k", style = "italic", weight = "light", 
            verticalalignment='center', horizontalalignment='left',rotation=0,fontdict=font3) #给散点加标签

plt.savefig('F:\code\date\submit\Fig6.jpg',bbox_inches = 'tight')
