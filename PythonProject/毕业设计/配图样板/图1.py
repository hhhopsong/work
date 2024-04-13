# -*- coding: utf-8 -*-
"""
Created on Sat Oct  7 14:09:19 2023

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

##################################
tf = np.loadtxt('F:\PC\TC\cma.txt',dtype='float')

##################################
select1 = ((1964<=tf[:,1])&(tf[:,1]<=2014)) & ((7<=tf[:,2])&(tf[:,2]<=9)) & ((100.0<=tf[:,-3])&(tf[:,-3]<=180.0)) & ((0.0<=tf[:,-4])&(tf[:,-4]<=30.0))

##################################
tf = tf[select1,:] 

###################################
tf1 = tf[tf[:,-1] == 1,:] 

###################################
nt = 51
ny=13
nx=33

fre= np.zeros((nt,ny,nx),dtype=int)

time = np.arange(nt) + 1964   
lat1 = ((np.arange(ny)*2.5)+0)[::-1]
lon1 = np.arange(nx)*2.5 +100

##################################
for k in range(len(tf1)):
    i = round((tf1[k,-3]-100)/2.5) 
    j = round((30-tf1[k,-4])/2.5)  
    t = int(tf1[k,1]-time[0])
    fre[t,j,i] = fre[t,j,i]+1

fre=np.sum(fre,axis=0)

##################################
ob_uv=xr.open_dataset(r'G:\reanalysis\ERA5\interp_1964_789_invertlat_timmean\1964-2014uv.nc')#Dimensions:  (level: 17, lat: 73, lon: 144, time: 876),1948-01-01 1948-02-01 ... 2020-12-01
ob_pr=xr.open_dataset(r'G:\reanalysis\ERA5\interp_1964_789_invertlat_timmean\1964-2014prate.nc')#Dimensions:  (level: 17, lat: 73, lon: 144, time: 876),1948-01-01 1948-02-01 ... 2020-12-01

ob_u2=xr.open_dataset(r'G:\reanalysis\NCEP\interp_1964_789_invertlat_timmean\uwnd.mon.mean.nc')#Dimensions:  (level: 17, lat: 73, lon: 144, time: 876),1948-01-01 1948-02-01 ... 2020-12-01
ob_v2=xr.open_dataset(r'G:\reanalysis\NCEP\interp_1964_789_invertlat_timmean\vwnd.mon.mean.nc')#Dimensions:  (level: 17, lat: 73, lon: 144, time: 876),1948-01-01 1948-02-01 ... 2020-12-01

##################################
ob_uwnd1=ob_uv['u'] 
ob_vwnd1=ob_uv['v'] 
ob_pr=ob_pr['mtpr'] 

ob_uwnd2=ob_u2['uwnd'] 
ob_vwnd2=ob_v2['vwnd'] 

##################################
tt=ob_uwnd1.loc[:,20:0,100:190] 

ob_uwnd1=ob_uwnd1.loc['1964-08-01':'2014-08-28':1].loc[:,850,90:-90,0:360]
ob_uwnd2=ob_uwnd2.loc['1964-08-01':'2014-08-28':1].loc[:,850,90:-90,0:360]
ob_pr=ob_pr.loc['1964-08-01':'2014-08-28':1].loc[:,90:-90,0:360]

ob_vwnd1=ob_vwnd1.loc['1964-08-01':'2014-08-28':1].loc[:,850,90:-90,0:360]
ob_vwnd2=ob_vwnd2.loc['1964-08-01':'2014-08-28':1].loc[:,850,90:-90,0:360]

Lat=tt['lat'] 
Lon=tt['lon'] 

lat=ob_uwnd1['lat']
lon=ob_uwnd1['lon']

ob_uwnd1=np.array(ob_uwnd1);      
ob_uwnd2=np.array(ob_uwnd2); 
ob_pr=np.array(ob_pr); 

##############################################Historical 850hpa纬向风
path1 = r"G:/interp_historical/interp_1964_789_timmean_invertlat/850hpa_ua"  
files = os.listdir(path1)

ua1=np.zeros((47,180,360))#涡度
m=0
for file in files:
  f=xr.open_dataset(path1 + "\\" + file)   
  ua=f['ua'] 
  ua=ua.loc[:,:,90:-90,0:360] 
  ua=np.mean(ua,axis=(0,1))
  ua1[m,:,:]=ua;m=m+1

##############################################Historical 850hpa经向风
path2 = r"G:/interp_historical/interp_1964_789_timmean_invertlat/850hpa_va"  
files = os.listdir(path2)

va1=np.zeros((47,180,360))#涡度
m=0
for file in files:
  f=xr.open_dataset(path2 + "\\" + file)   
  va=f['va'] 
  va=va.loc[:,:,90:-90,0:360]
  va=np.mean(va,axis=(0,1))
  va1[m,:,:]=va;m=m+1

ua1[ua1>100]=0
va1[va1>100]=0

######################################################Historical 降水
path3= r"G:/samemodel/interp_1964_789_timmean_invertlat/explain_MT/his/pr"  
files = os.listdir(path3)

pr1=np.zeros((35,180,360))#涡度
m=0
for file in files:
  f=xr.open_dataset(path3 + "\\" + file)   
  t=f['pr'] 
  t=t.loc[:,90:-90,0:360] 
  t=np.mean(t,axis=(0))
  pr1[m,:,:]=t;m=m+1

#####################################
uu8=np.mean(ob_uwnd1,axis=0) 
vv8=np.mean(ob_vwnd1,axis=0) 

uu1=np.nanmean(ua1,axis=0) 
vv1=np.nanmean(va1,axis=0) 

uu8=np.array(uu8);vv8=np.array(vv8) 
uu1=np.array(uu1);vv1=np.array(vv1) 

###################################
u8=np.array(uu8) 
v8=np.array(vv8)

u1=np.array(ua1) 
v1=np.array(va1)

u8[np.isnan(u8)] = 0;v8[np.isnan(v8)] = 0
u1[np.isnan(u1)] = 0;v1[np.isnan(v1)] = 0

from sfvp import VectorWind 
from sfvp import prep_data

mpl.rcParams['mathtext.default'] = 'regular'

u8, u8_info = prep_data(u8, 'yx')
v8, v8_info = prep_data(v8, 'yx')

u1, u1_info = prep_data(u1, 'tyx')
v1, v1_info = prep_data(v1, 'tyx')

#####################################
w1= VectorWind(u8, v8)
w2= VectorWind(u1, v1)

avor1=w1.vorticity()   
avor2=w2.vorticity()   
               
#####################################
avor1=avor1*1e05
avor2=avor2*1e05

avor1=np.nanmean(avor1,axis=2) 
avor2=np.nanmean(avor2,axis=2) 

MME=np.nanmean(pr1,axis=0)
ob_pr=np.nanmean(ob_pr,axis=0)

difu=uu1-uu8;
difv=vv1-vv8
difp=MME-ob_pr;

#####################################
difp=difp*24*3600

##########空间平滑
def spatial_smooth(data,pointnum):#pointnum为几次平滑
    for s in range(0,data.shape[0], 1):
        data[s,:]=np.convolve(data[s,:],np.ones(pointnum)/pointnum,mode='same')
    for s in range(0,data.shape[1],1):
        data[:,s]=np.convolve(data[:,s],np.ones(pointnum)/pointnum,mode='same')
    return data

fre=spatial_smooth(fre,3)
avor1=spatial_smooth(avor1,3)
difp=spatial_smooth(difp,3)

######################mask陆地
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

avor1=mask(avor1, -90, 90, 0, 359) #带入参数     
avor2=mask(avor2, -90, 90, 0, 359) #带入参数
difp=mask(difp, -90, 90, 0, 359) #带入参数

########################################画图
#以下为地图投影以及坐标轴的设置
fig = plt.figure(figsize=(13,18),dpi=400)

plt.rcParams['font.sans-serif']=['Times New Roman']
plt.subplots_adjust(wspace=0.1, hspace=0.32)#wspace、hspace左右、上下的间距

extent1=[100,180,0,35]#850hpa

xticks1=np.arange(extent1[0],extent1[1]+1,20)
yticks1=np.arange(extent1[2],extent1[3]+1,15)

#############ERA5
ax1 = fig.add_subplot(311,projection=ccrs.PlateCarree(central_longitude=180))
                      
#地图相关设置，包括边界，河流，海岸线，坐标
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE.with_scale('50m'),color='gray')
ax1.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')#添加陆地并且陆地部分全部填充成浅灰色
ax1.add_feature(cfeature.LAKES, alpha=0.9)

ax1.set_xticks(xticks1, crs=ccrs.PlateCarree())
ax1.set_yticks(yticks1, crs=ccrs.PlateCarree())

lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()

ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)

font = {'family' : 'Times New Roman','weight' : 'bold','size' : 18}

############################################相对涡度
fig1 =ax1.contourf(lon,lat,avor1,np.linspace(-1.5,1.5,16), zorder=0, extend = 'both', transform=ccrs.PlateCarree(),cmap=cmaps.BlueWhiteOrangeRed)

####################################生成TC频数等值线
fig2=ax1.contour(lon1,lat1,fre,levels=[2,4,6,8],linewidths=2,colors='gray',transform=ccrs.PlateCarree())#18为等值线的间隔距，值越大越密集
plt.clabel(fig2,fontsize=24,colors='k')# plt.clabel为设置标签颜色以及大小

zero=ax1.contour(lon,lat,uu8,transform=ccrs.PlateCarree(),colors='m',levels=[0],linewidths=3,alpha=1)
plt.plot(-31, 6.3, marker='o',color='lawngreen',
         markersize=18)  # 季风槽最东点

##############箭头边框设置
from matplotlib.patches import Rectangle

#########绘制箭头的矩形框
start_point = (-188.3, 0.6)#矩形框起点
RE=Rectangle(start_point,8, 6.7,linewidth=3,linestyle='-' ,zorder=1,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax1.add_patch(RE)  #显示矩形框

#############标序号的矩形框
start_point = (-259, 29)#矩形框起点
RE=Rectangle(start_point,6, 5,linewidth=3,linestyle='-' ,zorder=1,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax1.add_patch(RE)  #显示矩形框

ax1.text(-76, 31.5, "a)",color='k',ha="center", va="center",rotation=0,fontdict=font)
ax1.text(-77, 36.8, "ERA5",color='k',ha="center", va="center",rotation=0,fontdict=font)

##########################风场
n=5

Q1=ax1.quiver(lon[::n],lat[::n],uu8[::n,::n],vv8[::n,::n],
color='g',pivot='mid',scale=90,
width=0.0049,headwidth=4,headlength=5.5,headaxislength=4.5,
transform=ccrs.PlateCarree())

import matplotlib.patheffects as path_effects

Q1.set_path_effects([path_effects.PathPatchEffect
                      (edgecolor='white', facecolor='g', 
                        linewidth= 1.7 )])

ax1.quiverkey(Q1,  X=0.946, Y=0.09, U=5,angle = 0,  label='5m/s', 
              labelpos='N', color = 'green',labelcolor = 'k', fontproperties = font,linewidth=0.8)#linewidth=1为箭头的大小

#刻度线设置
xmajorLocator = MultipleLocator(20)#先定义xmajorLocator，再进行调用
ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
xminorLocator = MultipleLocator(10)
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ymajorLocator = MultipleLocator(15)
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
yminorLocator = MultipleLocator(5)
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度

# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签

#最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=20)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

font2 = {'family' : 'Times New Roman','weight' : 'bold','size' : 16}#设置横纵坐标的名称以及对应字体格式family'
#ax1.set_ylabel('historical DGPI', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来

#############MME
ax2 = fig.add_subplot(312,projection=ccrs.PlateCarree(central_longitude=180))

#地图相关设置，包括边界，河流，海岸线，坐标
ax2.set_extent(extent1, crs=ccrs.PlateCarree())
ax2.add_feature(cfeature.COASTLINE.with_scale('50m'),color='gray')
ax2.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')#添加陆地并且陆地部分全部填充成浅灰色
ax2.add_feature(cfeature.LAKES, alpha=0.9)

ax2.set_xticks(xticks1, crs=ccrs.PlateCarree())
ax2.set_yticks(yticks1, crs=ccrs.PlateCarree())

ax2.xaxis.set_major_formatter(lon_formatter)
ax2.yaxis.set_major_formatter(lat_formatter)

############################################相对涡度
fig3 =ax2.contourf(lon,lat,avor2,np.linspace(-1.5,1.5,16), zorder=0, extend = 'both', transform=ccrs.PlateCarree(),cmap=cmaps.BlueWhiteOrangeRed)
#fig3 =ax2.contourf(lon,lat,uu1,np.linspace(-6,6,21), zorder=0, extend = 'both', transform=ccrs.PlateCarree(),cmap=cmaps.BlueWhiteOrangeRed)

zero=ax2.contour(lon,lat,uu1,transform=ccrs.PlateCarree(),colors='m',levels=[0],linewidths=3,alpha=1)

plt.plot(-28.9, 8.9, marker='o',color='lawngreen',
         markersize=18)  # 季风槽最东点

#########绘制箭头的矩形框
start_point = (-188.3, 0.6)#矩形框起点
RE=Rectangle(start_point,8, 6.7,linewidth=3,linestyle='-' ,zorder=1,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax2.add_patch(RE)  #显示矩形框

#############标序号的矩形框
start_point = (-259, 29)#矩形框起点
RE=Rectangle(start_point,6, 5,linewidth=3,linestyle='-' ,zorder=1,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax2.add_patch(RE)  #显示矩形框

ax2.text(-76, 31.5, "b)",color='k',ha="center", va="center",rotation=0,fontdict=font)
ax2.text(-77, 36.8, "MME",color='k',ha="center", va="center",rotation=0,fontdict=font)

##########################风场
Q2=ax2.quiver(lon[::n],lat[::n],uu1[::n,::n],vv1[::n,::n],
color='g',pivot='mid',scale=90,
width=0.0049,headwidth=4,headlength=5.5,headaxislength=4.5,
transform=ccrs.PlateCarree())

import matplotlib.patheffects as path_effects

Q2.set_path_effects([path_effects.PathPatchEffect
                      (edgecolor='white', facecolor='green', 
                        linewidth= 1.7 )])

ax2.quiverkey(Q2,  X=0.946, Y=0.09, U=5,angle = 0,  label='5m/s', 
              labelpos='N', color = 'green',labelcolor = 'k', fontproperties = font,linewidth=0.8)#linewidth=1为箭头的大小

#刻度线设置
ax2.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
ax2.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ax2.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax2.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度

# ax2.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签

#最大刻度、最小刻度的刻度线长短，粗细设置
ax2.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax2.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax2.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=20)
labels = ax2.get_xticklabels() + ax2.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#############MME-ERA5
ax3 = fig.add_subplot(313,projection=ccrs.PlateCarree(central_longitude=180))

#地图相关设置，包括边界，河流，海岸线，坐标
ax3.set_extent(extent1, crs=ccrs.PlateCarree())
ax3.add_feature(cfeature.COASTLINE.with_scale('50m'),color='gray')
ax3.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray')#添加陆地并且陆地部分全部填充成浅灰色
ax3.add_feature(cfeature.LAKES, alpha=0.9)

ax3.set_xticks(xticks1, crs=ccrs.PlateCarree())
ax3.set_yticks(yticks1, crs=ccrs.PlateCarree())

ax3.xaxis.set_major_formatter(lon_formatter)
ax3.yaxis.set_major_formatter(lat_formatter)

#########绘制箭头的矩形框
start_point = (-188.3, 0.6)#矩形框起点
RE=Rectangle(start_point,8, 6.7,linewidth=3,linestyle='-' ,zorder=1,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax3.add_patch(RE)  #显示矩形框

#############标序号的矩形框
start_point = (-259, 29)#矩形框起点
RE=Rectangle(start_point,6, 5,linewidth=3,linestyle='-' ,zorder=1,\
              edgecolor='None',facecolor='white', transform=ccrs.PlateCarree()) #75表示105+75=180,经度终点，5+25=30，纬度终点
ax3.add_patch(RE)  #显示矩形框

ax3.text(-76, 31.5, "c)",color='k',ha="center", va="center",rotation=0,fontdict=font)
ax3.text(-74, 36.8, "MME-ERA5",color='k',ha="center", va="center",rotation=0,fontdict=font)

##############################################相对涡度
fig4 =ax3.contourf(lon,lat,difp,np.linspace(-3,3,21), zorder=0, extend = 'both', transform=ccrs.PlateCarree(),cmap=cmaps.BlueWhiteOrangeRed)

##########################风场
Q3=ax3.quiver(lon[::n],lat[::n],difu[::n,::n],difv[::n,::n],
color='g',pivot='mid',scale=30,
width=0.0052,headwidth=4,headlength=5.5,headaxislength=4.5,
transform=ccrs.PlateCarree())

import matplotlib.patheffects as path_effects

Q3.set_path_effects([path_effects.PathPatchEffect
                      (edgecolor='white', facecolor='green', 
                        linewidth= 1.8 )])

ax3.quiverkey(Q3,  X=0.946, Y=0.09, U=2,angle = 0,  label='2m/s', 
              labelpos='N', color = 'green',labelcolor = 'k', fontproperties = font,linewidth=0.8)#linewidth=1为箭头的大小

#刻度线设置
ax3.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
ax3.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ax3.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax3.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度

#最大刻度、最小刻度的刻度线长短，粗细设置
ax3.tick_params(which='major', length=11,width=2,color='darkgray')#最大刻度长度，宽度设置，
ax3.tick_params(which='minor', length=8,width=1.8,color='darkgray')#最小刻度长度，宽度设置
ax3.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外

#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=20)
labels = ax3.get_xticklabels() + ax3.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]

#ax3.set_ylabel('SSP585 DGPI ', font2 )#ax.bar那里虽然进行了设置，这里仍然要设置，否则不显示出来

t = {'family' : 'Times New Roman','weight' : 'bold','size' : 24}

###################colorbar位置
position = fig.add_axes([0.296, 0.63, 0.44, 0.011 ])#位置[左,下,右,上]
cb1 = plt.colorbar(fig1, cax=position,orientation='horizontal')
cb1.ax.tick_params(length=1, labelsize=18)#length为刻度线的长度

position = fig.add_axes([0.296, 0.35, 0.44, 0.011 ])#位置[左,下,右,上]
cb2 = plt.colorbar(fig3, cax=position,orientation='horizontal')
cb2.ax.tick_params(length=1, labelsize=18)#length为刻度线的长度

position = fig.add_axes([0.296, 0.08, 0.44, 0.011 ])#位置[左,下,右,上]
cb3 = plt.colorbar(fig4, cax=position,orientation='horizontal')
cb3.ax.tick_params(length=1, labelsize=18)#length为刻度线的长度

import matplotlib.ticker as ticker
tick_locator = ticker.MaxNLocator(nbins=7)  # colorbar上的刻度值个数


plt.savefig('F:\code\date\submit\Fig1.jpg',bbox_inches = 'tight')
