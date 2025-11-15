#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import earth_avg_radius
import cmaps
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.io.shapereader as shpreader
from cartopy.util import add_cyclic_point
import cartopy.feature as cfeature
from scipy.stats import linregress
import cmaps

from matplotlib.ticker import MaxNLocator
plt.rcParams['savefig.dpi'] = 300 # 图片像素
plt.rcParams['figure.dpi'] = 300 # 分辨率
plt.rcParams['font.sans-serif']=['Times New Roman'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号


# ## 获得位势高度扰动场

# In[33]:


def reg2(pc,lat,lon,pre2d):

    ## 回归系数,相关系数
    A = np.vstack([pc, np.ones(len(pc))]).T
    s,r,p = np.zeros((lat.shape[0],lon.shape[0])),np.zeros((lat.shape[0],lon.shape[0])),np.zeros((lat.shape[0],lon.shape[0]))
    for i in range(len(lat)):
        for j in range(len(lon)):
                s[i,j],_,r[i,j], p[i,j],_  = linregress(pc,pre2d[:,i,j])


    # 显著性检验
    pre2d = np.array(pre2d).reshape(pre2d.shape[0],pre2d.shape[1]*pre2d.shape[2])
    area = p
    pre_reg_cyc, lon_cyc = add_cyclic_point(s, coord = lon)
    nx, ny = np.meshgrid(lon_cyc, lat)
    
    return s,area,pre_reg_cyc,nx,ny


time_ = ['1979-01-01','2012-12-01']
level_ = 500
pc = np.load(r'E:\PE\index\A/pred_45.npy') 



with xr.open_dataset(r'E:\Data\2023/v-z_1960-2022_JJA_mean.nc') as f2:
      f2 = f2.sel(time=slice(time_[0],time_[1]))
      lat,lon = np.array(f2['lat']),np.array(f2['lon'])
      z = f2['z'] 
      z = z.sel(level=level_)
      z = z-z.mean(dim = 'time')
      z = np.array(z)/9.8
      s3,area3,pre_reg_cyc3,nx3,ny3 = reg2(pc,lat,lon,z)

hgt = s3
print(hgt.shape)


# 读取数据

# In[34]:


f_uwnd = xr.open_dataset(r'E:\Data\2023/u_1960-2022_JJA_mean.nc')
f_uwnd = f_uwnd.sel(time=slice(time_[0],time_[1]),level=level_)
f_uwnd = f_uwnd.mean(dim = 'time')
uwnd = np.array(f_uwnd['u'])


f_vwnd = xr.open_dataset(r'E:\Data\2023/v-z_1960-2022_JJA_mean.nc')
f_vwnd = f_vwnd.sel(time=slice(time_[0],time_[1]),level=level_)
f_vwnd = f_vwnd.mean(dim = 'time')
vwnd = np.array(f_vwnd['v'])


# T - N 通量计算

# In[35]:


a=6400000 #地球半径
omega=7.292e-5 #自转角速度
lev = level_/1000#p/p0
g = 9.8

dlon=(np.gradient(lon)*np.pi/180.0).reshape((1,-1))
dlat=(np.gradient(lat)*np.pi/180.0).reshape((-1,1))
coslat = (np.cos(np.array(lat)*np.pi/180)).reshape((-1,1))
sinlat = (np.sin(np.array(lat)*np.pi/180)).reshape((-1,1))
 
#计算科氏力
f=np.array(2*omega*np.sin(lat*np.pi/180.0)).reshape((-1,1))   
#计算|U|

wind = np.sqrt(uwnd**2+vwnd**2)

# #计算括号外的参数，a^2可以从括号内提出
c=(lev)*coslat/(2*a*a*wind)

za = s3
#Ψ`
streamf = g*za/f   

#计算各个部件，难度在于二阶导，变量的名字应该可以很容易看出我是在计算哪部分
dzdlon = np.gradient(streamf,axis = 1)/dlon
ddzdlonlon = np.gradient(dzdlon,axis = 1)/dlon
dzdlat = np.gradient(streamf,axis = 0)/dlat
ddzdlatlat = np.gradient(dzdlat,axis = 0)/dlat
ddzdlatlon = np.gradient(dzdlon,axis = 0)/dlat
print(dzdlon.shape)
#这是X,Y分量共有的部分
x_tmp = dzdlon*dzdlon-streamf*ddzdlonlon
y_tmp = dzdlon*dzdlat-streamf*ddzdlatlon
#计算两个分量
fx = c * ((uwnd/coslat/coslat)*x_tmp+vwnd*y_tmp/coslat)
fy = c * ((uwnd/coslat)*y_tmp+vwnd*(dzdlat*dzdlat-streamf*ddzdlatlat))


# 只保留25°以北的WAF
fx = fx[:27,:]
fy = fy[:27,:] 
lat_ = lat[:27]
print(lat_)


# 画风场

# In[36]:


def mapart(ax):
    #添加地图元素
    projection = ccrs.PlateCarree()  #创建投影
    ax.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=0.6, zorder=1,color='gainsboro')
    #设置为刻度格式
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 刻度格式
    ax.xaxis.set_major_formatter(LongitudeFormatter())  
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    # ax.tick_params(axis='both', labelsize=5, direction='out') ###设置刻度标记大小
    #画线范围和间隔
    ax.set_xticks(np.arange(0,330.1,60),crs=projection)
    ax.set_yticks(np.arange(-45,90.1,30),crs=projection)


# In[37]:


plt.figure(figsize = (9,5))
plt.subplots_adjust(hspace = 0.15)

ax1 = plt.subplot(211, projection = ccrs.PlateCarree(central_longitude = 180))
ax1.set_extent([0,-45, -45, 90], crs=ccrs.PlateCarree())
ax1.spines['geo'].set_linewidth(2)  ###设置边框粗细
mapart(ax1)



###绘制海岸线
ax1.coastlines(lw = 0.8,color='gray',alpha=0.3,zorder=2)
land = cfeature.NaturalEarthFeature('physical', 'land','50m',alpha=0.6,edgecolor='gainsboro',zorder=2,
                                                                facecolor=cfeature.COLORS['land']) ##除pre alpha=0.1，其他为0.6
                                         
# ax1.add_feature(land,facecolor='gainsboro',alpha=0.6) #改这里的color是面的颜色
###绘制海岸线


###画位势高度场
level=[15,12,9,6,1,0,-1,-6,-9,-12,-15] ###[35,20,16,10,5,-5,-10,-16,-20,-35]35,20,12,9,6,3,-3,-6,-9,-12,-20,-35
level = level[::-1]  
c1 = ax1.contourf(nx3, ny3, pre_reg_cyc3,levels=level, alpha=0.8,zorder=1,extend='both',cmap = cmaps.MPL_PuOr_r, transform = ccrs.PlateCarree())###海温的colorbar，cmap=色带
C = plt.colorbar(c1,shrink = 0.8,pad = 0.02,ticks = level)   ##shrink，从0-1，整个色条将会按照输入值被缩放。pad，色条与边框之间的距离 
C.ax.tick_params(labelsize=5,length=0) ###colorbar字体大小
C.outline.set_edgecolor('white') 
print(s3.min())

# c = ax1.contour(nx3,ny3,pre_reg_cyc3,zorder=15,linewidths=1,linestyles=['--','--','--','-','-','-','-'],colors=['b','b','b','k','r','r','r'],
                    #  levels=[-12,-8,-4,0,4,8,12],transform = ccrs.PlateCarree())
###画位势高度场


##画TN
c = ax1.quiver(lon[::2], lat_[::2], fx[::2,::2],fy[::2,::2],color='darkred',pivot='mid',width=0.0023,scale=1,headwidth=7,headlength=7.5,zorder = 40,transform = ccrs.PlateCarree())
ax1.quiverkey(c, X=0.9, Y = 1.07,U = 0.05,label='0.05 m$^2$/s$^2$',labelpos='E',fontproperties={'size': '10'})

##画TN



cb = ax1.contourf(lon,lat,area3, levels = [0,0.05,1],hatches = ['////',None],colors = "none",transform = ccrs.PlateCarree())

for collection in cb.collections:
    collection.set_edgecolor('w')
    collection.set_lw(55)
for collection in cb.collections:
    collection.set_linewidth(0)
#打点

plt.title('Reg 200hPaWAF&Z onto C index',fontsize=10,loc='left')
ax1.tick_params(axis='both', labelsize=10, direction='out',width=2) ###设置刻度字体大小和方向

plt.show()

