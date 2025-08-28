import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.mpl.ticker import LongitudeFormatter,LatitudeFormatter
import matplotlib.ticker as mticker
plt.rcParams['font.sans-serif']=['SimHei']
fig=plt.figure(figsize=(2,2),dpi=400)
ax=fig.add_axes([0,0,1,1],projection=ccrs.PlateCarree(central_longitude=110))

ax.add_feature(cf.LAND.with_scale('110m'))
ax.add_feature(cf.OCEAN.with_scale('110m'))
ax.add_feature(cf.COASTLINE.with_scale('110m'),lw=0.4)
ax.add_feature(cf.RIVERS.with_scale('110m'),lw=0.4)
################################################################
ax.set_xticks([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])#指定要显示的经纬度
ax.set_yticks([-90,-60,-30,0,30,60,90])
ax.xaxis.set_major_formatter(LongitudeFormatter())#刻度格式转换为经纬度样式
ax.yaxis.set_major_formatter(LatitudeFormatter())
ax.tick_params(axis='both',which='major',labelsize=3,direction='out',length=5,width=0.3,pad=0.2,top=True,right=True)
ax.xaxis.set_minor_locator(mticker.MultipleLocator(5))#刻度格式转换为经纬度样式
ax.yaxis.set_minor_locator(mticker.MultipleLocator(5))
ax.tick_params(axis='both',which='minor',direction='out',width=0.3,top=True,right=True)
ax.spines['geo'].set_linewidth(0.5)#调节边框粗细
ax.set_title('Python仿制NCL风格地图',fontsize=5)
ax.set_extent([-180, 180, -30, 80], crs=ccrs.PlateCarree())  # 设置地图范围
plt.savefig('hzk.pdf', bbox_inches='tight')