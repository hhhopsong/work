import os # 系统模块,读取文件路径
import geopandas as gpd
import salem
from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import xarray as xr
from eofs.standard import Eof
from scipy import interpolate   # 双线性插值
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm # 进度条
from tools.TN_WaveActivityFlux import TN_WAF

# 南海小地图
def adjust_sub_axes(ax_main, ax_sub, shrink, lr=1.0, ud=1.0):
    '''将ax_sub调整到ax_main的右下角.shrink指定缩小倍数。
    当ax_sub是GeoAxes时,需要在其设定好范围后再使用此函数'''

    bbox_main = ax_main.get_position()
    bbox_sub = ax_sub.get_position()
    wratio = bbox_main.width / bbox_sub.width
    hratio = bbox_main.height / bbox_sub.height
    wnew = bbox_sub.width * shrink
    hnew = bbox_sub.height * shrink
    bbox_new = mtransforms.Bbox.from_extents(
        bbox_main.x1 - lr * wnew, bbox_main.y0 + (ud - 1) * hnew,
        bbox_main.x1 - (lr - 1) * wnew, bbox_main.y0 + ud * hnew
    )
    ax_sub.set_position(bbox_new)

def spaCorr(X, Y):
    '''计算空间相关系数'''
    # 得出均不为nan的位置
    id = np.where((~np.isnan(X)) & (~np.isnan(Y)))
    X = X[id].flatten()
    Y = Y[id].flatten()
    return np.corrcoef(X, Y)[0, 1]


####23个模式图+MME图+观测图
# 绘图地图要素设置
fontfamily = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.3    # 边框粗细
#plt.tight_layout()
#plt.subplots_adjust(wspace=0.045, hspace=0.001)#wspace、hspace左右、上下的间距
wspce, hspce = 0.35, 0.003
fig = plt.figure(dpi=1000, figsize=(6, 3))
south_china_sea = False
south_china_sea_shink = 0.4
pic_shape = [4, 6]  # 子图行列数
# 经纬度范围
extent1 = [105, 128, 17, 33]  # 经度范围，纬度范围
xticks1 = np.arange(extent1[0], extent1[1]+1, 20)
yticks1 = np.arange(extent1[2], extent1[3]+1, 10)
proj = ccrs.PlateCarree()
# 等值线值
level1 = [-1., -.8, -.6, -.4, -.2, .2, .4, .6, .8, 1.]
# 主模态值
pc = 0
# 字体大小
font_title_size = 5 # 标题
font_title = {'family': 'Arial', 'weight': 'bold', 'size': font_title_size}
pad_title = 0.5  # 标题与图的距离
font_tick_size = 5  # 刻度
font_tick = {'family': 'Arial', 'weight': 'normal', 'size': font_tick_size}
pad_tick = 0.5  # 刻度与轴的距离
font_colorbar_size = 5  # colorbar
font_colorbar = {'family': 'Arial', 'weight': 'bold', 'size': font_colorbar_size}
line_width = 0.1    # 省界、国界线宽
tick_width = 0.3    # 刻度线宽
tick_clolor = 'black'    # 刻度线颜色
# 经纬度
xmajorLocator = MultipleLocator(10)  # x轴最大刻度
xminorLocator = MultipleLocator(2)  # x轴最小刻度
ymajorLocator = MultipleLocator(5)  # y轴最大刻度
yminorLocator = MultipleLocator(1)  # y轴最小刻度
# 时间范围
time = ['1979', '2014']
#数据路径
dataurl = r"C:/Users/10574/OneDrive/File/Graduation Thesis/ThesisData/CMIP6/historical/CMIP6_historical_tasmax/day"#数据路径
Model_Name = os.listdir(dataurl)
# 数据掩膜
zone = 'SouthChina'
shp = fr"D:/CODES/Python/PythonProject/map/shp/south_china/中国南方.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
# 获取站点网格
grids = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\CN05.1\CN05.1_Tmax_1961_2021_daily_025x025.nc")
grids_lon, grids_lat = np.meshgrid(grids['lon'], grids['lat'])
try:
    std_obs_term_78_extreHighDays = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_obs_term_78_extreHighDays.nc')
except:
######################################################################################

    obs_sort95 = grids['tmax'].sel(time=slice(time[0]+'-01-01', time[1]+'-12-31')).quantile(0.95, dim='time')
    obs_term = grids['tmax'].sel(time=slice(time[0]+'-01-01', time[1]+'-12-31'))
    obs_term_78 = obs_term.sel(time=obs_term.time.dt.month.isin([7, 8]))
    obs_term_78_extreHighDays = np.zeros((eval(time[1]) - eval(time[0]) + 1, obs_term_78.shape[1], obs_term_78.shape[2]))
    for i in range(eval(time[0]), eval(time[1]) + 1):
        obs_term_78_extreHighDays[i - eval(time[0]), :, :] = np.sum(np.where((obs_term_78 - obs_sort95) > 0, 1, 0)[(i - eval(time[0])) * 62:(i + 1 - eval(time[0])) * 62, :, :],axis=0)  # 极端高温日数
    std_obs_term_78_extreHighDays = (obs_term_78_extreHighDays - np.mean(obs_term_78_extreHighDays, axis=0)) / np.std(obs_term_78_extreHighDays, axis=0) # 标准化
    std_obs_term_78_extreHighDays_dataset = xr.DataArray(std_obs_term_78_extreHighDays, coords=[[str(i) for i in range(eval(time[0]), eval(time[1])+1)], grids['lat'], grids['lon']], dims=['time', 'lat', 'lon'])
    std_obs_term_78_extreHighDays_dataset.name = 'days'
    std_obs_term_78_extreHighDays_dataset.to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_obs_term_78_extreHighDays.nc')
    std_obs_term_78_extreHighDays = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_obs_term_78_extreHighDays.nc')
######################################################################################
std_obs_term_78_extreHighDays = std_obs_term_78_extreHighDays.salem.roi(shape=split_shp)
obs_term_78_extreHighDays_eof = Eof(std_obs_term_78_extreHighDays['days'].to_numpy())  # 进行eof分解
obs_78_eof = obs_term_78_extreHighDays_eof.eofs(eofscaling=2, neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
obs_78_s = obs_term_78_extreHighDays_eof.varianceFraction(neigs=2)


# 数据读取
eof = np.zeros((len(Model_Name), grids_lat.shape[0], grids_lon.shape[1]))
varFra = np.zeros(len(Model_Name))
for iModle in range(len(Model_Name)):
    ModelName = Model_Name[iModle]
    try:
        print(f'读取{iModle+1} {ModelName}缓存文件...')
        std_q_term_78_extreHighDays = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_{ModelName}_term_78_extreHighDays.nc')['days'].to_numpy()
        q_lat = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_{ModelName}_term_78_extreHighDays.nc')['lat'].to_numpy()
        q_lon = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_{ModelName}_term_78_extreHighDays.nc')['lon'].to_numpy()
    except:
    ######################################################################################

        url = os.listdir(dataurl + '/' + ModelName)
        q = xr.open_dataset(dataurl + '/' + ModelName + '/' + url[0])
        for iurl in tqdm(url[1:], desc=f'\t读取{iModle+1} {ModelName}模式 数据', unit='文件', position=0, colour='green'):
            q = xr.concat([q, xr.open_dataset(dataurl + '/' + ModelName + '/' + iurl)], dim='time')
        q_sort95 = q['tasmax'].sel(time=slice(time[0]+'-01-01', time[1] + '-12-31')).quantile(0.95, dim='time') - 273.15  # 95%分位数, 转换为摄氏度

        q_term = q['tasmax'].sel(time=slice(time[0]+'-01-01', time[1] + '-12-31')) - 273.15
        q_term_78 = q_term.sel(time=q_term.time.dt.month.isin([7, 8]))   # 7-8月气温
        q_term_78_extreHighDays = np.zeros((eval(time[1]) - eval(time[0]) + 1, q_term_78.shape[1], q_term_78.shape[2]))
        for i in tqdm(range(eval(time[0]),  eval(time[1]) + 1), desc=f'\t计算{iModle+1} {ModelName}模式 极端高温日数', unit='Year', position=0, colour='green'):
            q_term_78_extreHighDays[i - eval(time[0]), :, :] = np.sum(np.where((q_term_78 - q_sort95) > 0, 1, 0)[(i - eval(time[0]))*62:(i + 1 - eval(time[0]))*62, :, :], axis=0)    # 极端高温日数
        print(f'\t保存 {ModelName}模式 缓存')
        std_q_term_78_extreHighDays = (q_term_78_extreHighDays - np.mean(q_term_78_extreHighDays, axis=0)) / np.std(q_term_78_extreHighDays, axis=0) # 标准化
        std_q_term_78_extreHighDays_dataset = xr.DataArray(std_q_term_78_extreHighDays, coords=[[str(i) for i in range(eval(time[0]), eval(time[1])+1)], q['lat'], q['lon']], dims=['time', 'lat', 'lon'])
        std_q_term_78_extreHighDays_dataset.name = 'days'
        std_q_term_78_extreHighDays_dataset.to_netcdf(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_{ModelName}_term_78_extreHighDays.nc')
        std_q_term_78_extreHighDays = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_{ModelName}_term_78_extreHighDays.nc')['days'].to_numpy()
        q_lat = q['lat']
        q_lon = q['lon']

    ######################################################################################
    # 双线性插值到CN05.1网格
    ######################################################################################
    std_q_term_78_extreHighDays_interp = np.zeros((std_q_term_78_extreHighDays.shape[0], grids_lat.shape[0], grids_lon.shape[1]))
    num = 0
    for i in tqdm(std_q_term_78_extreHighDays, desc=f'\t插值 {ModelName}模式', unit='Year', position=0, colour='green'):
        interp = interpolate.RegularGridInterpolator((q_lat, q_lon), i, method='linear')
        std_q_term_78_extreHighDays_interp[num, :, :] = interp((grids_lat, grids_lon))
        num += 1
    std_q_term_78_extreHighDays_interp_dataset = xr.DataArray(std_q_term_78_extreHighDays_interp, coords=[[str(i) for i in range(eval(time[0]), eval(time[1])+1)], grids['lat'], grids['lon']], dims=['time', 'lat', 'lon'])
    ######################################################################################
    # 裁切中国范围数据，其余数据Nan化
    std_q_term_78_extreHighDays = std_q_term_78_extreHighDays_interp_dataset.salem.roi(shape=split_shp)

    # eof分解
    eof_78 = Eof(np.array(std_q_term_78_extreHighDays))   #进行eof分解
    EOF_78 = eof_78.eofs(eofscaling=2, neofs=2)  # 得到空间模态U eofscaling 对得到的场进行放缩 （1为除以特征值平方根，2为乘以特征值平方根，默认为0不处理） neofs决定输出的空间模态场个数
    #PC_78 = eof_78.pcs(pcscaling=1, npcs=2)  # 同上 npcs决定输出的时间序列个数
    s_78 = eof_78.varianceFraction(neigs=2)   # 得到前neig个模态的方差贡献
    q_term_78_extreHighDays_eof = EOF_78[pc, :, :]
    ######################################################################################
    # 绘图
    print(f'\t绘制 {ModelName}模式')
    varFra[iModle] = s_78[pc]
    eof[iModle, :, :] = q_term_78_extreHighDays_eof
    zone_corr = spaCorr(eof[iModle, :, :], obs_78_eof[pc, :, :])
    if zone_corr < 0:
        q_term_78_extreHighDays_eof = -q_term_78_extreHighDays_eof
        eof[iModle, :, :] = q_term_78_extreHighDays_eof
        zone_corr = spaCorr(eof[iModle, :, :], obs_78_eof[pc, :, :])
    ax1 = fig.add_subplot(pic_shape[0], pic_shape[1], iModle+1, projection=proj)
    ax1.set_extent(extent1, crs=proj)
    ax1.set_title(ModelName, font=font_title, loc='left', pad=pad_title)
    #ax1.set_title(f"{zone_corr:.2f}", font=font_title, loc='right', pad=pad_title)
    ax1.text(119, 17.5, f'{s_78[pc]*100:.2f}%', fontsize=5, fontweight='bold', color='black', zorder=20,
             transform=ccrs.PlateCarree(central_longitude=0))
    ax1.text(122.5, 30.5, f'{zone_corr:.2f}', fontsize=5, fontweight='bold', color='black', zorder=20,
             transform=ccrs.PlateCarree(central_longitude=0))
    a1 = ax1.contourf(grids['lon'], grids['lat'], q_term_78_extreHighDays_eof, cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='both', transform=proj)
    ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
    ax1.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\cnriver\1级河流.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='b', linewidth=0.2)
    draw_maps(get_adm_maps(level='国'), linewidth=line_width)
    #draw_maps(get_adm_maps(level='省'), linewidth=line_width*0.5)
    # 刻度线设置
    ax1.set_xticks(xticks1, crs=proj)
    ax1.set_yticks(yticks1, crs=proj)
    ax1.xaxis.set_tick_params(pad=pad_tick)
    ax1.yaxis.set_tick_params(pad=pad_tick)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
    ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
    ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
    ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    #最大刻度、最小刻度的刻度线长短，粗细设置
    ax1.tick_params(which='major', length=tick_width*5,width=tick_width,color=tick_clolor)#最大刻度长度，宽度设置，
    ax1.tick_params(which='minor', length=tick_width*2.5,width=tick_width*0.5,color=tick_clolor)#最小刻度长度，宽度设置
    ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
    plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外
    #设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=font_tick_size)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname(fontfamily) for label in labels]
    [label.set_fontweight('normal') for label in labels]

    # 南海小地图
    if south_china_sea:
        ax_sub = fig.add_axes(ax1.get_position(), projection=proj)
        ax_sub.set_extent([105, 125, 0, 25], crs=proj)
        a_sub = ax_sub.contourf(grids['lon'], grids['lat'], q_term_78_extreHighDays_eof, cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='both', transform=proj)
        ax_sub.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
        draw_maps(get_adm_maps(level='国'), linewidth=line_width)
        draw_maps(get_adm_maps(level='省'), linewidth=line_width*0.5)
        adjust_sub_axes(ax1, ax_sub, shrink=south_china_sea_shink)

# 绘制集合平均MME
mme = np.mean(eof, axis=0)
china = get_adm_maps(level='国', record="first", only_polygon=True, wgs84=True)
lons, lats = np.meshgrid(grids['lon'], grids['lat'])
mme = china.maskout(lons, lats, mme)
zone_corr = spaCorr(mme, obs_78_eof[pc, :, :])
ax1 = fig.add_subplot(pic_shape[0], pic_shape[1], 23, projection=proj)
ax1.set_extent(extent1, crs=proj)
ax1.set_title(f"MME", font=font_title, loc='left', pad=pad_title, color='r')
#ax1.set_title(f"{zone_corr:.2f}", font=font_title, loc='right', pad=pad_title,color='r')
ax1.text(119, 17.5, f'{varFra.mean()*100:.2f}%', fontsize=5, fontweight='bold', color='red', zorder=20,
         transform=ccrs.PlateCarree(central_longitude=0))
ax1.text(122.5, 30.5, f'{zone_corr:.2f}', fontsize=5, fontweight='bold', color='red', zorder=20,
         transform=ccrs.PlateCarree(central_longitude=0))
a1 = ax1.contourf(grids['lon'], grids['lat'], mme, cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='both', transform=proj)
ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
ax1.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\cnriver\1级河流.shp').geometries(), ccrs.PlateCarree(),
                   facecolor='none', edgecolor='b', linewidth=0.2)
draw_maps(get_adm_maps(level='国'), linewidth=line_width)
#draw_maps(get_adm_maps(level='省'), linewidth=line_width*0.5)
# 刻度线设置
ax1.set_xticks(xticks1, crs=proj)
ax1.set_yticks(yticks1, crs=proj)
ax1.xaxis.set_tick_params(pad=pad_tick)
ax1.yaxis.set_tick_params(pad=pad_tick)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)

ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
#最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=tick_width*5,width=tick_width,color=tick_clolor)#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=tick_width*2.5,width=tick_width*0.5,color=tick_clolor)#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=font_tick_size)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname(fontfamily) for label in labels]
[label.set_fontweight('normal') for label in labels]

# 南海小地图
if south_china_sea:
    ax_sub = fig.add_axes(ax1.get_position(), projection=proj)
    ax_sub.set_extent([105, 125, 0, 25], crs=proj)
    a_sub = ax_sub.contourf(grids['lon'], grids['lat'], mme, cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='both', transform=proj)
    ax_sub.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
    draw_maps(get_adm_maps(level='国'), linewidth=line_width)
    draw_maps(get_adm_maps(level='省'), linewidth=line_width*0.5)
    adjust_sub_axes(ax1, ax_sub, shrink=south_china_sea_shink)

# 绘制obs

ax1 = fig.add_subplot(pic_shape[0], pic_shape[1], 24, projection=proj)

ax1.set_extent(extent1, crs=proj)
ax1.set_title(f"obs", font=font_title, loc='left', pad=pad_title, color='r')
ax1.text(119, 17.5, f'{obs_78_s[pc]*100:.2f}%', fontsize=5, fontweight='bold', color='red', zorder=20,
         transform=ccrs.PlateCarree(central_longitude=0))
a1 = ax1.contourf(grids['lon'], grids['lat'], obs_78_eof[pc, :, :], cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='neither', transform=proj)
ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
ax1.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\cnriver\1级河流.shp').geometries(), ccrs.PlateCarree(),
                   facecolor='none', edgecolor='b', linewidth=0.2)
draw_maps(get_adm_maps(level='国'), linewidth=line_width)
#draw_maps(get_adm_maps(level='省'), linewidth=line_width*0.5)
# 刻度线设置
ax1.set_xticks(xticks1, crs=proj)
ax1.set_yticks(yticks1, crs=proj)
ax1.xaxis.set_tick_params(pad=pad_tick)
ax1.yaxis.set_tick_params(pad=pad_tick)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
ax1.yaxis.set_major_formatter(lat_formatter)

ax1.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
ax1.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
ax1.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
ax1.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
# ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
#最大刻度、最小刻度的刻度线长短，粗细设置
ax1.tick_params(which='major', length=tick_width*5,width=tick_width,color=tick_clolor)#最大刻度长度，宽度设置，
ax1.tick_params(which='minor', length=tick_width*2.5,width=tick_width*0.5,color=tick_clolor)#最小刻度长度，宽度设置
ax1.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=font_tick_size)
labels = ax1.get_xticklabels() + ax1.get_yticklabels()
[label.set_fontname(fontfamily) for label in labels]
[label.set_fontweight('normal') for label in labels]

# 南海小地图
if south_china_sea:
    ax_sub = fig.add_axes(ax1.get_position(), projection=proj)
    ax_sub.set_extent([105, 125, 0, 25], crs=proj)
    a_sub = ax_sub.contourf(grids['lon'], grids['lat'], obs_78_eof[pc, :, :], cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='both', transform=proj)
    ax_sub.add_feature(cfeature.LAND.with_scale('10m'),color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
    draw_maps(get_adm_maps(level='国'), linewidth=line_width)
    draw_maps(get_adm_maps(level='省'), linewidth=line_width*0.5)
    adjust_sub_axes(ax1, ax_sub, shrink=south_china_sea_shink)

#############
# color bar位置
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[左,下,右,上]
cb1 = plt.colorbar(a1, cax=fig.add_axes([0.125, 0.05, 0.775, 0.016]), orientation='horizontal')#orientation为水平或垂直
cb1.ax.tick_params(length=0, labelsize=font_colorbar_size, color='lightgray', pad=2)#length为刻度线的长度
# colorbar上的刻度值
tick_locator = ticker.FixedLocator([-1.0, -0.8, -0.6, -0.4, -0.2, 0.2, 0.4, 0.6, 0.8, 1.0])
cb1.locator = tick_locator
plt.subplots_adjust(wspace=wspce, hspace=hspce)#wspace、hspace左右、上下的间距

plt.savefig(fr'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\CMIP6_historical_{zone}_tamax_eof_{pc+1}.png', dpi=1000, bbox_inches='tight')
print('Finish')
