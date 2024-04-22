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
import seaborn as sns

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



####23个模式图+MME图+观测图
# 绘图地图要素设置
fontfamily = 'Arial'
plt.rcParams['font.sans-serif'] = ['Arial']
#plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.linewidth'] = 0.3    # 边框粗细
#plt.tight_layout()
#plt.subplots_adjust(wspace=0.045, hspace=0.001)#wspace、hspace左右、上下的间距
wspce, hspce = 0.3, 0.1
fig = plt.figure(dpi=1000, figsize=(6, 4.7))
south_china_sea = False
south_china_sea_shink = 0.4
pic_shape = [3, 3]  # 子图行列数
# 经纬度范围
extent1 = [105, 128, 17, 33]  # 经度范围，纬度范围
xticks1 = np.arange(extent1[0], extent1[1]+1, 20)
yticks1 = np.arange(extent1[2], extent1[3]+1, 10)
proj = ccrs.PlateCarree()
# 等值线值
level1 = [-45, -40, -35, -30, -25, -20, -15, -10, -5, -2, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45]
# 字体大小
font_title_size = 10 # 标题
font_title = {'family': 'Arial', 'weight': 'normal', 'size': font_title_size}
pad_title = 0.5  # 标题与图的距离
font_tick_size = 10  # 刻度
font_tick = {'family': 'Arial', 'weight': 'normal', 'size': font_tick_size}
pad_tick = 0.5  # 刻度与轴的距离
font_colorbar_size = 8  # colorbar
font_colorbar = {'family': 'Arial', 'weight': 'normal', 'size': font_colorbar_size}
line_width = 0.1    # 省界、国界线宽
tick_width = 0.3    # 刻度线宽
tick_clolor = 'black'    # 刻度线颜色
# 经纬度
xmajorLocator = MultipleLocator(10)  # x轴最大刻度
xminorLocator = MultipleLocator(2)  # x轴最小刻度
ymajorLocator = MultipleLocator(5)  # y轴最大刻度
yminorLocator = MultipleLocator(1)  # y轴最小刻度
# 数据掩膜
zone = 'SouthChina'
shp = fr"D:/CODES/Python/PythonProject/map/shp/south_china/中国南方.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
# 获取站点网格
grids = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\CN05.1\CN05.1_Tmax_1961_2021_daily_025x025.nc")
grids_lon, grids_lat = np.meshgrid(grids['lon'], grids['lat'])
# 95%分位数, 转换为摄氏度
# 时间范围
time = ['2021', '2099']
time1 = ['2021', '2040']
time2 = ['2041', '2060']
time3 = ['2081', '2099']
#数据路径
dataurl = r"C:/Users/10574/OneDrive/File/Graduation Thesis/ThesisData/CMIP6/ssp"#数据路径
dataurl_his = r"C:/Users/10574/OneDrive/File/Graduation Thesis/ThesisData/CMIP6/historical"#数据路径
ssp = ['ssp126', 'ssp245', 'ssp585']

# 数据读取
days = np.zeros((3, eval(time[1])-eval(time[0])+1, grids_lat.shape[0], grids_lon.shape[1]))

for i in range(len(ssp)):
    Model_Name = os.listdir(dataurl+'/'+ssp[i])
    day_ssp = np.zeros((len(Model_Name), eval(time[1])-eval(time[0])+1, grids_lat.shape[0], grids_lon.shape[1]))
    try:
        print(f'读取4MME缓存文件...')
        q_ssp_78_extreHighDays = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\MME_{ssp[i]}_78_extreHighDays.nc')['days']
        day_ssp_78_extreHighDays_diffmodel = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\{ssp[i]}_78_extreHighDays.nc')['days']
    except:
        for iModle in range(len(Model_Name)):
            ModelName = Model_Name[iModle]
            url = os.listdir(dataurl + '/' + ssp[i] + '/' + ModelName)
            if len(url) == 1:
                q = xr.open_dataset(dataurl + '/' + ssp[i] + '/' + ModelName + '/' + url[0])
            else:
                q = xr.open_dataset(dataurl + '/' + ssp[i] + '/' + ModelName + '/' + url[0])
                for iurl in tqdm(url[1:], desc=f'\t读取{iModle+1} {ModelName}模式 数据', unit='文件', position=0, colour='green'):
                    q = xr.concat([q, xr.open_dataset(dataurl + '/' + ssp[i] + '/' + ModelName + '/' + iurl)], dim='time')
            url_history = os.listdir(dataurl_his + '/CMIP6_historical_tasmax/day/' + ModelName)
            if len(url_history) == 1:
                q_history = xr.open_dataset(dataurl_his + '/CMIP6_historical_tasmax/day/' + ModelName + '/' + url_history[0])
            else:
                q_history = xr.open_dataset(dataurl_his + '/CMIP6_historical_tasmax/day/' + ModelName + '/' + url_history[0])
                for iurl in tqdm(url_history[1:], desc=f'\t读取{iModle+1} {ModelName}模式 数据', unit='文件', position=0, colour='green'):
                    q_history = xr.concat([q_history, xr.open_dataset(dataurl_his + '/CMIP6_historical_tasmax/day/' + ModelName + '/' + iurl)], dim='time')
            q_ssp = q['tasmax'].sel(time=slice(time[0]+'-01-01', time[1] + '-12-31'))
            q_ssp_78 = q_ssp.sel(time=q_ssp.time.dt.month.isin([7, 8]))   # 7-8月气温
            q_sort95 = q_history['tasmax'].sel(time=slice('1979-01-01', '2014-12-31')).quantile(0.95, dim='time')  # 95%分位数, 转换为摄氏度
            q_ssp_78_extreHighDays = np.zeros((eval(time[1]) - eval(time[0]) + 1, len(q['lat']), len(q['lon'])))
            for ii in tqdm(range(eval(time[0]),  eval(time[1]) + 1), desc=f'\t计算{iModle+1} {ModelName}模式 极端高温日数', unit='Year', position=0, colour='green'):
                q_ssp_78_extreHighDays[ii - eval(time[0]), :, :] = np.sum(np.where((q_ssp_78.to_numpy() - q_sort95.to_numpy()) > 0, 1, 0)[(ii - eval(time[0]))*62:(ii + 1 - eval(time[0]))*62, :, :], axis=0)    # 极端高温日数
            ######################################################################################
            # 双线性插值到CN05.1网格
            grid_q_ssp_days = np.zeros((eval(time[1]) - eval(time[0])+1, grids_lat.shape[0], grids_lon.shape[1]))
            num = 0
            for iii in tqdm(q_ssp_78_extreHighDays, desc=f'\t插值 {ModelName}模式', unit='Year', position=0, colour='green'):
                interp = interpolate.RegularGridInterpolator((q['lat'], q['lon']), iii, method='linear')
                grid_q_ssp_days[num, :, :] = interp((grids_lat, grids_lon))
                num += 1
            ######################################################################################
            day_ssp[iModle, :, :, :] = grid_q_ssp_days
        day_ssp_MME = np.mean(day_ssp, axis=0)
        print(f'\t保存4MME缓存')
        day_ssp_MME_datasaet = xr.DataArray(day_ssp_MME, coords=[[str(i) for i in range(eval(time[0]), eval(time[1]) + 1)], grids['lat'], grids['lon']], dims=['time', 'lat', 'lon'])
        day_ssp_MME_datasaet.name = 'days'
        day_ssp_MME_datasaet.to_netcdf(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\MME_{ssp[i]}_78_extreHighDays.nc')
        q_ssp_78_extreHighDays = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\MME_{ssp[i]}_78_extreHighDays.nc')['days']
        day_ssp_datasaet = xr.DataArray(day_ssp, coords=[Model_Name, [str(i) for i in range(eval(time[0]), eval(time[1]) + 1)], grids['lat'], grids['lon']], dims=['Model', 'time', 'lat', 'lon'])
        day_ssp_datasaet.name = 'days'
        day_ssp_datasaet.to_netcdf(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\{ssp[i]}_78_extreHighDays.nc')
        day_ssp_78_extreHighDays_diffmodel = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\{ssp[i]}_78_extreHighDays.nc')['days']
for i in range(len(ssp)):
    print(f'模拟{ssp[i]}情景')
    ######################################################################################
    # 裁切中国范围数据，其余数据Nan化
    q_ssp_78_extreHighDays = xr.open_dataset(fr'D:\CODES\Python\PythonProject\cache\Graduation Thesis\MME_{ssp[i]}_78_extreHighDays.nc')['days'].salem.roi(shape=split_shp)
    ######################################################################################
    q_7 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_7.npy')
    q_8 = np.load(r'D:\CODES\Python\PythonProject\毕业设计\文献图表复现\缓存文件\q_8.npy')
    q_78_obs = q_7 + q_8
    # 绘图 Nearterm
    q_ssp_78_2140 = q_ssp_78_extreHighDays.sel(time=slice(time1[0], time1[1]))
    #reg_days_1 = [[np.polyfit(np.arange(eval(time1[0]), eval(time1[1])+1, 1), q_ssp_78_2140[:, ilat, ilon] - q_78_obs[ilat, ilon], 1)[0] for ilon in range(len(grids['lon']))] for ilat in tqdm(range(len(grids['lat'])), desc='计算Delta EHDs/year', position=0, leave=True)]
    delta_2140 = q_ssp_78_2140.mean('time') - q_78_obs
    ax1 = fig.add_subplot(pic_shape[0], pic_shape[1], i*3+1, projection=proj)
    ax1.set_extent(extent1, crs=proj)
    if i == 0:
        ax1.set_title('Nearterm\n', fontdict=font_title, pad=pad_title, loc='center')
    a1 = ax1.contourf(grids['lon'], grids['lat'], delta_2140, cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='both', transform=proj)
    ax1.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
    ax1.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\cnriver\1级河流.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='b', linewidth=0.2)
    draw_maps(get_adm_maps(level='国'), linewidth=line_width)
    # 绘图 Midterm
    q_ssp_78_4160 = q_ssp_78_extreHighDays.sel(time=slice(time2[0], time2[1]))
    #reg_days_2 = [[np.polyfit(np.arange(eval(time2[0]), eval(time2[1]) + 1, 1),q_ssp_78_4160[:, ilat, ilon] - q_78_obs[ilat, ilon], 1)[0] for ilon in range(len(grids['lon']))] for ilat in tqdm(range(len(grids['lat'])), desc='计算Delta EHDs/year', position=0, leave=True)]
    delta_4160 = q_ssp_78_4160.mean('time') - q_78_obs
    ax2 = fig.add_subplot(pic_shape[0], pic_shape[1], i*3+2, projection=proj)
    ax2.set_extent(extent1, crs=proj)
    if i == 0:
        ax2.set_title('Midterm\n(a)SSP126', fontdict=font_title, pad=pad_title, loc='center')
    elif i == 1:
        ax2.set_title('(b)SSP245', fontdict=font_title, pad=pad_title, loc='center')
    elif i == 2:
        ax2.set_title('(c)SSP585', fontdict=font_title, pad=pad_title, loc='center')
    a2 = ax2.contourf(grids['lon'], grids['lat'], delta_4160, cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='both', transform=proj)
    ax2.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
    ax2.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\cnriver\1级河流.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='b', linewidth=0.2)
    draw_maps(get_adm_maps(level='国'), linewidth=line_width)
    # 绘图 Longterm
    q_ssp_78_8199 = q_ssp_78_extreHighDays.sel(time=slice(time3[0], time3[1]))
    #reg_days_3 = [[np.polyfit(np.arange(eval(time3[0]), eval(time3[1]) + 1, 1), q_ssp_78_8199[:, ilat, ilon] - q_78_obs[ilat, ilon], 1)[0] for ilon in range(len(grids['lon']))] for ilat in tqdm(range(len(grids['lat'])), desc='计算Delta EHDs/year', position=0, leave=True)]
    delta_8199 = q_ssp_78_8199.mean('time') - q_78_obs
    ax3 = fig.add_subplot(pic_shape[0], pic_shape[1], i*3+3, projection=proj)
    ax3.set_extent(extent1, crs=proj)
    if i == 0:
        ax3.set_title('Longterm\n', fontdict=font_title, pad=pad_title, loc='center')
    a3 = ax3.contourf(grids['lon'], grids['lat'], delta_8199, cmap=cmaps.BlueWhiteOrangeRed, levels=level1, extend='both', transform=proj)
    ax3.add_feature(cfeature.LAND.with_scale('10m'), color='lightgray', lw=line_width)# 添加陆地并且陆地部分全部填充成浅灰色
    ax3.add_geometries(Reader(r'D:\CODES\Python\PythonProject\map\cnriver\1级河流.shp').geometries(), ccrs.PlateCarree(), facecolor='none', edgecolor='b', linewidth=0.2)
    draw_maps(get_adm_maps(level='国'), linewidth=line_width)
    # ax1刻度线设置
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

    # ax2刻度线设置
    ax2.set_xticks(xticks1, crs=proj)
    ax2.set_yticks(yticks1, crs=proj)
    ax2.xaxis.set_tick_params(pad=pad_tick)
    ax2.yaxis.set_tick_params(pad=pad_tick)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)

    ax2.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
    ax2.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
    ax2.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
    ax2.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    #最大刻度、最小刻度的刻度线长短，粗细设置
    ax2.tick_params(which='major', length=tick_width*5,width=tick_width,color=tick_clolor)#最大刻度长度，宽度设置，
    ax2.tick_params(which='minor', length=tick_width*2.5,width=tick_width*0.5,color=tick_clolor)#最小刻度长度，宽度设置
    ax2.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)

    # ax3刻度线设置
    ax3.set_xticks(xticks1, crs=proj)
    ax3.set_yticks(yticks1, crs=proj)
    ax3.xaxis.set_tick_params(pad=pad_tick)
    ax3.yaxis.set_tick_params(pad=pad_tick)
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)

    ax3.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
    ax3.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
    ax3.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
    ax3.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
    # ax1.axes.xaxis.set_ticklabels([]) ##隐藏刻度标签
    #最大刻度、最小刻度的刻度线长短，粗细设置
    ax3.tick_params(which='major', length=tick_width*5,width=tick_width,color=tick_clolor)#最大刻度长度，宽度设置，
    ax3.tick_params(which='minor', length=tick_width*2.5,width=tick_width*0.5,color=tick_clolor)#最小刻度长度，宽度设置
    ax3.tick_params(which='both',bottom=True,top=False,left=True,labelbottom=True,labeltop=False)
    plt.rcParams['xtick.direction'] = 'out' #将x轴的刻度线方向设置向内或者外
    #设置坐标刻度值的大小以及刻度值的字体
    plt.tick_params(labelsize=font_tick_size)
    labels = ax1.get_xticklabels() + ax1.get_yticklabels()
    [label.set_fontname(fontfamily) for label in labels]
    [label.set_fontweight('normal') for label in labels]


#############
# color bar位置
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#位置[左,下,右,上]
cb1 = plt.colorbar(a1, cax=fig.add_axes([0.125, 0.05, 0.775, 0.02]), orientation='horizontal')#orientation为水平或垂直
cb1.ax.tick_params(length=0, labelsize=font_colorbar_size, color='lightgray', pad=2)#length为刻度线的长度
# colorbar上的刻度值
tick_locator = ticker.FixedLocator(level1)
cb1.locator = tick_locator
plt.subplots_adjust(wspace=wspce, hspace=hspce)#wspace、hspace左右、上下的间距

plt.savefig(fr'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\未来预估.png', dpi=1000, bbox_inches='tight')
plt.close()
obs = grids['tmax'].salem.roi(shape=split_shp).sel(time=slice('1979-01-01', '2021-12-31'))
q_obs_sort95 = grids['tmax'].salem.roi(shape=split_shp).sel(time=slice('1979-01-01', '2014-12-31')).quantile(0.95, dim='time')
obs_78 = obs.sel(time=obs.time.dt.month.isin([7, 8]))
obs_78_days = np.where((obs_78.to_numpy() - q_obs_sort95.to_numpy()) > 0, 1, 0)
obs_78_days_gridnums = np.ones((len(obs['lat']), len(obs['lon'])))
obs_78_days_gridnums = xr.DataArray(obs_78_days_gridnums, coords=[obs['lat'], obs['lon']], dims=['lat', 'lon'])
obs_78_days_gridnums.name = 'gridnums'
obs_78_days_gridnums = obs_78_days_gridnums.salem.roi(shape=split_shp).sum()
obs_78_days_avg = [obs_78_days[i*62:i*62+62].sum()/obs_78_days_gridnums for i in range(43)]

obs_78_days_avg = np.array(obs_78_days_avg)
projections_126 = xr.open_dataset(r"D:\CODES\Python\PythonProject\cache\Graduation Thesis\ssp126_78_extreHighDays.nc")['days'].mean(['lat', 'lon'])
projections_245 =xr.open_dataset(r"D:\CODES\Python\PythonProject\cache\Graduation Thesis\ssp245_78_extreHighDays.nc")['days'].mean(['lat', 'lon'])
projections_585 = xr.open_dataset(r"D:\CODES\Python\PythonProject\cache\Graduation Thesis\ssp585_78_extreHighDays.nc")['days'].mean(['lat', 'lon'])
projections = xr.concat([projections_126, projections_245, projections_585], dim='ssp')
projections = xr.DataArray(projections, coords=[ssp, Model_Name, [str(i) for i in range(eval(time[0]), eval(time[1]) + 1)]], dims=['ssp', 'model', 'time'])
palette = sns.xkcd_palette(["windows blue", "dusty purple", "red"])

sns.set(style='ticks')
fig = sns.relplot(x="time", y="days", hue="ssp", kind="line", data=projections.to_dataframe(), palette=palette, legend=False)
#fig_obs = sns.lineplot(x=[i for i in range(-42, 1)], y=obs_78_days_avg, color='gray', label='Observation')
# 图像大小
fig.fig.set_size_inches(6, 4.7)
ax = plt.gca()
# 设置横坐标的刻度范围和标记
ax.set_xlim(0, 81)
ax.set_xticks([1]+np.arange(5, 76, 5)+[79])
ax.set_xticklabels(["2021"] + [f"{i}" for i in range(2025, 2096, 5)] + ["2099"])
# 设置纵坐标的刻度范围和标记
ax.set_ylim(0, 62)
ax.set_yticks(range(0, 62, 5))
ax.set_yticklabels([f"{i}" for i in range(0, 62, 5)])
#plt.axvline(x=0, color='gray', linestyle='-', linewidth=1)
plt.axvline(x=20, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(x=40, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(x=60, color='gray', linestyle='--', linewidth=0.5)
plt.axvline(x=80, color='gray', linestyle='--', linewidth=0.5)
plt.xlabel('Year')
plt.ylabel('EHDs')
plt.legend(title='', loc='upper left', labels=['SSP126', 'SSP245', 'SSP585'])
plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\论文配图\未来预估折线图.png', dpi=1000, bbox_inches='tight')
plt.show()
print('Finish')
