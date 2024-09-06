from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 专门提供经纬度的
import numpy as np
import pymannkendall as mk
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
from cnmaps import get_adm_maps, draw_maps
from eofs.standard import Eof
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator
from toolbar.masked import masked   # 气象工具函数
import pandas as pd
import tqdm
import seaborn as sns

from toolbar.pre_whitening import ws2001

# 数据读取
data_year = ['1979', '2022']
# 读取CN05.1逐日最高气温数据
CN051_1 = xr.open_dataset(r"E:\data\CN05.1\1961_2021\CN05.1_Tmax_1961_2021_daily_025x025.nc")
CN051_2 = xr.open_dataset(r"E:\data\CN05.1\2022\CN05.1_Tmax_2022_daily_025x025.nc")
Tmax_cn051 = xr.concat([CN051_1, CN051_2], dim='time')
if eval(input("1)是否计算全国极端高温95分位相对阈值(0/1)?\n")):
    Tmax_sort95 = Tmax_cn051['tmax'].sel(time=slice(data_year[0]+'-01-01', data_year[1]+'-12-31')).quantile(0.95, dim='time')     # 全国极端高温95分位相对阈值
    Tmax_sort95.to_netcdf(r"D:\CODES\Python\Meteorological\paper1\cache\NationalHighTemp_95threshold.nc")
    del Tmax_sort95     # 释放Tmax_sort95占用内存,优化代码性能
Tmax_sort95 = xr.open_dataset(r"cache\NationalHighTemp_95threshold.nc")  # 读取缓存
if eval(input("2)是否计算全国极端高温日温度距平(0/1)?\n")):
    EHD = Tmax_cn051.sel(time=slice(data_year[0]+'-01-01', data_year[1]+'-12-31')) - Tmax_sort95  # 温度距平
    EHD = EHD.where(EHD >= 0, np.nan)  # 极端高温日温度距平
    EHD = EHD-EHD+1  # 数据二值化处理(1:极端高温,np.nan:非极端高温)
    EHD.to_netcdf(r"D:\CODES\Python\Meteorological\paper1\cache\EHD.nc")
    del EHD  # 释放EHD占用内存,优化代码性能
if eval(input("3)是否计算长江流域极端高温地区占比(0/1)?\n")):
    EHD = xr.open_dataset(r"cache\EHD.nc")  # 读取缓存
    EHD = masked(EHD, r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
    EHD = EHD.sel(time=EHD['time.month'].isin([6, 7, 8, 9]))  # 选择6、7、8、9月数据  # 格点数
    station_num = masked((CN051_2-CN051_2+1).sel(time='2022-01-01'), r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域站点数
    station_num = station_num.sum()  # 长江流域格点数
    EHDstations_zone = EHD.sum(dim=['lat', 'lon'])/station_num  # 长江流域逐日极端高温格点占比
    # 将数据按日序分组，并转换为DataArray格式
    EHDstations_zone = xr.DataArray(EHDstations_zone['tmax'].to_numpy().reshape([44, 122]), coords=[[str(i) for i in range(eval(data_year[0]), eval(data_year[1]) + 1)], [str(i) for i in range(1, 122 + 1)]], dims=['year', 'day'])
    EHDstations_zone.to_netcdf(r"D:\CODES\Python\Meteorological\paper1\cache\EHDstations_zone.nc")
    del EHDstations_zone  # 释放EHDstations_zone占用内存,优化代码性能
if eval(input("4)是否计算长江流域极端高温日数高发期去趋势变率(0/1)?\n")):
    EHD = xr.open_dataset(r"cache\EHD.nc")  # 读取缓存
    EHD = masked(EHD, r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
    # 截取目标时段数据
    EHD_7 = EHD.sel(time=EHD['time.month'].isin([7]))
    EHD_7 = EHD_7.sel(time=EHD_7['time.day'].isin(range(16, 32)))
    EHD_8 = EHD.sel(time=EHD['time.month'].isin([8]))
    EHD_8 = EHD_8.sel(time=EHD_8['time.day'].isin(range(1, 20)))
    # 合并数据,并按时间排序
    EHD_concat = xr.concat([EHD_7, EHD_8], dim='time').sortby('time')
    EHD_concat.fillna(0)  # 数据二值化处理(1:极端高温,0:非极端高温)
    EHD_concat = EHD_concat['tmax'].groupby('time.year').sum('time')  # 计算目标时段累计极端高温日数
    EHD_concat = masked(EHD_concat, r"C:\Users\10574\OneDrive\File\气象数据资料\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域EHD温度距平
    # 计算EOF
    eof = Eof(EHD_concat.to_numpy())  # 进行eof分解
    Modality = eof.eofs(eofscaling=2, neofs=2)
    PC = eof.pcs(pcscaling=1, npcs=2)
    s = eof.varianceFraction(neigs=2)
    # OLS趋势分析
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(np.arange(0, eval(data_year[1]) - eval(data_year[0]) + 1), PC[:, 0])
    OLS_detrended = PC[:, 0] - np.arange(0, eval(data_year[1]) - eval(data_year[0]) + 1) * slope - intercept
    # 预白化 + Sen's Slope
    k, b = mk.sens_slope(ws2001(PC[:, 0]))
    SEN_detrended = PC[:, 0] - np.arange(0, eval(data_year[1]) - eval(data_year[0]) + 1) * k - b
    np.save(r"cache\OLS_detrended.npy", OLS_detrended)
    np.save(r"cache\SEN_detrended.npy", SEN_detrended)
    del EHD_concat, eof, Modality, PC, s, slope, intercept, r_value, p_value, std_err, k, b  # 释放占用内存,优化代码性能
if eval(input("5)是否计算海温时间滚动差值(0/1)?\n")):
    key_month = 6  # 关键月份(临期月份),距离研究时段最近的前向月份
    pre = xr.open_dataset(r"E:\data\NOAA\ERSSTv5\sst.mnmean.nc")['sst']
    pre = pre.sel(time=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) + 1) + '-12-31'))
    times = 0
    # 研究月份外时间滚动差值(不含同期!!)
    for m1 in range(0, 12):
        M = key_month - m1  # 前向月份
        M_cross = 0  # 前向月份跨年标志
        if M <= 0:  # 向前跨年
            M += 12
            M_cross = 1
        for m2 in range(0, 12-m1):
            if m2 == 0:
                if M_cross == 0:
                    sst_output = pre.sel(time=slice(str(eval(data_year[0])) + '-01-01', str(eval(data_year[1])) + '-12-31'))
                    sst_output = sst_output.sel(time=sst_output['time.month'].isin([M]))
                    sst_output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\sst_diff\sst_{times+1}_{M}_{M}.nc")
                    times += 1
                    del sst_output
                elif M_cross == 1:
                    sst_output = pre.sel(time=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) - 1) + '-12-31'))
                    sst_output = sst_output.sel(time=sst_output['time.month'].isin([M]))
                    sst_output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\sst_diff\sst_{times+1}_{M}_{M}.nc")
                    times += 1
                    del sst_output
            else:
                m = M - m2  # 后向月份
                m_cross = M_cross  # 后向月份跨年标志(为何直接用=M_cross? 因为前向月份已跨年,后向月份必跨年)
                if m <= 0:
                    m += 12
                    m_cross = 1
                sst_forward = pre.sel(time=slice(str(eval(data_year[0]) - M_cross) + '-01-01', str(eval(data_year[1]) - M_cross) + '-12-31'))
                sst_forward = sst_forward.sel(time=sst_forward['time.month'].isin([M]))
                sst_backfore = pre.sel(time=slice(str(eval(data_year[0]) - m_cross) + '-01-01', str(eval(data_year[1]) - m_cross) + '-12-31'))
                sst_backfore = sst_backfore.sel(time=sst_backfore['time.month'].isin([m]))
                sst_output = sst_forward.to_numpy() - sst_backfore.to_numpy()
                sst_output = xr.DataArray(sst_output.data, coords=[('time', sst_forward['time.year'].data),
                                                              ('lat', sst_forward['lat'].data),
                                                              ('lon', sst_forward['lon'].data)]).to_dataset(name='sst')
                sst_output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\sst_diff\sst_{times+1}_{M}_{m}.nc")
                times += 1
                del sst_output, sst_forward, sst_backfore
if eval(input("6)是否计算陆地降水时间滚动差值(0/1)?\n")):
    key_month = 6  # 关键月份(临期月份),距离研究时段最近的前向月份
    pre = xr.open_dataset(r"E:\data\CRU\grid\pre\cru_ts4.08.1901.2023.pre.dat.nc")['pre']
    pre = pre.sel(time=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) + 1) + '-12-31'))
    times = 0
    # 研究月份外时间滚动差值(不含同期!!)
    for m1 in range(0, 12):
        M = key_month - m1  # 前向月份
        M_cross = 0  # 前向月份跨年标志
        if M <= 0:  # 向前跨年
            M += 12
            M_cross = 1
        for m2 in range(0, 12-m1):
            if m2 == 0:
                if M_cross == 0:
                    output = pre.sel(time=slice(str(eval(data_year[0])) + '-01-01', str(eval(data_year[1])) + '-12-31'))
                    output = output.sel(time=output['time.month'].isin([M]))
                    output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\pre_diff\pre_{times+1}_{M}_{M}.nc")
                    times += 1
                    del output
                elif M_cross == 1:
                    output = pre.sel(time=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) - 1) + '-12-31'))
                    output = output.sel(time=output['time.month'].isin([M]))
                    output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\pre_diff\pre_{times+1}_{M}_{M}.nc")
                    times += 1
                    del output
            else:
                m = M - m2  # 后向月份
                m_cross = M_cross  # 后向月份跨年标志(为何直接用=M_cross? 因为前向月份已跨年,后向月份必跨年)
                if m <= 0:
                    m += 12
                    m_cross = 1
                forward = pre.sel(time=slice(str(eval(data_year[0]) - M_cross) + '-01-01', str(eval(data_year[1]) - M_cross) + '-12-31'))
                forward = forward.sel(time=forward['time.month'].isin([M]))
                backfore = pre.sel(time=slice(str(eval(data_year[0]) - m_cross) + '-01-01', str(eval(data_year[1]) - m_cross) + '-12-31'))
                backfore = backfore.sel(time=backfore['time.month'].isin([m]))
                output = forward.to_numpy() - backfore.to_numpy()
                output = xr.DataArray(output.data, coords=[('time', forward['time.year'].data),
                                                              ('lat', forward['lat'].data),
                                                              ('lon', forward['lon'].data)]).to_dataset(name='pre')
                output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\pre_diff\pre_{times+1}_{M}_{m}.nc")
                times += 1
                del output, forward, backfore
if eval(input("7)是否计算全球降水时间滚动差值(0/1)?\n")):
    key_month = 6  # 关键月份(临期月份),距离研究时段最近的前向月份
    pre = xr.open_dataset(r"E:\data\NOAA\PREC\precip.mon.anom.nc")['precip']
    pre = pre.sel(time=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) + 1) + '-12-31'))
    times = 0
    # 研究月份外时间滚动差值(不含同期!!)
    for m1 in range(0, 12):
        M = key_month - m1  # 前向月份
        M_cross = 0  # 前向月份跨年标志
        if M <= 0:  # 向前跨年
            M += 12
            M_cross = 1
        for m2 in range(0, 12-m1):
            if m2 == 0:
                if M_cross == 0:
                    output = pre.sel(time=slice(str(eval(data_year[0])) + '-01-01', str(eval(data_year[1])) + '-12-31'))
                    output = output.sel(time=output['time.month'].isin([M]))
                    output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\glopre_diff\pre_{times+1}_{M}_{M}.nc")
                    times += 1
                    del output
                elif M_cross == 1:
                    output = pre.sel(time=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) - 1) + '-12-31'))
                    output = output.sel(time=output['time.month'].isin([M]))
                    output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\glopre_diff\pre_{times+1}_{M}_{M}.nc")
                    times += 1
                    del output
            else:
                m = M - m2  # 后向月份
                m_cross = M_cross  # 后向月份跨年标志(为何直接用=M_cross? 因为前向月份已跨年,后向月份必跨年)
                if m <= 0:
                    m += 12
                    m_cross = 1
                forward = pre.sel(time=slice(str(eval(data_year[0]) - M_cross) + '-01-01', str(eval(data_year[1]) - M_cross) + '-12-31'))
                forward = forward.sel(time=forward['time.month'].isin([M]))
                backfore = pre.sel(time=slice(str(eval(data_year[0]) - m_cross) + '-01-01', str(eval(data_year[1]) - m_cross) + '-12-31'))
                backfore = backfore.sel(time=backfore['time.month'].isin([m]))
                output = forward.to_numpy() - backfore.to_numpy()
                output = xr.DataArray(output.data, coords=[('time', forward['time.year'].data),
                                                              ('lat', forward['lat'].data),
                                                              ('lon', forward['lon'].data)]).to_dataset(name='precip')
                output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\glopre_diff\pre_{times+1}_{M}_{m}.nc")
                times += 1
                del output, forward, backfore
if eval(input("8)是否计算2m气温时间滚动差值(0/1)?\n")):
    key_month = 6  # 关键月份(临期月份),距离研究时段最近的前向月份
    pre = xr.open_dataset(r"E:\data\ERA5\ERA5_singleLev\ERA5_sgLEv.nc")['t2m']
    pre = pre.sel(date=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) + 1) + '-12-31'))
    pre = xr.DataArray(pre.data, coords=[('time', pd.to_datetime(pre['date'], format="%Y%m%d")),
                                            ('lat', pre['latitude'].data),
                                            ('lon', pre['longitude'].data)]).to_dataset(name='t2m')
    times = 0
    # 研究月份外时间滚动差值(不含同期!!)
    for m1 in range(0, 12):
        M = key_month - m1  # 前向月份
        M_cross = 0  # 前向月份跨年标志
        if M <= 0:  # 向前跨年
            M += 12
            M_cross = 1
        for m2 in range(0, 12-m1):
            if m2 == 0:
                if M_cross == 0:
                    output = pre.sel(time=slice(str(eval(data_year[0])) + '-01-01', str(eval(data_year[1])) + '-12-31'))
                    output = output.sel(time=output['time.month'].isin([M]))
                    output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\2mT\diff\2mT_{times+1}_{M}_{M}.nc")
                    times += 1
                    del output
                elif M_cross == 1:
                    output = pre.sel(time=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) - 1) + '-12-31'))
                    output = output.sel(time=output['time.month'].isin([M]))
                    output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\2mT\diff\2mT_{times+1}_{M}_{M}.nc")
                    times += 1
                    del output
            else:
                m = M - m2  # 后向月份
                m_cross = M_cross  # 后向月份跨年标志(为何直接用=M_cross? 因为前向月份已跨年,后向月份必跨年)
                if m <= 0:
                    m += 12
                    m_cross = 1
                forward = pre.sel(time=slice(str(eval(data_year[0]) - M_cross) + '-01-01', str(eval(data_year[1]) - M_cross) + '-12-31'))
                forward = forward.sel(time=forward['time.month'].isin([M]))
                backfore = pre.sel(time=slice(str(eval(data_year[0]) - m_cross) + '-01-01', str(eval(data_year[1]) - m_cross) + '-12-31'))
                backfore = backfore.sel(time=backfore['time.month'].isin([m]))
                output = forward['t2m'].to_numpy() - backfore['t2m'].to_numpy()
                output = xr.DataArray(output.data, coords=[('time', forward['time.year'].data),
                                                              ('lat', forward['lat'].data),
                                                              ('lon', forward['lon'].data)]).to_dataset(name='t2m')
                output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\2mT\diff\2mT_{times+1}_{M}_{m}.nc")
                times += 1
                del output, forward, backfore
if eval(input("9)是否计算各气压层UVZ时间滚动差值(0/1)?\n")):
    key_month = 6  # 关键月份(临期月份),距离研究时段最近的前向月份
    var_name = input("计算各气压层u?v?z?\n")
    pre = xr.open_dataset(r"E:\data\ERA5\ERA5_pressLev\era5_pressLev.nc").sel(
        date=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) + 1) + '-12-31'),
        pressure_level=[200, 500, 600, 700, 850],
        latitude=[90 - i*0.5 for i in range(361)], longitude=[i*0.5 for i in range(720)])[var_name]
    pre = xr.DataArray(pre.data, coords=[('time', pd.to_datetime(pre['date'], format="%Y%m%d")),
                                            ('lat', pre['latitude'].data),
                                            ('lon', pre['longitude'].data)]).to_dataset(name=var_name)
    times = 0
    # 研究月份外时间滚动差值(不含同期!!)
    for m1 in range(0, 12):
        M = key_month - m1  # 前向月份
        M_cross = 0  # 前向月份跨年标志
        if M <= 0:  # 向前跨年
            M += 12
            M_cross = 1
        for m2 in range(0, 12-m1):
            if m2 == 0:
                if M_cross == 0:
                    output = pre.sel(time=slice(str(eval(data_year[0])) + '-01-01', str(eval(data_year[1])) + '-12-31'))
                    output = output.sel(time=output['time.month'].isin([M]))
                    output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\uvz\{var_name}\diff\{var_name}_{times+1}_{M}_{M}.nc")
                    times += 1
                    del output
                elif M_cross == 1:
                    output = pre.sel(time=slice(str(eval(data_year[0]) - 1) + '-01-01', str(eval(data_year[1]) - 1) + '-12-31'))
                    output = output.sel(time=output['time.month'].isin([M]))
                    output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\uvz\{var_name}\diff\{var_name}_{times+1}_{M}_{M}.nc")
                    times += 1
                    del output
            else:
                m = M - m2  # 后向月份
                m_cross = M_cross  # 后向月份跨年标志(为何直接用=M_cross? 因为前向月份已跨年,后向月份必跨年)
                if m <= 0:
                    m += 12
                    m_cross = 1
                forward = pre.sel(time=slice(str(eval(data_year[0]) - M_cross) + '-01-01', str(eval(data_year[1]) - M_cross) + '-12-31'))
                forward = forward.sel(time=forward['time.month'].isin([M]))
                backfore = pre.sel(time=slice(str(eval(data_year[0]) - m_cross) + '-01-01', str(eval(data_year[1]) - m_cross) + '-12-31'))
                backfore = backfore.sel(time=backfore['time.month'].isin([m]))
                output = forward[var_name].to_numpy() - backfore[var_name].to_numpy()
                output = xr.DataArray(output.data, coords=[('time', forward['time.year'].data),
                                                              ('lat', forward['lat'].data),
                                                              ('lon', forward['lon'].data)]).to_dataset(name=var_name)
                output.to_netcdf(fr"D:\CODES\Python\Meteorological\paper1\cache\uvz\{var_name}\diff\{var_name}_{times+1}_{M}_{m}.nc")
                times += 1
                del output, forward, backfore

print("数据处理完成")
