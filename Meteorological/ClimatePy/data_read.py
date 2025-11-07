import numpy as np
import xarray as xr
import pandas as pd

def era5_p(data_path, begin_year, end_year, level, var_name):
    '''
    读取ERA5数据
    :param data_path:  ERA5数据路径
    :param begin_year:  开始年份
    :param end_year:  结束年份
    :param p:  压力层(list)
    :param var_name:  变量名称
    :return:
    '''
    try:
        era5 = xr.open_dataset(data_path).sel(
            date=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'),
            pressure_level=level,
            latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
        pre = xr.Dataset({var_name:(['time', 'level', 'lat', 'lon'], era5.data)},
                         coords={'time': pd.to_datetime(era5['date'], format="%Y%m%d"),
                                 'level': era5['pressure_level'].data,
                                 'lat': era5['latitude'].data,
                                 'lon': era5['longitude'].data})
    except:
        era5 = xr.open_dataset(data_path).sel(
            valid_time=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'),
            pressure_level=level,
            latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
        pre = xr.Dataset({var_name: (['time', 'level', 'lat', 'lon'], era5.data)},
                         coords={'time': pd.to_datetime(era5['valid_time'], format="%Y%m%d"),
                                 'level': era5['pressure_level'].data,
                                 'lat': era5['latitude'].data,
                                 'lon': era5['longitude'].data})
    return pre

def era5_hp(data_path, begin_year, end_year, level, var_name):
    '''
    读取ERA5数据
    :param data_path:  ERA5数据路径
    :param begin_year:  开始年份
    :param end_year:  结束年份
    :param p:  压力层(list)
    :param var_name:  变量名称
    :return:
    '''
    era5 = xr.open_dataset(data_path).sel(
        valid_time=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'),
        pressure_level=level,
        latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
    pre = xr.Dataset({var_name:(['time', 'level', 'lat', 'lon'], era5.data)},
                        coords={'time': pd.to_datetime(era5['valid_time'], format="%Y%m%d"),
                                'level': era5['pressure_level'].data,
                                'lat': era5['latitude'].data,
                                'lon': era5['longitude'].data})
    return pre

def era5_p_AfterOpen(data, begin_year, end_year, level, var_name):
    try:
        era5 = data.sel(date=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'), pressure_level=level,
            latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
        var = xr.Dataset({var_name:(['time', 'level', 'lat', 'lon'], era5.data)},
                        coords={'time': pd.to_datetime(era5['date'], format="%Y%m%d"),
                                'level': era5['pressure_level'].data,
                                'lat': era5['latitude'].data,
                                'lon': era5['longitude'].data})
    except:
        era5 = data.sel(valid_time=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'), pressure_level=level,
            latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
        var = xr.Dataset({var_name:(['time', 'level', 'lat', 'lon'], era5.data)},
                        coords={'time': pd.to_datetime(era5['valid_time'], format="%Y%m%d"),
                                'level': era5['pressure_level'].data,
                                'lat': era5['latitude'].data,
                                'lon': era5['longitude'].data})
    return var

def era5_AfterOpen(data, begin_year, end_year, var_name, level=None):
    try:
        if level == None:
            era5 = data.sel(date=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'),
                latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
            var = xr.Dataset({var_name: (['time', 'lat', 'lon'], era5.data)},
                             coords={'time': pd.to_datetime(era5['date'], format="%Y%m%d"),
                                     'lat': era5['latitude'].data,
                                     'lon': era5['longitude'].data})
        else:
            era5 = data.sel(date=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'), pressure_level=level,
                latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
            var = xr.Dataset({var_name:(['time', 'level', 'lat', 'lon'], era5.data)},
                            coords={'time': pd.to_datetime(era5['date'], format="%Y%m%d"),
                                    'level': era5['pressure_level'].data,
                                    'lat': era5['latitude'].data,
                                    'lon': era5['longitude'].data})
    except:
        if level == None:
            era5 = data.sel(valid_time=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'),
                latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
            var = xr.Dataset({var_name: (['time', 'lat', 'lon'], era5.data)},
                             coords={'time': pd.to_datetime(era5['valid_time'], format="%Y%m%d"),
                                    'lat': era5['latitude'].data,
                                    'lon': era5['longitude'].data})
        else:
            era5 = data.sel(valid_time=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'), pressure_level=level,
                latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
            var = xr.Dataset({var_name:(['time', 'level', 'lat', 'lon'], era5.data)},
                            coords={'time': pd.to_datetime(era5['valid_time'], format="%Y%m%d"),
                                    'level': era5['pressure_level'].data,
                                    'lat': era5['latitude'].data,
                                    'lon': era5['longitude'].data})
    return var

def era5_s(data_path, begin_year, end_year, var_name):
    '''
    读取ERA5数据
    :param data_path:  ERA5数据路径
    :param begin_year:  开始年份
    :param end_year:  结束年份
    :param var_name:  变量名称
    :return:
    '''
    try:
        era5 = xr.open_dataset(data_path).sel(
            date=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'),
            latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
        pre = xr.Dataset({var_name:(['time', 'lat', 'lon'], era5.data)},
                         coords={'time': pd.to_datetime(era5['date'], format="%Y%m%d"),
                                 'lat': era5['latitude'].data,
                                 'lon': era5['longitude'].data})
    except:
        era5 = xr.open_dataset(data_path).sel(
            valid_time=slice(str(begin_year) + '-01-01', str(end_year + 1) + '-12-31'),
            latitude=[90 - i * 0.5 for i in range(361)], longitude=[i * 0.5 for i in range(720)])[var_name]
        pre = xr.Dataset({var_name:(['time', 'lat', 'lon'], era5.data)},
                         coords={'time': pd.to_datetime(era5['valid_time'], format="%Y%m%d"),
                                 'lat': era5['latitude'].data,
                                 'lon': era5['longitude'].data})
    return pre


def era5_land(data_path, begin_year, end_year, var_name):
    '''
    读取ERA5数据
    :param data_path:  ERA5数据路径
    :param begin_year:  开始年份
    :param end_year:  结束年份
    :param var_name:  变量名称
    :return:
    '''
    try:
        era5 = xr.open_dataset(data_path).sel(
            date=slice(str(begin_year) + '-01-01', str(end_year) + '-12-31'),
            latitude=slice(None, None, 5), longitude=slice(None, None, 10))[var_name]
        pre = xr.Dataset({var_name:(['time', 'lat', 'lon'], era5.data)},
                         coords={'time': pd.to_datetime(era5['date'], format="%Y%m%d"),
                                 'lat': era5['latitude'].data,
                                 'lon': era5['longitude'].data})
    except:
        era5 = xr.open_dataset(data_path).sel(
            valid_time=slice(str(begin_year) + '-01-01', str(end_year) + '-12-31'),
            latitude=slice(None, None, 5), longitude=slice(None, None, 5))[var_name]
        pre = xr.Dataset({var_name:(['time', 'lat', 'lon'], era5.data)},
                         coords={'time': pd.to_datetime(era5['valid_time'], format="%Y%m%d"),
                                 'lat': era5['latitude'].data,
                                 'lon': era5['longitude'].data})
    return pre


def prec(data_path, begin_year, end_year):
    pre = xr.open_dataset(data_path)['precip']
    pre = pre.sel(time=slice(str(begin_year) + '-01-01', str(end_year) + '-12-31'))
    pre = xr.Dataset({'pre': (['time', 'lat', 'lon'], pre.data)},
                     coords={'time': pre['time'].data,
                             'lat': pre['lat'].data,
                             'lon': pre['lon'].data})
    return pre

def ersst(data_path, begin_year, end_year):
    ersst = xr.open_dataset(data_path)['sst']
    ersst = ersst.sel(time=slice(str(begin_year) + '-01-01', str(end_year) + '-12-31'))
    ersst = xr.Dataset({'sst': (['time', 'lat', 'lon'], ersst.data)},
                     coords={'time': ersst['time'].data,
                             'lat': ersst['lat'].data,
                             'lon': ersst['lon'].data})
    return ersst


if __name__ == '__main__':
    era5_land("E:/data/ERA5/ERA5_land/uv_2mTTd_sfp_pre_0.nc", 1961, 2022, 't2m')