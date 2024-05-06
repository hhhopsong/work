import xarray as xr
import xgrads as xg
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy.interpolate import interpolate
from tqdm import tqdm

force_file_address = '//wsl.localhost/Ubuntu-20.04/home/hopsong/lbm/data/frc'

# LBM网格
lbm_lon = [i * 2.8125 for i in range(128)]
lbm_lat = [-87.8638,-85.0965,-82.3129,-79.5256,-76.7369,-73.9475,-71.1577,-68.3678,-65.5776,-62.7873,
           -59.9970,-57.2066,-54.4162,-51.6257,-48.8352,-46.0447,-43.2542,-40.4636,-37.6731,-34.8825,
           -32.0919,-29.3014,-26.5108,-23.7202,-20.9296,-18.1390,-15.3484,-12.5578,-9.76715,-6.97653,
           -4.18592,-1.39531,1.39531,4.18592,6.97653,9.76715,12.5578,15.3484,18.1390,20.9296,23.7202,
           26.5108,29.3014,32.0919,34.8825,37.6731,40.4636,43.2542,46.0447,48.8352,51.6257,54.4162,
           57.2066,59.9970,62.7873,65.5776,68.3678,71.1577,73.9475,76.7369,79.5256,82.3129,85.0965,
           87.86384]
lon_grid, lat_grid = np.meshgrid(lbm_lon, lbm_lat)
level_sig = [0.99500, 0.97999, 0.94995, 0.89988, 0.82977, 0.74468, 0.64954, 0.54946, 0.45447, 0.36948, 0.29450,
         0.22953, 0.17457, 0.12440, 0.0846830, 0.0598005, 0.0449337, 0.0349146, 0.0248800, 0.00829901]
level_sigp = [isig * (1000 - 1) + 1 for isig in level_sig]
level_p = [1000, 950, 900, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5]

""":param address: str, 强迫场文件地址"""

def read_force_file(address=force_file_address):
    lbm = xg.open_mfdataset(f'{address}/draw.ctl')
    return lbm, level_sigp

""":param element: str, 强迫要素(涡度:ovor, 散度:odiv, 温度：otmp, 海平面气压/地形试验:ops=f, 湿度:osph=f)"""
def vertical_structure(element='t', show=False):
    """
    Show the vertical structure of the element.
    :param element: str, 气象要素('v', 'd', 't', 'p', 'q')
    :return: element: xr.DataArray, 垂直结构数据
    """
    lbm, level = read_force_file()
    element = lbm[element]
    element = np.where(element.to_numpy() == 0, np.nan, element.to_numpy().byteswap())
    element = xr.DataArray(element, coords=[lbm['time'], lbm['lev'], lbm['lat'], lbm['lon']], dims=['time', 'lev', 'lat', 'lon'])
    if show:
        element = element.mean(['lon', 'lat'])
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        # 设置y轴从1000hPa到0hPa
        ax.set_ylim(1000, 0)
        x_max = np.nanmax(element)
        x_min = np.nanmin(element)
        x_size = np.abs(x_max) if np.abs(x_max) > np.abs(-x_min) else np.abs(x_min)
        ax.set_xlim(-x_size*1.2, x_size*1.2)
        # 展示垂直结构
        element.plot(ax=ax, y='lev', yincrease=False, marker='o')
        plt.show()
    return element


def interp_to_lbm(data=None):
    """
    LBM网格插值函数
    :param data: xr.DataArray, 自定义强迫场数据, 数据顺序为(lev, lat, lon), lev为P坐标系
    :return: np.array, 插值后的数据
    """
    if data is None:
        raise ValueError('data参数不能为空')
    elif len(data['lev']) <= 1:
        raise ValueError('data垂直层次错误')
    elif len(data['lev']) > 1:
        data = xr.DataArray(data.to_numpy(), coords=[data['lev'], data['lat'], data['lon']], dims=['lev', 'lat', 'lon'])
        interp = interpolate.RegularGridInterpolator((data['lev'], data['lat'], data['lon']), data,method='linear')
        data_interp = interp((level_p, lat_grid, lon_grid))
        data_interp = np.where(np.isnan(data_interp), 0, data_interp)
        return data_interp

def vertical_profile(data=None, K=20, kvpr=2, vamp=8., vdil=20., vcnt=0.45, show=True):
    """
    生成强迫场的理想化垂直结构.
    :param data: xr.DataArray, 自定义垂直结构数据
    :param K: int, 垂直层数
    :param kvpr: int, 理想化结构类型（1:三角函数, 2:Gamma函数, 3:垂直均匀）
    :param vamp: float, 振幅(1为原数据振幅)
    :param vdil: float, 振幅衰减率
    :param vcnt: float, 振幅中心(sigma坐标)
    :param idealization_make: bool, 是否进行理想化结构生成
    :return: None
    """
    if data is not None:
        # data数据顺序为(lev), lev为P坐标系[1000-0hPa]
        if len(data['lev']) <= 1:
            raise ValueError('data垂直层次错误')
        data = xr.DataArray(data.to_numpy(), coords=[data['lev']], dims=['lev'])
        interp = interpolate.RegularGridInterpolator((data['lev']), data, method='linear')   # date插值到lbm网格上
        data_interp = interp(level_p)
        data_interp = np.where(np.isnan(data_interp), 0, data_interp)
        return [i for i in data_interp]
    structure = [vamp for i in range(K)]
    # 生成理想化垂直结构
    # 生成垂直结构
    for i in range(K):
        if kvpr == 1:
            # 三角函数
            structure[i] = vamp * np.sin(np.pi * level_sig[i])
        elif kvpr == 2:
            # Gamma函数
            structure[i] = vamp * np.exp(-vdil * (level_sig[i] - vcnt) ** 2)
        elif kvpr == 3:
            # 垂直均匀
            structure[i] = vamp
        else:
            raise ValueError('kvpr参数错误')
    if show:
        # 画图
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_title('理想强迫场垂直结构')
        # 设置y轴从1000hPa到0hPa,且为log坐标
        ax.set_ylim(1000, 0)
        x_max = np.nanmax(structure)
        x_min = np.nanmin(structure)
        x_size = np.abs(x_max) if np.abs(x_max) > np.abs(-x_min) else np.abs(x_min)
        ax.set_xlim(-x_size*1.2, x_size*1.2)
        structure_plot = xr.DataArray(structure, coords=[level_p], dims=['lev'])
        structure_plot.plot(ax=ax, y='lev', marker='o')
        plt.show()
    return structure


def horizontal_profile(data=None, khpr=1, hamp=0.25, xdil=23., ydil=6.5, xcnt=77., ycnt=-1.5):
    """
    生成强迫场的理想化水平结构.
    :param data: xr.DataArray, 自定义水平结构数据
    :param khpr: int, 理想化结构类型（1:椭圆函数, 2:区域一致）
    :param hamp: float, 振幅(1为原数据振幅)
    :param xdil: float, 强迫区域半径
    :param ydil: float, 强迫区域半径
    :param xcnt: float, 强迫中心经度(-180, 180)
    :param ycnt: float, 强迫中心纬度(-90, 90)
    :return: None
    """
    if data is not None:
        # data须为单一时次单层数据, 数据顺序为(lat, lon)
        if len(data['lev']) != 1:
            raise ValueError('data垂直层次错误')
        data = xr.DataArray(data.to_numpy(), coords=[data['lat'], data['lon']], dims=['lat', 'lon'])
        interp = interpolate.RegularGridInterpolator((data['lat'], data['lon']), data, method='linear')   # date插值到lbm网格上
        data_interp = interp((lat_grid, lon_grid))
        data_interp = np.where(np.isnan(data_interp), 0, data_interp)
        return data_interp
    # 生成理想化水平结构
    if xcnt < -180 or xcnt > 180 or ycnt < -90 or ycnt > 90:
        raise ValueError('xcnt或ycnt参数错误')
    elif xdil <= 0 or ydil <= 0:
        raise ValueError('xdil或ydil参数错误')
    hor_structure = np.zeros((len(lbm_lat), len(lbm_lon)))
    for i in range(len(lbm_lat)):
        for j in range(len(lbm_lon)):
            if khpr == 1:
                # 椭圆函数
                dx = (lbm_lon[j] - xcnt) ** 2 / xdil ** 2
                if lbm_lon[j] < 0. and xcnt + xdil >= 180.:
                    dx = (360. + lbm_lon[j] - xcnt) ** 2 / xdil ** 2
                elif lbm_lon[j] > 0. and xcnt - xdil <= -180.:
                    dx = ( lbm_lon[j] - 360. - xcnt )**2 / xdil**2
                dy = (lbm_lat[i] - ycnt) ** 2 / ydil ** 2
                d = dx + dy
                if d > 1.:
                    hor_structure[i, j] = 0.
                else:
                    d = np.sqrt(d)
                    hor_structure[i, j] = hamp * (1. - d)
            elif khpr == 2:
                # 区域平均
                if lbm_lat[i] >= ycnt - ydil and lbm_lat[i] <= ycnt + ydil:
                    dx = lbm_lon[j] - xcnt
                    if lbm_lon[j] < 0. and xcnt + xdil >= 180.:
                        dx = 360. + lbm_lon[j] - xcnt
                    elif lbm_lon[j] > 0. and xcnt - xdil <= -180.:
                        dx = lbm_lon[j] - 360. - xcnt
                    if np.abs(dx) <= xdil:
                        hor_structure[i, j] = hamp
                else:
                    hor_structure[i, j] = 0.
            else:
                raise ValueError('khpr参数错误')
    return hor_structure


def mk_frc(K=20, hor_structure=None, )

if __name__ == '__main__':
    #vertical_structure('t', show=True)
    vertical_profile(kvpr=2, vamp=8., vdil=20., vcnt=0.45)
pass
