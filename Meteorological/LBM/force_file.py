import xarray as xr
import xgrads as xg
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

force_file_address = '//wsl.localhost/Ubuntu-20.04/home/hopsong/lbm/data/frc'


def read_force_file(address=force_file_address):
    lbm = xg.open_mfdataset(f'{address}/draw.ctl')
    level = [0.99500, 0.97999, 0.94995, 0.89988, 0.82977, 0.74468, 0.64954, 0.54946, 0.45447, 0.36948, 0.29450,
        0.22953, 0.17457, 0.12440, 0.0846830, 0.0598005, 0.0449337, 0.0349146, 0.0248800, 0.00829901]
    level_p = [isig*(1000-1)+1 for isig in level]
    return lbm, level_p

""":param element: str, 强迫要素(涡度:ovor, 散度:odiv, 温度：otmp, 海平面气压/地形试验:ops=f, 湿度:osph=f)"""
def vertical_structure(element='t', ifshow=False):
    """
    Show the vertical structure of the element.
    :param element: str, 气象要素('v', 'd', 't', 'p', 'q')
    :return: None
    """
    lbm, level = read_force_file()
    element = lbm[element]
    element = xr.DataArray(element.to_numpy(), coords=[lbm['time'], level, lbm['lat'], lbm['lon']], dims=['time', 'lev', 'lat', 'lon'])
    if ifshow:
        element = element.mean(['lon', 'lat'])
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        # 设置y轴从1000hPa到0hPa
        ax.set_ylim(1000, 0)
        # 展示垂直结构
        element.plot(ax=ax, y='lev', yincrease=False)
        plt.show()

if __name__ == '__main__':
    vertical_structure('t', ifshow=True)

pass
