from pprint import *
import cinrad
from cinrad.visualize import PPI, Section
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def ppi_plot(data, radius):
    fig = PPI(data, dpi=75, add_city_names=True)
    fig.plot_range_rings(radius, color='black', linewidth=1.0)
    for i in range(0, radius - 3, 5):
        fig.plot_range_rings(radius, color='black', linewidth=1.0)
    liner = fig.geoax.gridlines(draw_labels=True, linewidth=1.0, linestyle='--', color='gray', alpha=0.5)
    liner.top_labels = False
    liner.right_labels = False
    liner.xformatter = LONGITUDE_FORMATTER
    liner.yformatter = LATITUDE_FORMATTER
    liner.xlabel_style = {'size': 3, 'color': 'black', 'weight': 'bold'}
    liner.ylabel_style = {'size': 3, 'color': 'black', 'weight': 'bold'}


# 读取数据
path = 'D:/雷达气象实习资料/雷达实验1/2007070306.bin'
data = cinrad.io.CinradReader(path, radar_type='SA')
# 任务一
radius = 400
r = data.get_data(2, radius, 'REF')
pprint(r)
ppi_plot(r, radius)
plt.show()
# 任务二

