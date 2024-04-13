from pprint import *
from cinrad import *
from cinrad.visualize import PPI, Section
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER



# 读取数据
path = 'D:/雷达气象实习资料/雷达实验1/2007070306.bin'
data = io.CinradReader(path, radar_type='SA')
# 任务一
radius = 400
data.set_code('Z9519')
r = [data.get_data(i, radius, 'REF') for i in data.angleindex_r]
vcs = calc.VCS(r)
sec = vcs.get_section(start_cart=(118.5, 32.5), end_cart=(117.8, 33.1))
# 画图
fig = Section(sec)
plt.show()
