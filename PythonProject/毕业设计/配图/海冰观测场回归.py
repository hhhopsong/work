from cartopy import crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # ר���ṩ��γ�ȵ�
from cartopy.util import add_cyclic_point
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patheffects as path_effects
import matplotlib.path as mpath
from cnmaps import get_adm_maps, draw_maps
from matplotlib import ticker
import cmaps
from matplotlib.ticker import MultipleLocator, FixedLocator
from eofs.standard import Eof
from scipy.ndimage import filters
from tqdm import tqdm
import geopandas as gpd
import salem
from tools.TN_WaveActivityFlux import TN_WAF
import pprint


std_q78 = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\std_q78.nc')
shp = fr"D:/CODES/Python/PythonProject/map/shp/south_china/�й��Ϸ�.shp"
split_shp = gpd.read_file(shp)
split_shp.crs = 'wgs84'
std_q78 = std_q78.salem.roi(shape=split_shp)
pc_index = 0
# eof�ֽ�
eof_78 = Eof(std_q78['tmax'].to_numpy())  # ����eof�ֽ�
EOF_78 = eof_78.eofs(eofscaling=2,
                     neofs=2)  # �õ��ռ�ģ̬U eofscaling �Եõ��ĳ����з��� ��1Ϊ��������ֵƽ������2Ϊ��������ֵƽ������Ĭ��Ϊ0������ neofs��������Ŀռ�ģ̬������
PC_78 = eof_78.pcs(pcscaling=1, npcs=2)  # ͬ�� npcs���������ʱ�����и���
s_78 = eof_78.varianceFraction(neigs=2)  # �õ�ǰneig��ģ̬�ķ����
# ���ݶ�ȡ
sat = xr.open_dataset(r"C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\ERA5\ERA5_2mTemperature_MeanSlp.nc")

# ������Ƭ
T2m = sat['t2m'].sel(time=slice('1979-01-01', '2014-12-31'))
T2m_78 = T2m.sel(time=T2m.time.dt.month.isin([7, 8]))
# ��γ��
lon_sat = T2m['longitude']
lat_sat = T2m['latitude']

# ���߰��·����ݽ���ÿ��ƽ��
sat_78 = T2m_78.groupby('time.year').mean('time')
sat_78 = np.array(sat_78)
try:
    # ��ȡ���ϵ��
    reg_sat = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sat.nc')
except:
    # �����ݻع鵽PC��
    reg_sat = [[np.polyfit(PC_78[:, pc_index], sat_78[:, ilat, ilon], 1)[0] for ilon in range(len(lon_sat))] for ilat in tqdm(range(len(lat_sat)), desc='����SAT', position=0, leave=True)]
    xr.DataArray(reg_sat, coords=[lat_sat, lon_sat], dims=['lat', 'lon']).to_netcdf(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sat.nc')
    ###�����ٶ�ȡ
    reg_sat = xr.open_dataset(r'D:\CODES\Python\PythonProject\cache\Graduation Thesis\reg_sat.nc')

# ����������0.05����
from scipy.stats import t

# �������ɶ�.
n = len(PC_78[:, 0])
# ʹ��t�������ع�ϵ���ĵ�������
# ����tֵ
Lxx = np.sum((PC_78[:, pc_index] - np.mean(PC_78[:, pc_index])) ** 2)
# SST
Sr_sat = reg_sat**2 * Lxx
St_sat = np.sum((sat_78 - np.mean(sat_78, axis=0)) ** 2, axis=0)
��_sat = np.sqrt((St_sat - Sr_sat) / (n - 2))
t_sat = reg_sat * np.sqrt(Lxx) / ��_sat

# �����ٽ�ֵ
t_critical = t.ppf(0.975, n - 2)
# ���������Լ���
p_sat78 = np.zeros((len(lat_sat), len(lon_sat)))
p_sat78.fill(np.nan)
p_sat78[np.abs(t_sat['__xarray_dataarray_variable__'].to_numpy()) > t_critical] = 1

# ��ͼ
# ##��ͼҪ������
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.subplots_adjust(wspace=0.1, hspace=0.001)  # wspace��hspace���ҡ����µļ��
font = {'family' : 'Arial','weight' : 'bold','size' : 12}
# plt.subplots_adjust(wspace=0.1, hspace=0.32)  # wspace��hspace���ҡ����µļ��
extent1 = [0, 360, 50, 90]  # ���ȷ�Χ��γ�ȷ�Χ
xticks1 = np.arange(extent1[0], extent1[1] + 1, 10)
yticks1 = np.arange(extent1[2], extent1[3] + 1, 10)

fig = plt.figure(figsize=(10, 10))

# ##ax1 Corr. PC1 & JA SST,2mT
level1 = [-1, -.7, -.4, -.1, -.05, .05, .1, .4, .7, 1]
ax1 = fig.add_subplot(1,1,1, projection=ccrs.NorthPolarStereo(central_longitude=90))
ax1.set_extent(extent1, crs=ccrs.PlateCarree())
# ȥ��fill_value 1e+20
# ȥ��180����!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!���Ͻؾ�
reg_sat, a1_sat_lon = add_cyclic_point(reg_sat['__xarray_dataarray_variable__'].to_numpy(), coord=lon_sat)
print('��ʼ���Ƶ�ͼ1')
ax1.set_title('(a)Reg 2mT', fontsize=20, loc='left')
a1 = ax1.contourf(a1_sat_lon, lat_sat, reg_sat, cmap=cmaps.GMT_polar[4:10]+cmaps.CBR_wet[0]+cmaps.GMT_polar[10:16], levels=level1, extend='both', transform=ccrs.PlateCarree(central_longitude=0))
ax1.add_feature(cfeature.COASTLINE.with_scale('110m'), linewidth=.3)  # ��Ӻ�����
ax1.add_geometries(Reader(shp).geometries(), ccrs.PlateCarree(), facecolor='none',edgecolor='black',linewidth=2)

p_sat, a1_lon_sat = add_cyclic_point(p_sat78, coord=lon_sat)
p_sat = np.where(p_sat == 1, 0, np.nan)

a1_uv = ax1.quiver(a1_lon_sat, lat_sat, p_sat, p_sat, scale=20, color='black', headlength=3,
                   regrid_shape=60, headaxislength=3, transform=ccrs.PlateCarree(central_longitude=0), width=0.005)

grid_lon = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=1, color='grey',linestyle='--')
grid_lon.xlocator = FixedLocator(np.linspace(-180,180,13))
grid_lon.ylocator = FixedLocator([0])
grid_lon.xlabel_style = {'size': 20}

grid_lat = ax1.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color='grey',linestyle='--')
grid_lon.xlocator = FixedLocator([])
grid_lat.ylocator = FixedLocator([65, 80])

theta = np.linspace(0, 2*np.pi, 100)
center, radius = [0.5, 0.5], 0.5
verts = np.vstack([np.sin(theta), np.cos(theta)]).T
circle = mpath.Path(verts * radius + center)
ax1.set_boundary(circle, transform=ax1.transAxes)

'''# �̶�������
ax1.set_xticks(xticks1, crs=proj)
lon_formatter = LongitudeFormatter()
lat_formatter = LatitudeFormatter()
ax1.xaxis.set_major_formatter(lon_formatter)
font = {'family': 'Arial', 'weight': 'bold', 'size': 28}

xmajorLocator = MultipleLocator(60)  # �ȶ���xmajorLocator���ٽ��е���
ax1.xaxis.set_major_locator(xmajorLocator)  # x�����̶�
xminorLocator = MultipleLocator(10)
ax1.xaxis.set_minor_locator(xminorLocator)  # x����С�̶�

# ax1.axes.xaxis.set_ticklabels([]) ##���ؿ̶ȱ�ǩ
# ���̶ȡ���С�̶ȵĿ̶��߳��̣���ϸ����
ax1.tick_params(which='major', length=11, width=2, color='darkgray')  # ���̶ȳ��ȣ�������ã�
ax1.tick_params(which='minor', length=8, width=1.8, color='darkgray')  # ��С�̶ȳ��ȣ��������
ax1.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
plt.rcParams['xtick.direction'] = 'out'  # ��x��Ŀ̶��߷����������ڻ�����
# �����̶�ֵ�����С
ax1.tick_params(axis='both', labelsize=28, colors='black')
# ��������̶�ֵ�Ĵ�С�Լ��̶�ֵ������
labels = ax1.get_xticklabels()
[label.set_fontname('Arial') for label in labels]
font2 = {'family': 'Arial', 'weight': 'bold', 'size': 28}'''

# color barλ��
# position = fig.add_axes([0.296, 0.08, 0.44, 0.011])#λ��[��,��,��,��]
position1 = fig.add_axes([0.296, 0.05, 0.44, 0.011])
cb1 = plt.colorbar(a1, cax=position1, orientation='horizontal')  # orientationΪˮƽ��ֱ
cb1.ax.tick_params(length=1, labelsize=14)  # lengthΪ�̶��ߵĳ���
cb1.locator = ticker.FixedLocator([-1, -.7, -.4, -.1, 0, .1, .4, .7, 1]) # colorbar�ϵĿ̶�ֵ����


plt.savefig(r'C:\Users\10574\OneDrive\File\Graduation Thesis\������ͼ\�����۲ⳡ�ع�.png', dpi=1000, bbox_inches='tight')
plt.show()
