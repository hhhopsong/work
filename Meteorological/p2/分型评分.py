import xarray as xr
import numpy as np
import tqdm as tq
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cmaps

from cartopy.io.shapereader import Reader
from toolbar.filter import MovingAverageFilter
from toolbar.masked import masked  # 气象工具函数
from toolbar.K_Mean import K_Mean
from toolbar.average_filter import nanmean_filter
import scipy

# 数据读取
data_year = ['1961', '2022']
# 读取CN05.1逐日最高气温数据
#CN051_1 = xr.open_dataset(r"E:\data\CN05.1\1961_2021\CN05.1_Tmax_1961_2021_daily_025x025.nc")
CN051_2 = xr.open_dataset(r"E:\data\CN05.1\2022\CN05.1_Tmax_2022_daily_025x025.nc")
#CN051 = xr.concat([CN051_1, CN051_2], dim='time')
try:
    Tmax_5Day_filt = xr.open_dataarray(fr"D:\PyFile\p2\data\Tmax_5Day_filt.nc")
except:
    Tmax = xr.concat([CN051_1, CN051_2], dim='time')
    Tmax = masked(Tmax, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp")  # 掩膜处理得长江流域温度
    Tmax = Tmax.sel(time=Tmax['time.month'].isin([6, 7, 8])).groupby('time.year')  # 截取夏季数据
    Tmax_5Day_filt = np.array([[[MovingAverageFilter(iyear[1]['tmax'].data[:, i, j], 'lowpass', [5], np.nan).filted()
                                 for j in range(283)] for i in range(163)] for iyear in tq.tqdm(Tmax)])  # 5天滑动平均
    Tmax_5Day_filt = Tmax_5Day_filt.transpose(0, 3, 1, 2)  # 转换为(year, day, lat, lon)格式
    Tmax_5Day_filt = xr.DataArray(Tmax_5Day_filt,
                                  coords=[[str(i) for i in range(eval(data_year[0]), eval(data_year[1]) + 1)],
                                          [str(i) for i in range(1, 88 + 1)],
                                          CN051_2['lat'].data,
                                          CN051_2['lon'].data],
                                  dims=['year', 'day', 'lat', 'lon'], )
    Tmax_5Day_filt.to_netcdf(fr"D:\PyFile\p2\data\Tmax_5Day_filt.nc")
    del Tmax
T_th = 0.90
t95 = masked(Tmax_5Day_filt, r"D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp").mean(dim=['year', 'day']).quantile(T_th)  # 夏季内 长江中下游流域 分位数
EHD = Tmax_5Day_filt - t95
EHD = EHD.where(EHD > 0, 0)  # 极端高温日温度距平
EHD = EHD.where(EHD == 0, 1)  # 数据二值化处理(1:极端高温, 0:非极端高温)
EHD = masked(EHD, r"D:\PyFile\map\self\长江_TP\长江_tp.shp")  # 掩膜处理得长江流域EHD温度距平
EHD = EHD.sel(day=EHD['day'][30:]) # 截取7月3日至8月28日数据
zone_stations = masked((CN051_2 - CN051_2 + 1).sel(time='2022-01-01'), r"D:\PyFile\map\self\长江_TP\长江_tp.shp").sum()['tmax'].data
EHDstations_zone = EHD.sum(dim=['lat', 'lon']) / zone_stations  # 长江流域逐日极端高温格点占比
S_q = 0.9
S_th = 0.3
EHD20 = EHD.where(EHDstations_zone >= S_th, np.nan)  # 提取极端高温日占比大于10%
# 获取EHD20的年份和日
EHD20_time = np.zeros((EHD20['year'].size, EHD20['day'].size))
for iyear in range(EHD20['year'].size):
    for iday in range(EHD20['day'].size):
        EHD20_time[iyear, iday] = f"{iyear + 1961}{iday:02d}"
bridge = []
EHD20_time = np.where(EHDstations_zone >= S_th, EHD20_time, np.nan)  # 提取极端高温日占比大于10%
for i in EHD20_time:
    for j in i:
        if not np.isnan(j):
            bridge.append(j)
EHD20_time = np.array(bridge)
EHD20 = masked(EHD20, r"D:\PyFile\map\self\长江_TP\长江_tp.shp")  # 减去非研究地区
EHD20 = EHD20.data.reshape(-1, 163 * 283)
EHD20 = pd.DataFrame(EHD20).dropna(axis=0, how='all')
EHD20_ = EHD20.dropna(axis=1, how='all')

# 字体为新罗马
plt.rcParams['font.family'] = 'Times New Roman'



def plot_test(data, max_clusters=10):
    """
    显示Variance肘部图和Silhouette系数图的双折线图
    :param data: 数据
    :param max_clusters: 最大聚类数
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn import metrics

    inertia = []
    explained_variance_ratio = []  # 用于存储解释方差占比
    silhouette_scores = []
    cluster_range = range(2, max_clusters + 1)
    flattened_data = data.reshape(data.shape[0], -1)

    for n_clusters in cluster_range:
        # 流水线
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # 标准化步骤
            ("kmeans", KMeans(n_clusters=n_clusters, random_state=42)),  # K均值聚类步骤
        ])
        pipeline.fit(flattened_data)
        kmeans = pipeline['kmeans']
        labels = pipeline['kmeans'].labels_
        inertia.append(kmeans.inertia_)
        explained_variance_ratio.append(kmeans.inertia_)  # 解释方差占比
        silhouette_scores.append(metrics.silhouette_score(flattened_data, labels))

    explained_variance_ratio = np.array(explained_variance_ratio)
    explained_variance_ratio /= zone_stations

    # 绘制双折线图，设置双y轴
    fig, ax1 = plt.subplots(figsize=(8, 4))

    ax1.plot(cluster_range, silhouette_scores, color='r', label='S', zorder=4)
    ax1.set_xlabel('Number of Clusters', fontsize=16)
    ax1.set_ylabel('S', color='r', fontsize=16)
    ax1.tick_params(axis='y', colors='r', labelsize=12)
    ax1.set_ylim(bottom=min(silhouette_scores) - 0.1 * abs(min(silhouette_scores)),
                 top=max(silhouette_scores) + 0.1 * abs(max(silhouette_scores)))
    ax1.set_xlim(left=min(cluster_range)-0.2, right=max(cluster_range)+0.2)
    ax1.scatter(3, silhouette_scores[1], color='red', marker='^')
    ax2 = ax1.twinx()
    ax2.plot(cluster_range, explained_variance_ratio, color='b', label='MSE', zorder=3)
    ax2.set_ylabel('MSE', color='b', fontsize=16)
    ax2.spines['left'].set_color('red')
    ax2.spines['right'].set_color('blue')
    ax2.tick_params(axis='y', colors='b', labelsize=12)
    ax2.set_ylim(bottom=min(explained_variance_ratio) - 0.025 * abs(min(explained_variance_ratio)),
                 top=max(explained_variance_ratio) + 0.025 * abs(max(explained_variance_ratio)))
    ax2.scatter(3, explained_variance_ratio[1], color='blue', marker='o')

    # 添加图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', edgecolor='none')
    ax1.set_title('MSE & Silhouette Coefficient', fontsize=18, loc='left')

    plt.xticks(np.arange(2, max_clusters + 1, 1))  # 整数x轴刻度
    fig.tight_layout()
    plt.savefig(fr"D:\PyFile\p2\pic\图3_1.png", dpi=600, bbox_inches='tight')
    plt.show()

plot_test(EHD20_.to_numpy(), max_clusters=10)
from matplotlib.ticker import MultipleLocator
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib import ticker, colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

K_s = 3
K = K_Mean(EHD20_.to_numpy(), K_s)
# 绘制三种聚类的平均分布图
fig = plt.figure(figsize=(10, 6))
time = [[] for i in range(K_s)]
abc_index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
lev = np.array([[800, 900, 1000, 1100, 1130, 1150],
                [400, 500, 600, 700, 800, 900],
                [170, 190, 210, 230, 250, 270]])
Tavg_weight = np.zeros((K_s, 163, 283))
KM_all = np.zeros((K_s, 163, 283))
for cluster in range(K_s):
    KM = []
    for i in K[cluster]['indices']:
        KM.append(EHD20.iloc[i].to_numpy())
        time[cluster].append(str(int(EHD20_time[i - 1])))
        print(EHD20_time[i - 1])
    Tavg_weight[cluster] = np.array(KM).mean(axis=0).reshape(163, 283)
    KM = np.array(KM).sum(axis=0)
    KM_all[cluster] = KM.reshape(163, 283)

#将KM_all按照维度0的平均值进行排序
sort_index = np.argsort([np.nanmax(KM_all[i]) for i in range(K_s)])[::-1]  # 降序排列
KM_all = KM_all[sort_index]  # 重新排列聚类顺序
Tavg_weight = Tavg_weight[sort_index]  # 同步排列权重矩阵
time = [time[i] for i in sort_index] # 同步调整时间顺序

type_name = ['MLB Type', 'ALL Type', 'UB Type']
for cluster in range(K_s):
    extent_CN = [88, 124, 22, 38]  # 中国大陆经度范围，纬度范围
    ax = fig.add_subplot(2, K_s, cluster + 1, projection=ccrs.PlateCarree())
    ax.set_title(f"{abc_index[cluster]}) {type_name[cluster]}", loc='left', fontsize=14, weight='bold')
    ax.add_geometries(Reader(
        r'D:\PyFile\map\地图边界数据\青藏高原边界数据总集\TPBoundary2500m_长江流域\TPBoundary2500m_长江流域.shp').geometries(),
                      ccrs.PlateCarree(), facecolor='gray', edgecolor='black', linewidth=.5)
    ax.add_geometries(Reader(r'D:\PyFile\map\地图线路数据\长江\长江.shp').geometries(), ccrs.PlateCarree(),
                      facecolor='none', edgecolor='blue', linewidth=0.2)
    ax.add_geometries(Reader(r'D:\PyFile\map\地图边界数据\长江区1：25万界线数据集（2002年）\长江区.shp').geometries(),
                    ccrs.PlateCarree(), facecolor='none', edgecolor='black', linewidth=.5)
    ax.set_extent(extent_CN)
    # 刻度线设置
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
    proj = ccrs.PlateCarree()   # 投影方式
    xticks1=np.arange(extent_CN[0], extent_CN[1]+1, 10)
    yticks1=np.arange(extent_CN[2], extent_CN[3]+1, 10)
    ax.set_xticks(xticks1, crs=proj)
    if cluster == 0: ax.set_yticks(yticks1, crs=proj)  # 设置经纬度坐标,只在第一个图上显示y轴坐标
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    xmajorLocator = MultipleLocator(10)#先定义xmajorLocator，再进行调用
    ax.xaxis.set_major_locator(xmajorLocator)#x轴最大刻度
    xminorLocator = MultipleLocator(1)
    ax.xaxis.set_minor_locator(xminorLocator)#x轴最小刻度
    ymajorLocator = MultipleLocator(4)
    ax.yaxis.set_major_locator(ymajorLocator)#y轴最大刻度
    yminorLocator = MultipleLocator(1)
    ax.yaxis.set_minor_locator(yminorLocator)#y轴最小刻度
    # 调整刻度值字体大小
    ax.tick_params(axis='both', labelsize=12, colors='black')
    # 最大刻度、最小刻度的刻度线长短，粗细设置
    ax.tick_params(which='major', length=3.5, width=1, color='black')  # 最大刻度长度，宽度设置，
    ax.tick_params(which='minor', length=2, width=.9, color='black')  # 最小刻度长度，宽度设置
    ax.tick_params(which='both', bottom=True, top=False, left=True, labelbottom=True, labeltop=False)
    custom_colors = ["#FDDDB1", "#FDB57E", "#F26E4C", "#CA1E14", "#7F0000"]
    # if cluster == 0: custom_colors = ["#d1e5f0", "#92c5de", "#67b7df", "#4393c3", "#2166ac"]
    # elif cluster == 1: custom_colors = ["#ebb7cc", "#eb88af", "#eb6198", "#c55280", "#923c5e"]
    # elif cluster == 2: custom_colors = ["#D6B2F0", "#ca8ef6", "#ab78d0", "#a94ac9", "#7e3795"]
    custom_cmap = colors.ListedColormap(custom_colors)
    norm = mcolors.BoundaryNorm(lev[cluster], custom_cmap.N)
    con = ax.contourf(CN051_2['lon'], CN051_2['lat'], KM_all[cluster],
                      cmap=custom_cmap, transform=ccrs.PlateCarree(),
                      levels=lev[cluster], extend='max', norm=norm)
    ax.contour(CN051_2['lon'], CN051_2['lat'], KM_all[cluster],
                colors='w', linewidths=0.1, transform=ccrs.PlateCarree(), linestyles='solid',
                levels=lev[cluster][1:-1])
    # 色标
    ax_colorbar = inset_axes(ax, width="55%", height="5%", loc='upper right', bbox_to_anchor=(-0.03, 0.17, 1, 1),
                             bbox_transform=ax.transAxes, borderpad=0)
    cb1 = plt.colorbar(con, cax=ax_colorbar, orientation='horizontal', drawedges=True)
    cb1.locator = ticker.FixedLocator(lev[cluster])
    #cb1.set_label('EHDs', fontsize=0, loc='left')
    cb1.set_ticklabels(lev[cluster])
    cb1.ax.tick_params(length=0, labelsize=10, direction='in')  # length为刻度线的长度

    print(f'---{cluster}---' * 10)
Tavg_weight = xr.Dataset({'W': (['type', 'lat', 'lon'], Tavg_weight)},
                       coords={'type': [i for i in range(1, K_s + 1)],
                               'lat': CN051_2['lat'].data,
                               'lon': CN051_2['lon'].data})
Tavg_weight.to_netcdf(fr"D:\PyFile\p2\data\Tavg_weight.nc")

Time_type = np.zeros((62, K_s))
for i in range(K_s):
    for iyear in time[i]:
        index = eval(iyear[:4]) - 1961  # 年份索引
        Time_type[index, i] += 1
Time_type = xr.Dataset({'K': (['year', 'type'], Time_type)},
                       coords={'year': [i for i in range(1961, 2023)], 'type': [i for i in range(1, K_s + 1)]})
'''try:
    Time_type.to_netcdf(fr'D:\PyFile\p2\data\Time_type_AverFiltAll{T_th}%_{S_th}%_{K_s}.nc')
except:
    pass'''

import seaborn as sns
from scipy import stats

# 加载数据集
data = xr.open_dataset(fr'D:\PyFile\p2\data\Time_type_AverFiltAll{T_th}%_{S_th}%_{K_s}.nc')
data = Time_type

# 将数据集转换为 Pandas DataFrame 以便更容易操作
df = data.to_dataframe().reset_index()

# 按年和类型分组，并对 K 值求和
grouped_data = df.groupby(['year', 'type'])['K'].sum().unstack()

# 计算每年的总和和每种类型的占比
total_by_year = grouped_data.sum(axis=1)  # 每年的总和
proportion_by_type = grouped_data.div(total_by_year, axis=0)  # 每种类型的占比

# 定义折线图的颜色
contrasting_colors = ['blue', 'red', 'green']

# 开始绘制图表
ax1 = fig.add_subplot(2, 1, 2)

# 绘制柱状图（单色表示每年的总天数）
bars = ax1.bar(grouped_data.index, total_by_year, color='lightgray', alpha=0.8, edgecolor='black', label='')
ax1.set_title('d) Days of types', loc='left', fontsize=14, weight='bold')  # 设置标题
ax1.set_xlim(1960, 2023)
ax1.set_xlabel('Year', fontsize=12)  # 设置 x 轴标签
ax1.set_ylim(0, 63)
ax1.set_ylabel('Days', fontsize=12)  # 设置 y 轴标签
ax1.tick_params(axis='x', rotation=0, labelsize=12)  # 设置 x 轴刻度标签旋转和大小
ax1.tick_params(axis='y', labelsize=12)  # 设置 y 轴刻度标签大小
ax1.set_xticks(range(1961, 1961+len(total_by_year), 5))  # 设置 x 轴的刻度点间隔为 5 年
ax1.set_xticklabels(total_by_year.index[::5])  # 设置 x 轴的刻度标签

# 在柱状图顶部添加数值标注
for bar in bars:
    height = bar.get_height()
    if height > 0:
        ax1.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}', ha='center', va='bottom', fontsize=8, weight='bold')

# 添加网格线，使图表更加美观
#ax1.grid(axis='y', linestyle='--', alpha=0.7)

# 颜色
colors = ['#2166ac', '#f968a1', '#8c62aa']

# 绘制并排柱状图（不同颜色表示每种类型的占比）
x = np.arange(len(grouped_data.index))  # the label locations
width = 0.2  # the width of the bars
ax = ax1.twinx()  # 创建第二个 y 轴

# 绘制每种类型的柱状图
for i, col in enumerate(grouped_data.columns):
    ax.bar(
        1961 - width  + x + i * width,  # Offset each type's bars
        grouped_data[col],
        width=width,
        label=f'{type_name[col-1]}',
        color=colors[i],
        edgecolor='none',
        alpha=0.8
    )

ax.set_ylim(0, 63)
ax.yaxis.set_visible(False)  # ax隐藏y轴标签
ax.legend(fontsize=10, loc='upper right', bbox_to_anchor=(1, 1.12), edgecolor='none', ncol=3)

ax_reg = ax.twinx()
# 获取 type=1 的数据并转换为 Pandas DataFrame
type_1_data = data.sel(type=1)['K'].to_dataframe().reset_index()
# 确保 x 和 y 数据长度匹配
x = type_1_data['year']
y = type_1_data['K']
# 确保 y 没有 NaN 值
y = y.fillna(0)
ax_reg = sns.regplot(data=type_1_data, x=x, y=y, ax=ax_reg, scatter=False, ci=0, line_kws={"linestyle": "--", "color": colors[0]})  # 长江流域极端高温格点逐年占比
ax_reg.yaxis.set_visible(False)  # ax2隐藏y轴标签
ax_reg.set_ylim(0, 63)

ax_reg = ax.twinx()
# 获取 type=1 的数据并转换为 Pandas DataFrame
type_2_data = data.sel(type=2, year=slice(1961, 2021))['K'].to_dataframe().reset_index()
# 确保 x 和 y 数据长度匹配
x = type_2_data['year']
y = type_2_data['K']
# 确保 y 没有 NaN 值
y = y.fillna(0)
ax_reg = sns.regplot(data=type_2_data, x=x, y=y, ax=ax_reg, scatter=False, ci=0, line_kws={"linestyle": "--", "color": colors[1]})  # 长江流域极端高温格点逐年占比
ax_reg.yaxis.set_visible(False)  # ax2隐藏y轴标签
ax_reg.set_ylim(0, 63)

ax_reg = ax.twinx()
# 获取 type=1 的数据并转换为 Pandas DataFrame
type_3_data = data.sel(type=3)['K'].to_dataframe().reset_index()
# 确保 x 和 y 数据长度匹配
x = type_3_data['year']
y = type_3_data['K']
# 确保 y 没有 NaN 值
y = y.fillna(0)
ax_reg = sns.regplot(data=type_3_data, x=x, y=y, ax=ax_reg, scatter=False, ci=0, line_kws={"linestyle": "--", "color": colors[2]})  # 长江流域极端高温格点逐年占比
ax_reg.yaxis.set_visible(False)  # ax2隐藏y轴标签
ax_reg.set_ylim(0, 63)

# Add gridlines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 子图间距
fig.subplots_adjust(hspace=0.01)

# 自动调整布局，避免标签重叠
plt.tight_layout()

# 显示图表
plt.savefig(fr"D:\PyFile\p2\pic\图3.png", dpi=600, bbox_inches='tight')
plt.show()
