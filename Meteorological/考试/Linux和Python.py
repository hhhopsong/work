import matplotlib.pyplot as plt
import xarray as xr

data = xr.open_dataset(r"E:\data\self\ZUV_850_DJF_1991_2020.nc")
# 提取变量和坐标数据
z = data['z']  # 提取z变量（位势高度）
u = data['u']  # 提取u变量（经向风）
v = data['v']  # 提取v变量（纬向风）
latitude = data['latitude']  # 纬度
longitude = data['longitude']  # 经度

# 将变量和坐标转换为NumPy数组以便绘图
z_data = z.values  # z的数值
u_data = u.values  # u的数值
v_data = v.values  # v的数值
lat = latitude.values  # 纬度数组
lon = longitude.values  # 经度数组

# 创建绘图窗口
fig, ax = plt.subplots(figsize=(10, 8))  # 图的大小为10x8英寸
font_size = 18 # 字体大小
# 绘制z的填色图（contourf），使用coolwarm色彩方案
cmap = plt.cm.coolwarm  # 选择colormap
contour = ax.contourf(lon, lat, z_data, cmap=cmap)  # 填色图，纬度为x轴，经度为y轴

# 添加水平放置的颜色条
colorbar = plt.colorbar(contour, orientation='horizontal', pad=0.1)  # 在图下方添加颜色条
colorbar.ax.tick_params(labelsize=font_size)
# 绘制矢量风场图（quiver），叠加在z的填色图上
quiver = ax.quiver(lon[::3], lat[::2], u_data[::2, ::3], v_data[::2, ::3], scale=200, zorder=1)  # 调整scale以控制矢量长度
ax.quiverkey(quiver, 0.88, 1.03, 10, "10 m/s", labelpos="E", coordinates="axes", fontproperties={'size': font_size})  # 在右上角添加风矢量标准箭头

# 设置x轴和y轴的刻度和标签
ax.set_yticks([30, 40, 50, 60])  # 设置纬度刻度
ax.set_yticklabels(["30°N", "40°N", "50°N", "60°N"], fontsize=font_size)  # 设置纬度刻度标签
ax.set_xticks([80, 90, 100, 110, 120, 130])  # 设置经度刻度
ax.set_xticklabels(["80°E", "90°E", "100°E", "110°E", "120°E", "130°E"], fontsize=font_size)  # 设置经度刻度标签

# 网格
ax.grid(True)  # 添加网格线

# 设置x轴和y轴的标签，以及图的标题
ax.set_xlabel("Longitude", fontsize=font_size)  # x轴标签
ax.set_ylabel("Latitude", fontsize=font_size)  # y轴标签
ax.set_title("Z&UV at 850hPa", loc='left', fontsize=font_size)  # 图标题

# 调整布局以避免内容重叠
plt.tight_layout()

# 保存绘图结果
plt.savefig("E:\data\self\z_uv_850hPa.png", dpi=600, bbox_inches='tight')

# 显示绘图结果
plt.show()