import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'stix'

MLR = np.load(fr"{PYFILE}/p2/data/MLR-type_time.npy")
AR = np.load(fr"{PYFILE}/p2/data/AR-type_time.npy")
UR = np.load(fr"{PYFILE}/p2/data/UR-type_time.npy")

MLR = [i[4:] for i in MLR]
AR = [i[4:] for i in AR]
UR = [i[4:] for i in UR]
# 统计MLR中相同元素的数量
# count结果按照元素名称排列
# 统计每天数量
MLR_count = pd.Series(MLR).value_counts().sort_index() / (2022-1961+1)
AR_count = pd.Series(AR).value_counts().sort_index() / (2022-1961+1)
UR_count = pd.Series(UR).value_counts().sort_index() / (2022-1961+1)
# 生成7.3到8.29的列表
day = [f'{i:0>2}' for i in range(1, 59)]
# 补全缺失日期，没有的记为 0
MLR_count = MLR_count.reindex(day, fill_value=0)
AR_count  = AR_count.reindex(day, fill_value=0)
UR_count  = UR_count.reindex(day, fill_value=0)
df = pd.DataFrame({
    "day": day * 3,
    "count": list(MLR_count.values) + list(AR_count.values) + list(UR_count.values),
    "type": ["MLR"] * len(day) + ["AR"] * len(day) + ["UR"] * len(day)
})

x = np.arange(len(day))
fig, ax = plt.subplots(figsize=(np.array([12, 5])*.5))
## '#2166ac', '#ff5370', '#13a252'
line1, = ax.plot(x, MLR_count.values, linewidth=2, label='MLR-type', color='#2166ac')
line2, = ax.plot(x, AR_count.values, linewidth=2, label='AR-type', color='#ff5370')
line3, = ax.plot(x, UR_count.values, linewidth=2, label='UR-type', color='#13a252')


def gradient_fill_between(ax, x, y, y0=0, color='C0', alpha_top=0.8, zorder=1):
    import matplotlib.colors as mcolors
    from matplotlib.path import Path
    from matplotlib.patches import PathPatch
    x = np.asarray(x)
    y = np.asarray(y)

    # 先构造填充区域的路径
    verts = np.concatenate([
        np.column_stack([x, y]),
        np.column_stack([x[::-1], np.full_like(x, y0)])
    ])
    path = Path(verts)
    patch = PathPatch(path, facecolor='none', edgecolor='none')
    ax.add_patch(patch)

    # 构造从“透明白”到底部 -> “目标颜色”到顶部的渐变
    n = 256
    rgb = mcolors.to_rgb(color)
    grad = np.ones((n, 1, 4))
    grad[..., 0] = rgb[0]
    grad[..., 1] = rgb[1]
    grad[..., 2] = rgb[2]
    grad[..., 3] = np.linspace(0, alpha_top, n)[:, None]  # 底部透明，顶部更实

    # 把渐变图画出来，再裁剪到 fill_between 区域
    im = ax.imshow(
        grad,
        aspect='auto',
        origin='lower',
        extent=[x.min(), x.max(), y0, max(y.max(), y0)],
        zorder=zorder
    )
    im.set_clip_path(patch)
    return im

gradient_fill_between(ax, x, MLR_count.values, y0=0, color=line1.get_color(), alpha_top=0.25)
gradient_fill_between(ax, x, AR_count.values,  y0=0, color=line2.get_color(), alpha_top=0.25)
gradient_fill_between(ax, x, UR_count.values,  y0=0, color=line3.get_color(), alpha_top=0.25)

for spine in ax.spines.values():
    spine.set_linewidth(2)  # 设置边框线宽

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Clim_daily of three types EHTBW", loc="left") #日分布
ax.set_xticks(x)
ax.set_xticklabels(day, rotation=45)
ax.legend(edgecolor='none', fontsize=8)
ax.set_ylim(0, 0.48)
ax.set_yticks([0, 0.12, 0.24, 0.36, 0.48])
ax.set_yticklabels([ "0     ", "0.12", "0.24", "0.36", "0.48"], rotation=0)
# 设定x轴坐标标签
ax.set_xlim(-2, 59)
ax.set_xticks([-2, 12, 29, 43, 59])
ax.set_xticklabels([ "07/01", "07/15", "08/01", "08/15", "08/31"], rotation=0)

plt.tight_layout()
plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r6.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r6.png", bbox_inches='tight', dpi=600)
plt.show()

