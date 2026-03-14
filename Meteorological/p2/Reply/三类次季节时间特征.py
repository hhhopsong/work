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
MLR_count = pd.Series(MLR).value_counts().sort_index() / pd.Series(MLR).value_counts().values.max()
AR_count = pd.Series(AR).value_counts().sort_index() / pd.Series(AR).value_counts().values.max()
UR_count = pd.Series(UR).value_counts().sort_index() / pd.Series(UR).value_counts().values.max()
# 生成7.3到8.28的列表
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

ax.fill_between(x, MLR_count.values, 0, alpha=0.25, color=line1.get_color())
ax.fill_between(x, AR_count.values, 0, alpha=0.25, color=line2.get_color())
ax.fill_between(x, UR_count.values, 0, alpha=0.25, color=line3.get_color())

ax.set_xlabel("")
ax.set_ylabel("")
ax.set_title("Clim_daily of three types EHTBW", loc="left") #日分布
ax.set_xticks(x)
ax.set_xticklabels(day, rotation=45)
ax.legend(edgecolor='none', fontsize=8)
ax.set_ylim(0, 1.1)
ax.set_xlim(0, 57)
# 设定x轴坐标标签
ax.set_xticks([0, 12, 29, 43, 57])
ax.set_xticklabels([ "07/03", "07/15", "08/01", "08/15", "08/28"], rotation=0)

plt.tight_layout()
plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r6.pdf", bbox_inches='tight')
plt.savefig(fr"{PYFILE}/p2/pic/reply/fig_r6.png", bbox_inches='tight', dpi=600)
plt.show()

