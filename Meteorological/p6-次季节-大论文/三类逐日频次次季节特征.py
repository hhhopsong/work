import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# ============================================================
# 路径设置
# ============================================================
PYFILE = r"/volumes/TiPlus7100/PyFile"
DATA = r"/volumes/TiPlus7100/data"


# ============================================================
# 绘图字体设置
# ============================================================
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 读取数据
# ============================================================
MLR = np.load(
    fr"{PYFILE}/p2/data/MLR-type_time.npy",
    allow_pickle=True
)

AR = np.load(
    fr"{PYFILE}/p2/data/AR-type_time.npy",
    allow_pickle=True
)

UR = np.load(
    fr"{PYFILE}/p2/data/UR-type_time.npy",
    allow_pickle=True
)


# ============================================================
# 提取日期编号
# 例如原始字符串为 xxxx01，则保留 01
# ============================================================
MLR = [str(i)[4:] for i in MLR]
AR = [str(i)[4:] for i in AR]
UR = [str(i)[4:] for i in UR]


# ============================================================
# 计算1961—2022年平均每日发生次数
# ============================================================
n_years = 2022 - 1961 + 1

MLR_count = (
    pd.Series(MLR, dtype="object")
    .value_counts()
    .sort_index()
    / n_years
)

AR_count = (
    pd.Series(AR, dtype="object")
    .value_counts()
    .sort_index()
    / n_years
)

UR_count = (
    pd.Series(UR, dtype="object")
    .value_counts()
    .sort_index()
    / n_years
)


# ============================================================
# 生成01—58的日期编号
# 对应07/03—08/29，共58天
# ============================================================
day = [f"{i:02d}" for i in range(1, 59)]


# 补全缺失日期，没有事件的日期记为0
MLR_count = MLR_count.reindex(day, fill_value=0)
AR_count = AR_count.reindex(day, fill_value=0)
UR_count = UR_count.reindex(day, fill_value=0)


# 可选：整理为DataFrame，便于后续保存或统计
df = pd.DataFrame(
    {
        "day": day * 3,
        "count": (
            list(MLR_count.values)
            + list(AR_count.values)
            + list(UR_count.values)
        ),
        "type": (
            ["MLR"] * len(day)
            + ["AR"] * len(day)
            + ["UR"] * len(day)
        ),
    }
)


# ============================================================
# 三行小多图：比并列柱状图更清晰
# ============================================================
x = np.arange(len(day))

# Nature / NPG 风格配色
nature_colors = {
    "MLR": "#3C5488",
    "AR":  "#E64B35",
    "UR":  "#00A087",
}

series_list = [
    ("MLR-type", MLR_count.values, nature_colors["MLR"]),
    ("AR-type",  AR_count.values,  nature_colors["AR"]),
    ("UR-type",  UR_count.values,  nature_colors["UR"]),
]

fig, axes = plt.subplots(
    nrows=3,
    ncols=1,
    figsize=(6.5, 4.5),
    sharex=True,
    sharey=True,
    gridspec_kw={
        "hspace": 0.10
    }
)

for ax, (label, values, color) in zip(axes, series_list):

    # 柱状图
    ax.bar(
        x,
        values,
        width=0.82,
        color=color,
        edgecolor="none",
        alpha=0.90,
        zorder=3
    )


    # 类型名称放在子图左上角
    ax.text(
        0.015,
        0.95,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color=color,
        ha="left",
        va="center"
    )

    # y轴范围和刻度
    ax.set_ylim(0, 0.48)
    ax.set_yticks([0, 0.12, 0.24, 0.36, 0.48])

    # 网格线
    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.30
    )

    # Nature风格边框
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.8,
        labelsize=8
    )

# 除最后一幅图外，隐藏x轴线和刻度
for ax in axes[:-1]:
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(
        axis="x",
        bottom=False,
        labelbottom=False
    )

# x轴刻度
axes[-1].set_xlim(-1, len(day))
axes[-1].set_xticks([0, 12, 29, 43, 57])
axes[-1].set_xticklabels(
    ["07/03", "07/15", "08/01", "08/15", "08/29"],
    fontsize=9
)

# 总标题
fig.suptitle(
    "Climatological daily distribution of three EHTBW types",
    x=0.10,
    y=0.98,
    ha="left",
    fontsize=12,
    fontweight="bold"
)

# 公共y轴标题
fig.text(
    0.015,
    0.50,
    "Frequency",
    rotation=90,
    va="center",
    ha="center",
    fontsize=10
)

plt.subplots_adjust(
    left=0.12,
    right=0.98,
    bottom=0.12,
    top=0.90
)
plt.savefig(
    fr"{PYFILE}/p6/pic/fig_r6.pdf",
    bbox_inches="tight",
)

plt.savefig(
    fr"{PYFILE}/p6/pic/fig_r6.png",
    bbox_inches="tight",
    dpi=600,
)

plt.show()
