import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from time import time
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

# K均值聚类初始化方法的评分。
def bench_k_means(kmeans, name, data, labels):
    """
    参数
    ----------
    kmeans : KMeans对象
    name : str
    data : 形状为(n_samples, n_features)的数组
        用于聚类的数据。
    labels : 形状为(n_samples,)的ndarray
    """
    t0 = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - t0
    results = [name, fit_time, estimator[-1].inertia_]

    # 定义聚类评估指标
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]
    results += [m(labels, estimator[-1].labels_) for m in clustering_metrics]  # 计算并添加聚类评估指标结果到结果列表

    # Silhouette系数需要使用完整的数据集
    results += [
        metrics.silhouette_score(
            data,
            estimator[-1].labels_,
            metric="euclidean",
            sample_size=300,
        )
    ]

    # 显示结果
    formatter_result = (
        "{:9s}\t{:.3f}s\t{:.0f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}"
    )
    print(formatter_result.format(*results))


def plot_test(data, max_clusters=15):
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

    # 绘制双折线图，设置双y轴
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(cluster_range, silhouette_scores, marker='o', color='b', label='S', zorder=4)
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('S')
    ax1.tick_params(axis='y')
    ax1.set_ylim(bottom=min(silhouette_scores) - 0.1 * abs(min(silhouette_scores)),
                 top=max(silhouette_scores) + 0.1 * abs(max(silhouette_scores)))
    ax1.set_xlim(left=min(cluster_range), right=max(cluster_range))
    ax2 = ax1.twinx()
    ax2.plot(cluster_range, explained_variance_ratio, marker='^', color='r', label='SSE', zorder=3)
    ax2.set_ylabel('SSE')
    ax2.tick_params(axis='y')
    ax2.set_ylim(bottom=min(explained_variance_ratio) - 0.1 * abs(min(explained_variance_ratio)),
                 top=max(explained_variance_ratio) + 0.1 * abs(max(explained_variance_ratio)))

    # 添加图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', edgecolor='none')
    ax1.set_title('a) SSE & Silhouette Coefficient', fontsize=14, loc='left')

    plt.xticks(np.arange(2, max_clusters + 1, 1))  # 整数x轴刻度
    fig.tight_layout()
    plt.show()

    score = xr.Dataset({
        'inertia': ('cluster', inertia),
        'var': ('cluster', explained_variance_ratio),
        'scores': ('cluster', silhouette_scores),
    }, coords={'cluster': cluster_range})

    return score

def K_Mean(data, n_clusters=3):
    """
    K均值聚类
    :param data: 数据
    :param n_clusters: 聚类数
    :return: 聚类结果
    """
    flattened_data = data.reshape(data.shape[0], -1)

    # 流水线
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 标准化步骤
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42)),  # K均值聚类步骤
    ])
    pipeline.fit(flattened_data)

    # 聚类标签
    labels = pipeline['kmeans'].labels_

    # 构造输出结果
    cluster_results = {}
    for cluster in range(n_clusters):
        # 找到属于该聚类的样本索引
        cluster_indices = np.where(labels == cluster)[0]

        # 计算该类别的平均分布图
        mean_distribution = np.mean(data[cluster_indices], axis=0)

        # 存储结果
        cluster_results[cluster] = {
            "indices": cluster_indices,                # 某聚类的样本索引
            "mean_distribution": mean_distribution,    # 该类的平均分布图
            "shape": mean_distribution.shape,          # 原始分布图的形状
            "labels": labels                           # 逐样本所属的聚类
        }

    # 按聚类标签分类原始分布图
    for cluster in range(n_clusters):
        cluster_indices = np.where(labels == cluster)[0]
        #print(f"Cluster {cluster}: 图像索引 {cluster_indices}")

        # 可视化某个聚类类型的分布
        mean_distribution = np.mean(data[cluster_indices], axis=0)
        plt.figure(figsize=(10, 1))
        plt.title(f"Cluster {cluster} 平均分布")
        if mean_distribution.ndim == 1:  # 修正为单行显示
            plt.imshow(mean_distribution.reshape(1, -1), cmap='hot', aspect='auto')
        else:
            plt.imshow(mean_distribution, cmap='hot')
        plt.colorbar()
        plt.show()

    return cluster_results


if __name__ == '__main__':
    data = np.random.rand(100, 2)
    plot_test(data)
    result = K_Mean(data, 3)
