import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from time import time
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler

# 用于评估K均值聚类初始化方法的基准测试函数。
def bench_k_means(kmeans, name, data, labels):
    """
    参数
    ----------
    kmeans : KMeans实例对象
        已经设置初始化方法的KMeans实例对象。
    name : str
        策略的名称，将用于在表格中显示结果。
    data : 形状为(n_samples, n_features)的数组
        用于聚类的数据。
    labels : 形状为(n_samples,)的ndarray
        用于计算聚类指标的真实标签。
    """
    t0 = time()  # 记录开始时间
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)  # 创建管道，并拟合数据
    fit_time = time() - t0  # 计算拟合时间
    results = [name, fit_time, estimator[-1].inertia_]  # 初始化结果列表，包括名称、拟合时间、聚类不确定性度量

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


def plot_test(data, max_clusters=10):
    """
    显示Variance肘部图和Silhouette系数图
    :param data: 数据
    :param max_clusters: 最大聚类数
    """
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

    # 绘制肘部图（使用解释方差占比）
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, explained_variance_ratio, marker='o')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Explained Variance Ratio')
    # 整数x轴刻度
    plt.xticks(np.arange(2, max_clusters + 1, 1))
    plt.grid()
    plt.show()

    # 绘制Silhouette系数图
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, silhouette_scores, marker='o')
    plt.title('Silhouette Coefficient')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    # 整数x轴刻度
    plt.xticks(np.arange(2, max_clusters + 1, 1))
    plt.grid()
    plt.show()


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
            "indices": cluster_indices,                # 属于该聚类的样本索引
            "mean_distribution": mean_distribution,    # 该类的平均分布图
            "shape": mean_distribution.shape,          # 保留原始分布图形状
            "labels": labels                           # 每个样本的聚类标签
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
    K_Mean(data, 3)
