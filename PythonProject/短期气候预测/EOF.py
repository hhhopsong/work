import pprint

import numpy as np
from xgrads import open_CtlDataset
import xarray as xr


def read(DataSet):
    print('数据读取中...')
    DataLoc = DataSet
    try:
        data = xr.open_dataset(DataLoc)
    except:
        open_CtlDataset(DataLoc).to_netcdf(DataLoc[:-3] + 'nc')
        DataLoc = DataLoc[:-3] + 'nc'
        data = xr.open_dataset(DataLoc)
    return data


def transf(data, var, KS):
    # This Subroutine Provides Initial F By KS
    # KS=-1:Self; KS=0:Departure; KS=1:Standardized Departure
    print('数据预处理中...')
    GridPointsNum = len(data['lon']) * len(data['lat'])
    TimeSeriesLength = len(data['time'])
    H = data[var]
    if KS == -1:
        return data, len(data['lat']), len(data['lon']), TimeSeriesLength
    elif KS == 0 or KS == 1:
        avg = np.mean(H, axis=0)  # 平均高度场
        Departure = np.zeros((TimeSeriesLength, len(data['lat']), len(data['lon'])))
        for i in range(TimeSeriesLength):
            Departure[i] = H[i] - avg  # 高度距平场
        if KS == 0:
            return Departure, len(data['lat']), len(data['lon']), TimeSeriesLength
        Std = np.zeros((TimeSeriesLength, len(data['lat']), len(data['lon'])))
        for y in range(len(data['lat'])):
            for x in range(len(data['lon'])):
                Sx = np.sqrt((1 / TimeSeriesLength) * np.sum(Departure[:, y, x] ** 2))  # 标准差
                Std[:, y, x] = Departure[:, y, x] / Sx  # 标准化
        return Std, len(data['lat']), len(data['lon']), TimeSeriesLength


def forma(data, lat, lon, TimeSeriesLength):
    # 计算协方差矩阵forma(N,M,MNH,F,A)
    # 用Jacobi方法计算协方差矩阵的特征值与特征向量
    # 将特征值从大到小排列arrang
    # 计算特征向量的时间系数
    print('计算协方差矩阵中...')
    X = np.zeros((lon * lat, TimeSeriesLength))
    for t in range(TimeSeriesLength):
        for j in range(lat):
            for i in range(lon):
                X[lon * j + i, t] = data[t, j, i]
    A = np.dot(X, X.T)
    print('计算特征值与特征向量中...')
    Feat, V = np.linalg.eig(A)
    print('计算时间系数矩阵中...')
    Z = np.dot(V.T, X)
    arrang = []
    for i in range(len(Feat)):
        arrang.append([Feat[i], V[i], Z[i]])
    arrang.sort(key=lambda x: x[0], reverse=True)
    return arrang


def outer(data, lat, lon, TimeSeriesLength):
    # 计算每个特征向量的方差贡献
    print('计算方差贡献中...')
    output = []
    lam_all = sum([i[0] for i in data])
    for i in range(len(data)):
        V_field = np.zeros((lat, lon))
        T_list = np.zeros(TimeSeriesLength)
        for y in range(lat):
            for x in range(lon):
                V_field[y, x] = data[i][1][y * lon + x]
        T_list = data[i][2]
        output.append([V_field, T_list, data[i][0] / lam_all])
    return output


def eof(Dataset, var, KS):
    # Dataset(str)为文件路径, var(str)为目标变量名
    # KS=-1:原始形式; KS=0:距平形式 KS=1:标准化形式
    # lon_num, lat_num(int)为经纬格点数目
    # 返回([模态场1(numpy.mat), 时间系数(numpy.mat), 方差贡献率1(float)], ...)
    data = transf(read(Dataset), var, KS)
    arrang = forma(data[0], data[1], data[2], data[3])
    return outer(arrang, data[1], data[2], data[3])
