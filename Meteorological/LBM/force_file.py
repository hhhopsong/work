import time

import xarray as xr
import xgrads as xg
import cmaps
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from scipy.interpolate import interpolate
from tqdm import tqdm
import tqdm as tq
from f2py.dsphe import G2W
from cartopy.util import add_cyclic_point
from f2py.dim import NTR, NMDIM, KMAX, NMAX, NVAR, MMAX, LMAX
import torch


device = torch.device('cuda')
force_file_address = '//wsl.localhost/Ubuntu-20.04/home/hopsong/lbm/data/frc'

# LBM网格
lbm_lon = [i * 2.8125 for i in range(128)]
lbm_lat = [-87.8638, -85.0965, -82.3129, -79.5256, -76.7369, -73.9475, -71.1577, -68.3678, -65.5776, -62.7873,
           -59.9970, -57.2066, -54.4162, -51.6257, -48.8352, -46.0447, -43.2542, -40.4636, -37.6731, -34.8825,
           -32.0919, -29.3014, -26.5108, -23.7202, -20.9296, -18.1390, -15.3484, -12.5578, -9.76715, -6.97653,
           -4.18592, -1.39531, 1.39531, 4.18592, 6.97653, 9.76715, 12.5578, 15.3484, 18.1390, 20.9296, 23.7202,
           26.5108, 29.3014, 32.0919, 34.8825, 37.6731, 40.4636, 43.2542, 46.0447, 48.8352, 51.6257, 54.4162,
           57.2066, 59.9970, 62.7873, 65.5776, 68.3678, 71.1577, 73.9475, 76.7369, 79.5256, 82.3129, 85.0965,
           87.86384]
lon_grid, lat_grid = np.meshgrid(lbm_lon, lbm_lat)
level_sig = [0.99500, 0.97999, 0.94995, 0.89988, 0.82977, 0.74468, 0.64954, 0.54946, 0.45447, 0.36948, 0.29450,
             0.22953, 0.17457, 0.12440, 0.0846830, 0.0598005, 0.0449337, 0.0349146, 0.0248800, 0.00829901]
level_sigp = [isig * (1000 - 1) + 1 for isig in level_sig]
level_p = [1000, 950, 900, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10, 7, 5]
S2D = 86400.
K = 20  # 垂直层数

""":param address: str, 强迫场文件地址"""


def read_force_file(address=force_file_address):
    lbm = xg.open_mfdataset(f'{address}/draw.ctl')
    return lbm, level_sigp


def grid2wave(data=None, lat=64, N=128, M=42, K_=20, re=False, ops=True, HGRAD='POSO', debug=False):

    """
    格点数据转谱系数
    :param lat: int, 纬向格点数
    :param N: int, 经向格点数
    :param data: np.array, 格点数据(lev, 64, 128)
    :return: np.array, 谱系数
    """
    Z = np.zeros((NMDIM, KMAX))
    start = time.time()
    try:
        if ops:
            Z = G2W(torch.tensor(Z).to(device), GDATA=torch.tensor(data).to(device), HGRAD='    ', HFUNC='POSO', KMAXD=torch.tensor(1).to(device))
            end = time.time()
            if debug:
                print('G2W函数运行时间:{}'.format(end - start))
            return Z.cpu().numpy()
        else:
            Z = G2W(torch.tensor(Z).to(device), GDATA=torch.tensor(data).to(device), HGRAD=HGRAD, HFUNC='POSO', KMAXD=torch.tensor(K_).to(device))
            end = time.time()
            if debug:
                print('G2W函数运行时间:{}'.format(end - start))
            return Z.cpu().numpy()

    except:
        raise ValueError('G2W函数运行失败')
    '''Z = np.zeros((lat * M, len(data)), dtype=complex)
    Z.fill(complex(0, 0))
    if data is None:
        raise ValueError('data参数不能为空')
    try:
        one = np.load(f'D:\CODES\Python\Meteorological\LBM\g2w.npy')
    except:
        if not re:
            one = np.ones((K_, lat, N), dtype=complex)
            for K in tqdm(range(len(data)), desc='Grid to Wave(PRE):', unit='层', position=0, colour='green'):
                for ilat in range(lat):
                    for k in range(M):
                        for j in range(N):
                            λ = 2 * np.pi * j / N
                            one[K, ilat, j] = one[K, ilat, j] * (np.cos(k * λ) * complex(1, 0) - np.sin(k * λ) * complex(0, 1))
            np.save(f'D:\CODES\Python\Meteorological\LBM\g2w.npy', one)
            one = np.load(f'D:\CODES\Python\Meteorological\LBM\g2w.npy')
        if re:
            raise ValueError('逆向功能尚未开发')
    for K in range(len(data)):
        for ilat in range(lat):
            for k in range(M):
                for j in range(N):
                    Z[ilat * M + k, K] += data[K, ilat, j] * one[K, ilat, j]
    Z = Z / N
    return Z'''



def vertical_structure(data=force_file_address, element='t', show=False):
    """
    Show the vertical structure of the element.
    :param element: str, 气象要素('v', 'd', 't', 'p', 'q')
    :return: element: xr.DataArray, 垂直结构数据
    """
    lbm, level = read_force_file(address=data)
    element = lbm[element]
    element = np.where(element.to_numpy() == 0, np.nan, element.to_numpy().byteswap())
    element = xr.DataArray(element, coords=[lbm['time'], lbm['lev'], lbm['lat'], lbm['lon']],
                           dims=['time', 'lev', 'lat', 'lon'])
    if show:
        element = element.mean(['lon', 'lat'])
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        # 设置y轴从1000hPa到0hPa
        ax.set_ylim(1000, 0)
        x_max = np.nanmax(element)
        x_min = np.nanmin(element)
        x_size = np.abs(x_max) if np.abs(x_max) > np.abs(-x_min) else np.abs(x_min)
        ax.set_xlim(-x_size * 1.2, x_size * 1.2)
        # 展示垂直结构
        element.plot(ax=ax, y='lev', yincrease=False, marker='o')
        plt.show()
    return element


def interp_to_lbm(data=None):
    """
    LBM网格插值函数
    :param data: xr.DataArray, 自定义强迫场数据, 数据顺序为(lev, lat, lon), lev为P坐标系
    :return: np.array, 插值后的数据
    """
    if data is None:
        raise ValueError('data参数不能为空')
    elif len(data['lev']) <= 1:
        raise ValueError('data垂直层次错误')
    elif len(data['lev']) > 1:
        data = xr.DataArray(data.to_numpy(), coords=[data['lev'], data['lat'], data['lon']], dims=['lev', 'lat', 'lon'])
        interp = interpolate.RegularGridInterpolator((data['lev'], data['lat'], data['lon']), data, method='linear')
        data_interp = interp((level_p, lat_grid, lon_grid))
        data_interp = np.where(np.isnan(data_interp), 0, data_interp)
        return data_interp


def vertical_profile(data=None, K=20, kvpr=2, vamp=8., vdil=20., vcnt=0.45, show=False):
    """
    生成强迫场的理想化垂直结构.
    :param data: xr.DataArray, 自定义垂直结构数据
    :param K: int, 垂直层数
    :param kvpr: int, 理想化结构类型（1:三角函数, 2:Gamma函数, 3:垂直均匀）
    :param vamp: float, 振幅(1为原数据振幅)
    :param vdil: float, 振幅衰减率
    :param vcnt: float, 振幅中心(sigma坐标)
    :param idealization_make: bool, 是否进行理想化结构生成
    :return: None
    """
    if data is not None:
        # data数据顺序为(lev), lev为P坐标系[1000-0hPa]
        if len(data['lev']) <= 1:
            raise ValueError('data垂直层次错误')
        data = xr.DataArray(data.to_numpy(), coords=[data['lev']], dims=['lev'])
        interp = interpolate.RegularGridInterpolator((data['lev']), data, method='linear')  # date插值到lbm网格上
        data_interp = interp(level_p)
        data_interp = np.where(np.isnan(data_interp), 0, data_interp)
        return [i for i in data_interp]
    structure = [vamp for i in range(K)]
    # 生成理想化垂直结构
    # 生成垂直结构
    for i in range(K):
        if kvpr == 1:
            # 三角函数
            structure[i] = vamp * np.sin(np.pi * level_sig[i])
        elif kvpr == 2:
            # Gamma函数
            structure[i] = vamp * np.exp(-vdil * (level_sig[i] - vcnt) ** 2)
        elif kvpr == 3:
            # 垂直均匀
            structure[i] = vamp
        else:
            raise ValueError('kvpr参数错误')
    if show:
        # 画图
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_title('理想强迫场垂直结构')
        # 设置y轴从1000hPa到0hPa,且为log坐标
        ax.set_ylim(1000, 0)
        x_max = np.nanmax(structure)
        x_min = np.nanmin(structure)
        x_size = np.abs(x_max) if np.abs(x_max) > np.abs(-x_min) else np.abs(x_min)
        ax.set_xlim(-x_size * 1.2, x_size * 1.2)
        structure_plot = xr.DataArray(structure, coords=[level_p], dims=['lev'])
        structure_plot.plot(ax=ax, y='lev', marker='o')
        plt.show()
    return structure


def horizontal_profile(data=None, khpr=1, hamp=0.25, xdil=23., ydil=6.5, xcnt=77., ycnt=-1.5):
    """
    生成强迫场的理想化水平结构.
    :param data: xr.DataArray, 自定义水平结构数据
    :param khpr: int, 理想化结构类型（1:椭圆函数, 2:区域一致）
    :param hamp: float, 振幅(1为原数据振幅)
    :param xdil: float, 强迫区域半径
    :param ydil: float, 强迫区域半径
    :param xcnt: float, 强迫中心经度(-180, 180)
    :param ycnt: float, 强迫中心纬度(-90, 90)
    :return: None
    """
    if data is not None:
        # data须为单一时次单层数据, 数据顺序为(lat, lon)
        if len(data['lev']) != 1:
            raise ValueError('data垂直层次错误')
        data = xr.DataArray(data.to_numpy(), coords=[data['lat'], data['lon']], dims=['lat', 'lon'])
        interp = interpolate.RegularGridInterpolator((data['lat'], data['lon']), data, method='linear')  # date插值到lbm网格上
        data_interp = interp((lat_grid, lon_grid))
        data_interp = np.where(np.isnan(data_interp), 0, data_interp)
        return data_interp
    # 生成理想化水平结构
    if xcnt < -180 or xcnt > 180 or ycnt < -90 or ycnt > 90:
        raise ValueError('xcnt或ycnt参数错误')
    elif xdil <= 0 or ydil <= 0:
        raise ValueError('xdil或ydil参数错误')
    hor_structure = np.zeros((len(lbm_lat), len(lbm_lon)))
    for i in range(len(lbm_lat)):
        for j in range(len(lbm_lon)):
            if khpr == 1:
                # 椭圆函数
                dx = (lbm_lon[j] - xcnt) ** 2 / xdil ** 2
                if lbm_lon[j] < 0. and xcnt + xdil >= 180.:
                    dx = (360. + lbm_lon[j] - xcnt) ** 2 / xdil ** 2
                elif lbm_lon[j] > 0. and xcnt - xdil <= -180.:
                    dx = (lbm_lon[j] - 360. - xcnt) ** 2 / xdil ** 2
                dy = (lbm_lat[i] - ycnt) ** 2 / ydil ** 2
                d = dx + dy
                if d > 1.:
                    hor_structure[i, j] = 0.
                else:
                    d = np.sqrt(d)
                    hor_structure[i, j] = hamp * (1. - d)
            elif khpr == 2:
                # 区域平均
                if lbm_lat[i] >= ycnt - ydil and lbm_lat[i] <= ycnt + ydil:
                    dx = lbm_lon[j] - xcnt
                    if lbm_lon[j] < 0. and xcnt + xdil >= 180.:
                        dx = 360. + lbm_lon[j] - xcnt
                    elif lbm_lon[j] > 0. and xcnt - xdil <= -180.:
                        dx = lbm_lon[j] - 360. - xcnt
                    if np.abs(dx) <= xdil:
                        hor_structure[i, j] = hamp
                else:
                    hor_structure[i, j] = 0.
            else:
                raise ValueError('khpr参数错误')
    return hor_structure


def mk_grads(data=None, url=None, structure=None, hor_structure=None, ver_structure=None, ovor=0, odiv=0, otmp=0, ops=0, osh=0):
    """
    生成强迫场GrADS文件
    :param data:    xr.DataArray, 自定义强迫场数据(涡度, 散度, 温度, 海平面气压, 水汽)
    :param url:    str, GrADS文件地址
    :param structure:   np.array, 自定义强迫场结构(涡度, 散度, 温度, 海平面气压, 水汽)
    :param hor_structure:   np.array, 水平结构
    :param ver_structure:   np.array, 垂直结构
    :param ovor:    float, 涡度强迫
    :param odiv:    float, 散度强迫
    :param otmp:    float, 温度强迫
    :param ops:    float, 海平面气压强迫(地形试验)
    :param osh:   float, 水汽强迫
    :return: np.array, 强迫场
    """
    Kv = K
    if ops:
        Kv = 1
    GFrct = np.zeros((5, Kv, len(lbm_lat), len(lbm_lon)))
    if data is not None:
        if ops and len(data['lev']) > 1:
            raise ValueError('date为多层数据时不能进行地形试验')
        for ie in range(5):
            for i in range(Kv):
                GFrct[ie, i, :, :] = data.loc[ie, i, :, :].to_numpy() / S2D
        frc = xr.DataArray(GFrct.astype(np.float32), coords=[['v', 'd', 't', 'p', 'q'], range(Kv), lbm_lat, lbm_lon],
                           dims=['var', 'lev', 'lat', 'lon'])
        if url is None:
            return frc.to_numpy()
        else:
            print("强迫场数据已写入GrADS文件")
            frc.to_netcdf(url)
            return frc.to_numpy()
    elif hor_structure is None or ver_structure is None:
        raise ValueError('hor_structure和ver_structure不能为空')
    else:
        for ie in range(5):
            for k in range(Kv):
                for i in range(len(lbm_lat)):
                    for j in range(len(lbm_lon)):
                        GFrct[ie, k, i, j] = hor_structure[i, j] * ver_structure[k] / S2D
    if structure:
        if len(np.shape(structure)) != (5, K, len(lbm_lat), len(lbm_lon)):
            raise ValueError('structure数据维度错误')
        else:
            GFrct = structure
    # 写入GrADS文件
    for iK in range(K):
        frc = np.zeros((5, Kv, len(lbm_lat), len(lbm_lon)))
        if ovor:
            frc[0, iK, :, :] = GFrct[0, iK, :, :] * ovor
        if odiv:
            frc[1, iK, :, :] = GFrct[1, iK, :, :] * odiv
        if otmp:
            frc[2, iK, :, :] = GFrct[2, iK, :, :] * otmp
        if ops:
            frc[3, iK, :, :] = GFrct[3, iK, :, :] * ops
        if osh:
            frc[4, iK, :, :] = GFrct[4, iK, :, :] * osh
        frc = frc.astype(np.float32)
        frc = xr.DataArray(frc, coords=[['v', 'd', 't', 'p', 'q'], range(Kv), lbm_lat, lbm_lon],
                           dims=['var', 'lev', 'lat', 'lon']).to_dataset(name='frc')
        print("强迫场数据已写入GrADS文件")
        if url is None:
            return frc
        else:
            print("强迫场数据已写入GrADS文件")
            frc.to_netcdf(url+r'\frc.nc')
            return frc


def SetNMO2(Mmax, Lmax, Nmax, Mint):
    """
    设置非线性模态
    :param Mmax: int, 纬向波数
    :param Lmax: int, 最大垂直波数
    :param Nmax: int, 全波数(二维指数)
    :param Mint: int, 波数步长
    :return: NMO: np.array, 非线性模态
    """
    Nmh = 0
    if Mmax is None:
        Mmax = Nmax
    NMO = np.zeros((2, Mmax+ 1, Lmax + 1))
    for l in range(Lmax + 1):
        Mend = np.min(np.array([int(Mmax), int(Nmax - l)]))
        for m in range(0, Mend + 1, Mint):
            Nmh += 1
            if Mmax == 0:
                NMO[0, m, l] = Nmh
                NMO[1, m, l] = Nmh
            else:
                NMO[0, m, l] = 2 * Nmh - 1
                NMO[1, m, l] = 2 * Nmh
    return NMO.astype(int)


#PWM 的强迫向量被重新排序，由 owall=f 定义的边界条件控制。在这种情况下，强迫向量的顺序是：v, d, t, p, q。

def mk_wave(Gfrct, Mmax=None, Lmax=42, Nmax=42, Mint=1, ovor=False, odiv=False, otmp=False, ops=False, osh=False, owall=True, oclassic=True, debug=False):
    """
    进行强迫场的谱系数计算
    :param Gfrct: np.array, 强迫场
    :param Mmax: int, 纬向波数
    :param Lmax: int, 最大垂直波数
    :param Nmax: int, 全波数(二维指数)
    :param Mint: int, 波数步长
    :param ovor: bool, 涡度强迫
    :param odiv: bool, 散度强迫
    :param otmp: bool, 温度强迫
    :param ops: bool, 海平面气压强迫
    :param osh: bool, 水汽强迫
    :param owall: bool, 矩阵求解器应用于整个矩阵/对角矩阵
    :param oclassic: bool, 是否为经典强迫（不含水汽)
    :param debug: bool, 调试模式
    :return: 二进制文件: frc.mat
    """
    K_ = K
    if ops:
        K_ = 1
    else:
        K_ = K
    if Mmax is None:
        Mmax = Nmax
    if len(Gfrct[0, :K_, :, :]) != K_:
        raise ValueError('Gfrct垂直层次错误')
    Ntr = NTR# 三角形截断谱分量个数
    Jw = np.zeros((Ntr+1), dtype=int)
    MAXN = 2 * Nmax * (NVAR * KMAX + 1)
    Wfrcf = np.zeros(((Lmax+1) * (Nmax+1), K_))
    Wxvor = np.zeros((MAXN, K_, Ntr + 1))
    Wxdiv = np.zeros((MAXN, K_, Ntr + 1))
    Wxtemp = np.zeros((MAXN, K_, Ntr + 1))
    Wxps = np.zeros((MAXN, Ntr + 1))
    Wxsph = np.zeros((MAXN, K_, Ntr + 1))
    NMO = np.zeros((2, MMAX + 1, LMAX + 1), dtype=int)
    NMO = SetNMO2(Mmax, Lmax, Nmax, Mint)
    iW = -1
    result = []
    # 先调整维度顺序
    Gfrct = Gfrct.transpose(0, 1, 3, 2).reshape(Gfrct.shape[0], Gfrct.shape[1], (128+1) * 64)

    g2w_ovor = grid2wave(Gfrct[0, :K_, :].T, ops=ops, debug=debug)
    g2w_div = grid2wave(Gfrct[1, :K_, :].T, ops=ops, debug=debug)
    g2w_temp = grid2wave(Gfrct[2, :K_, :].T, ops=ops, debug=debug)
    g2w_ps = grid2wave(Gfrct[3, :K_, :].T, ops=ops, debug=debug)
    g2w_sph = grid2wave(Gfrct[4, :K_, :].T, ops=ops, debug=debug)

    for m in tq.trange(Ntr + 1):
        Lend = np.min(np.array([int(Lmax), int(Nmax - m)]))
        for iK in range(K_):
            iW = -1
            for l in range(Lend + 1):
                if m == 0 and l == 0:
                    continue
                i = NMO[0, m, l]
                j = NMO[1, m, l]
                iW += 1
                if ovor:
                    Wxvor[iW, iK, m] = g2w_ovor[i, iK]
                if odiv:
                    Wxdiv[iW, iK, m] = g2w_div[i, iK]
                if otmp:
                    Wxtemp[iW, iK, m] = g2w_temp[i, iK]
                if ops:
                    Wxps[iW, m] = g2w_ps[i, 0]
                if osh:
                    Wxsph[iW, iK, m] = g2w_sph[i, iK]
                if m==0:
                    continue
                iW += 1
                if ovor:
                    Wxvor[iW, iK, m] = g2w_ovor[j, iK]
                if odiv:
                    Wxdiv[iW, iK, m] = g2w_div[j, iK]
                if otmp:
                    Wxtemp[iW, iK, m] = g2w_temp[j, iK]
                if ops:
                    Wxps[iW, m] = g2w_ps[j, 0]
                if osh:
                    Wxsph[iW, iK, m] = g2w_sph[j, iK]
        if not owall:
            if oclassic:
                result.append(Wxvor[0:iW, :K_, m])
                result.append(Wxdiv[0:iW, :K_, m])
                result.append(Wxtemp[0:iW, :K_, m])
                result.append(Wxps[0:iW, m])
            else:
                result.append(Wxvor[0:iW, :K_, m])
                result.append(Wxdiv[0:iW, :K_, m])
                result.append(Wxtemp[0:iW, :K_, m])
                result.append(Wxps[0:iW, m])
                result.append(Wxsph[0:iW, :K_, m])
        else:
            Jw[m] = iW
    if owall:
        if oclassic:
            for m in range(Ntr + 1):
                result.append(Wxvor[0:Jw[m], :K_, m])
                result.append(Wxdiv[0:Jw[m], :K_, m])
                result.append(Wxtemp[0:Jw[m], :K_, m])
                result.append(Wxps[0:Jw[m], m])
        else:
            for m in range(Ntr + 1):
                result.append(Wxvor[0:Jw[m], :K_, m])
                result.append(Wxdiv[0:Jw[m], :K_, m])
                result.append(Wxtemp[0:Jw[m], :K_, m])
                result.append(Wxps[0:Jw[m], m])
                result.append(Wxsph[0:Jw[m], :K_, m])
    np.concatenate([arr.ravel() for arr in result], dtype=np.float64).tofile(force_file_address+'/frc.mat')
    if owall:
        print('Get matrix file (all)')
    else:
        print('Get matrix file (pwm)')
    return result



if __name__ == '__main__':
    v = vertical_profile(kvpr=2, vamp=8., vdil=20., vcnt=0.45)  # 生成强迫场的理想化垂直结构
    h = horizontal_profile(khpr=1, hamp=0.25, xdil=23., ydil=6.5, xcnt=77., ycnt=-1.5)  # 生成强迫场的理想化水平结构
    frc = mk_grads(hor_structure=h, ver_structure=v, ovor=0, odiv=0, otmp=1, ops=0, osh=0)  # 生成强迫场
    frc_mat = mk_wave(np.array(add_cyclic_point(frc.to_dataarray()[0], coord=frc['lon'])[0]), Lmax=64, Nmax=42, Mint=1, ovor=False, odiv=False, otmp=True, ops=False, osh=False, owall=True, oclassic=True, debug=False)  # 生成谱资料
