import warnings
import pandas as pd
import numpy as np
import xarray as xr
import tqdm as tq

from metpy.units import units
import metpy.calc as mpcalc
import metpy.constants as constants

class TemperatureBudget:
    """
    U : xarray.DataArray
        水平纬向风场['time, 'level', 'lat', 'lon']
    V : xarray.DataArray
        水平经向风场['time, 'level', 'lat', 'lon']
    W : xarray.DataArray
        垂直速度场 (Pa/s) ['time, 'level', 'lat', 'lon']
    T : xarray.DataArray
        温度异常场,时间上在目标月份前后各取一个月以进行差分['time, 'level', 'lat', 'lon']
    ----------

    用以计算温度收支方程:\n
    ∂T/∂t = -V·∇T + ωσ + Q/Cp \n

    T为温度\n
    t为时间(s)\n
    V为水平速度矢量\n
    ∇T为温度水平梯度\n
    ω为垂直速度\n
    σ表示静力稳定度\n
    Q表示非绝热加热率\n
    Cp表示定压比热容

    ----------

    【另附扰动方程：∂T'/∂t = -(V·∇T)' + (ωσ)' + Q'/Cp】\n
    '表示扰动量

    """
    def __init__(self, U: xr.DataArray, V: xr.DataArray, W: xr.DataArray, T: xr.DataArray):
        # 忽略RuntimeWarning
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        # 常量
        self.date = U.time.values
        self.lev = U.level.values
        self.lon = U.lon.values
        self.lat = U.lat.values
        self.U = np.array(U) * units.m / units.s
        self.V = np.array(V) * units.m / units.s
        self.W = np.array(W) * (units.Pa / units.s)
        self.T = np.array(T) * units.K
        # 结果
        self.data = self.main()
        self.dTdt = self.data['dTdt']
        self.adv_T = self.data['adv_T']
        self.ver = self.data['ver']
        self.Q = self.data['Q']

    def main(self):
        """
        计算温度收支方程
        Returns
        -------
        data : xarray.Dataset
            温度收支方程['year', 'level', 'lat', 'lon']
            dTdt: 温度倾向
            adv_T: 温度平流
            ver: 垂直速度扰动
            Q: 非绝热加热率
        """
        pressure = np.array(self.lev).reshape((len(self.lev), 1, 1)) * 100 * units.Pa
        month_days_dict = {
            1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        meta_start_year = pd.to_datetime(self.date)[0].year == 1961 and pd.to_datetime(self.date)[0].month == 1 # 是否为1961年1月
        if meta_start_year:
            data_all = np.zeros((4, len(self.date)-1, len(self.lev), len(self.lat), len(self.lon)))
        else:
            data_all = np.zeros((4, len(self.date)-2, len(self.lev), len(self.lat), len(self.lon)))
        date_nums = 0
        for i in range(len(self.date)):
            time = month_days_dict[pd.to_datetime(self.date)[i].month] * 24 * 60 * 60 * units.s
            # 温度倾向
            if i == 0 or i == len(self.date) - 1:
                if meta_start_year and i == 0:
                    dTdt = self.T[i + 1] - self.T[i] # 时间前向差分
                else:
                    continue  # 跳过第一个和最后一个不可计算中央差的时次
            else:
                dTdt = self.T[i + 1] - self.T[i - 1]  # 时间中央差分
                dTdt = dTdt / 2.
            dTdt = dTdt / time
            # 温度平流
            dx, dy = mpcalc.lat_lon_grid_deltas(self.lon, self.lat)
            adv_T = np.zeros((len(self.lev), len(self.lat), len(self.lon)))
            for ilev in range(len(self.lev)):
                adv_T[ilev] = mpcalc.advection(self.T[i, ilev, :, :], self.U[i, ilev, :, :], self.V[i, ilev, :, :], dx=dx, dy=dy, x_dim=-1, y_dim=-2)
            adv_T = adv_T * units.K / units.s
            # 静力稳定度
            T_K = self.T[i, :, :, :]
            ss = ((constants.dry_air_gas_constant * T_K) / constants.dry_air_spec_heat_press / pressure - np.gradient(T_K, axis=0) / np.gradient(pressure, axis=0))
            ver = self.W[i, :, :, :] * ss
            # 非绝热加热
            Q = dTdt - adv_T - ver
            data_all[:, date_nums] = np.array([dTdt, adv_T, ver, Q]) * units.K / units.s
            date_nums += 1
        # DataSet格式化
        data = xr.Dataset({
            'dTdt': (['time', 'level', 'lat', 'lon'], data_all[0]),
            'adv_T': (['time', 'level', 'lat', 'lon'], data_all[1]),
            'ver': (['time', 'level', 'lat', 'lon'], data_all[2]),
            'Q': (['time', 'level', 'lat', 'lon'], data_all[3])},
            coords={'level': self.lev, 'lat': self.lat, 'lon': self.lon, 'time': self.date[1-meta_start_year:-1]})
        return data

    def to_nc(self, path):
        """
        保存为nc文件
        Parameters
        ----------
        path : str
            保存路径
        """
        self.data.to_netcdf(path)


if __name__ == '__main__':
    U = xr.open_dataset(r'E:\data\ERA5\ERA5_pressLev\single_var\U.nc')['u']
    V = xr.open_dataset(r'E:\data\ERA5\ERA5_pressLev\single_var\V.nc')['v']
    W = xr.open_dataset(r'E:\data\ERA5\ERA5_pressLev\single_var\W.nc')['w']
    T = xr.open_dataset(r'E:\data\ERA5\ERA5_pressLev\single_var\T.nc')['t']

    for i in tq.trange(1961, 2023):
        u = U.sel(time=slice(str(i - 1) + '-12', str(i + 1) + '-01'))
        v = V.sel(time=slice(str(i - 1) + '-12', str(i + 1) + '-01'))
        w = W.sel(time=slice(str(i - 1) + '-12', str(i + 1) + '-01'))
        t = T.sel(time=slice(str(i - 1) + '-12', str(i + 1) + '-01'))
        budget = TemperatureBudget(u, v, w, t)
        budget.to_nc(fr'E:\data\ERA5\ERA5_pressLev\single_var\t_budget\t_budget_{i}.nc')

    '''# 指定目录路径
    dir_path = r"E:\data\ERA5\ERA5_pressLev\single_var\t_budget"
    # 获取目录下所有.nc文件（按文件名排序确保时间顺序）
    all_files = sorted([os.path.join(dir_path, f)
                        for f in os.listdir(dir_path)
                        if f.endswith(".nc")])
    # 使用xarray打开多个文件（自动沿时间维度合并）
    # parallel=True 启用并行读取
    # chunks 设置分块策略（根据内存情况调整）
    combined = xr.open_mfdataset(
        all_files,
        combine="by_coords",  # 自动根据坐标合并
        parallel=True,
        chunks={"time": 100}  # 示例分块大小，按需调整
    )

    # 查看合并后的数据集结构
    print(combined)

    # 输出合并后的文件（建议使用不同文件名）
    output_path = os.path.join(dir_path, "combined.nc")
    combined.to_netcdf(output_path)

    print(f"文件已合并保存至：{output_path}")'''