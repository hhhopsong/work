import warnings
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
        # 常量
        self.date = U.time.values
        self.lev = U.level.values
        self.lon = np.array([i for i in range(360)])  # 1×1分辨率
        self.lat = np.array([i-90 for i in range(181)]) # 1×1分辨率

        self.U = U.interp(lat=self.lat, lon=self.lon) * units.m / units.s
        self.V = V.interp(lat=self.lat, lon=self.lon) * units.m / units.s
        self.W = W.interp(lat=self.lat, lon=self.lon) / 100 * (units.hPa / units.s)
        self.T = T.interp(lat=self.lat, lon=self.lon) * units.K



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
        pressure = np.array(self.lev).reshape((len(self.lev), 1, 1)) * units.hPa
        month_days_dict = {
            1: 31, 2: 28.25, 3: 31, 4: 30, 5: 31, 6: 30,
            7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31
        }
        data_all = np.zeros((4, len(self.date), len(self.lev), len(self.lat), len(self.lon)))
        for i in tq.trange(len(self.date), desc='计算中...', unit='时次'):
            time = month_days_dict[self.date.month.values[i]] * 24 * 60 * 60 * units.s
            # 温度倾向
            if i == 0:
                dTdt = self.T[i + 1] * units.K - self.T[i] * units.K  # 时间前向差分
            elif i == len(self.date) - 1:
                dTdt = self.T[i] * units.K - self.T[i - 1] * units.K  # 时间后向差分
            else:
                dTdt = self.T[i + 1] * units.K - self.T[i - 1] * units.K  # 时间中央差分
                dTdt = dTdt / 2.
            # 温度平流
            dx, dy = mpcalc.lat_lon_grid_deltas(self.lon, self.lat)
            adv_T = np.zeros((len(self.lev), len(self.lat), len(self.lon)))
            for ilev in range(len(self.lev)):
                adv_T[ilev] = mpcalc.advection(self.T[i, ilev, :, :], self.U[i, ilev, :, :], self.V[i, ilev, :, :], dx=dx, dy=dy, x_dim=-1, y_dim=-2)
            adv_T = adv_T
            # 静力稳定度
            T_K = self.T[i, :, :, :]
            ss = ((constants.dry_air_gas_constant * T_K) / constants.dry_air_spec_heat_press / pressure
                  - np.gradient(T_K, axis=0) / np.gradient(pressure, axis=0))
            ver = self.W[i, :, :, :] * ss
            # 非绝热加热率
            Q = ((dTdt - adv_T * time - ver * time) * constants.dry_air_spec_heat_press) / time
            data_all[:, i] = np.array([dTdt, adv_T, ver, Q])
        # DataSet格式化
        data = xr.Dataset({
            'dTdt': (['time', 'level', 'lat', 'lon'], data_all[0]),
            'adv_T': (['time', 'level', 'lat', 'lon'], data_all[1]),
            'ver': (['time', 'level', 'lat', 'lon'], data_all[2]),
            'Q': (['time', 'level', 'lat', 'lon'], data_all[3])},
            coords={'level': self.lev, 'lat': self.lat, 'lon': self.lon, 'time': self.date})
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
    budget = TemperatureBudget(U, V, W, T)

    budget.to_nc(r'E:\data\ERA5\ERA5_pressLev\temperature_budget.nc')