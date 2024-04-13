import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import earth_avg_radius


def TN_WAF(Geopotential_climatic, U_climatic, V_climatic, Geopotential, lon=np.array([]), lat=np.array([]), PressLevel=200, mode=2):
    # 计算的是单一时次的TN波作用通量, 请注意输入的数据
    p = PressLevel * units('hPa')  # 也有用300hPa的
    p0 = 1000 * units('hPa')    # 标准气压
    if mode == 1:
        """
        https://cloud.tencent.com/developer/article/1826835?from=information.detail.%E7%BC%96%E7%A8%8B%E7%BB%86%E8%8A%82
        该函数用于计算T-N波作用通量
        作者：某南大气象学博士
        """
        # 目标日期和目标气压的位势
        Geopotential = xr.DataArray(Geopotential, coords=[lat, lon], dims=['lat', 'lon'])
        Geopotential.loc[90, :] = np.nan
        Geopotential.loc[-90, :] = np.nan
        Geopotential_climatic = xr.DataArray(Geopotential_climatic, coords=[lat, lon], dims=['lat', 'lon'])
        Geopotential_climatic.loc[90, :] = np.nan
        Geopotential_climatic.loc[-90, :] = np.nan
        Φ = Geopotential * units('m**2/s**2')
        # 目标月和目标气压位势的气候态
        Φ_climatic = Geopotential_climatic * units('m**2/s**2')
        # 目标月和目标气压基本流场U分量的气候态
        u_climatic = U_climatic * units('m/s')
        # 目标月和目标气压基本流场V分量的气候态
        v_climatic = V_climatic * units('m/s')

        # 经纬度转为弧度制
        lon_deg = lon
        lat_deg = lat
        lon_rad = np.deg2rad(lon_deg) * units('1')
        lat_rad = np.deg2rad(lat_deg) * units('1')

        # 科氏参数
        f = mpcalc.coriolis_parameter(lat)
        cosφ = np.cos(lat_rad)

        # 位势的扰动
        Φ_prime = Φ - Φ_climatic
        # 将需要对弧度制经纬度求偏导的量的坐标都换成弧度制经纬度
        Φ_prime = xr.DataArray(Φ_prime, coords=[lat_rad, lon_rad], dims=['lat', 'lon'])
        f = xr.DataArray(f, coords=[lat_rad], dims=['lat'])
        cosφ = xr.DataArray(cosφ, coords=[lat_rad], dims=['lat'])
        # uv 90度和-90度的数据清空,并设定为0
        u_climatic = xr.DataArray(u_climatic, coords=[lat_rad, lon_rad], dims=['lat', 'lon'])
        v_climatic = xr.DataArray(v_climatic, coords=[lat_rad, lon_rad], dims=['lat', 'lon'])
        u_climatic.loc[np.deg2rad(90), :] = np.nan
        u_climatic.loc[np.deg2rad(-90), :] = np.nan
        v_climatic.loc[np.deg2rad(90), :] = np.nan
        v_climatic.loc[np.deg2rad(-90), :] = np.nan
        # 目标月和目标气压基本流场的气候态
        wind_climatic = mpcalc.wind_speed(u_climatic, v_climatic)
        wind_climatic = xr.DataArray(wind_climatic, coords=[lat_rad, lon_rad], dims=['lat', 'lon'])
        # 准地转流函数相对于气候场的扰动
        Ψ_prime = Φ_prime / f

        # 一顿偏导猛如虎
        dΨ_prime_dλ = Ψ_prime.differentiate('lon')
        ddΨ_prime_ddλ = dΨ_prime_dλ.differentiate('lon')
        dΨ_prime_dφ = Ψ_prime.differentiate('lat')
        ddΨ_prime_ddφ = dΨ_prime_dφ.differentiate('lat')
        ddΨ_prime_dλφ = dΨ_prime_dλ.differentiate('lat')
        # T-N波作用通量的水平分量公共部分
        temp1 = p / p0 * cosφ / (2 * wind_climatic * earth_avg_radius**2)
        temp2 = dΨ_prime_dλ * dΨ_prime_dφ - Ψ_prime * ddΨ_prime_dλφ

        # T-N波作用通量的水平分量
        fx = temp1 * (u_climatic / cosφ**2 * (dΨ_prime_dλ**2 - Ψ_prime * ddΨ_prime_ddλ) + v_climatic / cosφ * temp2)
        fy = temp1 * (u_climatic / cosφ * temp2 + v_climatic * (dΨ_prime_dφ**2 - Ψ_prime * ddΨ_prime_ddφ))

        # 把弧度制经纬度,再换成角度制,便于画图
        lon_ = np.array(lon)
        fx = xr.DataArray(fx, coords=[lat_deg, lon_], dims=['lat', 'lon']).sortby(['lon', 'lat'])
        fy = xr.DataArray(fy, coords=[lat_deg, lon_], dims=['lat', 'lon']).sortby(['lon', 'lat'])
        if 90 in lat_deg:
            # 90度的数据清空,并设定为nan
            fx.loc[90, :] = np.nan
            fy.loc[90, :] = np.nan
        if -90 in lat_deg:
            # -90度的数据清空,并设定为nan
            fx.loc[-90, :] = np.nan
            fy.loc[-90, :] = np.nan
        # 赤道附近的数据清空，并设定为nan(ps:因为赤道附近f很小,地转风分量很小,很小的风速扰动会得出很大的结果,所以设为nan)
        fx.loc[-20:20, :] = np.nan
        fy.loc[-20:20, :] = np.nan
        Ψ_prime.loc[np.deg2rad(20):np.deg2rad(-20), :] = np.nan
        return fx, fy
    if mode == 2:
        """
        https://www.jianshu.com/p/97042d9019c0?ivk_sa=1024320u
        该函数用于计算T-N波作用通量
        作者:摸鱼咯
        """
        a = 6.371004e6 * units("m")     # 地球半径
        omega = 7.292e-5 * units("1/s")  # 自转角速度
        lev = PressLevel / 1000 * units("1")  # 气压层
        f = np.array(2 * omega * np.sin(np.array(lat) * np.pi / 180.)).reshape((-1, 1)) * units("1/s")    # 科氏参数
        U_climatic = np.array(U_climatic) * units("m/s")    # U风
        V_climatic = np.array(V_climatic) * units("m/s")    # V风
        Geopotential = np.array(Geopotential) * units("m**2 / s**2")    # 高度场
        Geopotential_climatic = np.array(Geopotential_climatic) * units("m**2 / s**2")    # 高度气候场
        dlon = (np.gradient(np.array(lon)) * np.pi / 180.).reshape(1, -1) * units("1")    # 经度间隔
        dlat = (np.gradient(np.array(lat)) * np.pi / 180.).reshape(-1, 1) * units("1")    # 纬度间隔
        coslat = np.cos(np.array(lat) * np.pi / 180.).reshape(-1, 1) * units("1")    # cos(lat)
        sinlat = np.sin(np.array(lat) * np.pi / 180.).reshape(-1, 1) * units("1")    # sin(lat)
        z_tmp = Geopotential_climatic    # 计算z_tmp
        wind = np.sqrt(U_climatic ** 2 + V_climatic ** 2)   # 计算|U|
        c = lev * coslat / (2 * a * a * wind)    # 计算c
        za = np.array(Geopotential) - np.array(z_tmp)   # 计算za
        streamf = za / f    # 计算流函数
        # 求偏导
        dzdlon = np.gradient(streamf, axis=1) / dlon
        dzdlat = np.gradient(streamf, axis=0) / dlat
        ddzdlonlon = np.gradient(dzdlon, axis=1) / dlon
        ddzdlatlat = np.gradient(dzdlat, axis=0) / dlat
        ddzdlatdlon = np.gradient(dzdlat, axis=1) / dlon
        # X,Y公共部分
        x_tmp1 = dzdlon * dzdlon - streamf * ddzdlonlon
        x_tmp2 = dzdlat * dzdlat - streamf * ddzdlatlat
        y_tmp = dzdlon * dzdlat - streamf * ddzdlatdlon
        # 计算X,Y
        fx = c * ((U_climatic / coslat / coslat) * x_tmp1 + V_climatic * y_tmp / coslat)
        fy = c * ((U_climatic / coslat) * y_tmp + V_climatic * x_tmp2)
        fx[0, :] = np.nan
        fy[0, :] = np.nan
        streamf[0, :] = np.nan
        fx[-1, :] = np.nan
        fy[-1, :] = np.nan
        streamf[-1, :] = np.nan
        lon, lat = np.meshgrid(lon, lat)
        fx = np.where(np.abs(lat) < 20, np.nan, fx)
        fy = np.where(np.abs(lat) < 20, np.nan, fy)
        streamf = np.where(np.abs(lat) < 20, np.nan, streamf)
        return fx, fy
