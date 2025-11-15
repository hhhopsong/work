import numpy as np
import xarray as xr
import metpy.calc as mpcalc
from metpy.units import units
from metpy.constants import earth_avg_radius
from scipy.ndimage import filters

def TN_WAF(Geopotential_climatic, U_climatic, V_climatic, Geopotential, lon=np.array([]), lat=np.array([]), PressLevel=200, mode=2):
    """
    计算的是单一时次的TN波作用通量, 请注意输入的数据格式为list或者numpy.array(推荐),代码参考了下列样例,并做了勘误。\n

    计算模式 1 作者:气象学家(网络昵称)
        https://cloud.tencent.com/developer/article/1826835?from=information.detail.%E7%BC%96%E7%A8%8B%E7%BB%86%E8%8A%82\n
    计算模式 2 作者:摸鱼咯(网络昵称)
        https://www.jianshu.com/p/97042d9019c0?ivk_sa=1024320u
    :param Geopotential_climatic: 气候态位势高度场
    :param U_climatic: 气候态U风
    :param V_climatic: 气候态V风
    :param Geopotential: 目标时次位势高度场
    :param lon: 经度
    :param lat: 纬度
    :param PressLevel: 气压层
    :param mode: 计算模式
    :return: fx, fy
    """
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
        coslat = (np.cos(np.array(lat) * np.pi / 180.).reshape(-1, 1)) * units("1")    # cos(lat)
        sinlat = (np.sin(np.array(lat) * np.pi / 180.).reshape(-1, 1)) * units("1")    # sin(lat)
        z_tmp = Geopotential_climatic   # 计算z_tmp
        wind = np.sqrt(U_climatic ** 2 + V_climatic ** 2)   # 计算|U|
        c = lev * coslat / (2 * a * a * wind)    # 计算c
        za = np.array(Geopotential)  # 计算za
        streamf = za * units("m**2 / s**2") / f    # 计算流函数
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


def TN_WAF_3D(Uc, Vc, GEOa, Tc=None, u_threshold=None, return_streamf=False, filt=0, filtmode='strict'):
    """
    计算的是三维的TN波作用通量, 请注意输入的数据格式为3D-DataArray,代码参考了下列样例,并做了勘误。\n
    https://www.bilibili.com/read/cv15633261/?spm_id_from=333.999.collection.opus.click
    :param Uc:    气候态U风
    :param Vc:  气候态V风
    :param GEOa:    位势高度场扰动场
    :param Tc:  气候态温度
    :param u_threshold:    风速阈值
    :param return_streamf:    是否返回流函数
    :param filt:    是否进行滤波
    :param filtmode:    滤波模式['mix','strict']
    :return:    Fx, Fy, Fz
    """
    ### 常量
    gc = 290.0  # 气体常数
    g = 9.80665  # 重力加速度
    re = 6378388.0  # 地球半径
    sclhgt = 8000.0  # 大气标高
    omega = 7.292e-5  # 角速度

### 函数  给定气候态和位势扰动；输入三维(level,lat,lon)的DataArray数据
    ### 数据维度和坐标
    data_shape =GEOa.shape
    data_coords=GEOa.coords

    ### 检验维度是否匹配
    if len(data_shape) != 3:
        raise ValueError('数据维度不匹配,请检查数据维度是否为三维(level,lat,lon)')

    ### 气候态和扰动
    UVc=np.sqrt(Uc**2+Vc**2)

    if not data_shape[0]==1:
        Tc  =xr.where(abs(Tc  ['lat'])<=20,np.nan,Tc  ).transpose('level','lat','lon')
    Uc  =xr.where(abs(Uc  ['lat'])<=20,np.nan,Uc  ).transpose('level','lat','lon')
    Vc  =xr.where(abs(Vc  ['lat'])<=20,np.nan,Vc  ).transpose('level','lat','lon')
    UVc =xr.where(abs(UVc ['lat'])<=20,np.nan,UVc ).transpose('level','lat','lon')
    PSI_global = GEOa.transpose('level','lat','lon')
    GEOa=xr.where(abs(GEOa['lat'])<=20,np.nan,GEOa).transpose('level','lat','lon')


    lon=np.array(GEOa['lon'  ])[np.newaxis,np.newaxis,:         ]
    lat=np.array(GEOa['lat'  ])[np.newaxis,:         ,np.newaxis]
    pp =np.array(GEOa['level'])[:         ,np.newaxis,np.newaxis]

    if not data_shape[0]==1:
        Tc  =np.array(Tc  )
    Uc  =np.array(Uc  )
    Vc  =np.array(Vc  )
    UVc =np.array(UVc )
    GEOa=np.array(GEOa)

    if not u_threshold is None:
        if not data_shape[0]==1:
            Tc  =np.where(Uc>=0,Tc  ,np.nan)
        Uc  =np.where(Uc>=u_threshold,Uc  ,np.nan)
        Vc  =np.where(Uc>=u_threshold,Vc  ,np.nan)
        UVc =np.where(Uc>=u_threshold,UVc ,np.nan)
        GEOa=np.where(Uc>=u_threshold,GEOa,np.nan)

    ### 坐标、常数补充
    ## 坐标差分
    dlon  =np.deg2rad(np.gradient(lon,axis=2))
    dlat  =np.deg2rad(np.gradient(lat,axis=1))
    coslat=np.array(np.cos(np.deg2rad(lat)))
    sinlat=np.array(np.sin(np.deg2rad(lat)))
    if not data_shape[0]==1:
        dlev=np.gradient(-sclhgt*np.log(pp/1000.0),axis=0)

    ## 科氏参数
    f=2*omega*sinlat

    ## N^2
    if not data_shape[0]==1:
        N2=np.array(gc*(pp/1000.0)**0.286)/sclhgt*np.gradient(Tc*(1000.0/pp)**0.286,axis=0)/\
            (np.gradient(-sclhgt*np.log(pp/1000.0),axis=0))

    ## PSI
    PSIa=GEOa/f
    PSI_global = (PSI_global.where(PSI_global['lat']>=0, 0) - PSI_global.where(PSI_global['lat']<0, 0))/f
    PSI_global = np.array(PSI_global)

    ### 差分、计算TN通量三个分量
    ## 差分
    dzdlon=np.gradient(PSIa,axis=2)/dlon
    dzdlat=np.gradient(PSIa,axis=1)/dlat

    ddzdlonlon=np.gradient(dzdlon,axis=2)/dlon
    ddzdlonlat=np.gradient(dzdlon,axis=1)/dlat
    ddzdlatlat=np.gradient(dzdlat,axis=1)/dlat

    if not data_shape[0]==1:
        dzdlev    =np.gradient(PSIa  ,axis=0)/dlev
        ddzdlonlev=np.gradient(dzdlon,axis=0)/dlev
        ddzdlatlev=np.gradient(dzdlat,axis=0)/dlev

    ## 分量的u/v组分
    xuterm=dzdlon*dzdlon-PSIa*ddzdlonlon
    xvterm=dzdlon*dzdlat-PSIa*ddzdlonlat

    yuterm=xvterm
    yvterm=dzdlat*dzdlat-PSIa*ddzdlatlat

    if not data_shape[0]==1:
        zuterm=dzdlon*dzdlev-PSIa*ddzdlonlev
        zvterm=dzdlat*dzdlev-PSIa*ddzdlatlev

    ## 分量
    coef=pp*coslat/1000.0/2.0/UVc
    Fx=coef*(xuterm*Uc/(re*coslat)**2+xvterm*Vc/(re**2*coslat))
    Fy=coef*(yuterm*Uc/(re**2*coslat)+yvterm*Vc/re**2)
    if not data_shape[0]==1:
        Fz=coef*(f**2/N2*(zuterm*Uc/(re*coslat)+zvterm*Vc/re))

    ## 转为dataarray
    Fx=xr.DataArray(
        Fx,
        dims  =('level','lat','lon'),
        coords=data_coords
    )
    Fy=xr.DataArray(
        Fy,
        dims  =('level','lat','lon'),
        coords=data_coords
    )
    if not data_shape[0]==1:
        Fz=xr.DataArray(
            Fz,
            dims  =('level','lat','lon'),
            coords=data_coords
        )


    # 平滑处理
    if filt:
        Fx0 = Fx.copy()
        Fy0 = Fy.copy()
        if not data_shape[0] == 1:
            Fz0 = Fz.copy()
        ## 裁取非nan数据
        index = []
        I = 0
        for i in np.where(np.isnan(Fx).any(axis=2))[1][1:]:
            if i - I > 1:
                index.append(I+1)
                index.append(i)
            I = i
        I = 0
        index_ = []
        for i in np.where(np.isnan(Fy).any(axis=2))[1][1:]:
            if i - I > 1:
                index_.append(I+1)
                index_.append(i)
            I = i
        if len(index)%4 != 0 or len(index_)%4 != 0:
            raise ValueError('经纬度裁剪异常!')
        ## 合并多层index
        times = len(index) // 4
        indexAll = np.array(index).reshape(times, 4).copy()
        index_All = np.array(index_).reshape(times, 4).copy()
        index = [0, 0, 0, 0]
        index_ = [0, 0, 0, 0]
        index[0] = indexAll[:, 0].max()
        index[1] = indexAll[:, 1].min()
        index[2] = indexAll[:, 2].max()
        index[3] = indexAll[:, 3].min()
        index_[0] = index_All[:, 0].max()
        index_[1] = index_All[:, 1].min()
        index_[2] = index_All[:, 2].max()
        index_[3] = index_All[:, 3].min()

        index[0] = index_[0] if index_[0] > index[0] else index[0]
        index[1] = index_[1] if index_[1] < index[1] else index[1]
        index[2] = index_[2] if index_[2] > index[2] else index[2]
        index[3] = index_[3] if index_[3] < index[3] else index[3]

        ## 北半球平滑
        Fxn = Fx[:, index[0]:index[1], :]
        Fxn = filters.gaussian_filter(Fxn, filt, mode='wrap')
        Fyn = Fy[:, index[0]:index[1], :]
        Fyn = filters.gaussian_filter(Fyn, filt, mode='wrap')
        if not data_shape[0]==1:
            Fzn = Fz[:, index[0]:index[1], :]
            Fzn = filters.gaussian_filter(Fzn, filt, mode='wrap')
        ## 南半球平滑
        Fxs = Fx[:, index[2]:index[3], :]
        Fxs = filters.gaussian_filter(Fxs, filt, mode='wrap')
        Fys = Fy[:, index[2]:index[3], :]
        Fys = filters.gaussian_filter(Fys, filt, mode='wrap')
        if not data_shape[0]==1:
            Fzs = Fz[:, index[2]:index[3], :]
            Fzs = filters.gaussian_filter(Fzs, filt, mode='wrap')
        ## 合并
        Fn_nan1 = np.zeros_like(Fx[:, :index[0], :])
        Fn_nan1.fill(np.nan)
        Fn_nan2 = np.zeros_like(Fx[:, index[1]:index[2], :])
        Fn_nan2.fill(np.nan)
        Fn_nan3 = np.zeros_like(Fx[:, index[3]:, :])
        Fn_nan3.fill(np.nan)
        Fx = np.concatenate([Fn_nan1, Fxn, Fn_nan2, Fxs, Fn_nan3], axis=1)
        Fy = np.concatenate([Fn_nan1, Fyn, Fn_nan2, Fys, Fn_nan3], axis=1)
        if not data_shape[0]==1:
            Fz = np.concatenate([Fn_nan1, Fzn, Fn_nan2, Fzs, Fn_nan3], axis=1)

        if filtmode == 'mix':
            # 将平滑结果后的nan值位置加上原数据的同位置值
            Fx = np.where(np.isnan(Fx), Fx0, Fx)
            Fy = np.where(np.isnan(Fy), Fy0, Fy)
            if not data_shape[0]==1:
                Fz = np.where(np.isnan(Fz), Fz0, Fz)

    ### 返回结果
    if return_streamf:
        if not data_shape[0]==1:
            return Fx, Fy, Fz, PSI_global
        else:
            return Fx[0], Fy[0], PSI_global[0]
    else:
        if not data_shape[0]==1:
            return Fx, Fy, Fz
        else:
            return Fx[0], Fy[0]