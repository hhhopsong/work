import numpy as np
import xarray as xr


def transform(data, lon_name='lon', type='180->360'):
    """
    将经纬度从180->360或360->180转换
    Parameters
    ----------
    data : xarray.DataArray
        数据
    lon_name : str
        经度名称
    type : str
        转换类型，180->360或360->180
    Returns
    -------
    data : xarray.DataArray
        转换后的数据
    """
    if type == '180->360':
        data.coords[lon_name] = np.mod(data[lon_name], 360.)
        return data.reindex({lon_name: np.sort(data[lon_name])})
    elif type == '360->180':
        data.coords[lon_name] = data[lon_name].where(data[lon_name] <= 180, data[lon_name] - 360)
        return data.reindex({lon_name: np.sort(data[lon_name])})
    else:
        raise ValueError('type must be 180->360 or 360->180')