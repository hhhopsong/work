from significance_test import corr_test
import numpy as np
import xarray as xr

def lead_lag_sst_lon(dataset_x):
    '''
    计算超前滞后相关系数
    :param dataset:
    :return:
    '''
    # 数据读取
    sst = xr.open_dataset(dataset_x)  # NetCDF-4文件路径不可含中文
    # 截取sst数据为5N-5S，40E-80W
    time_data = sst['time']
    lon_sst = sst['lon']
    sst_term = sst.sel(time=slice(f'{time_data[0]}', f'{time_data[1]}'))
    sst_lastyear = sst.sel(time=slice(f'{time_data[0] - 1}', f'{time_data[1] - 1}'))
    sst_nextyear = sst.sel(time=slice(f'{time_data[0] + 1}', f'{time_data[1] + 1}'))
    # 计算sst经向平均值
    sst_term_lonavg = sst_term.mean(dim='lat')
    sst_lastyear_lonavg = sst_lastyear.mean(dim='lat')
    sst_nextyear_lonavg = sst_nextyear.mean(dim='lat')
    PC = np.load(r"D:\PyFile\paper1\OLS35.npy") # 读取时间序列
    # 计算sst距平
    sst_term_anom = sst_term_lonavg - sst_term_lonavg.mean(dim='time')
    sst_lastyear_anom = sst_lastyear_lonavg - sst_lastyear_lonavg.mean(dim='time')
    sst_nextyear_anom = sst_nextyear_lonavg - sst_nextyear_lonavg.mean(dim='time')
    # 计算sst距平与EOF的超前滞后相关系数，滞后范围为5
    num_lead_lag_corr = 18
    lead_lag_corr = np.zeros((num_lead_lag_corr, len(lon_sst)))
    lead_lag_corr.fill(np.nan)
    sst_leadlag = np.zeros(((time_data[1] - time_data[0] + 1) * 18, len(lon_sst)))
    for i in range(time_data[1] - time_data[0] + 1):
        sst_leadlag[i*18:i*18+18, :] = np.append(sst_lastyear_anom[i*12+9:i*12+12, :], np.append(sst_term_anom[i*12:i*12+12, :], sst_nextyear_anom[i*12:i*12+3, :], axis=0), axis=0)
    for i in range(num_lead_lag_corr):
        for j in range(len(lon_sst)):
            lead_lag_corr[i, j] = np.corrcoef(sst_leadlag[i::num_lead_lag_corr, j], PC)[0, 1]
    lead_lag_corr2 = np.zeros((num_lead_lag_corr, len(lon_sst)))
    lead_lag_corr2.fill(np.nan)
    for i in range(num_lead_lag_corr):
        for j in range(len(lon_sst)):
            lead_lag_corr2[i, j] = np.corrcoef(sst_leadlag[i::num_lead_lag_corr, j], PC)[0, 1]
    # 进行显著性检验
    p_lead_lag_corr = corr_test(PC, lead_lag_corr, alpha=0.1)


if __name__ == '__main__':
    dataset_x = r"E:\data\NOAA\ERSSTv5\sst.mnmean.nc"
    lead_lag_sst_lon(dataset_x)