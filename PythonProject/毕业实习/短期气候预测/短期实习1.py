import struct

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from xgrads import open_CtlDataset
from cdo import Cdo

import pprint as pp


def sc(Xi, Xj):
    Y = np.zeros((len(Xj), 2))
    for i in range(len(Xj)):
        Y[i] = [i, np.corrcoef(Xi, Xj[i])[0, 1]]
    return sorted(Y.tolist(), key=lambda x: x[1], reverse=True)[0]


def sd(Xi, Xj):
    Y = np.zeros((len(Xj), 2))
    for i in range(len(Xj)):
        Y[i] = [i, np.sqrt(np.sum((np.array(Xi) - np.array(Xj[i])) ** 2))]
    return sorted(Y.tolist(), key=lambda x: x[1], reverse=False)[0]


def sde(Xi, Xj):
    Xijk = np.zeros((len(Xj), len(Xi)))
    Y = np.zeros((len(Xj), 2))
    for i in range(len(Xj)):
        Xijk[i, :] = np.array(Xi) - np.array(Xj[i])
        Fij = np.sum(Xijk) / Xijk.shape[0]
        Eij = np.sum(np.abs(Xijk)) / Xijk.shape[0]
        Sij = np.sum(np.abs(Xijk - Fij)) / Xijk.shape[0]
        Y[i] = [i, (Sij - Eij) / 2]
    return sorted(Y, key=lambda x: x[1], reverse=False)[0]


def predict(X_hgt, X_sst, hgt1, hgt2, hgt3, sst1, sst2, hgt1_, hgt2_, hgt3_, sst1_, sst2_, hgt3560_150150_,
            hgt1030_150150_, hgt1030_100150_, hgt3060_100150_, hgt4060_70110_, sst4050_180140_, sst1010_175150_):
    pred_in = [[hgt1_['hgt'].values[i], hgt2_['hgt'].values[i], hgt3_['hgt'].values[i], sst1_['sst'].values[i],
                sst2_['sst'].values[i]] for i in range(len(hgt1_['hgt']))]
    pred = np.zeros((len(X_hgt['hgt']), 3, 2))
    for i in range(len(X_hgt['hgt'])):
        hgt3560_150150 = X_hgt.sel(lat=slice(35, 60)).sel(lon=slice(150, 210)).sel(time=X_hgt.time_data[i].values)
        hgt1030_150150 = X_hgt.sel(lat=slice(10, 30)).sel(lon=slice(150, 210)).sel(time=X_hgt.time_data[i].values)
        hgt1030_100150 = X_hgt.sel(lat=slice(10, 30)).sel(lon=slice(100, 150)).sel(time=X_hgt.time_data[i].values)
        hgt3060_100150 = X_hgt.sel(lat=slice(30, 60)).sel(lon=slice(100, 150)).sel(time=X_hgt.time_data[i].values)
        hgt4060_70110 = X_hgt.sel(lat=slice(40, 60)).sel(lon=slice(70, 110)).sel(time=X_hgt.time_data[i].values)
        sst4050_180140 = X_sst.sel(lat=slice(40, 50)).sel(lon=slice(180, 220)).sel(time=X_hgt.time_data[i].values)
        sst1010_175150 = X_sst.sel(lat=slice(-10, 10)).sel(lon=slice(175, 210)).sel(time=X_hgt.time_data[i].values)
        hgt1_p = (hgt3560_150150.mean("lat").mean("lon") - hgt3560_150150_.mean()) - (
                hgt1030_150150.mean('lat').mean('lon') - hgt1030_150150_.mean())
        hgt1_p = (hgt1_p - hgt1.mean()) / hgt1.std()
        hgt2_p = (hgt1030_100150.mean('lat').mean('lon') - hgt1030_100150_.mean()) - (
                hgt3060_100150.mean('lat').mean('lon') - hgt3060_100150_.mean())
        hgt2_p = (hgt2_p - hgt2.mean()) / hgt2.std()
        hgt3_p = (hgt1030_100150.mean('lat').mean('lon') - hgt1030_100150_.mean()) - (
                hgt4060_70110.mean('lat').mean('lon') - hgt4060_70110_.mean())
        hgt3_p = (hgt3_p - hgt3.mean()) / hgt3.std()
        sst1_p = sst4050_180140.mean('lat').mean('lon') - sst4050_180140_.mean()
        sst1_p = (sst1_p - sst1.mean()) / sst1.std()
        sst2_p = sst1010_175150.mean('lat').mean('lon') - sst1010_175150_.mean()
        sst2_p = (sst2_p - sst2.mean()) / sst2.std()
        year_in = [hgt1_p['hgt'].values, hgt2_p['hgt'].values, hgt3_p['hgt'].values, sst1_p['sst'].values,
                   sst2_p['sst'].values]
        pred[i, 0, :] = sc(year_in, pred_in)
        pred[i, 1, :] = sd(year_in, pred_in)
        pred[i, 2, :] = sde(year_in, pred_in)
    return pred


hgt = xr.open_dataset(r"C:\Users\10574\Desktop\data\win500h1952-2001.nc").sel(time=slice('1952-01-01', '1991-12-31'))
sst = xr.open_dataset(r"C:\Users\10574\Desktop\data\sst1952-2001.nc").sel(time=slice('1952-01-01', '1991-12-31'))
hgt3560_150150 = hgt.sel(lat=slice(35, 60)).sel(lon=slice(150, 210))
hgt1030_150150 = hgt.sel(lat=slice(10, 30)).sel(lon=slice(150, 210))
hgt1030_100150 = hgt.sel(lat=slice(10, 30)).sel(lon=slice(100, 150))
hgt3060_100150 = hgt.sel(lat=slice(30, 60)).sel(lon=slice(100, 150))
hgt4060_70110 = hgt.sel(lat=slice(40, 60)).sel(lon=slice(70, 110))
sst4050_180140 = sst.sel(lat=slice(40, 50)).sel(lon=slice(180, 220))
sst1010_175150 = sst.sel(lat=slice(-10, 10)).sel(lon=slice(175, 210))
hgt1 = (hgt3560_150150.mean("lat").mean("lon") - hgt3560_150150.mean()) - (
        hgt1030_150150.mean('lat').mean('lon') - hgt1030_150150.mean())
hgt1_ = (hgt1 - hgt1.mean()) / hgt1.std()
hgt2 = (hgt1030_100150.mean('lat').mean('lon') - hgt1030_100150.mean()) - (
        hgt3060_100150.mean('lat').mean('lon') - hgt3060_100150.mean())
hgt2_ = (hgt2 - hgt2.mean()) / hgt2.std()
hgt3 = (hgt1030_100150.mean('lat').mean('lon') - hgt1030_100150.mean()) - (
        hgt4060_70110.mean('lat').mean('lon') - hgt4060_70110.mean())
hgt3_ = (hgt3 - hgt3.mean()) / hgt3.std()
sst1 = sst4050_180140.mean('lat').mean('lon') - sst4050_180140.mean()
sst1_ = (sst1 - sst1.mean()) / sst1.std()
sst2 = sst1010_175150.mean('lat').mean('lon') - sst1010_175150.mean()
sst2_ = (sst2 - sst2.mean()) / sst2.std()
pred_hgt = xr.open_dataset(r"C:\Users\10574\Desktop\data\win500h1952-2001.nc").sel(
    time=slice('1992-01-01', '2001-12-31'))
pred_sst = xr.open_dataset(r"C:\Users\10574\Desktop\data\sst1952-2001.nc").sel(time=slice('1992-01-01', '2001-12-31'))
pred_result = predict(pred_hgt, pred_sst, hgt1, hgt2, hgt3, sst1, sst2, hgt1_, hgt2_, hgt3_, sst1_, sst2_,
                      hgt3560_150150, hgt1030_150150, hgt1030_100150, hgt3060_100150, hgt4060_70110, sst4050_180140,
                      sst1010_175150)
with open(r".\雨型.txt", encoding='gbk') as f:
    rain = f.readlines()
for i in range(len(rain)):
    rain[i] = rain[i].strip().split("\t")
rain_type = [[rain[i][0], rain[i].index("1")] for i in range(len(rain))]
print('\t\t\t1992\t1993\t1994\t1995\t1996\t1997\t1998\t1999\t2000\t2001')
print('观测雨型', end='\t\t')
for i in range(10):
    print(rain_type[i + 40][1], end='\t\t')
print('\n预测雨型')
print('\tSC', end='\t\t')
for i in range(10):
    print(rain_type[int(pred_result[i, 0, 0])][1], end='\t\t')
print('\n\tSD', end='\t\t')
for i in range(10):
    print(rain_type[int(pred_result[i, 1, 0])][1], end='\t\t')
print('\n\tSDE', end='\t\t')
for i in range(10):
    print(rain_type[int(pred_result[i, 2, 0])][1], end='\t\t')
