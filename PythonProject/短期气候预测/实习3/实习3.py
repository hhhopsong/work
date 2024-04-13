import pprint
import numpy as np
import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import xarray as xr
from xgrads import open_CtlDataset


with open('LonLat.txt', 'r+', encoding='utf-8') as data:
    data0 = data.readlines()
for i in range(len(data0)):
    data0[i] = [i.strip() for i in data0[i].split('\t')[:4]] + ['0'] + ['1']
    data0[i][2], data0[i][3], data0[i][4] = data0[i][2].split()[0], data0[i][2].split()[1], data0[i][3]
    try:
        data0[i].pop(5)
    except:
        pass
    try:
        data0[i].pop(6)
    except:
        pass
loc = data0

