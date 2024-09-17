from cartopy.util import add_cyclic_point

from LBM.force_file import horizontal_profile as hp
from LBM.force_file import vertical_profile as vp
from LBM.force_file import mk_grads, mk_wave, interp3d_lbm
import xarray as xr
import numpy as np



frc_loc = 'frc'

v = vp(kvpr=2, vamp=8., vdil=20., vcnt=0.45)  # 生成强迫场的理想化垂直结构
h = hp(khpr=1, hamp=0.25, xdil=23., ydil=6.5, xcnt=77., ycnt=-1.5)  # 生成强迫场的理想化水平结构
frc = mk_grads(hor_structure=h, url=frc_loc, ver_structure=v, ovor=0, odiv=0, otmp=1, ops=0, osh=0)  # 生成强迫场
frc_nc = interp3d_lbm(frc)
frc_nc.to_netcdf(r'D:\CODES\Python\Meteorological\frc_nc\frc.t42l20.Tingyang.nc')

