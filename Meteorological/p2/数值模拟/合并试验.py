import xarray as xr

frc1 = xr.open_dataset(r'C:\Users\86136\Desktop\frc.t42l20.nc')
frc2 = xr.open_dataset(r'D:\lbm\main\data\Forcing\frc.t42l20.nc')

result = frc1 + frc2
result.to_netcdf(r'D:\lbm\main\data\Forcing\frc.t42l20.nc', format='NETCDF3_CLASSIC')

frc1_p = xr.open_dataset(r'C:\Users\86136\Desktop\frc_p.t42l20.nc')
frc2_p = xr.open_dataset(r'D:\lbm\main\data\Forcing\frc_p.t42l20.nc')

result_p = frc1_p + frc2_p
result_p.to_netcdf(r'D:\lbm\main\data\Forcing\frc_p.t42l20.nc', format='NETCDF3_CLASSIC')