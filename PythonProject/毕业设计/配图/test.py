import xarray as xr

prec = xr.open_dataset(r'C:\Users\10574\OneDrive\File\Graduation Thesis\ThesisData\PREC\precip.mon.anom.nc')
pre = prec['precip'].sel(time=slice('1979-01-01', '2014-12-31'))
pre = pre.sel(time=pre.time.dt.month.isin([7, 8]))
pass