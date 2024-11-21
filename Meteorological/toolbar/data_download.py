import cdsapi


def download(year):
    dataset = "derived-era5-pressure-levels-daily-statistics"
    request = {
        "product_type": "reanalysis",
        "variable": [
            "geopotential",
            "u_component_of_wind",
            "v_component_of_wind"
        ],
        "year": [str(year)],
        "month": ["06", "07", "08"],
        "day": [
            "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
            "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
            "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
        ],
        "pressure_level": ["200"],
        "daily_statistic": "daily_mean",
        "time_zone": "utc+00:00",
        "frequency": "6_hourly"
    }
    target = f'E:\data\ERA5\ERA5_pressLev\daily\era5_uvz_daily_{year}.nc'
    client = cdsapi.Client()
    client.retrieve(dataset, request, target)

downing = 0
for i in range(1961, 2025):
    if downing % 12 != 0:
        download(i)
        downing += 1
        print(f'{i}年数据下载中')
    else:
        pass