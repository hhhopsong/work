from cnmaps import get_adm_maps

name = "Yangtze-River middle and lower"
####请更改name为你想要的地图名称，如"china"、"world"、"beijing"等
china = (
    get_adm_maps(province='湖北省', record= "first", only_polygon=True, wgs84=True)
    + get_adm_maps(province='湖南省', record= "first", only_polygon=True, wgs84=True)
    + get_adm_maps(province='江西省', record= "first", only_polygon=True, wgs84=True)
    + get_adm_maps(province='安徽省', record= "first", only_polygon=True, wgs84=True)
    + get_adm_maps(province='江苏省', record= "first", only_polygon=True, wgs84=True)
    + get_adm_maps(province='上海市', record= "first", only_polygon=True, wgs84=True)
    + get_adm_maps(province='浙江省', record= "first", only_polygon=True, wgs84=True)
        )
china.to_file('./shp/' + name + '/' + name + '.geojson')  # 默认为 geojson 格式文件
china.to_file('./shp/' + name + '/' + name + '.shp', engine='ESRI Shapefile')  # 也可以指定 shapefile 格式文件
