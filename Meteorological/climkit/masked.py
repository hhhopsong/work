import geopandas as gpd
import salem


# 裁切范围数据，其余数据Nan化
def masked(data, shp, crs='wgs84'):
    """
    掩膜函数
    :param data: 原始数据
    :param shp: SHP矢量文件
    :param crs: 地理坐标系格式
    :return: 掩膜数据矩阵
    """
    split_shp = gpd.read_file(shp)
    split_shp.crs = crs
    return data.salem.roi(shape=split_shp)

