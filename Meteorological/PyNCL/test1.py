import numpy as np





def _is_list_or_tuple(arg):
    if ( ((type(arg) == list) or (type(arg) == tuple)) ):
        return True
    else:
        return False

def _is_numpy_array(arg):
    if isinstance(arg ,np.ndarray):
        return True
    else:
        return False

#
# 测试我们是否可以加载掩码数组。
#
try:
  from numpy import ma
  HAS_MA = True
except:
  HAS_MA = False

def _is_numpy_ma(arg):
    if HAS_MA and ma.isMaskedArray(arg):
        return True
    else:
        return False



def _get_integer_version(strversion):
# 将语义化版本号转换为数值型版本标识
# 示例："1.10.4" → 1*10000 + 10*100 + 4 = 11004
    d = strversion.split('.')
    if len(d) > 2:
       v = int(d[0]) * 10000 + int(d[1]) * 100 + int(d[2][0])
    elif len(d) is 2:
       v = int(d[0]) * 10000 + int(d[1]) * 100
    else:
       v = int(d[0]) * 10000
    return v
IS_NEW_MA = _get_integer_version(np.__version__) > 10004


def _get_arr_and_fv(arr):
# 该函数返回一个NumPy数组及其填充值（如果输入是掩码数组）
# 否则直接返回原数组，并用'None'表示无填充值
#
# 后续计划添加对NioVariables类型的识别，届时可以检查"_FillValue"属性并使用该属性值
  if _is_numpy_ma(arr):
    if IS_NEW_MA:
      return arr.filled(),arr.fill_value
    else:
      return arr.filled(),arr.fill_value()
  else:
    return arr,None


def vector(wks, uarray, varray, rlistc=None):
    """
    创建并在地图上绘制矢量图，返回生成的图形标识符

    plot = Ngl.vector_map(wks, u, v, res=None)

    参数说明：
    wks -- 由Ngl.open_wks调用返回的图形工作站标识符
    u,v -- 矢量数据（允许使用掩码数组）
    res -- 包含PyNGL资源的Resources类实例（可选参数）
    """
    #
    # 确保输入数组为二维
    #
    if len(uarray.shape) != 2 or len(varray.shape) != 2:
        print("vector - arrays must be 2D")
        return None

    # Get NumPy array from masked arrays, if necessary.
    uar2, uar_fill_value = _get_arr_and_fv(uarray)
    var2, var_fill_value = _get_arr_and_fv(varray)

    _set_spc_defaults(1)
    rlist = _crt_dict(rlistc)

    # 将资源字典分离为：
    # 1. 适用于矢量场(VectorField)的资源
    # 2. 适用于矢量图(VectorPlot)的资源
    #
    rlist1 = {}
    rlist2 = {}
    rlist3 = {}
    for key in list(rlist.keys()):
        rlist[key] = _convert_from_ma(rlist[key])
        if (key[0:2] == "vf"):
            rlist1[key] = rlist[key]
        elif (key[0:3] == "ngl"):
            _set_spc_res(key[3:], rlist[key])
        else:
            rlist2[key] = rlist[key]
            #
            # 当图形可能叠加到不规则绘图类（进行线性化或对数化处理）时，
            # 需要记录所有刻度线资源，以便后续重新应用到IrregularPlot类
            #
            if (key[0:2] == "vp" or key[0:2] == "tm" or key[0:6] == "pmTick"):
                rlist3[key] = rlist[key]

    # 如有必要，设置缺失值资源
    _set_msg_val_res(rlist1, uar_fill_value, "vector_u")
    _set_msg_val_res(rlist1, var_fill_value, "vector_v")

    _set_vector_res(rlist, rlist2)  # 设置附加矢量图资源
    _set_labelbar_res(rlist, rlist2, True)  # 设置附加色标资源
    _set_tickmark_res(rlist, rlist3)  # 设置附加刻度线资源

    #
    #  调用 wrapped 函数并返回。
    #
    ivct = vector_wrap(wks, uar2, var2, "double", "double", \
                       uar2.shape[0], uar2.shape[1], 0, \
                       pvoid(), "", 0, pvoid(), "", 0, 0, pvoid(), pvoid(), \
                       rlist1, rlist2, rlist3, pvoid())
    del rlist
    del rlist1
    del rlist2
    del rlist3
    return _lst2pobj(ivct)


################################################################

def vector_map(wks, uarray, varray, rlistc=None):
    """
  Creates and draws vectors over a map, and returns a PlotId of the plot
  created.

  plot = Ngl.vector_map(wks, u, v, res=None)

  wks -- The identifier returned from calling Ngl.open_wks

  u,v -- The vector data. Masked arrays allowed.

  res -- An optional instance of the Resources class having PyNGL
         resources as attributes.
    """
    #
    #  Make sure the arrays are 2D.
    #
    if len(uarray.shape) != 2 or len(varray.shape) != 2:
        print("vector_map - arrays must be 2D")
        return None

    # Get NumPy array from masked arrays, if necessary.
    uar2, uar_fill_value = _get_arr_and_fv(uarray)
    var2, var_fill_value = _get_arr_and_fv(varray)

    _set_spc_defaults(1)
    rlist = _crt_dict(rlistc)

    #  Separate the resource dictionary into those resources
    #  that apply to VectorField, MapPlot, and VectorPlot.
    #
    rlist1 = {}
    rlist2 = {}
    rlist3 = {}
    for key in list(rlist.keys()):
        rlist[key] = _convert_from_ma(rlist[key])
        if (key[0:2] == "vf"):
            rlist1[key] = rlist[key]
        elif ((key[0:2] == "mp") or (key[0:2] == "vp") or (key[0:3] == "pmA") or \
              (key[0:3] == "pmO") or (key[0:3] == "pmT") or (key[0:2] == "tm") or \
              (key[0:2] == "ti")):
            rlist3[key] = rlist[key]
        elif (key[0:3] == "ngl"):
            _set_spc_res(key[3:], rlist[key])
        else:
            rlist2[key] = rlist[key]

    # Set missing value resources, if necessary
    _set_msg_val_res(rlist1, uar_fill_value, "vector_u")
    _set_msg_val_res(rlist1, var_fill_value, "vector_v")

    _set_map_res(rlist, rlist3)  # Set some addtl map resources
    _set_vector_res(rlist, rlist2)  # Set some addtl vector resources
    _set_labelbar_res(rlist, rlist2, True)  # Set some addtl labelbar resources

    #
    # Test for masking a lambert conformal plot.
    #
    mask_list = _test_for_mask_lc(rlist, rlist3)

    #
    #  Call the wrapped function and return.
    #
    ivct = vector_map_wrap(wks, uar2, var2, "double", "double", \
                           uar2.shape[0], uar2.shape[1], 0, \
                           pvoid(), "", 0, pvoid(), "", 0, 0, pvoid(), pvoid(), \
                           rlist1, rlist2, rlist3, pvoid())

    livct = _lst2pobj(ivct)

    if mask_list["MaskLC"]:
        livct = _mask_lambert_conformal(wks, livct, mask_list, rlist3)

    del rlist
    del rlist1
    del rlist2
    del rlist3
    return (livct)

################################################################
