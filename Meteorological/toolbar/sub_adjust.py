import matplotlib.transforms as mtransforms

def adjust_sub_axes(ax_main, ax_sub, shrink, lr=1.0, ud=1.0, width=1.0, height=1.0):
    '''
    将ax_sub调整到ax_main的右下角.shrink指定缩小倍数。
    当ax_sub是GeoAxes时,需要在其设定好范围后再使用此函数
    :param ax_main: 主图
    :param ax_sub: 子图
    :param shrink: 缩小倍数
    :param lr: 左右间距(>1：左偏, <1：右偏)
    :param ud: 上下间距(>1：上偏, <1：下偏)
    :param width: 宽度缩放比例
    :param height: 高度比例
    '''
    bbox_main = ax_main.get_position()
    bbox_sub = ax_sub.get_position()
    wratio = bbox_main.width / bbox_sub.width
    hratio = bbox_main.height / bbox_sub.height
    wnew = bbox_sub.width * shrink * width
    hnew = bbox_sub.height * shrink * height
    bbox_new = mtransforms.Bbox.from_extents(
        bbox_main.x1 - lr*wnew, bbox_main.y0 + (ud-1)*hnew,
        bbox_main.x1 - (lr-1)*wnew, bbox_main.y0 + ud*hnew
    )
    ax_sub.set_position(bbox_new)