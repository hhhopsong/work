import numpy as np
import numba
from numba import jit
from LBM.f2py.usphe import SPW2G, SPG2W
from LBM.f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI
import torch


device = torch.device('cuda')


def copy(datai, idim):
    if isinstance(datai, torch.Tensor):
        device = datai.device
        datai = datai.cpu().numpy()
        result = np.array(datai[:idim])
        return torch.tensor(result, dtype=datai.dtype, device=device)
    else:
        return np.array(datai[:idim])


def W2G(GDATA, WDATA, HGRAD, HFUNC, KMAXD):
    # 初始化内部工作变量
    ZDATA = torch.zeros((IDIM * JDIM, KMAXD), device=device)
    WORK = torch.zeros((IDIM * JDIM, KMAXD), device=device)
    QSINLA = torch.zeros(JDIM, device=device)
    QGW = torch.zeros(JDIM, device=device)
    QPNM = torch.zeros((NMAX + 2, MMAX + 1), device=device)
    QDPNM = torch.zeros((NMAX + 2, MMAX + 1), device=device)

    # 初始化内部保存变量
    PNM = torch.zeros(JMXHF * NMDIM, device=device)
    DPNM = torch.zeros(JMXHF * NMDIM, device=device)
    TRIGS = torch.zeros(IDIM * 2, device=device)
    IFAX = torch.zeros(10, dtype=torch.int32, device=device)
    NMO = torch.zeros((2, MMAX + 1, LMAX + 1), dtype=torch.int32, device=device)
    GWX = torch.zeros(JDIM, device=device)
    GWDEL = torch.zeros(JDIM, device=device)

    OSET = False
    OFIRST = True

    if OFIRST:
        print(' @@@ DSPHE: SPHERICAL TRANSFORM INTFC. 93/12/07')
        OFIRST = False

    if not OSET:
        print(' ### W2G: SPSTUP MUST BE CALLED BEFORE')
        return

    if HGRAD[0] == 'Y':
        GDATA = SPW2G(GDATA, WDATA, DPNM, NMO, TRIGS, IFAX, HGRAD, HFUNC, IMAX, JMAX, KMAXD, IDIM, JDIM, LMAX, MMAX,
                      NMAX, MINT, NMDIM, JMXHF, ZDATA, WORK)
    else:
        GDATA = SPW2G(GDATA, WDATA, PNM, NMO, TRIGS, IFAX, HGRAD, HFUNC, IMAX, JMAX, KMAXD, IDIM, JDIM, LMAX, MMAX,
                      NMAX, MINT, NMDIM, JMXHF, ZDATA, WORK)
    return GDATA


def G2W(WDATA, GDATA, HGRAD, HFUNC, KMAXD):
    # 初始化内部保存变量
    PNM = torch.zeros((NMDIM, JMXHF), device=device)
    DPNM = torch.zeros((NMDIM, JMXHF), device=device)
    TRIGS = torch.zeros(IDIM * 2, device=device)
    IFAX = torch.zeros(10, dtype=torch.int32, device=device)
    NMO = torch.zeros((2, MMAX + 1, LMAX + 1), dtype=torch.int32, device=device)
    GWX = torch.zeros(JDIM, device=device)
    GWDEL = torch.zeros(JDIM, device=device)
    ZDATA = torch.zeros((IDIM * JDIM, KMAX), device=device)
    WORK = torch.zeros((IDIM * JDIM, KMAX), device=device)
    OSET = True  # 逻辑变量，用于判断是否已经使用球谐函数

    if not OSET:
        print(' ### G2W: SPSTUP MUST BE CALLED BEFORE')
        return

    if HGRAD[0] == 'Y':
        WDATA = SPG2W(WDATA, GDATA, DPNM, NMO, TRIGS, IFAX, GWDEL, HGRAD, HFUNC, IMAX, JMAX, KMAXD, IDIM, JDIM, LMAX,
                      MMAX, NMAX, MINT, NMDIM, JMXHF, ZDATA, WORK)
    elif HGRAD[0] == 'X':
        WDATA = SPG2W(WDATA, GDATA, PNM, NMO, TRIGS, IFAX, GWDEL, HGRAD, HFUNC, IMAX, JMAX, KMAXD, IDIM, JDIM, LMAX,
                      MMAX, NMAX, MINT, NMDIM, JMXHF, ZDATA, WORK)
    else:
        WDATA = SPG2W(WDATA, GDATA, PNM, NMO, TRIGS, IFAX, GWX, HGRAD, HFUNC, IMAX, JMAX, KMAXD, IDIM, JDIM, LMAX,
                      MMAX, NMAX, MINT, NMDIM, JMXHF, ZDATA, WORK)
    return WDATA
