import numpy as np


def W2G(GDATA, WDATA, HGRAD, HFUNC, KMAXD):
    from f2py.usphe import SPW2G, SPG2W
    from f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI


    # 初始化内部工作变量
    ZDATA = np.zeros((IDIM * JDIM, KMAXD))
    WORK = np.zeros((IDIM * JDIM, KMAXD))
    QSINLA = np.zeros(JDIM)
    QGW = np.zeros(JDIM)
    QPNM = np.zeros((NMAX + 2, MMAX + 1))
    QDPNM = np.zeros((NMAX + 2, MMAX + 1))

    # 初始化内部保存变量
    PNM = np.zeros(JMXHF * NMDIM)
    DPNM = np.zeros(JMXHF * NMDIM)
    TRIGS = np.zeros(IDIM * 2)
    IFAX = np.zeros(10, dtype=np.int32)
    NMO = np.zeros((2, MMAX + 1, LMAX + 1), dtype=np.int32)
    GWX = np.zeros(JDIM)
    GWDEL = np.zeros(JDIM)

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
    from f2py.usphe import SPW2G, SPG2W
    from f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI


    # 初始化内部保存变量
    PNM = np.zeros(JMXHF * NMDIM)
    DPNM = np.zeros(JMXHF * NMDIM)
    TRIGS = np.zeros(IDIM * 2)
    IFAX = np.zeros(10, dtype=np.int32)
    NMO = np.zeros((2, MMAX + 1, LMAX + 1), dtype=np.int32)
    GWX = np.zeros(JDIM)
    GWDEL = np.zeros(JDIM)
    ZDATA = np.zeros((IDIM * JDIM, KMAX), dtype=np.float64)
    WORK = np.zeros((IDIM * JDIM, KMAX), dtype=np.float64)
    OSET = False

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
