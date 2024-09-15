import numpy as np
#from ufftp import fft99x
#from copy import copy


# 全局常量
KMAXD = 100
NMDIMD = 12000
OLSET = False
MLIST = np.zeros(NMDIMD, dtype=np.int32)
JLIST = np.zeros(NMDIMD, dtype=np.int32)


def SPW2G(GDATA, WDATA, PNM, NMO, TRIGS, IFAX, HGRAD, HFUNC, IMAX, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, ZDATA, WORK):
    from f2py.ufftp import fft99x
    from f2py.copy import copy
    from f2py.usphe import SPW2G, SPG2W
    from f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI

    if IMAX == 1 or JMAX == 1:
        print(' ### SPW2G: THIS ROUTINE IS FOR 3 DIM.')
        return

    # 处理LOFFS和LDPNM标志
    LOFFS = HFUNC[-1] == 'O'
    if LOFFS:
        if KMAXD < KMAX:
            print(' ### SPW2G: WORK AREA(KMAXD) TOO SMALL < ', KMAX)
            return
    LDPNM = HGRAD[0] == 'Y'

    DOFFS = np.zeros(KMAXD)
    if LOFFS:
        for K in range(KMAX):
            DOFFS[K] = WDATA[NMO[1, 0, 0], K]
            WDATA[NMO[1, 0, 0], K] = 0.

    ZDATA = SPW2Z(ZDATA, WDATA, PNM, NMO, LDPNM, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, WORK)
    if LOFFS:
        for K in range(KMAX):
            WDATA[NMO[1, 0, 0], K] = DOFFS[K]

    if HGRAD[0] == 'X':
        ZDATA = GRADX(ZDATA, IDIM, JDIM, KMAX, MMAX, MINT, WORK)
        LOFFS = False

    new_work, new_zdata = fft99x(WORK, ZDATA, TRIGS, IFAX, 1, IDIM, IMAX, JDIM * KMAX, 1)
    WORK = new_work
    ZDATA = new_zdata

    for K in range(KMAX):
        WORKZ = WORK[0, K]
        for IJ in range(IDIM * JMAX, IDIM * JDIM):
            WORK[IJ, K] = WORKZ
        for I in range(IMAX, IDIM):
            for IJ in range(0, IDIM * JDIM, IDIM):
                WORK[IJ + I, K] = WORKZ

    if LOFFS:
        for K in range(KMAX):
            for IJ in range(IDIM * JDIM):
                WORK[IJ, K] = WORK[IJ, K] + DOFFS[K]

    if HFUNC[0] == 'A':
        for K in range(KMAX):
            for IJ in range(IDIM * JDIM):
                GDATA[IJ, K] = GDATA[IJ, K] + WORK[IJ, K]
    elif HFUNC[0] == 'S':
        for K in range(KMAX):
            for IJ in range(IDIM * JDIM):
                GDATA[IJ, K] = GDATA[IJ, K] - WORK[IJ, K]
    elif HFUNC[0] == 'N':
        for K in range(KMAX):
            for IJ in range(IDIM * JDIM):
                GDATA[IJ, K] = -WORK[IJ, K]
    else:
        for K in range(KMAX):
            for IJ in range(IDIM * JDIM):
                GDATA[IJ, K] = WORK[IJ, K]
    return GDATA


def SPG2W(WDATA, GDATA, PNM, NMO, TRIGS, IFAX, GW, HGRAD, HFUNC, IMAX, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, ZDATA, WORK):
    from f2py.ufftp import fft99x
    from f2py.copy import copy
    from f2py.usphe import SPW2G, SPG2W
    from f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI

    if IMAX == 1 or JMAX == 1:
        print(' ### SPG2W: THIS ROUTINE IS FOR 3 DIM.')
        return

    LOFFS = HFUNC[-1] == 'O'
    if LOFFS:
        if KMAXD < KMAX:
            print(' ### SPG2W: WORK AREA(KMAXD) TOO SMALL < ', KMAX)
            return
    LDPNM = HGRAD[0] == 'Y'

    DOFFS = np.zeros(KMAXD)
    if LOFFS:
        for K in range(KMAX):
            DOFFS[K] = GDATA[0, K]
            for IJ in range(IDIM * JDIM):
                WORK[IJ, K] = GDATA[IJ, K] - DOFFS[K]
    else:
        WORK = copy(GDATA, IDIM * JDIM * KMAX)

    new_work, new_zdata = fft99x(WORK, ZDATA, TRIGS, IFAX, 1, IDIM, IMAX, JDIM * KMAX, 0)
    WORK = new_work
    ZDATA = new_zdata

    if HGRAD[0] == 'X':
        ZDATA = GRADX(ZDATA, IDIM, JDIM, KMAX, MMAX, MINT, WORK)
        LOFFS = False

    WDATA = SPZ2W(WDATA, ZDATA, PNM, NMO, GW, LDPNM, HFUNC, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, WORK)

    if LOFFS:
        if HFUNC[0] == 'N' or HFUNC[0] == 'S':
            for K in range(KMAX):
                WDATA[NMO[1, 0, 0], K] = WDATA[NMO[1, 0, 0], K] - DOFFS[K]
        else:
            for K in range(KMAX):
                WDATA[NMO[1, 0, 0], K] = WDATA[NMO[1, 0, 0], K] + DOFFS[K]

    for K in range(KMAX):
        WDATA[NMO[2, 0, 0], K] = 0.0

    return WDATA


def SPW2Z(ZDATA, WDATA, PNM, NMO, LDPNM, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, ZDW):
    from f2py.usphe import SPW2G, SPG2W
    from f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI
    global OLSET, MLIST, JLIST
    if not OLSET:
        OLSET = True
        if NMDIMD < NMDIM:
            print(' ### SPW2Z: WORK AREA(NMDIMD) TOO SMALL < ', NMDIM)
            return

        index = 0
        for M in range(0, MMAX, MINT):
            LEND = min(LMAX, NMAX - M)
            MM = M // MINT
            for L in range(0, LEND):
                MLIST[index] = 2 * MM + 1
                MLIST[index + 1] = 2 * MM + 2
                if L % 2 == 0:
                    JLIST[index] = 0
                    JLIST[index + 1] = 0
                else:
                    JLIST[index] = (JMAX + 1) // 2
                    JLIST[index + 1] = (JMAX + 1) // 2
                index += 2

    ZDW = np.zeros((IDIM, JDIM, KMAX))
    ZDATA = np.zeros((IDIM, JDIM, KMAX))

    if KMAX < (JMAX + 1) // 2:
        for NM in range(NMDIM):
            IM = MLIST[NM]
            for K in range(KMAX):
                for J in range((JMAX + 1) // 2):
                    JP = JLIST[NM] + J
                    ZDW[IM, JP, K] = ZDW[IM, JP, K] + PNM[NM, J] * WDATA[NM, K]
    else:
        for NM in range(NMDIM):
            IM = MLIST[NM]
            for J in range((JMAX + 1) // 2):
                JP = JLIST[NM] + J
                for K in range(KMAX):
                    ZDW[IM, JP, K] = ZDW[IM, JP, K] + PNM[NM, J] * WDATA[NM, K]

    for J in range(1, (JMAX + 1) // 2 + 1):
        JN = J
        JS = JMAX + 1 - J
        if not LDPNM:
            JE = J
            JO = (JMAX + 1) // 2 + J
        else:
            JE = (JMAX + 1) // 2 + J
            JO = JN
        for K in range(KMAX):
            for IM in range(1, IDIM + 1):
                ZDATA[IM, JS, K] = ZDW[IM, JE, K] - ZDW[IM, JO, K]
                ZDATA[IM, JN, K] = ZDW[IM, JE, K] + ZDW[IM, JO, K]

    for K in range(1, KMAX + 1):
        for J in range(1, JDIM + 1):
            ZDATA[2, J, K] = 0.0

    return ZDATA


def SPZ2W(WDATA, ZDATA, PNM, NMO, GW, LDPNM, HFUNC, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, ZDW):
    from f2py.usphe import SPW2G, SPG2W
    from f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI
    global OLSET, MLIST, JLIST
    if not OLSET:
        OLSET = True
        if NMDIMD < NMDIM:
            print(' ### SPZ2W: WORK AREA(NMDIMD) TOO SMALL < ', NMDIM)
            return

        index = 0
        for M in range(0, MMAX, MINT):
            LEND = min(LMAX, NMAX - M)
            MM = M // MINT
            for L in range(0, LEND):
                MLIST[index] = 2 * MM + 1
                MLIST[index + 1] = 2 * MM + 2
                if L % 2 == 0:
                    JLIST[index] = 0
                    JLIST[index + 1] = 0
                else:
                    JLIST[index] = (JMAX + 1) // 2
                    JLIST[index + 1] = (JMAX + 1) // 2
                index += 2

    if HFUNC[0]!= 'A' and HFUNC[0]!= 'S':
        WDATA = np.zeros((NMDIM, KMAX))

    for J in range(1, (JMAX + 1) // 2 + 1):
        JN = J
        JS = JMAX + 1 - J
        if not LDPNM:
            JE = J
            JO = (JMAX + 1) // 2 + J
        else:
            JE = (JMAX + 1) // 2 + J
            JO = JN
        for K in range(1, KMAX + 1):
            for IM in range(1, IDIM + 1):
                ZDW[IM, JE, K] = GW[J] * (ZDATA[IM, JN, K] + ZDATA[IM, JS, K])
                ZDW[IM, JO, K] = GW[J] * (ZDATA[IM, JN, K] - ZDATA[IM, JS, K])

    if HFUNC[0] == 'N' or HFUNC[0] == 'S':
        if KMAX < NMDIM:
            for J in range(1, (JMAX + 1) // 2 + 1):
                for K in range(1, KMAX + 1):
                    for NM in range(1, NMDIM + 1):
                        IM = MLIST[NM]
                        JP = JLIST[NM] + J
                        WDATA[NM, K] = WDATA[NM, K] - PNM[NM, J] * ZDW[IM, JP, K]
        else:
            for J in range(1, (JMAX + 1) // 2 + 1):
                for NM in range(1, NMDIM + 1):
                    for K in range(1, KMAX + 1):
                        IM = MLIST[NM]
                        JP = JLIST[NM] + J
                        WDATA[NM, K] = WDATA[NM, K] - PNM[NM, J] * ZDW[IM, JP, K]
    else:
        if KMAX < NMDIM:
            for J in range(1, (JMAX + 1) // 2 + 1):
                for K in range(1, KMAX + 1):
                    for NM in range(1, NMDIM + 1):
                        IM = MLIST[NM]
                        JP = JLIST[NM] + J
                        WDATA[NM, K] = WDATA[NM, K] + PNM[NM, J] * ZDW[IM, JP, K]
        else:
            for J in range(1, (JMAX + 1) // 2 + 1):
                for NM in range(1, NMDIM + 1):
                    for K in range(1, KMAX + 1):
                        IM = MLIST[NM]
                        JP = JLIST[NM] + J
                        WDATA[NM, K] = WDATA[NM, K] + PNM[NM, J] * ZDW[IM, JP, K]

    return WDATA


def GRADX(ZDATA, IDIM, JDIM, KMAX, MMAX, MINT, ZDW):
    from f2py.ufftp import fft99x
    from f2py.copy import copy
    from f2py.usphe import SPW2G, SPG2W
    from f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI


    ZDW = copy(ZDATA, IDIM * JDIM * KMAX)
    for M in range(0, MMAX, MINT):
        MM = M // MINT
        MR = 2 * MM + 1
        MI = 2 * MM + 2
        for K in range(KMAX):
            for J in range(JDIM):
                ZDATA[MR, J, K] = -float(M) * ZDW[MI, J, K]
                ZDATA[MI, J, K] = float(M) * ZDW[MR, J, K]
    return ZDATA
