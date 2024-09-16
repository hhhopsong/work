import numpy as np
import numba
from numba import jit
from LBM.f2py.ufftp import fft99x
from LBM.f2py.dim import IDIM, JDIM, IMAX, JMAX, LMAX, MMAX, NMAX, MINT, JMXHF, NMDIM, KMAX, KDIM, IJKDIM, MMXMI
import torch


# 全局常量
KMAXD = 20
NMDIMD = NMDIM
OLSET = False
MLIST = torch.zeros(NMDIMD, dtype=torch.int32)
JLIST = torch.zeros(NMDIMD, dtype=torch.int32)
device = torch.device('cuda')


 # jit，numba装饰器中的一种
def copy(datai, idim):
    if isinstance(datai, torch.Tensor):
        device = datai.device
        datai = datai.cpu().numpy()
        result = np.array(datai[:idim])
        return torch.tensor(result, dtype=datai.dtype, device=device)
    else:
        return np.array(datai[:idim])



def SPW2G(GDATA, WDATA, PNM, NMO, TRIGS, IFAX, HGRAD, HFUNC, IMAX, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, ZDATA, WORK):
    if IMAX == 1 or JMAX == 1:
        print(' ### SPW2G: THIS ROUTINE IS FOR 3 DIM.')
        return

    # 处理LOFFS和LDPNM标志
    LOFFS = HFUNC[-1] == 'O'
    if LOFFS:
        # 假设这里的KMAXD是一个预定义的常量或者全局变量
        # 如果不是，需要正确定义它
        if KMAXD < KMAX:
            print(' ### SPW2G: WORK AREA(KMAXD) TOO SMALL < ', KMAX)
            return
    LDPNM = HGRAD[0] == 'Y'

    DOFFS = torch.zeros(KMAX, device=device)
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
    if IMAX == 1 or JMAX == 1:
        print(' ### SPG2W: THIS ROUTINE IS FOR 3 DIM.')
        return

    LOFFS = HFUNC[-1] == 'O'
    if LOFFS:
        if KMAXD < KMAX:
            print(' ### SPG2W: WORK AREA(KMAXD) TOO SMALL < ', KMAX)
            return
    LDPNM = HGRAD[0] == 'Y'

    DOFFS = torch.zeros(KMAX, device=device)
    if LOFFS:
        DOFFS = GDATA[0, :KMAX]
        WORK[:, :KMAX] = GDATA[:, :KMAX] - DOFFS.view(1, -1).repeat(IDIM * JDIM, 1)
    else:
        WORK = GDATA.clone()

    new_work, new_zdata = fft99x(WORK, ZDATA, TRIGS, IFAX, 1, IDIM, IMAX, JDIM * KMAX, 0)
    WORK = new_work.view(WORK.shape)
    ZDATA = new_zdata.view(ZDATA.shape)

    if HGRAD[0] == 'X':
        ZDATA = GRADX(ZDATA, IDIM, JDIM, KMAX, MMAX, MINT, WORK)
        LOFFS = False

    WDATA = SPZ2W(WDATA, ZDATA, PNM, NMO, GW, LDPNM, HFUNC, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, WORK)

    if LOFFS:
        if HFUNC[0] == 'N' or HFUNC[0] == 'S':
            WDATA[NMO[0, 0, 0], :KMAX] = WDATA[NMO[0, 0, 0], :KMAX] - DOFFS
        else:
            WDATA[NMO[0, 0, 0], :KMAX] = WDATA[NMO[0, 0, 0], :KMAX] + DOFFS

    WDATA[NMO[1, 0, 0], :KMAX] = 0.0

    return WDATA


def SPW2Z(ZDATA, WDATA, PNM, NMO, LDPNM, JMAX, KMAX, IDIM, JDIM, LMAX, MMAX, NMAX, MINT, NMDIM, JMXHF, ZDW):
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

    ZDW = torch.zeros((IDIM, JDIM, KMAX), device=device)
    ZDATA = torch.zeros((IDIM, JDIM, KMAX), device=device)

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
        WDATA = torch.zeros((NMDIM, KMAX), device=device)

    ZDW = ZDW.view(IDIM, JDIM, KMAX)
    # 创建索引张量
    Js = torch.arange(1, (JMAX + 1) // 2 + 1, device=device)
    JNs = Js
    JSs = JMAX + 1 - Js
    if not LDPNM:
        JEs = Js
        JOs = (JMAX + 1) // 2 + Js
    else:
        JEs = (JMAX + 1) // 2 + Js
        JOs = JNs
    Ks = torch.arange(1, KMAX + 1, device=device)
    IMs = torch.arange(1, IDIM + 1, device=device)

    # 使用广播机制计算
    for J in range(len(Js)):
        JE = JEs[J] - 1
        JO = JOs[J] - 1
        JN = JNs[J] - 1
        JS = JSs[J] - 1
        PNM = torch.tensor(PNM, device=device)
        part1 = GW[J - 1] * (ZDATA.view(IDIM, JDIM, KMAX)[:, JN, :] + ZDATA.view(IDIM, JDIM, KMAX)[:, JS, :])
        part2 = GW[J - 1] * (ZDATA.view(IDIM, JDIM, KMAX)[:, JN, :] - ZDATA.view(IDIM, JDIM, KMAX)[:, JS, :])
        ZDW[:, JE, :] = part1
        ZDW[:, JO, :] = part2

    if HFUNC[0] == 'N' or HFUNC[0] == 'S':
        if KMAX < NMDIM:
            Js = torch.arange(1, (JMAX + 1) // 2 + 1, device=device)
            JLIST = torch.tensor(JLIST, dtype=torch.long, device=device)
            MLIST = torch.tensor(MLIST, dtype=torch.long, device=device)
            Ks = torch.arange(1, KMAX + 1, device=device)
            NMs = torch.arange(1, NMDIM + 1, device=device)
            for J in range(len(Js)):
                JP = JLIST[NMs - 1] + Js[J]
                IM = MLIST[NMs - 1]
                part = PNM[:, J] * ZDW[IM - 1, JP - 1, :]
                WDATA[:, :] = WDATA[:, :] - part
        else:
            MLIST = torch.tensor(MLIST, dtype=torch.long, device=device)
            JLIST = torch.tensor(JLIST, dtype=torch.long, device=device)
            WDATA = torch.tensor(WDATA, dtype=torch.float32, device=device)

            Js = torch.arange(1, (JMAX + 1) // 2 + 1, device=device)

            JP = JLIST + Js
            IM = MLIST
            part = PNM[:, Js - 1].unsqueeze(1) * ZDW[IM - 1, JP - 1]
            WDATA = WDATA - part.sum(dim=1)
    else:
        if KMAX < NMDIM:
            Js = torch.arange(1, (JMAX + 1) // 2 + 1, device=device)
            JLIST = torch.tensor(JLIST, dtype=torch.long, device=device)
            MLIST = torch.tensor(MLIST, dtype=torch.long, device=device)
            for J in range(len(Js)):
                JP = JLIST + Js[J]
                IM = MLIST
                part = PNM[:, J - 1].unsqueeze(1) * ZDW[IM - 1, JP - 1]
                WDATA = WDATA + part
        else:
            # 将数据移动到GPU
            MLIST = torch.tensor(MLIST, dtype=torch.long, device=device)
            JLIST = torch.tensor(JLIST, dtype=torch.long, device=device)
            WDATA = torch.tensor(WDATA, dtype=torch.float, device=device)

            J_indices = torch.arange(1, (JMAX + 1) // 2 + 1, device=device)
            NM_indices = torch.arange(1, NMDIM + 1, device=device)

            # 使用广播机制计算结果
            for J in J_indices:
                JP = JLIST[:, None] + J
                for NM in NM_indices:
                    IM = MLIST[NM - 1]
                    # 使用广播计算乘法和加法
                    WDATA[NM - 1] += (PNM[NM - 1, J - 1] * ZDW[IM - 1, JP - 1]).sum(dim=1)
    return WDATA


def GRADX(ZDATA, IDIM, JDIM, KMAX, MMAX, MINT, ZDW):
    ZDW = ZDATA.clone()
    ZDATA = torch.tensor(ZDATA, dtype = torch.float32, device=device)
    ZDW = torch.tensor(ZDW, dtype = torch.float32, device=device)
    Ms = torch.arange(0, MMAX, MINT, device=device)
    MMs = Ms // MINT
    MRs = 2 * MMs + 1
    MIs = 2 * MMs + 2
    ZDATA[MRs[:, None, None], torch.arange(JDIM)[None, :, None], torch.arange(KMAX)[None, None, :]] = -Ms[:, None, None] * ZDW[MIs[:, None, None], torch.arange(JDIM)[None, :, None], torch.arange(KMAX)[None, None, :]]
    ZDATA[MIs[:, None, None], torch.arange(JDIM)[None, :, None], torch.arange(KMAX)[None, None, :]] = Ms[:, None, None] * ZDW[MRs[:, None, None], torch.arange(JDIM)[None, :, None], torch.arange(KMAX)[None, None, :]]
    return ZDATA


