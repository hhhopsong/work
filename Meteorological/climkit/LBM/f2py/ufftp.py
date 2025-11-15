import numpy as np
import math
import torch

device = torch.device('cuda')

def fft99x(g, z, trigs, ifax, inc, jump, n, lot, isign):
    new_g = torch.tensor(g.flatten(), dtype=torch.complex64, device = device)
    new_z = torch.tensor(z.flatten(), dtype=torch.complex64, device = device)
    if n == 1:
        if isign == 0:
            for k in range(lot):
                new_z[k] = new_g[k]
        else:
            for k in range(lot):
                new_g[k] = new_z[k]
        return new_g, new_z
    if jump > inc:
        incn = (lot * jump) // n
    else:
        incn = inc
    if incn % 16 == 0:
        incn = incn - 1
    incn = max(incn, lot)
    if isign == 0:
        # 计算la和lb的偏移量
        la_offsets = torch.arange(1, n * inc + 1, inc).view(-1, 1).to(device)
        lb_offsets = torch.arange(0, n * incn, incn).view(-1, 1).to(device)
        # 计算所有的索引并进行赋值
        for i in range(n):
            la_offset = la_offsets[i]
            lb_offset = lb_offsets[i]
            la_indices = la_offset + torch.arange(0, lot * jump, jump).view(-1, 1).to(device)
            lb_indices = lb_offset + torch.arange(lot).view(-1, 1).to(device)
            new_z[lb_indices.view(-1)] = new_g[la_indices.view(-1)]
        new_z = new_z.reshape(-1, incn).transpose(0, 1)
        new_z = torch.fft.fft(new_z)
        new_z = new_z.flatten()
        for k in range(lot):
            new_g[k] = new_z[k]
            new_g[k+incn] = new_z[k+incn*(n - 1)]
        # 计算lb的偏移量
        lb_offsets = torch.arange(2 * incn, n * incn, incn).view(-1, 1).to(device)
        for i in range(len(lb_offsets)):
            lb_offset = lb_offsets[i]
            lb_indices = lb_offset + torch.arange(lot).view(-1, 1).to(device)
            source_indices = lb_indices - incn
            new_g[lb_indices.view(-1)] = new_z[source_indices.view(-1)]
        la_offsets = torch.arange(1, n * inc + 1, inc).view(-1, 1).to(device)
        lb_offsets = torch.arange(0, n * incn, incn).view(-1, 1).to(device)
        for i in range(n):
            la_offset = la_offsets[i]
            lb_offset = lb_offsets[i]
            la_indices = la_offset + torch.arange(0, lot * jump, jump).view(-1, 1).to(device)
            lb_indices = lb_offset + torch.arange(lot).view(-1, 1).to(device)
            new_z[la_indices.view(-1)] = new_g[lb_indices.view(-1)]
        return new_g, new_z
    else:
        # 计算la和lb的偏移量
        la_offsets = torch.arange(1, n * inc + 1, inc).view(-1, 1).to(device)
        lb_offsets = torch.arange(0, n * incn, incn).view(-1, 1).to(device)
        # 计算所有的索引并进行赋值
        for i in range(n):
            la_offset = la_offsets[i]
            lb_offset = lb_offsets[i]
            la_indices = la_offset + torch.arange(0, lot * jump, jump).view(-1, 1).to(device)
            lb_indices = lb_offset + torch.arange(lot).view(-1, 1).to(device)
            new_g[lb_indices.view(-1)] = new_z[la_indices.view(-1)]
        for k in range(lot):
            new_z[k] = new_g[k]
            new_z[k+incn*(n - 1)] = new_g[k+incn]
        lb_offsets = torch.arange(2 * incn, n * incn, incn).view(-1, 1).to(device)
        for i in range(len(lb_offsets)):
            lb_offset = lb_offsets[i]
            lb_indices = lb_offset + torch.arange(lot).view(-1, 1).to(device)
            target_indices = lb_indices - incn
            new_z[target_indices.view(-1)] = new_g[lb_indices.view(-1)]
        new_z = new_z.reshape(-1, incn).transpose(0, 1)
        new_z = torch.fft.ifft(new_z)
        new_z = new_z.flatten()
        # 预先计算偏移量
        la_offsets = torch.arange(1, n * inc + 1, inc).to(device).view(-1, 1)
        lb_offsets = torch.arange(0, n * incn, incn).to(device).view(-1, 1)
        for i in range(n):
            la_offset = la_offsets[i]
            lb_offset = lb_offsets[i]
            la_indices = la_offset + torch.arange(0, lot * jump, jump).view(-1, 1).to(device)
            lb_indices = lb_offset + torch.arange(lot).view(-1, 1).to(device)
            new_g[la_indices.view(-1)] = new_z[lb_indices.view(-1)]
        return new_g, new_z


def rfftfm(n, inc, jump, lot, r, wa, ifac, wsave):
    if n == 1:
        return None
    if jump!= 1:
        result = rftf2m(n, inc, jump, lot, r, wa, ifac, wsave)
        return result
    else:
        result = rftf3m(n, inc, lot, r, wa, ifac, wsave)
        return result


def rftf2m(n, inc, jump, lot, r, wa, ifac, wsave):
    # 将相关数据转换为torch张量并移到指定设备上
    r = torch.tensor(r, dtype=torch.complex64, device=device)
    wa = torch.tensor(wa, dtype=torch.float32, device=device)
    ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    wsave = torch.tensor(wsave, dtype=torch.complex64, device=device)

    incn = lot * jump // n if jump > inc else inc
    if incn % 16 == 0:
        incn = incn - 1
    incn = max(incn, lot)
    n4 = (n // 4) * 4
    if n4 >= 4:
        iabase = 0
        ibbase = inc
        icbase = inc * 2
        idbase = inc * 3
        jabase = 0
        jbbase = incn
        jcbase = incn * 2
        jdbase = incn * 3
        inq = 4 * inc
        inqn = 4 * incn
        for k in range(0, n4, 4):
            ia = iabase
            ib = ibbase
            ic = icbase
            id = idbase
            ja = jabase
            jb = jbbase
            jc = jcbase
            jd = jdbase
            for l in range(lot):
                wsave[ja] = r[ia]
                wsave[jb] = r[ib]
                wsave[jc] = r[ic]
                wsave[jd] = r[id]
                ia = ia + jump
                ib = ib + jump
                ic = ic + jump
                id = id + jump
                ja = ja + 1
                jb = jb + 1
                jc = jc + 1
                jd = jd + 1
            iabase = iabase + inq
            ibbase = ibbase + inq
            icbase = icbase + inq
            idbase = idbase + inq
            jabase = jabase + inqn
            jbbase = jbbase + inqn
            jcbase = jcbase + inqn
            jdbase = jdbase + inqn
    if n4!= n:
        iabase = n4 * inc
        jabase = n4 * incn
        for k in range(n4, n):
            ia = iabase
            ja = jabase
            for l in range(lot):
                wsave[ja] = r[ia]
                ia = ia + jump
                ja = ja + 1
            iabase = iabase + inc
            jabase = jabase + incn
    wsave, wa, ifac, r = rftf1m(n, incn, lot, wsave, wa, ifac, r)
    cf = 1.0 / n
    if n4 >= 4:
        iabase = 0
        ibbase = inc
        icbase = inc * 2
        idbase = inc * 3
        jabase = 0
        jbbase = incn
        jcbase = incn * 2
        jdbase = incn * 3
        inq = 4 * inc
        inqn = 4 * incn
        for k in range(0, n4, 4):
            ia = iabase
            ib = ibbase
            ic = icbase
            id = idbase
            ja = jabase
            jb = jbbase
            jc = jcbase
            jd = jdbase
            for l in range(lot):
                r[ia] = cf * wsave[ja]
                r[ib] = cf * wsave[jb]
                r[ic] = cf * wsave[jc]
                r[id] = cf * wsave[jd]
                ia = ia + jump
                ib = ib + jump
                ic = ic + jump
                id = id + jump
                ja = ja + 1
                jb = jb + 1
                jc = jc + 1
                jd = jd + 1
            iabase = iabase + inq
            ibbase = ibbase + inq
            icbase = icbase + inq
            idbase = idbase + inq
            jabase = jabase + inqn
            jbbase = jbbase + inqn
            jcbase = jcbase + inqn
            jdbase = jdbase + inqn
    if n4!= n:
        iabase = n4 * inc
        jabase = n4 * incn
        for k in range(n4, n):
            ia = iabase
            ja = jabase
            for l in range(lot):
                r[ia] = cf * wsave[ja]
                ia = ia + jump
                ja = ja + 1
            iabase = iabase + inc
            jabase = jabase + incn
    return r, wa, ifac, wsave


def rftf3m(n, inc, lot, c, wa, ifac, ch):
    # 判断GPU是否可用并设置设备
    device = torch.device('cuda')

    # 将相关数据转换为torch张量并移到指定设备上
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    wa = torch.tensor(wa, dtype=torch.float32, device=device)
    ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    if ch is not None:
        ch = torch.tensor(ch, dtype=torch.complex64, device=device)

    na, c, wa, ifac, ch = rftf9m(n, inc, lot, None, c, wa, ifac, ch)
    cf = 1.0 / n
    n4 = (n // 4) * 4
    if na == 1:
        if n4 >= 4:
            for k in torch.arange(1, n4, 4).to(device):
                for l in range(lot):
                    c[l][k] = cf * c[l][k]
                    c[l][k + 1] = cf * c[l][k + 1]
                    c[l][k + 2] = cf * c[l][k + 2]
                    c[l][k + 3] = cf * c[l][k + 3]
        if n4!= n:
            for k in torch.arange(n4 + 1, n).to(device):
                for l in range(lot):
                    c[l][k] = cf * c[l][k]
    else:
        if n4 >= 4:
            for k in torch.arange(1, n4, 4).to(device):
                for l in range(lot):
                    c[l][k] = cf * ch[l][k]
                    c[l][k + 1] = cf * ch[l][k + 1]
                    c[l][k + 2] = cf * ch[l][k + 2]
                    c[l][k + 3] = cf * ch[l][k + 3]
        if n4!= n:
            for k in torch.arange(n4 + 1, n).to(device):
                for l in range(lot):
                    c[l][k] = cf * ch[l][k]
    if device.type == 'cuda':
        c = c
        if ch is not None:
            ch = ch
        wa = wa
        ifac = ifac
    return c, wa, ifac, ch

 # jit，numba装饰器中的一种
def rftf1m(n, inc, lot, c, wa, ifac, ch):
    # 将相关数据转换为torch张量并移到指定设备上
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    wa = torch.tensor(wa, dtype=torch.float32, device=device)
    ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    if ch is not None:
        ch = torch.tensor(ch, dtype=torch.complex64, device=device)

    na, c, wa, ifac, ch = rftf9m(n, inc, lot, None, c, wa, ifac, ch)
    if na == 1:
        return c, wa, ifac, ch
    n4 = (n // 4) * 4
    if n4 >= 4:
        for k in range(1, n4, 4):
            for l in range(lot):
                c[l][k] = ch[l][k]
                c[l][k + 1] = ch[l][k + 1]
                c[l][k + 2] = ch[l][k + 2]
                c[l][k + 3] = ch[l][k + 3]
    if n4!= n:
        for k in range(n4 + 1, n):
            for l in range(lot):
                c[l][k] = ch[l][k]
    return c, wa, ifac, ch


def rftf9m(n, inc, lot, na, c, wa, ifac, ch):
    # 将相关数据转换为torch张量并移到指定设备上
    ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    wa = torch.tensor(wa, dtype=torch.float32, device=device)
    if c is not None:
        c = torch.tensor(c, dtype=torch.complex64, device=device)
    if ch is not None:
        ch = torch.tensor(ch, dtype=torch.complex64, device=device)

    nf = ifac[2]
    if na is None:
        na = 1
    l2 = n
    iw = n
    for k1 in range(1, nf):
        kh = nf - k1
        ip = ifac[kh + 3]
        l1 = l2 // ip
        ido = n // l2
        idl1 = ido * l1
        iw = iw - (ip - 1) * ido
        na = 1 - na
        if ip == 4:
            ix2 = iw + ido
            ix3 = ix2 + ido
            if na == 0:
                c, ch = radf4m(inc, lot, ido, l1, c, ch, wa[iw], wa[ix2], wa[ix3])
            else:
                c, ch = radf4m(inc, lot, ido, l1, ch, c, wa[iw], wa[ix2], wa[ix3])
        elif ip == 2:
            if na == 0:
                c, ch = radf2m(inc, lot, ido, l1, c, ch, wa[iw])
            else:
                c, ch = radf2m(inc, lot, ido, l1, ch, c, wa[iw])
        elif ip == 3:
            ix2 = iw + ido
            if na == 0:
                c, ch = radf3m(inc, lot, ido, l1, c, ch, wa[iw], wa[ix2])
            else:
                c, ch = radf3m(inc, lot, ido, l1, ch, c, wa[iw], wa[ix2])
        elif ip == 5:
            ix2 = iw + ido
            ix3 = ix2 + ido
            ix4 = ix3 + ido
            if na == 0:
                c, ch = radf5m(inc, lot, ido, l1, c, ch, wa[iw], wa[ix2], wa[ix3], wa[ix4])
            else:
                c, ch = radf5m(inc, lot, ido, l1, ch, c, wa[iw], wa[ix2], wa[ix3], wa[ix4])
        else:
            if ido == 1:
                na = 1 - na
            if na == 0:
                c, ch = radfgm(inc, lot, ido, ip, l1, idl1, c, c, c, ch, ch, wa[iw])
                na = 1
            else:
                c, ch = radfgm(inc, lot, ido, ip, l1, idl1, ch, ch, ch, c, c, wa[iw])
                na = 0
        l2 = l1
    return na, c, wa, ifac, ch


def radf4m(inc, lot, ido, l1, c, ch, ca1, ca2, ca3):
    # 将相关数据转换为torch张量并移到指定设备上
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    ca1 = torch.tensor(ca1, dtype=torch.float32, device=device)
    ca2 = torch.tensor(ca2, dtype=torch.float32, device=device)
    ca3 = torch.tensor(ca3, dtype=torch.float32, device=device)

    for i in range(lot):
        for j in torch.arange(ido).to(device):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            index3 = (2 * ido + j) * l1 * inc
            index4 = (3 * ido + j) * l1 * inc
            ca = ca1[j]
            cb = ca2[j]
            cc = ca3[j]
            ch11 = c[i][index1] + c[i][index3]
            ch12 = c[i][index1] - c[i][index3]
            ch21 = c[i][index2] + c[i][index4]
            ch22 = c[i][index2] - c[i][index4]
            ch[i][index1] = ch11 + ch21
            ch[i][index3] = ch11 - ch21
            ch[i][index2] = (ch12 - ch22) * ca+(ch12 + ch22) * cb
            ch[i][index4] = (ch12 - ch22) * cb-(ch12 + ch22) * ca
    return c, ch


def radf2m(inc, lot, ido, l1, c, ch, ca1):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    ca1 = torch.tensor(ca1, dtype=torch.float32, device=device)

    for i in range(lot):
        for j in torch.arange(ido).to(device):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            ca = ca1[j]
            ch1 = c[i][index1]+c[i][index2]
            ch2 = (c[i][index1]-c[i][index2]) * ca
            ch[i][index1] = ch1
            ch[i][index2] = ch2
    return c, ch


def radf3m(inc, lot, ido, l1, c, ch, ca1, ca2):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    ca1 = torch.tensor(ca1, dtype=torch.float32, device=device)
    ca2 = torch.tensor(ca2, dtype=torch.float32, device=device)

    tpi = 8 * torch.atan(torch.tensor(1.0))
    arg1 = tpi / 3
    ca1_ = torch.cos(arg1)
    sa1 = torch.sin(arg1)
    ca2_ = torch.cos(2 * arg1)
    sa2 = torch.sin(2 * arg1)
    for i in range(lot):
        for j in torch.arange(ido).to(device):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            index3 = (2 * ido + j) * l1 * inc
            ca = ca1[j]
            cb = ca2[j]
            ch1 = c[i][index1]+c[i][index2]+c[i][index3]
            ch2 = c[i][index1]+c[i][index2]*ca1_+c[i][index3]*ca2_
            ch3 = c[i][index1]+c[i][index2]*ca2_+c[i][index3]*ca1_
            ch[i][index1] = ch1
            ch[i][index2] = (ch2 - ch3) * sa1
            ch[i][index3] = (ch3 - ch2) * sa2
    return c, ch


def radf5m(inc, lot, ido, l1, cc, ch, wa1, wa2, wa3, wa4):
    # 将相关数据转换为torch张量并移到指定设备上
    cc = torch.tensor(cc, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    wa1 = torch.tensor(wa1, dtype=torch.float32, device=device)
    wa2 = torch.tensor(wa2, dtype=torch.float32, device=device)
    wa3 = torch.tensor(wa3, dtype=torch.float32, device=device)
    wa4 = torch.tensor(wa4, dtype=torch.float32, device=device)

    tr11 = (-1.0 + torch.sqrt(torch.tensor(5.0))) / 4.0
    ti11 = torch.sqrt(1.0 - tr11 * tr11)
    tr12 = (-1.0 - torch.sqrt(torch.tensor(5.0))) / 4.0
    ti12 = torch.sqrt(1.0 - tr12 * tr12)

    for k in range(l1):
        for l in range(lot):
            cr2 = cc[l][0][k][4] + cc[l][0][k][1]
            cr3 = cc[l][0][k][3] + cc[l][0][k][2]
            ci5 = cc[l][0][k][4] - cc[l][0][k][1]
            ci4 = cc[l][0][k][3] - cc[l][0][k][2]
            ch[l][0][0][k] = cc[l][0][k][0] + cr2 + cr3
            ch[l][0][2][k] = ti11 * ci5 + ti12 * ci4
            ch[l][ido - 1][1][k] = cc[l][0][k][0] + tr11 * cr2 + tr12 * cr3
            ch[l][0][4][k] = ti12 * ci5 - ti11 * ci4
            ch[l][ido - 1][3][k] = cc[l][0][k][0] + tr12 * cr2 + tr11 * cr3

    if ido > 1:
        idp2 = ido + 2
        for k in range(l1):
            for i in torch.arange(2, ido, 2).to(device):
                ic = idp2 - i
                for l in range(lot):
                    ca2 = wa1[i - 2] * cc[l][i - 1][k][1]
                    cu2 = wa1[i - 2] * cc[l][i][k][1]
                    ca3 = wa2[i - 2] * cc[l][i - 1][k][2]
                    cu3 = wa2[i - 2] * cc[l][i][k][2]
                    ca4 = wa3[i - 2] * cc[l][i - 1][k][3]
                    cu4 = wa3[i - 2] * cc[l][i][k][3]
                    ca5 = wa4[i - 2] * cc[l][i - 1][k][4]
                    cu5 = wa4[i - 2] * cc[l][i][k][4]
                    cb2 = wa1[i - 1] * cc[l][i - 1][k][1]
                    cv2 = wa1[i - 1] * cc[l][i][k][1]
                    cb3 = wa2[i - 1] * cc[l][i - 1][k][2]
                    cv3 = wa2[i - 1] * cc[l][i][k][2]
                    cb4 = wa3[i - 1] * cc[l][i - 1][k][3]
                    cv4 = wa3[i - 1] * cc[l][i][k][3]
                    cb5 = wa4[i - 1] * cc[l][i - 1][k][4]
                    cv5 = wa4[i - 1] * cc[l][i][k][4]
                    ch[l][ic - 1][1][k] = ca2 + cv2
                    ch[l][ic][1][k] = cu2 - cb2
                    ch[l][i - 1][2][k] = ca3 + cv3
                    ch[l][i][2][k] = cu3 - cb3
                    ch[l][ic - 1][3][k] = ca4 + cv4
                    ch[l][ic][3][k] = cu4 - cb4
                    ch[l][i - 1][4][k] = ca5 + cv5
                    ch[l][i][4][k] = cu5 - cb5

        for k in range(l1):
            for i in torch.arange(2, ido, 2).to(device):
                ic = idp2 - i
                for l in range(lot):
                    cc[l][i - 1][k][1] = ch[l][ic - 1][1][k] + ch[l][i - 1][4][k]
                    cc[l][i][k][1] = ch[l][ic][1][k] + ch[l][i][4][k]
                    cc[l][i - 1][k][2] = ch[l][i - 1][2][k] + ch[l][ic - 1][3][k]
                    cc[l][i][k][2] = ch[l][i][2][k] + ch[l][ic][3][k]
                    cc[l][i - 1][k][3] = ch[l][i][2][k] - ch[l][ic][3][k]
                    cc[l][i][k][3] = ch[l][ic - 1][3][k] - ch[l][i - 1][2][k]
                    cc[l][i - 1][k][4] = ch[l][ic][1][k] - ch[l][i][4][k]
                    cc[l][i][k][4] = ch[l][i - 1][4][k] - ch[l][ic - 1][1][k]

        for k in range(l1):
            for i in torch.arange(2, ido, 2).to(device):
                ic = idp2 - i
                for l in range(lot):
                    ctr2 = tr11 * cc[l][i - 1][k][1] + tr12 * cc[l][i - 1][k][2]
                    cti2 = tr11 * cc[l][i][k][1] + tr12 * cc[l][i][k][2]
                    ctr3 = tr12 * cc[l][i - 1][k][1] + tr11 * cc[l][i - 1][k][2]
                    cti3 = tr12 * cc[l][i][k][1] + tr11 * cc[l][i][k][2]
                    tr2 = ctr2 + cc[l][i - 1][k][0]
                    ti2 = cti2 + cc[l][i][k][0]
                    tr3 = ctr3 + cc[l][i - 1][k][0]
                    ti3 = cti3 + cc[l][i][k][0]
                    tr4 = ti12 * cc[l][i - 1][k][4] - ti11 * cc[l][i - 1][k][3]
                    ti4 = ti12 * cc[l][i][k][4] - ti11 * cc[l][i][k][3]
                    sr23 = cc[l][i - 1][k][1] + cc[l][i - 1][k][2]
                    si23 = cc[l][i][k][1] + cc[l][i][k][2]
                    tr5 = ti11 * cc[l][i - 1][k][4] + ti12 * cc[l][i - 1][k][3]
                    ti5 = ti11 * cc[l][i][k][4] + ti12 * cc[l][i][k][3]
                    ch[l][i - 1][0][k] = cc[l][i - 1][k][0] + sr23
                    ch[l][i][0][k] = cc[l][i][k][0] + si23
                    ch[l][ic - 1][1][k] = tr2 - tr5
                    ch[l][ic][1][k] = ti5 - ti2
                    ch[l][i - 1][2][k] = tr2 + tr5
                    ch[l][i][2][k] = ti2 + ti5
                    ch[l][ic - 1][3][k] = tr3 - tr4
                    ch[l][ic][3][k] = ti4 - ti3
                    ch[l][i - 1][4][k] = tr3 + tr4
                    ch[l][i][4][k] = ti3 + ti4
    return cc, ch


def radfgm(inc, lot, ido, ip, l1, idl1, cc, c1, c2, ch, ch2, wa):
    # 将相关数据转换为torch张量并移到指定设备上
    cc = torch.tensor(cc, dtype=torch.complex64, device=device)
    c1 = torch.tensor(c1, dtype=torch.complex64, device=device)
    c2 = torch.tensor(c2, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    ch2 = torch.tensor(ch2, dtype=torch.complex64, device=device)
    wa = torch.tensor(wa, dtype=torch.float32, device=device)

    tpi = 8.0 * torch.atan(torch.tensor(1.0))
    arg = tpi / ip
    dcp = torch.cos(arg)
    dsp = torch.sin(arg)
    ipph = (ip + 1) // 2
    ipp2 = ip + 2
    idp2 = ido + 2

    if ido == 1:
        for ik in range(idl1):
            for l in range(lot):
                ch2[l][ik][0] = c2[l][ik][0]
        for j in range(1, ip):
            for k in torch.arange(l1).to(device):
                for l in range(lot):
                    ch[l][0][k][j] = c1[l][0][k][j]
        for j in range(1, ip):
            for k in torch.arange(l1).to(device):
                for i in range(2, ido, 2):
                    idij = i - 1
                    for l in range(lot):
                        ch[l][i - 1][k][j] = wa[idij - 1] * c1[l][i - 1][k][j] + wa[idij] * c1[l][i][k][j]
                        ch[l][i][k][j] = wa[idij - 1] * c1[l][i][k][j] - wa[idij] * c1[l][i - 1][k][j]
        for j in range(1, ipph):
            jc = ipp2 - j
            for k in torch.arange(l1).to(device):
                for l in range(lot):
                    c1[l][0][k][j] = ch[l][0][k][j] + ch[l][0][k][jc]
                    c1[l][0][k][jc] = ch[l][0][k][jc] - ch[l][0][k][j]
        ar1 = 1.0
        ai1 = 0.0
        for m in range(1, ipph):
            mc = ipp2 - m
            ar1h = dcp * ar1 - dsp * ai1
            ai1 = dcp * ai1 + dsp * ar1
            ar1 = ar1h
            for ik in range(idl1):
                for l in range(lot):
                    ch2[l][ik][m] = c2[l][ik][0] + ar1 * c2[l][ik][1]
                    ch2[l][ik][mc] = ai1 * c2[l][ik][ip - 1]
            dc2 = ar1
            ds2 = ai1
            ar2 = ar1
            ai2 = ai1
            for j in range(2, ipph):
                jc = ipp2 - j
                ar2h = dc2 * ar2 - ds2 * ai2
                ai2 = dc2 * ai2 + ds2 * ar2
                ar2 = ar2h
                for ik in range(idl1):
                    for l in range(lot):
                        ch2[l][ik][m] = ch2[l][ik][m] + ar2 * c2[l][ik][j]
                        ch2[l][ik][mc] = ch2[l][ik][mc] + ai2 * c2[l][ik][jc]
        for j in range(1, ipph):
            for ik in range(idl1):
                for l in range(lot):
                    ch2[l][ik][0] = ch2[l][ik][0] + c2[l][ik][j]
        for k in torch.arange(l1).to(device):
            for i in range(ido):
                for l in range(lot):
                    cc[l][i][0][k] = ch[l][i][k][0]
        for j in range(1, ipph):
            jc = ipp2 - j
            j2 = j * 2
            for k in torch.arange(l1).to(device):
                for l in range(lot):
                    cc[l][ido - 1][j2 - 2][k] = ch[l][0][k][j]
                    cc[l][0][j2 - 1][k] = ch[l][0][k][jc]
        return cc, c1, c2, ch, ch2, wa
    else:
        for ik in range(idl1):
            for l in range(lot):
                ch2[l][ik][0] = c2[l][ik][0]
        for j in range(1, ip):
            for k in torch.arange(l1).to(device):
                for l in range(lot):
                    ch[l][0][k][j] = c1[l][0][k][j]
        is_ = -ido
        for j in range(1, ip):
            is_ = is_ + ido
            for k in torch.arange(l1).to(device):
                for i in range(2, ido, 2):
                    idij = is_ + i - 1
                    for l in range(lot):
                        ch[l][i - 1][k][j] = wa[idij - 1] * c1[l][i - 1][k][j] + wa[idij] * c1[l][i][k][j]
                        ch[l][i][k][j] = wa[idij - 1] * c1[l][i][k][j] - wa[idij] * c1[l][i - 1][k][j]
        for j in range(1, ipph):
            jc = ipp2 - j
            for k in torch.arange(l1).to(device):
                for i in range(2, ido, 2):
                    for l in range(lot):
                        c1[l][i - 1][k][j] = ch[l][i - 1][k][j] + ch[l][i - 1][k][jc]
                        c1[l][i][k][j] = ch[l][i][k][j] + ch[l][i][k][jc]
                        c1[l][i - 1][k][jc] = ch[l][i][k][j] - ch[l][i][k][jc]
                        c1[l][i][k][jc] = ch[l][i - 1][k][jc] - ch[l][i - 1][k][j]
        ar1 = 1.0
        ai1 = 0.0
        for m in range(1, ipph):
            mc = ipp2 - m
            ar1h = dcp * ar1 - dsp * ai1
            ai1 = dcp * ai1 + dsp * ar1
            ar1 = ar1h
            for ik in range(idl1):
                for l in range(lot):
                    ch2[l][ik][m] = c2[l][ik][0] + ar1 * c2[l][ik][1]
                    ch2[l][ik][mc] = ai1 * c2[l][ik][ip - 1]
            dc2 = ar1
            ds2 = ai1
            ar2 = ar1
            ai2 = ai1
            for j in range(2, ipph):
                jc = ipp2 - j
                ar2h = dc2 * ar2 - ds2 * ai2
                ai2 = dc2 * ai2 + ds2 * ar2
                ar2 = ar2h
                for ik in range(idl1):
                    for l in range(lot):
                        ch2[l][ik][m] = ch2[l][ik][m] + ar2 * c2[l][ik][j]
                        ch2[l][ik][mc] = ch2[l][ik][mc] + ai2 * c2[l][ik][jc]
        for j in range(1, ipph):
            for ik in range(idl1):
                for l in range(lot):
                    ch2[l][ik][0] = ch2[l][ik][0] + c2[l][ik][j]
        for k in torch.arange(l1).to(device):
            for i in range(ido):
                for l in range(lot):
                    cc[l][i][0][k] = ch[l][i][k][0]
        for j in range(1, ipph):
            jc = ipp2 - j
            j2 = j * 2
            for k in torch.arange(l1).to(device):
                for l in range(lot):
                    cc[l][ido - 1][j2 - 2][k] = ch[l][0][k][j]
                    cc[l][0][j2 - 1][k] = ch[l][0][k][jc]
        for j in range(1, ipph):
            jc = ipp2 - j
            j2 = j * 2
            for k in torch.arange(l1).to(device):
                for i in range(2, ido, 2):
                    ic = idp2 - i
                    for l in range(lot):
                        cc[l][i - 1][j2 - 1][k] = ch[l][i - 1][k][j] + ch[l][i - 1][k][jc]
                        cc[l][i][j2 - 1][k] = ch[l][i][k][j] + ch[l][i][k][jc]
                        cc[l][ic - 1][j2 - 2][k] = ch[l][i - 1][k][j] - ch[l][i - 1][k][jc]
                        cc[l][ic][j2 - 2][k] = ch[l][i][k][jc] - ch[l][i][k][j]
        return cc, c1, c2, ch, ch2, wa

def rfftim(n, trigs, ifax):
    new_trigs = []
    for trig in trigs:
        if isinstance(trig, list):
            new_sub_trig = []
            for sub_trig in trig:
                new_sub_trig.append(torch.tensor(sub_trig, dtype=torch.complex64, device=device))
            new_trigs.append(new_sub_trig)
        else:
            new_trigs.append(torch.tensor(trig, dtype=torch.complex64, device=device))
    new_ifax = torch.tensor(ifax, dtype=torch.int64, device=device)
    if n == 1:
        return new_trigs, new_ifax
    new_trigs, new_ifax = rfti1m(n, new_trigs, new_ifax)
    return new_trigs, new_ifax


def rfti1m(n, wa, ifac):
    if isinstance(wa, list):
        new_wa = []
        for sub_wa in wa:
            if isinstance(sub_wa, list):
                new_sub_wa = []
                for item in sub_wa:
                    new_sub_wa.append(torch.tensor(item, dtype=torch.complex64, device=device))
                new_wa.append(new_sub_wa)
            else:
                new_wa.append(torch.tensor(sub_wa, dtype=torch.complex64, device=device))
    else:
        new_wa = torch.tensor(wa, dtype=torch.complex64, device=device)
    new_ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    ntryh = torch.tensor([4, 2, 3, 5], dtype=torch.int64, device=device)
    tpi = 8 * torch.atan(torch.tensor(1.0))
    nl = n
    nf = 0
    j = 0
    while True:
        j = j + 1
        if j <= 4:
            ntry = ntryh[j - 1]
        else:
            ntry = ntry + 2
        nq = nl // ntry
        nr = nl - ntry * nq
        if nr == 0:
            nf = nf + 1
            new_ifac[nf + 2] = ntry
            nl = nq
            if nl > 1:
                continue
            break
        else:
            continue
    for i in torch.arange(2, nf).to(device):
        if new_ifac[i + 2] == 2:
            new_ifac[i + 2] = 4
            new_ifac[3] = 2
    new_ifac[1] = n
    new_ifac[2] = nf
    if nf == 1:
        return new_wa, new_ifac
    l1 = 1
    is_ = 0
    for k in torch.arange(1, nf).to(device):
        ip = new_ifac[k + 2]
        ido = n // (l1 * ip)
        for j in torch.arange(1, ip).to(device):
            arggld = (j * l1) * (tpi / n)
            for ifi in torch.arange(1, (ido - 1) // 2 + 1).to(device):
                arg = ifi * arggld
                new_wa[2 * ifi + is_ - 1] = torch.cos(arg)
                new_wa[2 * ifi + is_] = torch.sin(arg)
            is_ = is_ + ido
        l1 = l1 * ip
    return new_wa, new_ifac

def rfftbm(n, inc, jump, lot, r, wa, ifac, wsave):
    new_r = []
    for sub_r in r:
        new_sub_r = []
        for item in sub_r:
            new_sub_r.append(torch.tensor(item, dtype=torch.complex64, device=device))
        new_r.append(new_sub_r)
    new_wa = []
    for sub_wa in wa:
        new_sub_wa = []
        for item in sub_wa:
            new_sub_wa.append(torch.tensor(item, dtype=torch.float32, device=device))
        new_wa.append(new_sub_wa)
    new_ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    new_wsave = []
    for sub_wsave in wsave:
        new_sub_wsave = []
        for item in sub_wsave:
            new_sub_wsave.append(torch.tensor(item, dtype=torch.complex64, device=device))
        new_wsave.append(new_sub_wsave)
    if n == 1:
        return new_r, new_wa, new_ifac, new_wsave
    if jump!= 1:
        new_r, new_wa, new_ifac, new_wsave = rftb2m(n, inc, jump, lot, new_r, new_wa, new_ifac, new_wsave)
    else:
        new_r, new_wa, new_ifac, new_wsave = rftb1m(n, inc, lot, new_r, new_wa, new_ifac, new_wsave)
    return new_r, new_wa, new_ifac, new_wsave



def rftb2m(n, inc, jump, lot, r, wa, ifac, wsave):
    r = torch.tensor(r, dtype=torch.complex64, device=device)
    wa = torch.tensor(wa, dtype=torch.float32, device=device)
    ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    wsave = torch.tensor(wsave, dtype=torch.complex64, device=device)

    if jump > inc:
        incn = (lot * jump) // n
    else:
        incn = inc
    if incn % 16 == 0:
        incn = incn - 1
    incn = max(incn, lot)
    n4 = (n // 4) * 4
    if n4 >= 4:
        iabase = 0
        ibbase = inc
        icbase = inc * 2
        idbase = inc * 3
        jabase = 0
        jbbase = incn
        jcbase = incn * 2
        jdbase = incn * 3
        inq = 4 * inc
        inqn = 4 * incn
        for k in torch.arange(0, n4, 4).to(device):
            ia = iabase
            ib = ibbase
            ic = icbase
            id = idbase
            ja = jabase
            jb = jbbase
            jc = jcbase
            jd = jdbase
            for l in range(lot):
                wsave[ja] = r[ia]
                wsave[jb] = r[ib]
                wsave[jc] = r[ic]
                wsave[jd] = r[id]
                ia = ia + jump
                ib = ib + jump
                ic = ic + jump
                id = id + jump
                ja = ja + 1
                jb = jb + 1
                jc = jc + 1
                jd = jd + 1
            iabase = iabase + inq
            ibbase = ibbase + inq
            icbase = icbase + inq
            idbase = idbase + inq
            jabase = jabase + inqn
            jbbase = jbbase + inqn
            jcbase = jcbase + inqn
            jdbase = jdbase + inqn
    if n4!= n:
        iabase = n4 * inc
        jabase = n4 * incn
        for k in torch.arange(n4, n).to(device):
            ia = iabase
            ja = jabase
            for l in range(lot):
                wsave[ja] = r[ia]
                ia = ia + jump
                ja = ja + 1
            iabase = iabase + inc
            jabase = jabase + incn
    r, wa, ifac, wsave = rftb1m(n, incn, lot, wsave, wa, ifac, r)
    if n4 >= 4:
        iabase = 0
        ibbase = inc
        icbase = inc * 2
        idbase = inc * 3
        jabase = 0
        jbbase = incn
        jcbase = incn * 2
        jdbase = incn * 3
        inq = 4 * inc
        inqn = 4 * incn
        for k in torch.arange(0, n4, 4).to(device):
            ia = iabase
            ib = ibbase
            ic = icbase
            id = idbase
            ja = jabase
            jb = jbbase
            jc = jcbase
            jd = jdbase
            for l in range(lot):
                r[ia] = wsave[ja]
                r[ib] = wsave[jb]
                r[ic] = wsave[jc]
                r[id] = wsave[jd]
                ia = ia + jump
                ib = ib + jump
                ic = ic + jump
                id = id + jump
                ja = ja + 1
                jb = jb + 1
                jc = jc + 1
                jd = jd + 1
            iabase = iabase + inq
            ibbase = ibbase + inq
            icbase = icbase + inq
            idbase = idbase + inq
            jabase = jabase + inqn
            jbbase = jbbase + inqn
            jcbase = jcbase + inqn
            jdbase = jdbase + inqn
    if n4!= n:
        iabase = n4 * inc
        jabase = n4 * incn
        for k in torch.arange(n4, n).to(device):
            ia = iabase
            ja = jabase
            for l in range(lot):
                r[ia] = wsave[ja]
                ia = ia + jump
                ja = ja + 1
            iabase = iabase + inc
            jabase = jabase + incn
    return r, wa, ifac, wsave

# jit，numba装饰器中的一种
def rftb1m(n, inc, lot, c, wa, ifac, ch):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    wa = torch.tensor(wa, dtype=torch.float32, device=device)
    ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)

    na = rftb9m(n, inc, lot, None, c, wa, ifac, ch)
    if na == 1:
        return c, wa, ifac, ch
    n4 = (n // 4) * 4
    if n4 >= 4:
        for k in torch.arange(0, n4, 4).to(device):
            for l in range(lot):
                c[l][k] = ch[l][k]
                c[l][k + 1] = ch[l][k + 1]
                c[l][k + 2] = ch[l][k + 2]
                c[l][k + 3] = ch[l][k + 3]
    if n4!= n:
        for k in torch.arange(n4, n).to(device):
            for l in range(lot):
                c[l][k] = ch[l][k]
    return c, wa, ifac, ch

def rftb9m(n, inc, lot, na, c, wa, ifac, ch):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    wa = torch.tensor(wa, dtype=torch.float32, device=device)
    ifac = torch.tensor(ifac, dtype=torch.int64, device=device)
    if ch is not None:
        ch = torch.tensor(ch, dtype=torch.complex64, device=device)

    nf = ifac[2]
    if na is None:
        na = 1
    l1 = 1
    iw = 1
    for k1 in range(1, nf):
        ip = ifac[k1 + 2]
        l2 = ip * l1
        ido = n // l2
        idl1 = ido * l1
        na = 1 - na
        if ip == 4:
            ix2 = iw + ido
            ix3 = ix2 + ido
            if na == 0:
                c, ch = radb4m(inc, lot, ido, l1, c, ch, wa[iw - 1], wa[ix2 - 1], wa[ix3 - 1])
            else:
                c, ch = radb4m(inc, lot, ido, l1, ch, c, wa[iw - 1], wa[ix2 - 1], wa[ix3 - 1])
        elif ip == 2:
            if na == 0:
                c, ch = radb2m(inc, lot, ido, l1, c, ch, wa[iw - 1])
            else:
                c, ch = radb2m(inc, lot, ido, l1, ch, c, wa[iw - 1])
        elif ip == 3:
            ix2 = iw + ido
            if na == 0:
                c, ch = radb3m(inc, lot, ido, l1, c, ch, wa[iw - 1], wa[ix2 - 1])
            else:
                c, ch = radb3m(inc, lot, ido, l1, ch, c, wa[iw - 1], wa[ix2 - 1])
        elif ip == 5:
            ix2 = iw + ido
            ix3 = ix2 + ido
            ix4 = ix3 + ido
            if na == 0:
                c, ch = radb5m(inc, lot, ido, l1, c, ch, wa[iw - 1], wa[ix2 - 1], wa[ix3 - 1], wa[ix4 - 1])
            else:
                c, ch = radb5m(inc, lot, ido, l1, ch, c, wa[iw - 1], wa[ix2 - 1], wa[ix3 - 1], wa[ix4 - 1])
        else:
            if na == 0:
                c, ch = radbgm(inc, lot, ido, ip, l1, idl1, c, c, c, ch, ch, wa[iw - 1])
                na = 1
            else:
                c, ch = radbgm(inc, lot, ido, ip, l1, idl1, ch, ch, ch, c, c, wa[iw - 1])
                na = 0
            if ido == 1:
                na = 1
    return na, c, wa, ifac, ch


def radb4m(inc, lot, ido, l1, c, ch, ca1, ca2, ca3):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    ca1 = torch.tensor(ca1, dtype=torch.float32, device=device)
    ca2 = torch.tensor(ca2, dtype=torch.float32, device=device)
    ca3 = torch.tensor(ca3, dtype=torch.float32, device=device)

    for i in range(lot):
        for j in torch.arange(ido).to(device):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            index3 = (2 * ido + j) * l1 * inc
            index4 = (3 * ido + j) * l1 * inc
            ca = ca1[j]
            cb = ca2[j]
            cc = ca3[j]
            ch11 = c[i][index1] + c[i][index3]
            ch12 = c[i][index1] - c[i][index3]
            ch21 = c[i][index2] + c[i][index4]
            ch22 = c[i][index2] - c[i][index4]
            ch[i][index1] = ch11 + ch21
            ch[i][index3] = ch11 - ch21
            ch[i][index2] = (ch12 - ch22) * ca+(ch12 + ch22) * cb
            ch[i][index4] = (ch12 - ch22) * cb-(ch12 + ch22) * ca
    return c, ch


def radb2m(inc, lot, ido, l1, c, ch, ca1):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    ca1 = torch.tensor(ca1, dtype=torch.float32, device=device)

    for i in range(lot):
        for j in torch.arange(ido).to(device):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            ca = ca1[j]
            ch1 = c[index1] + c[index2]
            ch2 = (c[index1] - c[index2]) * ca
            ch[index1] = ch1
            ch[index2] = ch2
    return c, ch


def radb3m(inc, lot, ido, l1, c, ch, ca1, ca2):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    ca1 = torch.tensor(ca1, dtype=torch.float32, device=device)
    ca2 = torch.tensor(ca2, dtype=torch.float32, device=device)

    tpi = 8 * torch.atan(torch.tensor(1.0))
    arg1 = tpi / 3
    ca1_ = torch.cos(arg1)
    sa1 = torch.sin(arg1)
    ca2_ = torch.cos(2 * arg1)
    sa2 = torch.sin(2 * arg1)

    for i in range(lot):
        for j in torch.arange(ido).to(device):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            index3 = (2 * ido + j) * l1 * inc
            ca = ca1[j]
            cb = ca2[j]
            ch1 = c[index1] + c[index2] + c[index3]
            ch2 = c[index1] + c[index2] * ca1_ + c[index3] * ca2_
            ch3 = c[index1] + c[index2] * ca2_ + c[index3] * ca1_
            ch[index1] = ch1
            ch[index2] = (ch2 - ch3) * sa1
            ch[index3] = (ch3 - ch2) * sa2
    return c, ch


def radb5m(inc, lot, ido, l1, c, ch, ca1, ca2, ca3, ca4):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    ch = torch.tensor(ch, dtype=torch.complex64, device=device)
    ca1 = torch.tensor(ca1, dtype=torch.float32, device=device)
    ca2 = torch.tensor(ca2, dtype=torch.float32, device=device)
    ca3 = torch.tensor(ca3, dtype=torch.float32, device=device)
    ca4 = torch.tensor(ca4, dtype=torch.float32, device=device)

    tpi = 8 * torch.atan(torch.tensor(1.0))
    arg1 = tpi / 5
    ca1_ = torch.cos(arg1)
    sa1 = torch.sin(arg1)
    arg2 = 2 * arg1
    ca2_ = torch.cos(arg2)
    sa2 = torch.sin(arg2)
    arg3 = 3 * arg1
    ca3_ = torch.cos(arg3)
    sa3 = torch.sin(arg3)
    arg4 = 4 * arg1
    ca4_ = torch.cos(arg4)
    sa4 = torch.sin(arg4)

    for i in range(lot):
        for j in torch.arange(ido).to(device):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            index3 = (2 * ido + j) * l1 * inc
            index4 = (3 * ido + j) * l1 * inc
            index5 = (4 * ido + j) * l1 * inc
            ca = ca1[j]
            cb = ca2[j]
            cc = ca3[j]
            cd = ca4[j]
            ch1 = c[index1] + c[index2] + c[index3] + c[index4] + c[index5]
            ch2 = c[index1] + c[index2] * ca1_ + c[index3] * ca2_ + c[index4] * ca3_ + c[index5] * ca4_
            ch3 = c[index1] + c[index2] * ca2_ + c[index3] * ca4_ + c[index4] * ca1_ + c[index5] * ca3_
            ch4 = c[index1] + c[index2] * ca3_ + c[index3] * ca1_ + c[index4] * ca4_ + c[index5] * ca2_
            ch5 = c[index1] + c[index2] * ca4_ + c[index3] * ca3_ + c[index4] * ca2_ + c[index5] * ca1_
            ch[index1] = ch1
            ch[index2] = (ch2 - ch5) * sa1
            ch[index3] = (ch3 - ch4) * sa2
            ch[index4] = (ch4 - ch3) * sa3
            ch[index5] = (ch5 - ch2) * sa4
    return c, ch

def radbgm(inc, lot, ido, ip, l1, idl1, c, ch1, ch2, ch3, cg1, cg2, ca1):
    c = torch.tensor(c, dtype=torch.complex64, device=device)
    ch1 = torch.tensor(ch1, dtype=torch.complex64, device=device)
    ch2 = torch.tensor(ch2, dtype=torch.complex64, device=device)
    ch3 = torch.tensor(ch3, dtype=torch.complex64, device=device)
    cg1 = torch.tensor(cg1, dtype=torch.complex64, device=device)
    cg2 = torch.tensor(cg2, dtype=torch.complex64, device=device)
    ca1 = torch.tensor(ca1, dtype=torch.float32, device=device)

    tpi = 8 * torch.atan(torch.tensor(1.0))
    for i in range(lot):
        for j in torch.arange(ido).to(device):
            index = j * l1 * inc
            index1 = j * idl1 * inc
            arg = j * tpi / ip
            ca = torch.cos(arg)
            sa = torch.sin(arg)
            for k in torch.arange(ip).to(device):
                indexk = (index + k * ido * l1)
                indexk1 = (index1 + k * ido)
                ch1_val = c[indexk]
                ch2_val = cg1[indexk1] * ca - cg2[indexk1] * sa
                ch3_val = cg1[indexk1] * sa + cg2[indexk1] * ca
                ch1[indexk] = ch1_val + ch2_val
                ch2[indexk] = ch1_val - ch2_val
                ch3[indexk] = ch3_val
    return c, ch1, ch2, ch3
