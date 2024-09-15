import numpy as np
import math


def fft99x(g, z, trigs, ifax, inc, jump, n, lot, isign):
    new_g = list(g)
    new_z = list(z)
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
        for i in range(n):
            la = (i * inc) + 1
            lb = (i * incn)
            for k in range(lot):
                new_z[lb + k] = new_g[la - 1]
                la = la + jump
        new_z = np.array(new_z)
        new_z = np.fft.fft(new_z.reshape(-1, incn).transpose())
        new_z = new_z.flatten()
        for k in range(lot):
            new_g[k] = new_z[k]
            new_g[k+incn] = new_z[k+incn*(n - 1)]
        for i in range(2, n):
            lb = (i * incn)
            for k in range(lot):
                new_g[lb + k] = new_z[lb + k - incn]
        for i in range(n):
            la = (i * inc) + 1
            lb = (i * incn)
            for k in range(lot):
                new_z[la - 1] = new_g[lb + k]
                la = la + jump
        return new_g, new_z
    else:
        for i in range(n):
            la = (i * inc) + 1
            lb = (i * incn)
            for k in range(lot):
                new_g[lb + k] = new_z[la - 1]
                la = la + jump
        for k in range(lot):
            new_z[k] = new_g[k]
            new_z[k+incn*(n - 1)] = new_g[k+incn]
        for i in range(2, n):
            lb = (i * incn)
            for k in range(lot):
                new_z[lb + k - incn] = new_g[lb + k]
        new_z = np.array(new_z)
        new_z = np.fft.ifft(new_z.reshape(-1, incn).transpose())
        new_z = new_z.flatten()
        for i in range(n):
            la = (i * inc) + 1
            lb = (i * incn)
            for k in range(lot):
                new_g[la - 1] = new_z[lb + k]
                la = la + jump
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
    na, c, wa, ifac, ch = rftf9m(n, inc, lot, None, c, wa, ifac, ch)
    cf = 1.0 / n
    n4 = (n // 4) * 4
    if na == 1:
        if n4 >= 4:
            for k in range(1, n4, 4):
                for l in range(lot):
                    c[l][k] = cf * c[l][k]
                    c[l][k + 1] = cf * c[l][k + 1]
                    c[l][k + 2] = cf * c[l][k + 2]
                    c[l][k + 3] = cf * c[l][k + 3]
        if n4!= n:
            for k in range(n4 + 1, n):
                for l in range(lot):
                    c[l][k] = cf * c[l][k]
    else:
        if n4 >= 4:
            for k in range(1, n4, 4):
                for l in range(lot):
                    c[l][k] = cf * ch[l][k]
                    c[l][k + 1] = cf * ch[l][k + 1]
                    c[l][k + 2] = cf * ch[l][k + 2]
                    c[l][k + 3] = cf * ch[l][k + 3]
        if n4!= n:
            for k in range(n4 + 1, n):
                for l in range(lot):
                    c[l][k] = cf * ch[l][k]
    return c, wa, ifac, ch


def rftf1m(n, inc, lot, c, wa, ifac, ch):
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
    for i in range(lot):
        for j in range(ido):
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
    for i in range(lot):
        for j in range(ido):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            ca = ca1[j]
            ch1 = c[i][index1]+c[i][index2]
            ch2 = (c[i][index1]-c[i][index2]) * ca
            ch[i][index1] = ch1
            ch[i][index2] = ch2
    return c, ch


def radf3m(inc, lot, ido, l1, c, ch, ca1, ca2):
    tpi = 8 * np.arctan(1)
    arg1 = tpi / 3
    ca1_ = np.cos(arg1)
    sa1 = np.sin(arg1)
    ca2_ = np.cos(2 * arg1)
    sa2 = np.sin(2 * arg1)
    for i in range(lot):
        for j in range(ido):
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
    tr11 = (-1.0 + math.sqrt(5.0)) / 4.0
    ti11 = math.sqrt(1.0 - tr11 * tr11)
    tr12 = (-1.0 - math.sqrt(5.0)) / 4.0
    ti12 = math.sqrt(1.0 - tr12 * tr12)

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
            for i in range(2, ido, 2):
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
            for i in range(2, ido, 2):
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
            for i in range(2, ido, 2):
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
    tpi = 8.0 * math.atan(1.0)
    arg = tpi / ip
    dcp = math.cos(arg)
    dsp = math.sin(arg)
    ipph = (ip + 1) // 2
    ipp2 = ip + 2
    idp2 = ido + 2

    if ido == 1:
        for ik in range(idl1):
            for l in range(lot):
                ch2[l][ik][0] = c2[l][ik][0]
        for j in range(1, ip):
            for k in range(l1):
                for l in range(lot):
                    ch[l][0][k][j] = c1[l][0][k][j]
        for j in range(1, ip):
            for k in range(l1):
                for i in range(2, ido, 2):
                    idij = i - 1
                    for l in range(lot):
                        ch[l][i - 1][k][j] = wa[idij - 1] * c1[l][i - 1][k][j] + wa[idij] * c1[l][i][k][j]
                        ch[l][i][k][j] = wa[idij - 1] * c1[l][i][k][j] - wa[idij] * c1[l][i - 1][k][j]
        for j in range(1, ipph):
            jc = ipp2 - j
            for k in range(l1):
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
        for k in range(l1):
            for i in range(ido):
                for l in range(lot):
                    cc[l][i][0][k] = ch[l][i][k][0]
        for j in range(1, ipph):
            jc = ipp2 - j
            j2 = j * 2
            for k in range(l1):
                for l in range(lot):
                    cc[l][ido - 1][j2 - 2][k] = ch[l][0][k][j]
                    cc[l][0][j2 - 1][k] = ch[l][0][k][jc]
        return cc, c1, c2, ch, ch2, wa
    else:
        for ik in range(idl1):
            for l in range(lot):
                ch2[l][ik][0] = c2[l][ik][0]
        for j in range(1, ip):
            for k in range(l1):
                for l in range(lot):
                    ch[l][0][k][j] = c1[l][0][k][j]
        is_ = -ido
        for j in range(1, ip):
            is_ = is_ + ido
            for k in range(l1):
                for i in range(2, ido, 2):
                    idij = is_ + i - 1
                    for l in range(lot):
                        ch[l][i - 1][k][j] = wa[idij - 1] * c1[l][i - 1][k][j] + wa[idij] * c1[l][i][k][j]
                        ch[l][i][k][j] = wa[idij - 1] * c1[l][i][k][j] - wa[idij] * c1[l][i - 1][k][j]
        for j in range(1, ipph):
            jc = ipp2 - j
            for k in range(l1):
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
        for k in range(l1):
            for i in range(ido):
                for l in range(lot):
                    cc[l][i][0][k] = ch[l][i][k][0]
        for j in range(1, ipph):
            jc = ipp2 - j
            j2 = j * 2
            for k in range(l1):
                for l in range(lot):
                    cc[l][ido - 1][j2 - 2][k] = ch[l][0][k][j]
                    cc[l][0][j2 - 1][k] = ch[l][0][k][jc]
        for j in range(1, ipph):
            jc = ipp2 - j
            j2 = j * 2
            for k in range(l1):
                for i in range(2, ido, 2):
                    ic = idp2 - i
                    for l in range(lot):
                        cc[l][i - 1][j2 - 1][k] = ch[l][i - 1][k][j] + ch[l][i - 1][k][jc]
                        cc[l][i][j2 - 1][k] = ch[l][i][k][j] + ch[l][i][k][jc]
                        cc[l][ic - 1][j2 - 2][k] = ch[l][i - 1][k][j] - ch[l][i - 1][k][jc]
                        cc[l][ic][j2 - 2][k] = ch[l][i][k][jc] - ch[l][i][k][j]
        return cc, c1, c2, ch, ch2, wa



def rfftim(n, trigs, ifax):
    new_trigs = list(trigs)
    new_ifax = list(ifax)
    if n == 1:
        return new_trigs, new_ifax
    new_trigs, new_ifax = rfti1m(n, new_trigs, new_ifax)
    return new_trigs, new_ifax


def rfti1m(n, wa, ifac):
    new_wa = list(wa)
    new_ifac = list(ifac)
    ntryh = [4, 2, 3, 5]
    tpi = 8 * np.arctan(1)
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
            new_ifac[nf+2] = ntry
            nl = nq
            if nl > 1:
                continue
            break
        else:
            continue
    for i in range(2, nf):
        if new_ifac[i+2] == 2:
            new_ifac[i+2] = 4
            new_ifac[3] = 2
    new_ifac[1] = n
    new_ifac[2] = nf
    if nf == 1:
        return new_wa, new_ifac
    l1 = 1
    is_ = 0
    for k in range(1, nf):
        ip = new_ifac[k+2]
        ido = n // (l1 * ip)
        for j in range(1, ip):
            arggld = (j * l1) * (tpi / n)
            for ifi in range(1, (ido - 1) // 2 + 1):
                arg = ifi * arggld
                new_wa[2*ifi+is_ - 1] = np.cos(arg)
                new_wa[2*ifi+is_] = np.sin(arg)
            is_ = is_ + ido
        l1 = l1 * ip
    return new_wa, new_ifac


def rfftbm(n, inc, jump, lot, r, wa, ifac, wsave):
    new_r = list(r)
    new_wa = list(wa)
    new_ifac = list(ifac)
    new_wsave = list(wsave)
    if n == 1:
        return new_r, new_wa, new_ifac, new_wsave
    if jump!= 1:
        new_r, new_wa, new_ifac, new_wsave = rftb2m(n, inc, jump, lot, new_r, new_wa, new_ifac, new_wsave)
    else:
        new_r, new_wa, new_ifac, new_wsave = rftb1m(n, inc, lot, new_r, new_wa, new_ifac, new_wsave)
    return new_r, new_wa, new_ifac, new_wsave


def rftb2m(n, inc, jump, lot, r, wa, ifac, wsave):
    new_r = list(r)
    new_wa = list(wa)
    new_ifac = list(ifac)
    new_wsave = list(wsave)
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
                new_wsave[ja] = new_r[ia]
                new_wsave[jb] = new_r[ib]
                new_wsave[jc] = new_r[ic]
                new_wsave[jd] = new_r[id]
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
                new_wsave[ja] = new_r[ia]
                ia = ia + jump
                ja = ja + 1
            iabase = iabase + inc
            jabase = jabase + incn
    new_r, new_wa, new_ifac, new_wsave = rftb1m(n, incn, lot, new_wsave, new_wa, new_ifac, new_r)
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
                new_r[ia] = new_wsave[ja]
                new_r[ib] = new_wsave[jb]
                new_r[ic] = new_wsave[jc]
                new_r[id] = new_wsave[jd]
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
                new_r[ia] = new_wsave[ja]
                ia = ia + jump
                ja = ja + 1
            iabase = iabase + inc
            jabase = jabase + incn
    return new_r, new_wa, new_ifac, new_wsave


def rftb1m(n, inc, lot, c, wa, ifac, ch):
    new_c = [list(sub_list) for sub_list in c]
    new_wa = list(wa)
    new_ifac = list(ifac)
    new_ch = [list(sub_list) for sub_list in ch]
    na = rftb9m(n, inc, lot, None, new_c, new_wa, new_ifac, new_ch)
    if na == 1:
        return new_c, new_wa, new_ifac, new_ch
    n4 = (n // 4) * 4
    if n4 >= 4:
        for k in range(0, n4, 4):
            for l in range(lot):
                new_c[l][k] = new_ch[l][k]
                new_c[l][k+1] = new_ch[l][k+1]
                new_c[l][k+2] = new_ch[l][k+2]
                new_c[l][k+3] = new_ch[l][k+3]
    if n4!= n:
        for k in range(n4, n):
            for l in range(lot):
                new_c[l][k] = new_ch[l][k]
    return new_c, new_wa, new_ifac, new_ch


def rftb9m(n, inc, lot, na, c, wa, ifac, ch):
    new_c = [list(sub_list) for sub_list in c]
    new_wa = list(wa)
    new_ifac = list(ifac)
    new_ch = [list(sub_list) for sub_list in ch]
    nf = new_ifac[2]
    na = 1
    l1 = 1
    iw = 1
    for k1 in range(1, nf):
        ip = new_ifac[k1+2]
        l2 = ip * l1
        ido = n // l2
        idl1 = ido * l1
        na = 1 - na
        if ip == 4:
            ix2 = iw + ido
            ix3 = ix2 + ido
            if na == 0:
                new_c, new_ch = radb4m(inc, lot, ido, l1, new_c, new_ch, new_wa[iw - 1], new_wa[ix2 - 1], new_wa[ix3 - 1])
            else:
                new_c, new_ch = radb4m(inc, lot, ido, l1, new_ch, new_c, new_wa[iw - 1], new_wa[ix2 - 1], new_wa[ix3 - 1])
        elif ip == 2:
            if na == 0:
                new_c, new_ch = radb2m(inc, lot, ido, l1, new_c, new_ch, new_wa[iw - 1])
            else:
                new_c, new_ch = radb2m(inc, lot, ido, l1, new_ch, new_c, new_wa[iw - 1])
        elif ip == 3:
            ix2 = iw + ido
            if na == 0:
                new_c, new_ch = radb3m(inc, lot, ido, l1, new_c, new_ch, new_wa[iw - 1], new_wa[ix2 - 1])
            else:
                new_c, new_ch = radb3m(inc, lot, ido, l1, new_ch, new_c, new_wa[iw - 1], new_wa[ix2 - 1])
        elif ip == 5:
            ix2 = iw + ido
            ix3 = ix2 + ido
            ix4 = ix3 + ido
            if na == 0:
                new_c, new_ch = radb5m(inc, lot, ido, l1, new_c, new_ch, new_wa[iw - 1], new_wa[ix2 - 1], new_wa[ix3 - 1], new_wa[ix4 - 1])
            else:
                new_c, new_ch = radb5m(inc, lot, ido, l1, new_ch, new_c, new_wa[iw - 1], new_wa[ix2 - 1], new_wa[ix3 - 1], new_wa[ix4 - 1])
        else:
            if na == 0:
                new_c, new_ch = radbgm(inc, lot, ido, ip, l1, idl1, new_c, new_c, new_c, new_ch, new_ch, new_wa[iw - 1])
                na = 1
            else:
                new_c, new_ch = radbgm(inc, lot, ido, ip, l1, idl1, new_ch, new_ch, new_ch, new_c, new_c, new_wa[iw - 1])
                na = 0
            if ido == 1:
                na = 1


def radb4m(inc, lot, ido, l1, c, ch, ca1, ca2, ca3):
    new_c = [list(sub_list) for sub_list in c]
    new_ch = [list(sub_list) for sub_list in ch]
    for i in range(lot):
        for j in range(ido):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            index3 = (2 * ido + j) * l1 * inc
            index4 = (3 * ido + j) * l1 * inc
            ca = ca1[j]
            cb = ca2[j]
            cc = ca3[j]
            ch11 = new_c[i][index1] + new_c[i][index3]
            ch12 = new_c[i][index1] - new_c[i][index3]
            ch21 = new_c[i][index2] + new_c[i][index4]
            ch22 = new_c[i][index2] - new_c[i][index4]
            new_ch[i][index1] = ch11 + ch21
            new_ch[i][index3] = ch11 - ch21
            new_ch[i][index2] = (ch12 - ch22) * ca+(ch12 + ch22) * cb
            new_ch[i][index4] = (ch12 - ch22) * cb-(ch12 + ch22) * ca
    return new_c, new_ch


def radb2m(inc, lot, ido, l1, c, ch, ca1):
    new_c = [list(sub_list) for sub_list in c]
    new_ch = [list(sub_list) for sub_list in ch]
    for i in range(lot):
        for j in range(ido):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            ca = ca1[j]
            ch1 = new_c[i][index1]+new_c[i][index2]
            ch2 = (new_c[i][index1]-new_c[i][index2]) * ca
            new_ch[i][index1] = ch1
            new_ch[i][index2] = ch2
    return new_c, new_ch


def radb3m(inc, lot, ido, l1, c, ch, ca1, ca2):
    new_c = [list(sub_list) for sub_list in c]
    new_ch = [list(sub_list) for sub_list in ch]
    tpi = 8 * np.arctan(1)
    arg1 = tpi / 3
    ca1_ = np.cos(arg1)
    sa1 = np.sin(arg1)
    ca2_ = np.cos(2 * arg1)
    sa2 = np.sin(2 * arg1)
    for i in range(lot):
        for j in range(ido):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            index3 = (2 * ido + j) * l1 * inc
            ca = ca1[j]
            cb = ca2[j]
            ch1 = new_c[i][index1]+new_c[i][index2]+new_c[i][index3]
            ch2 = new_c[i][index1]+new_c[i][index2]*ca1_+new_c[i][index3]*ca2_
            ch3 = new_c[i][index1]+new_c[i][index2]*ca2_+new_c[i][index3]*ca1_
            new_ch[i][index1] = ch1
            new_ch[i][index2] = (ch2 - ch3) * sa1
            new_ch[i][index3] = (ch3 - ch2) * sa2
    return new_c, new_ch


def radb5m(inc, lot, ido, l1, c, ch, ca1, ca2, ca3, ca4):
    new_c = [list(sub_list) for sub_list in c]
    new_ch = [list(sub_list) for sub_list in ch]
    tpi = 8 * np.arctan(1)
    arg1 = tpi / 5
    ca1_ = np.cos(arg1)
    sa1 = np.sin(arg1)
    arg2 = 2 * arg1
    ca2_ = np.cos(arg2)
    sa2 = np.sin(arg2)
    arg3 = 3 * arg1
    ca3_ = np.cos(arg3)
    sa3 = np.sin(arg3)
    arg4 = 4 * arg1
    ca4_ = np.cos(arg4)
    sa4 = np.sin(arg4)
    for i in range(lot):
        for j in range(ido):
            index1 = j * l1 * inc
            index2 = (ido + j) * l1 * inc
            index3 = (2 * ido + j) * l1 * inc
            index4 = (3 * ido + j) * l1 * inc
            index5 = (4 * ido + j) * l1 * inc
            ca = ca1[j]
            cb = ca2[j]
            cc = ca3[j]
            cd = ca4[j]
            ch1 = new_c[i][index1]+new_c[i][index2]+new_c[i][index3]+new_c[i][index4]+new_c[i][index5]
            ch2 = new_c[i][index1]+new_c[i][index2]*ca1_+new_c[i][index3]*ca2_+new_c[i][index4]*ca3_+new_c[i][index5]*ca4_
            ch3 = new_c[i][index1]+new_c[i][index2]*ca2_+new_c[i][index3]*ca4_+new_c[i][index4]*ca1_+new_c[i][index5]*ca3_
            ch4 = new_c[i][index1]+new_c[i][index2]*ca3_+new_c[i][index3]*ca1_+new_c[i][index4]*ca4_+new_c[i][index5]*ca2_
            ch5 = new_c[i][index1]+new_c[i][index2]*ca4_+new_c[i][index3]*ca3_+new_c[i][index4]*ca2_+new_c[i][index5]*ca1_
            new_ch[i][index1] = ch1
            new_ch[i][index2] = (ch2 - ch5) * sa1
            new_ch[i][index3] = (ch3 - ch4) * sa2
            new_ch[i][index4] = (ch4 - ch3) * sa3
            new_ch[i][index5] = (ch5 - ch2) * sa4
    return new_c, new_ch


def radbgm(inc, lot, ido, ip, l1, idl1, c, ch1, ch2, ch3, cg1, cg2, ca1):
    new_c = [list(sub_list) for sub_list in c]
    new_ch1 = [list(sub_list) for sub_list in ch1]
    new_ch2 = [list(sub_list) for sub_list in ch2]
    new_ch3 = [list(sub_list) for sub_list in ch3]
    new_cg1 = [list(sub_list) for sub_list in cg1]
    new_cg2 = [list(sub_list) for sub_list in cg2]
    tpi = 8 * np.arctan(1)
    for i in range(lot):
        for j in range(ido):
            index = j * l1 * inc
            index1 = j * idl1 * inc
            arg = j * tpi / ip
            ca = np.cos(arg)
            sa = np.sin(arg)
            for k in range(ip):
                indexk = (index + k * ido * l1)
                indexk1 = (index1 + k * ido)
                ch1 = new_c[i][indexk]
                ch2 = new_cg1[i][indexk1] * ca - new_cg2[i][indexk1] * sa
                ch3 = new_cg1[i][indexk1] * sa + new_cg2[i][indexk1] * ca
                new_ch1[i][indexk] = ch1+ch2
                new_ch2[i][indexk] = ch1 - ch2
                new_ch3[i][indexk] = ch3
    return new_c, new_ch1, new_ch2, new_ch3

