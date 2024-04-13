import matplotlib.pyplot as plt
import numpy as np
from array import array
from matplotlib import colors
import tkinter as tk
from tkinter import filedialog


def main():

    file = openFile()
    k = int(input('请选择仰角 1 3 5 6 7 8 9 10 11：'))
    el, az, rl, dbz = saDecoder(file,k)
    el, az, rl, dbz = dataLink(el, az, rl, dbz)
    x, y, h = sph2cord(el, az, rl)
    plotFunction(x, y, dbz, k)


def openFile():

    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename()


def saDecoder(file, k):

    f = open(file, 'rb')
    data = np.asarray(array('B', f.read()))
    data = data.reshape(len(data)//2432, 2432)

    if data[0,72] == 11:
        phi = [0.50, 0.50, 1.45, 1.45, 2.40, 3.35, 4.30, 5.25, 6.2, 7.5, 8.7, 10, 12, 14, 16.7, 19.5]
    if data[0, 72] == 21:
        phi = [0.50, 0.50, 1.45, 1.45, 2.40, 3.35, 4.30, 6.00, 9.00, 14.6, 19.5]
    if data[0, 72] == 31:
        phi = [0.50, 0.50, 1.50, 1.50, 2.50, 2.50, 3.50, 4.50]
    if data[0, 72] == 32:
        phi = [0.50, 0.50, 2.50, 3.50, 4.50]

    el = np.zeros((len(data), 460))  #仰角
    az = np.zeros((len(data), 460))  #方位角
    rl = np.zeros((len(data), 460))  #径向长度
    dbz = np.zeros((len(data), 460))  #反射率

    count = 0
    while count < len(data):
        el_number = data[count,44] + 256 * data[count,45] #仰角序数
        az_value = (data[count,36] + 256 * data[count,37]) / 8 * 180 / 4096  #方位角
        d_value = data[count,54] + 256 * data[count,55] #库长

        if d_value == 0:
            count += 1
            continue
        else:
            count += 1

        i = 0
        while i < 460:
            el[count-1, i] = phi[el_number-1]
            az[count-1, i] = az_value
            rl[count-1, i] = i + 1
            #计算反射率
            if i > d_value:
                dbz[count-1, i] = -9900
            else:
                if data[count-1, 128+i] == 0:  #无回波数据
                    dbz[count-1, i] = -9900
                elif data[count-1, 128+i] == 1:  #距离模糊
                    dbz[count-1, i] = -9901
                else:
                    dbz[count-1, i] = (data[count-1, 128+i] - 2) / 2 - 32
            i += 1

    m = 1
    while m < len(data):
        if data[m,44] > (k-1):
            break
        m += 1
    n = m
    while n < len(data):
        if data[n,44] > k:
            break
        n += 1



    elVlues = el[m:n,0:230]  #对应第k个仰角的仰角值
    azValues = az[m:n,0:230] #对应的方位角
    rlValues = rl[m:n,0:230] #对应的径向长度
    dbzValues = dbz[m:n,0:230] #对应的回波强度

    return elVlues, azValues, rlValues, dbzValues


def sph2cord(el, az, r):

    e, a = np.deg2rad([el, az])
    x = r * np.cos(e) * np.sin(a)
    y = r * np.cos(e) * np.cos(a)
    h = r * np.sin(e)

    return x, y, h


def plotFunction(x, y, dbz, k):

    phi = [0.50, 0.50, 1.45, 1.45, 2.40, 3.35, 4.30, 6.00, 9.00, 14.6, 19.5]
    cdict = ['#606060', '#01ADA5', '#C0C0FE', '#7B72EF', '#1F27D1',
             '#A6FDA8', '#00EA00', '#10921A', '#FCF465', '#C9C903', '#8C8C00',
             '#FFACAC', '#FE655C', '#EE0231', '#D58FFE', '#AA25FA', '#FFFFFF']
    cmap = colors.ListedColormap(cdict)
    norm = colors.Normalize(vmin=-15,vmax=70)
    x = np.concatenate((x, [x[0]]))  # 闭合
    y = np.concatenate((y, [y[0]]))  # 闭合
    plt.pcolor(x,y,dbz,norm=norm,cmap=cmap)
    plt.title('Reflectivity'+'('+str(phi[k-1])+')')
    plt.axis('square')
    plt.colorbar()
    plt.savefig(str(phi[k-1])+'.png')
    plt.show()

def dataLink(el,az,rl,dbz):

    El = np.zeros((360, 230))
    Az = np.zeros((360, 230))
    Rl = np.zeros((360, 230))
    DBZ = np.zeros((360, 230))

    for i in range(0, 361):
        err = np.abs(az[:, 0] - i)
        id = np.argmin(err)
        El[i-1,:] = el[id,:]
        Az[i-1,:] = az[id,:]
        Rl[i-1,:] = rl[id,:]
        DBZ[i-1,:] = dbz[id,:]

    return El, Az, Rl, DBZ


if __name__ == '__main__':
    main()