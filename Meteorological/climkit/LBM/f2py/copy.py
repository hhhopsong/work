def copy(datai, idim):
    """
    这个Fortran子例程COPY的功能是将输入数组DATAI中的元素逐个复制到输出数组DATAO中，数组的大小由IDIM指定。
    :param datai:
    :param idim:
    :return: datao
    """
    datao = [0]*idim
    for i in range(idim):
        datao[i]=datai[i]
    return datao
