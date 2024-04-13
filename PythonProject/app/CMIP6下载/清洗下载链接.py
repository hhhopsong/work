
fw = open("C:\\Users\\10574\\Desktop\\gov-url.txt", 'w') # 创建url保存文件
with open("C:\\Users\\10574\\Desktop\\gov.txt", 'r') as fr: # 读取所有下载链接信息
    for line in fr.readlines(): # 按行读取
        line = line.strip('\n').split(' ') # 去掉换行符并分割
        url = line[1].replace("'", '') # 清洗并获取待下载地址
        fw.writelines(url + '\n') # 将下载地址写入保存文件中
fw.close() # 关闭文件
