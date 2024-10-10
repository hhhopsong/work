# Coding By Danyhug at 2022-6-10 15:45:28
# 方正教务成绩查询
# https://github.com/Danyhug/zfCCJ
import time
import requests
import os
from selenium import webdriver as web

# 书写你的cookie
cookie = ''
# 书写教学综合信息服务平台的域名，我们学校的是http://jw.xxxxx.edu.cn:8111
domain = 'https://nxdyjs.nuist.edu.cn/gmis5/student/'


try:
    os.mkdir('pdf')
except:
    pass
finally:
    os.chdir('pdf')

def query(stuID: int, **kwargs) -> bool:
    """
    查询成绩
    :param num: 功能选择，1或2
    :param stuID: 学号
    :param kwargs: 起始区间学号
    :return: 返回是否查询成功
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36 Edg/129.0.0.0",
        'cookie': 'csrftoken=YAqSc1ecfpzrFw9A56i0JLjmhqSbTcc8K0ugGwI1qB4MyJoBHY18VERoQBfckDMj; ASP.NET_SessionId=lgk0u5v2hjzlgjlnjlvawcfg; iPlanetDirectoryPro=DfZVuEmnrLeRu6qqtjXyceX4GDnjr704; .ASPXAUTH=9222EE3EDEE93860CF8D0396BC04E0DF5BD1C46B261DC898730E7EED0C1577E86458DAD2FC27DE6B42FB145C7518F8C2B6D270316B786C4DC26B72F06469E6EB8FF9A8B35407451C11789724072C423C07B894868C85695C56B8C1540E08E560158D0027DC47F4F2152B8CD11A186DA811F74D601B0F85C5F9CB0611C06D4959E5D51F5FDCB8AC8DA3BB4B33EA2B6E44AA5B068A105814BEE5578F6C007C589160669B47D468F3C3E91D00C2E6744D1CBDE89E4965CA555D58544603CC40EA25EF534A1175936632E5A2356663AD40918075FDA213A86AB732981A53EFB0DA4437A96DB2C5F9C0DA14F62A4A3B436216B40342BF84868D33640D763603343852; __SINDEXCOOKIE__=c2e502fe7bc457b6934d28c4e13d25bf'
    }
    url = domain + 'yggl/kwhdbm'
    '''data = {
        'gsdygx': '12787-zw-mrgs',
        'ids': stuID,  # 学号
        'dyrq': '2002-03-28',
        'btmc': '学业成绩单',
        'cjdySzxs': 'dyrq,btmc,bwnr,dyfsdkc,xdydjksxm,bjgbdyxxkcxz'
    }'''

    # 拼接url
    url = url.format(stuID)
    res = requests.post(url=url,headers=headers).text
    print(res)
    # 点击报名按钮
    btn = web.find_element_by_xpath('//*[@id="content"]/')
    res = requests.get(btn, headers=headers).text
    print(btn)

query(0)
'''
    if res.find('成功') != -1:
        # 20105010550.pdf
        tempname = res.split('\\')
        downloadName = tempname[1] + tempname[2] + tempname[3].split('#')[0]

        url = '{}{}'.format(domain, downloadName)
        fileBlob = requests.get(url, headers=headers).content
        with open('{}.pdf'.format(stuID), 'wb+') as f:
            f.write(fileBlob)
            print('写入成功')
    else:
        print('获取成绩出错', url, res)
        return False


while True:
    print('选择功能：')
    print('1. 查询某同学成绩')
    print('2. 查询某区间成绩')

    n = input('我选择功能为：')

    if n == '1':
        print('已选择查询某同学成绩')

        # 学号
        stuID = int(input('请输入该同学的学号：'))
        query(stuID=stuID)

    elif n == '2':
        print('已选择查询某区间成绩')

        start = int(input('输入区间学号开始值：'))
        end = int(input('输入区间学号结束值：')) + 1

        # 循环查询
        for id in range(start, end):
            print('当前查询的学号为', id)
            time.sleep(2)
            query(stuID=id)
'''