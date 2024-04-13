import jieba
import pprint

def join_split(file, split_str):
    """
    将文件内容以split_str分割,并将上一次分割之后录入的字符串合并,返回合并后的字符串
    file: 文件
    :param split_str: 分割字符
    :return: list(str, str ...)
    """
    result = []
    element = ''
    for i in file:
        if split_str not in i:
            element += i
        else:
            element += i
            result.append(element)
            element = ''
    if len(result) == 1:
        print('WARNING: 文件未进行分割,请检查文件内容是否正确')
    return result


if __name__ == '__main__':
    with open('习概题库.txt', 'r', encoding='utf-8') as f:
        data0 = f.readlines()
    questions = join_split(data0, 'D')


    with open('习概.txt', 'r', encoding='gbk') as f:
        data1 = f.readlines()
    ans = join_split(data1, '正确答案')
    for i in range(len(ans)):
        ans[i] = ans[i].strip()
        if i < 10:
            ans[i] = ans[i][1:]
        elif i < 100:
            ans[i] = ans[i][2:]
        elif i < 200:
            ans[i] = ans[i][3:]
        elif i < 210:
            ans[i] = ans[i][1:]
        elif i < 300:
            ans[i] = ans[i][2:]
        elif i < 400:
            ans[i] = ans[i][3:]
    results = []
    for i in questions:
        flag = False
        ques_unit = i.split('】')[1].split('A')[0].strip()
        for j in ans:
            ans_unit = j.split('\n')[0][:-11]
            if ques_unit == ans_unit:
                ques_abcd = i.split('\n')
                anss_abcd = ('A' + j.split('A')[1]).split('\n')
                ans_sort = []
                for k in j.split('正确答案')[1].split('我的答案')[0].strip():
                    for ans_abcd_unit in anss_abcd:
                        if k in ans_abcd_unit:
                            for ques_abcd_unit in ques_abcd:
                                try:
                                    if ans_abcd_unit.split('、')[1].strip() in ques_abcd_unit:
                                        ans_sort.append(ques_abcd_unit.split('、')[0].strip() + ''.join(ans_abcd_unit.split('、')[1:]).strip() + '\n')
                                        break
                                except IndexError:
                                    #print('\033[93mWARNING: 请检查文本格式是否正确\033[0m')
                                    pass
                ans_sort = sorted(ans_sort)
                i += '\n正确答案\n'
                for sorts in ans_sort:
                    i += sorts
                results.append(i)
                flag = True
                break
        if not flag:
            ques_units = jieba.lcut(ques_unit)
            ans_max = ''
            percent_max = 0
            for j in ans:
                percent = 0
                ans_unit = j.split('A')[0].split('\n')[1].strip()[:-11]
                for k in ques_units:
                    if k in ans_unit:
                        percent += (len(k) / len(ques_unit) + len(k) / len(ans_unit)) / 2
                if percent > percent_max:
                    percent_max = percent
                    ans_max = '拟合答案' + j.split('正确答案')[1].split('我的答案')[0].strip() + f'\t\t题目拟合度:{percent_max*100:.2f}%\n'
            i += ans_max
            results.append(i)

    with open('题库(含答案).txt', 'w', encoding='gbk') as f:
        for i in results:
            f.write(i)
            f.write('\n')
        print('题库(含答案).txt 已生成')
