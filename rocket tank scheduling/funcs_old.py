# -*- coding: utf-8 -*-
# @Time : 2021/11/12 22:00
# @Author : hhq
# @File : funcs_old.py
import copy
import scipy.stats as stats
'''实现截尾正态分布'''
import time
import matplotlib.pyplot as plt
import copy
import numpy as np
import xlrd
# 设备编号
Equip_no = {'Milling': [0], 'FW1': [1], 'CHECK': [2, 3], 'CNC_v_l': [4], 'FW2': [5, 6], 'TEST': [7], 'XX_4Dock': [8, 9],
            'XX_5Dock': [10], 'XX_6Dock': [10]}
ma_num = 0
for k, v in Equip_no.items():
    ma_num += len(v)  # 设备总数
JOB_NUM = [13, 8, 5]  # 三种贮箱的任务数量
# jobs = [{'单底': 3, '短壳': 2, '筒段': 1}, {'单底': 2, '短壳': 2, '筒段': 5}, {'单底': 4, '短壳': 2, '筒段': 3}]
jobs = [[1, 2, 3], [5, 2, 2], [3, 2, 4]]  # 三种贮箱对应的筒段、短壳、单底需要数量
Components_time_table = [[4, 8, 4, 6], [4, 8, 4, 6], [24, 8, 24, 8], [146, 16]]  # 单底、短壳、筒段零件的加工时间
# 操作时间服从结尾正态分布，0.8~1.2倍的时间。单底焊接以5%的概率返工
Process = [['Milling', 'FW1', 'CHECK', 'CNC_v_l'], ['Milling', 'FW1', 'CHECK', 'CNC_v_l'],
           ['FW2', 'CHECK', 'CNC_v_l', 'TEST']]
Tube_section = {'Sid': 4}  # 筒段  壁板
Short_jar = {'Sid': 4}  # 短壳  壁板
Single_bottom = {'Mp': 6, 'Tc': 1, 'Pf': 1}  # 单底  瓜瓣，顶盖，型材框
iter_max = 10

pop_size = 10  # 种群规模
mu = 0.8  # 变异概率
cr = 0.1  # 交叉概率
sr = 0.5

def data_process(chrom):
    # comd = copy.deepcopy(chrom)
    re_name_list = []
    li = []
    for i in range(len(chrom)):  # 未扩展的工序序列，26个贮箱按照零件数量均进行扩展
        three_compoents_num = jobs[chrom[i]]   # 三种零件数量
        for num in range(len(three_compoents_num)):
            li += [num+1 for j in range(three_compoents_num[num])]  # 对零件i+1根据其数量扩展
        re_name_list.append([li])  #
# 把每个贮箱看作四种零件，前三种并行，最后一种在前三种完成后进行
# 将原数据扩展成11，12，13...41，42，...的样式。11表示筒段的第一道工序



def new_time_table(job, order_index):  # 可以进行时间表
    new_table = []
    for k, num in enumerate(job):  # 任务索引及其数量
        for i in range(num):
            new_table.append(Components_time_table[k])
    new_table = np.array(new_table)  # 将列表转为数组，方便进行数组内元素顺序调整
    new_table = new_table[order_index]
    return new_table


def stochastic(sto_order):  # 三种零部件分别用0，1，2表示,扩展sto_order
    li = []
    for i in sto_order:
        leng = len(Components_time_table[i])
        li += [i for j in range(leng)]
    sequence = li
    return sequence


def pop_init2(pop_size, job):  # 种群初始化，个体为具体任务（非工序）序列索引即order_index
    pop = np.zeros([pop_size, sum(job)], int)
    for i in range(pop_size):  # 针对个体i
        null = []
        for j in range(len(job)):  # 每种零件
            lo = [j for k in range(job[j])]  # 零件数量
            null += lo  # random_index生成子任务随机排序索引
        np.random.shuffle(null)
        pop[i] = null
    return pop

def pop_init(pop_size, job):  # 种群初始化，个体为具体任务（非工序）序列索引即order_index
    pop = np.zeros([pop_size, sum(job)], int)  # job = [13,8,5]
    for i in range(pop_size):  # 针对个体i计算
        null = []
        lo = []
        for j in range(len(job)):  # 每种零件，len(job)=3
            for k in range(job[j]):  # 零件数量,13,8,5
                lo.append(j)
        null += lo  # random_index生成子任务随机排序索引
        np.random.shuffle(null)
        pop[i] = null
    return pop

def FIT(seq):  # 26个贮箱排序的完工时间，一个染色体的时间  MA_start,MA_end为0时刻
    MA_start = np.zeros(ma_num)  # 全程机器开始时间
    MA_end = np.zeros(ma_num)  # 全程机器结束时间
    CT = np.zeros(len(seq))
    sub_order = []  # 26个贮箱的子零件排序集合
    machine_LIST=[]
    '''加上四种零件的各工序加工位置数组'''
    for i in range(len(seq)):  # jobs是各贮箱关于三类零件的数量的配比
        job = jobs[seq[i]]  # 0代表贮箱四，1代表贮箱五，2代表贮箱六，是代号为i的贮箱的零件数量集合
        s, e, order, ma_list = Sub_time(job, MA_start, MA_end)  # 根据任务——贮箱类型获取最佳子任务（筒段、短壳、单底）的结束时间,第一个零件的第一个工序的开始时间

        s, e, ma_i_list = FULL_time(seq[i], s, e)  # 根据开始时间以及子任务更新各机器时间，同时每单个贮箱的最大结束时间也可据此获得
        '''此处加上各个工序的开始结束时间以及加工位置'''
        # 每种贮箱的完工周期为最后一道工序的结束时间减去最早开始的工序的开始时间
        # CT[i] = max(e) - min(MA_start[0], MA_start[5], MA_start[6])
        CT[i] = max(e)
        sub_order.append(order+[3, 3])
        machine_LIST.append(ma_list+ma_i_list)
        MA_start, MA_end = s, e  # 更新开始时间和结束时间

    return max(MA_end), CT, sub_order, machine_LIST


def Sub_time(job, s, e):  # 获取最佳子任务（筒段、短壳、单底）的结束时间，最佳子排序的时间
    '''
    实现子任务设备上的连续加工，时间累计。如果是第一个子任务，则初始时间为0，否则在前面基础上进行计算
    :param job:
    :param s:
    :param e:
    :return:
    '''
    iter_max = 1  # job为三种贮箱的零件数量
    sub_pop = pop_init(pop_size, job)  # 子任务种群
    best_fit, best_chrom = best_one(sub_pop, s, e)  # 在当前时间表上计算最佳适应度以及最佳染色体（索引排序）

    pop = sub_pop  # 子种群
    # print(pop)
    for iter in range(iter_max):
        ss = copy.deepcopy(s)
        ee = copy.deepcopy(e)
        # 交叉
        for k in range(pop_size):
            if np.random.rand() > 0:
                new_chrom = cross(pop[k], best_chrom)
                pop[k] = new_chrom
        # 变异
        for j in range(pop_size):
            if np.random.rand() < 1:
                new_chrom = mutation(pop[j])
                pop[j] = new_chrom
        new_fit, new_chrom = best_one(pop, ss, ee)
        if new_fit < best_fit:
            best_chrom = new_chrom
            best_fit = new_fit
    ss = copy.deepcopy(s)
    ee = copy.deepcopy(e)
    order = stochastic(best_chrom)
    st, en, ma_list = c_max(order, ss, ee)
    # print(go_end)
    '''返回值加上加工位置列表'''
    return st, en, order, ma_list


def FULL_time(i, s, e):  # # 计算个体中每个贮箱后两道工序的完工时间
    '''
    :param i: i代表贮箱类型，
    :param e: 当前各设备结束时间
    :return: 完成最后两道工序后的结束时间
    '''
    Process = [['XX_4Dock', 'XX_4Dock'], ['XX_5Dock', 'XX_5Dock'], ['XX_6Dock', 'XX_6Dock']]
    LOAD = [146, 16]  # 同台设备测试导致时间急剧累积，产生过长时间
    end_x_y = [0, 0]
    ma_i_list = copy.deepcopy(LOAD)
    # s = copy.deepcopy(e)  # 初始化一个开始时间
    for p in range(len(Process[i])):  # 另一种思路，根据贮箱类型直接计算
        chosen_equip = Equip_no[Process[i][p]]  # 选择的设备
        start = e[chosen_equip]  # 可选设备对应编号集合
        ma_no = chosen_equip[np.argmin(start)]  # 选择的设备对应编号
        # print(chosen_equip)  后两道工序开始时间为总对接设备当前时间和子零件前道工序结束时间的最大值
        if p == 0:
            s[ma_no] = max(e[4], e[ma_no], e[7])
        else:
            s[ma_no] = max(e[4], e[ma_no], e[7], end_x_y[p-1])
        loadtime = LOAD[p]
        e[ma_no] = s[ma_no] + loadtime
        end_x_y[p] = e[ma_no]
        ma_i_list[p] = ma_no
    return s, e, ma_i_list


def best_one(pop, s, e):  # 子任务种群求适应度
    FIT = np.zeros(len(pop))  # 适应度 pop为012集合，代表筒段数、短壳数、单底数，不能直接用于计算适应度，要转化为0000*N0，1111*N1，2222*N2的不同组合
    for i in range(len(pop)):  # 对子任务遍历
        ss = copy.deepcopy(s)
        ee = copy.deepcopy(e)
        # new_table = new_time_table(job, pop[i])  # 按照索引顺序对子任务时间表整理排序
        sequence = stochastic(pop[i])  # 按照索引生成工序扩展序列,pop[i]为0，1，2的不同数量排序
        START, END, ma_list = c_max(sequence, ss, ee)  # 在s,e基础上计算出结束时间
        FIT[i] = max(END)
    best_fit = min(FIT)
    best_chrom = pop[np.argmin(FIT)]
    return best_fit, best_chrom


def mutation(order_index):
    point1 = np.random.randint(0, len(order_index))  # 随机选出0-len(order_index)之间的整数
    point2 = np.random.randint(0, len(order_index))  # 随机选出0-len(order_index)之间的整数
    while point1 > point2 or point1 == point2:
        point1 = np.random.randint(0, len(order_index))
        point2 = np.random.randint(0., len(order_index))
    order_index[point1], order_index[point2] = order_index[point2], order_index[point1]
    return order_index


def cross(a, best):  # 交叉的不再是order_index，而是012组合的交叉

    point1 = np.random.randint(0, len(a))  # 随机选出0-len(a)之间的整数
    point2 = np.random.randint(0, len(a))  # 随机选出0-len(order_index)之间的整数
    c = copy.deepcopy(a)
    d = copy.deepcopy(best)

    while point1 > point2 or point1 == point2:
        point1 = np.random.randint(0, len(a))
        point2 = np.random.randint(0., len(a))
    # fra = [a[point1:point2]]
    c[point1:point2] = d[point1:point2]
    # best[point1:point2] = fra
    form = list(d[:point1])
    back = list(d[point2:])
    whole = form + back
    np.random.shuffle(whole)
    c[:point1] = whole[:len(form)]
    c[point2:] = whole[-len(back):]
    return c


import scipy.stats as stats
'''实现截尾正态分布'''
def normfun(mu, sigma):
    lower, upper = mu - sigma, mu + sigma
    X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    t = X.rvs(1)[0]
    return t

def update_Comp(Components_time_table):  # 将原先的加工时间表改为服从截尾正态分布的时间表
    new_com = copy.deepcopy(Components_time_table)
    for row in range(len(Components_time_table)):
        for li in range(len(Components_time_table[row])):
            mu, sigma = Components_time_table[row][li], Components_time_table[row][li] / 10
            new_com[row][li] = normfun(mu, sigma)
    return new_com

def c_max(order, s, e):  # order根据下文应该是三种任一贮箱零件的排序，表示成001210...  # st表示贮箱号，
    # order为扩展的零件列表，数值表示零件号，出现的次数表示工序数
    START = copy.deepcopy(s)  # 所有设备开始时间
    END = copy.deepcopy(e)  # 设备结束时间
    # P, M, T = [], [], []  统计order不同元素个数，即其中有多少零件或工件
    end_x_y = []  # 各工序的结束时间
    ma_list = copy.deepcopy(order)  # 每个工序的加工位置
    for j in order:
        co_time = copy.deepcopy(Components_time_table[j])  # 零件j各工序的加工时间复制
        end_x_y.append(co_time)
    i = 0
    while i < len(order)-1:
        job = order[i]  # 零件号
        pro_num = len(new_com[job])  # 该任务零件含有的工序数或加工步骤数  new_com为服从截尾正态分布的新时间表
        no_job = list(order[:i + 1]).count(job)  # 该零件出现的次数
        no_job = no_job % pro_num  # 余数为任务的第几个工序
        process_name = Process[job][no_job - 1]  # 工序名，python索引从0开始，减去1为相应工序, -1表示最后一道工序
        machine = Equip_no[process_name]  # 注意python索引，找到对应可选设备编号 Equip_no为字典，{工序：[加工位置1,2,...]}
        start = END[machine]  # 找到可选设备的结束时间
        ma_no = machine[np.argmin(start)]  # 选出加工结束早的设备号，按照python习惯从0开始
        ma_list[i] = ma_no
        # ma_list[]？？？？
        if no_job == 1:
            START[ma_no] = END[ma_no]  # 开始时间
            # if i == 0:
            #     START[ma_no] = END[ma_no]  # 开始时间
            #     go_end.append(START[ma_no])
            # else:
            #     START[ma_no] = END[ma_no]  # 开始时间
        elif no_job == 0:  # 余数为0表示最后一个工序
            no_job += pro_num  # 表示最后一个工序
            START[ma_no] = max(END[ma_no], end_x_y[job][no_job - 2])  # 相应开始时间
        else:
            START[ma_no] = max(END[ma_no], end_x_y[job][no_job - 2])
        END[ma_no] = START[ma_no] + new_com[job][no_job - 1]  # 对应设备的结束时间
        # if i == len(order)-1:
        #     go_end.append(END[ma_no])
        end_x_y[job][no_job - 1] = END[ma_no]  # 该工序的结束时间
        if job == 2 and no_job == 2 and np.random.rand() <= 0.05:
            i -= 1
        else:
            i += 1
    return START, END, ma_list

'''做出画图函数'''
class Cij:
    def __init__(self, name, StartTime, LoadTime):
        self.name = name
        self.StartTime = StartTime
        self.LoadTime = LoadTime
        self.EndTime = StartTime + LoadTime

def PLOT(BC, sub_order, machine_table):  # 做出各工序的加工甘特图,machine_table表示各贮箱的各零件各工序的加工位置
    for i, task in enumerate(BC):  # 对应染色体索引和贮箱类型代号
        order = sub_order[i]  # 对应贮箱零件排序,扩展列表
        for j, part in enumerate(order):  # 遍历零件工序的索引和对应零件类型,part为零件代号
            machine = machine_table[i][j]  # 对应贮箱-零件-工序对应机器集合
            part_num = list(order[:j + 1]).count(part)  # 工序part在扩展列表中出现的次数，表示工序总数
            pro_num = len(new_com[part])  # 该零件工序总数
            operation = part_num % pro_num  # 余数为任务的第几个工序
            load_time = Components_time_table[part][operation - 1]  # 遍历此零件的加工时间列表,获取工序索引和时间
            locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)] = Cij(name='c{}_{}_{}_{}'.format(i, part, operation, machine), StartTime=0, LoadTime=load_time, )

    load_time_tables = []  # 加工工序，开始，结束时间
    M_time = np.zeros(ma_num)  # 初始化所有机器时间为0
    for i, _ in enumerate(BC):
        order = sub_order[i]
        for j, part in enumerate(order):  # 遍历零件索引及其类型
            machine = machine_table[i][j]  # 对应贮箱-零件-工序对应机器集合
            part_num = list(order[:j + 1]).count(part)  # 工序part在扩展列表中出现的次数，表示工序总数
            pro_num = len(new_com[part])  # 该零件工序总数
            operation = part_num % pro_num  # 余数为任务的第几个工序
            # load_time = Components_time_table[part][operation - 1]  # 遍历此零件的加工时间列表,获取工序索引和时间
            # 开始时间
            if operation == 1:
                locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].StartTime = M_time[machine]
            elif operation == 0:
                locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].StartTime = max(
                    M_time[machine],  # 该工序所在加工位置机器的时间
                    locals()['c{}_{}_{}_{}'.format(i, part, operation + pro_num - 2, machine)].EndTime)  # 该工序的前道工序的完工时间
            else:
                locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].StartTime = max(
                    M_time[machine],  # 该工序所在加工位置机器的时间
                    locals()['c{}_{}_{}_{}'.format(i, part, operation-2, machine)].EndTime)  # 该工序的前道工序的完工时间
            # 结束时间
            locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].EndTime = \
                locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].StartTime + locals()[
                    'c{}_{}_{}_{}'.format(i, part, operation, machine)].LoadTime
            M_time[machine] = locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].EndTime
            load_time_tables.append([locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].name,  # 每次结束均记录（工序名，机器号），开始，结束
                                     [locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].StartTime,
                                      locals()['c{}_{}_{}_{}'.format(i, part, operation, machine)].EndTime]])

    T = []

    for i in load_time_tables:
        T.append(i[-1][-1])

    return load_time_tables, max(T)  # load_time_tables 代表所有工件每个工序加工位置及其开始和结束加工时间


def BEST_ONE(pop):  # 最佳贮箱完工排序个体
    FS = np.zeros(len(pop))  # 每个个体适应度初始化集合
    BC = pop[0]  # 最佳个体
    BF, _, sub_order, MA_LIST = FIT(BC)  # 最佳个体适应度
    for i in range(1, len(pop)):
        FS[i], _, sub, best_list = FIT(pop[i])  # 个体适应度
        if FS[i] < BF:  # 更好的染色体
            BF = FS[i]
            BC = pop[i]
            sub_order = sub
            MA_LIST = best_list
    return BF, BC, sub_order, MA_LIST



new_com = update_Comp(Components_time_table)

def ave_cycle(chrom, v):
    kind_num = len(JOB_NUM)
    ave_list = np.zeros(kind_num)
    for i in range(len(chrom)):
        ave_list[chrom[i]] += v[i]
    for j in range(len(JOB_NUM)):
        ave_list[j] /= JOB_NUM[j]

    return ave_list




# 读取excel文件
def excel():
    wb = xlrd.open_workbook('加工列表.xls')  # 打开Excel文件
    sheet = wb.sheets()[0]  # 通过excel表格名称(rank)获取工作表
    dat = []  # 创建空list
    for a in range(1, sheet.nrows):  # 循环读取表格内容（每次读取一行数据）
        # cells = sheet.row_values(a)  # 每行数据赋值给cells
        data = sheet.row_values(a)[0][3]  # 因为表内可能存在多列数据，0代表第一列数据，1代表第二列，以此类推
        dat.append(data)  # 把每次循环读取的数据插入到list
    return dat