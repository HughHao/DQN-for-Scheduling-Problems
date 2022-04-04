# -*- coding: utf-8 -*-
# @Time : 2021/10/12 16:56
# @Author : hhq
# @File : jifen.py

# 积分计算
# from scipy import integrate
# def f(x):
#     return x + 1
# v, err = integrate.quad(f, 1, 2)
# print(v)
# def sum_job(jobs):
#     sum = 0
#     for i in jobs:
#         sum += i['数量']
#     return sum
'''
进行贮箱车间调度的运筹优化
'''
'''# 初始一个加工顺序
li = []
for i, u in enumerate(JOB_NUM):
    li += (np.ones([1, u], int) * (i+1)).tolist()
INIT_ORDER = []
for j in li:  # 将列表中各工件独立列表加起来
    INIT_ORDER += j'''

import numpy as np
# 设备数据
Milling = [0]  # 铣边
FW1 = [1]  # 焊接1
CHECK = [2, 3]  # 无损检测区
CNC_v_l = [4]  # 数控立车
FW2 = [5, 6]  # 焊接2
TEST = [7]  # 试验
XX_4Dock = [8, 9]  # XX_4的总对接
XX_56Dock = [10]  # XX_5,6的总对接
Tube_section = {'Sid': 4}  # 筒段  壁板
Short_jar = {'Sid': 4}  # 短壳  壁板
Single_bottom = {'Mp': 6, 'Tc': 1, 'Pf': 1}  # 单底  瓜瓣，顶盖，型材框

JOB_NUM = [13, 8, 5]  # 三种贮箱的任务数量
# jobs = [{'单底': 3, '短壳': 2, '筒段': 1}, {'单底': 2, '短壳': 2, '筒段': 5}, {'单底': 4, '短壳': 2, '筒段': 3}]
jobs = [[3, 2, 1], [2, 2, 5], [4, 2, 3]]  # 三种贮箱对应的单底、短壳、筒段需要数量
# 以上零件每种都有数十件，每完成一件贮箱的配套即可进行下面的流程加工
# 贮箱配套——》贮箱总对接|检测——》试验——贮箱
Sb_num = sum([JOB_NUM[i] * jobs[i][0] for i in range(3)])  # 三种贮箱单底数量 代号0
Sj_num = sum([JOB_NUM[i] * jobs[i][1] for i in range(3)])  # 三种贮箱短壳数量 代号1
Ts_num = sum([JOB_NUM[i] * jobs[i][2] for i in range(3)])  # 三种贮箱筒段数量 代号2
'''最后的总对接表示成以上的形式，便于后面的编码实现'''
jobs_num = [Sb_num, Sj_num, Ts_num, JOB_NUM[0], JOB_NUM[1], JOB_NUM[2]]  # 一共75个单底，52个短壳，68个筒段，进行调度生产，后面表示三种贮箱分别编码的数量
# 对以上有序顺序进行随机排序，同种工件不同排列按照同种顺序处理  # 相同工件之间不进行工序区分
def stochastic():  # 三种零部件分别用0，1，2表示，三种贮箱分别用3，4，5表示
    li = np.ones(sum(jobs_num), int)
    order_index = np.arange(sum(jobs_num))
    np.random.shuffle(order_index)
    s = 0
    for i, v in enumerate(jobs_num):
        if i < 3:
            li[order_index[s:(s+v)]] = i
            s += v
        else:
            li[order_index[s:]]

    return li
# 工艺路线，表示工序顺序以及加工时间
# 壁板1投料-筒段配套-》铣边-》焊接-》检测-》立车-筒段               \
# 壁板2投料-短壳配套-》铣边-》焊接-》检测-》立车-短壳            ————  贮箱配套 —— 总对接|检测 ——》试验——贮箱
# 瓜瓣、顶盖、型材框投料-单底配套-》焊接《-》检测立车-》试验-单底     /
time_table = [[4, 8, 4, 6], [4, 8, 4, 6], [24, 8, 24, 8], [146, 16]]
# 加工时间
# 1.遍历各零部件，开始计算完工时间
class Cij:
    def __init__(self, name, StartTime, LoadTime):  # 定义工作节点类 name为Cij：第i个工件在第j个机器上加工，StartTime为开始时间，LoadTime为加工时间，EndTime为加工结束时间
        self.name = name
        self.StartTime = StartTime
        self.LoadTime = LoadTime
        self.EndTime = StartTime + LoadTime
# 定义最大流程时间函数，适合多不同种同工序数的零件，每个工序只有一台机器可选择，需要零件所有工序的排序process_order以及各零件不同工序的加工时间列表T[Jobs_num * process_num]
def c_max(n):  # n根据下文应该是单条染色体  # 循环赋值函数，将工件数，机器数与加工时间进行绑定
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
    for job, i in enumerate(time_table):  # job为工件索引，i为每个工件所需加工流程（各机器上的时间）
        for process, loadtime in enumerate(i):
            if job < 2:
                if process < 2:  # 前两道工序可选机器一台，时间固定
                    locals()['c{}_{}'.format(job, process)] = Cij(name='c{}_{}'.format(job, process), StartTime=0,
                                                                  LoadTime=loadtime)
                elif process == 2:  # 第三道工序可选机器两台，时间一样
                    locals()['c{}_{}'.format(job, 2)] = Cij(name='c{}_{}'.format(job, 2), StartTime=0,
                                                                  LoadTime=loadtime)
                    locals()['c{}_{}'.format(job, 3)] = Cij(name='c{}_{}'.format(job, 3), StartTime=0,
                                                                  LoadTime=loadtime)
                elif process == 3:  # 最后一工序一台机器
                    locals()['c{}_{}'.format(job, 4)] = Cij(name='c{}_{}'.format(job, 4), StartTime=0,
                                                                  LoadTime=loadtime)
            elif job == 2:
                if process == 0:
                    locals()['c{}_{}'.format(job, 5)] = Cij(name='c{}_{}'.format(job, 5), StartTime=0,
                                                            LoadTime=loadtime)
                    locals()['c{}_{}'.format(job, 6)] = Cij(name='c{}_{}'.format(job, 6), StartTime=0,
                                                            LoadTime=loadtime)
                elif process == 1:
                    locals()['c{}_{}'.format(job, 2)] = Cij(name='c{}_{}'.format(job, 2), StartTime=0,
                                                            LoadTime=loadtime)
                    locals()['c{}_{}'.format(job, 3)] = Cij(name='c{}_{}'.format(job, 3), StartTime=0,
                                                            LoadTime=loadtime)
                elif process == 2:
                    locals()['c{}_{}'.format(job, 4)] = Cij(name='c{}_{}'.format(job, 4), StartTime=0,
                                                            LoadTime=loadtime)
                else:
                    locals()['c{}_{}'.format(job, 7)] = Cij(name='c{}_{}'.format(job, 7), StartTime=0,
                                                            LoadTime=loadtime)
            else:
                if process == 0:
                    locals()['c{}_{}'.format(job, 8)] = Cij(name='c{}_{}'.format(job, 8), StartTime=0,
                                                            LoadTime=loadtime)
                    locals()['c{}_{}'.format(job, 9)] = Cij(name='c{}_{}'.format(job, 9), StartTime=0,
                                                            LoadTime=loadtime)
                    locals()['c{}_{}'.format(job, 10)] = Cij(name='c{}_{}'.format(job, 10), StartTime=0,
                                                            LoadTime=loadtime)
                elif process == 1:
                    locals()['c{}_{}'.format(job, 7)] = Cij(name='c{}_{}'.format(job, 7), StartTime=0,
                                                            LoadTime=loadtime)
# 2.按照某种规则，开始计算总对接的贮箱型号，除法取整
'''把三个零部件当作三个不同工件，分别进行调度先对一个工件的零部件进行调度，求出总对接前的最短加工时间'''








# 定义最大流程时间函数，适合多不同种同工序数的零件，每个工序只有一台机器可选择，需要零件所有工序的排序process_order以及各零件不同工序的加工时间列表T[Jobs_num * process_num]
def c_max(n):  # n根据下文应该是单条染色体
    # 循环赋值函数，将工件数，机器数与加工时间进行绑定
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
    for job, i in enumerate(time_table):  # job为工件索引，i为每个工件所需加工流程（各机器上的时间）
        for machine, loadtime in enumerate(i):
            locals()['c{}_{}'.format(job + 1, machine + 1)] = Cij(name='c{}_{}'.format(job + 1, machine + 1),
                                                                  StartTime=0, LoadTime=loadtime, )
            # "{1} {0} {1}".format("hello", "world")
            # 'world hello world'
    # Python的locals()函数会以dict类型  返回  当前位置的全部局部变量。
    # 加工流程记录表。
    load_time_tables = []
    for num, job in enumerate(n):  # 工件索引及工件号
        for machine in range(1, machines + 1):
            if num == 0 and machine == 1:  # 索引号，机器号=工序号
                locals()['c{}_{}'.format(job, machine)].StartTime = 0  # 首个工件的第一道工序加工开始时间为0
                locals()['c{}_{}'.format(job, machine)].EndTime = locals()['c{}_{}'.format(job, machine)].StartTime + \
                                                                  locals()['c{}_{}'.format(job, machine)].LoadTime
                load_time_tables.append([locals()['c{}_{}'.format(job, machine)].name, [
                    locals()['c{}_{}'.format(job, machine)].StartTime,
                    locals()['c{}_{}'.format(job, machine)].EndTime]])
            elif num == 0 and machine > 1:   # 工件job的的后序工序的开始时间为其上道工序的结束时间
                locals()['c{}_{}'.format(job, machine)].StartTime = locals()['c{}_{}'.format(job, machine - 1)].EndTime
                locals()['c{}_{}'.format(job, machine)].EndTime = locals()['c{}_{}'.format(job, machine)].StartTime + \
                                                                  locals()['c{}_{}'.format(job, machine)].LoadTime
                load_time_tables.append([locals()['c{}_{}'.format(job, machine)].name, [
                    locals()['c{}_{}'.format(job, machine)].StartTime,
                    locals()['c{}_{}'.format(job, machine)].EndTime]])
            elif num > 0 and machine == 1:  # 后面工件的第一道工序开始时间为前面工件的第一道工序结束时间
                locals()['c{}_{}'.format(job, machine)].StartTime = locals()[
                    'c{}_{}'.format(n[num - 1], machine)].EndTime
                locals()['c{}_{}'.format(job, machine)].EndTime = locals()['c{}_{}'.format(job, machine)].StartTime + \
                                                                  locals()['c{}_{}'.format(job, machine)].LoadTime
                load_time_tables.append([locals()['c{}_{}'.format(job, machine)].name, [
                    locals()['c{}_{}'.format(job, machine)].StartTime,
                    locals()['c{}_{}'.format(job, machine)].EndTime]])

            elif num > 0 and machine > 1:  # 后面工件的后道工序为其上道工序以及所在机器的结束时间较大值
                locals()['c{}_{}'.format(job, machine)].StartTime = max(
                    locals()['c{}_{}'.format(n[num - 1], machine)].EndTime,
                    locals()['c{}_{}'.format(job, machine - 1)].EndTime)
                locals()['c{}_{}'.format(job, machine)].EndTime = locals()['c{}_{}'.format(job, machine)].StartTime + \
                                                                  locals()['c{}_{}'.format(job, machine)].LoadTime
                load_time_tables.append([locals()['c{}_{}'.format(job, machine)].name, [
                    locals()['c{}_{}'.format(job, machine)].StartTime,
                    locals()['c{}_{}'.format(job, machine)].EndTime]])
    return load_time_tables, load_time_tables[-1][-1][-1]  # load_time_tables代表所有工件每个工序加工位置及其开始和结束加工时间


def fitness(n):
    return 1 / (c_max(n)[1])