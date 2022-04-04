# -*- coding: utf-8 -*-
# @Time : 2021/10/4 15:33
# @Author : hhq
# @File : topology_jsp.py
import copy
import datetime
import numpy as np
import random
import re
import matplotlib.pyplot as plt
from matplotlib import rcParams


# jobs = 25  # 工件数
# machines = 10  # 机器数
# t_table = np.random.randint(1, 100, (jobs, machines))  # 加工时间列表
# m_table = np.random.randint(1, 11, (jobs, machines))
# 加工位置初始化，每个工序一个位置
# m_table = []
# m = list(range(1, machines + 1))
# for i in range(jobs):
#     m_table.append(random.shuffle(m))
# m_table = np.random.randint(1, machines+1, (jobs, machines))
# jobs = 6  # 工件数
# machines = 6  # 机器数
# t_table = [[1,3,6,7,3,6],[8,5,10,10,10,4],[5,4,8,9,1,7],[5,5,5,3,8,9],[9,3,5,4,3,1],[3,3,9,10,4,1]]
# m_table=[[3,1,2,4,6,5],[2,3,5,6,1,4],[3,4,6,1,2,5],[2,1,3,4,5,6],[3,2,5,6,1,4],[2,4,6,1,5,3]]
# jobs = 10  # 工件数
# machines = 10  # 机器数
# t_table = [[29,78,9,36,49,11,62,56,44,21],[43,90,75,11,69,28,46,46,72,30],[91,85,39,74,90,10,12,89,45,33],
#            [81,95,71,99,9,52,85,98,22,43],[14,6,22,61,26,69,21,49,72,53],[84,2,52,95,48,72,47,65,6,25],
#            [46,37,61,13,32,21,32,89,30,55],[31,86,46,74,32,88,19,48,36,79],[76,69,76,51,85,11,40,89,26,74],
#            [85,13,61,7,64,76,47,52,90,45]]
# m_table = [[1,2,3,4,5,6,7,8,9,10],[1,3,5,10,4,2,7,6,8,9],[2,1,4,3,9,6,8,7,10,5],[2,3,1,5,7,9,8,4,10,6],
#            [3,1,2,6,4,5,9,8,10,7],[3,2,6,4,9,10,1,7,5,8],[2,1,4,3,7,6,10,9,8,5],[3,1,2,6,5,7,9,10,8,4],
#            [1,2,4,6, 3, 10,7,8,5,9],[2,1,3,7,9,10,6,4,5,8]]
# jobs = 20
# machines = 5
# t_table = [[29,9,49,62,44],[43,75,69,46,72],[91,39,90,12,45],[81,71,9,85,22],[14,22,26,21,72],[84,52,48,47,6],
#            [46,61,32,32,30],[31,46,32,19,36],[76,76,85,40,26],[85,61,64,47,90],[78,36,11,56,21],[90,11,28,46,30],
#            [85,74,10,89,33],[95,99,52,98,43],[6,61,69,49,53],[2,95,72,65,29],[37,13,21,89,55],[86,74,88,48,79],
#            [69,51,11,89,74],[13,7,76,52,45]]
# m_table = [[1,2,3,4,5],[1,2,4,3,5],[2,1,3,5,4],[2,1,5,3,4],[3,2,1,4,5],[3,2,5,1,4],[2,1,3,4,5],[3,2,1,4,5],
#            [1,4,3,2,5],[2,3,1,4,5],[2,4,1,5,3],[3,1,2,4,5],[1,3,2,4,5],[3,1,2,4,5],[1,2,5,3,4],[2,1,4,5,3],
#            [1,3,2,4,5],[1,2,5,3,4],[2,3,1,4,5],[1,2,3,4,5]]

# from case_generated import jobs, machines, m_table, t_table
jobs = 6
machines = 10
t_table = [[82, 91, 88, 72, 28, 29, 82, 99, 8, 81], [82, 51, 28, 3, 72, 9, 65, 29, 93, 8], [40, 48, 29, 16, 48, 20, 56, 40, 11, 71], [43, 67, 81, 45, 81, 99, 50, 89, 38, 60], [67, 26, 27, 78, 88, 43, 16, 53, 62, 15], [92, 11, 78, 13, 69, 54, 67, 75, 56, 68]]
m_table=[[2, 7, 10, 6, 1, 9, 5, 8, 3, 4], [10, 4, 7, 6, 3, 9, 5, 1, 8, 2], [7, 3, 1, 2, 9, 4, 10, 6, 5, 8], [2, 9, 8, 1, 4, 7, 6, 10, 5, 3], [2, 1, 5, 10, 4, 7, 8, 3, 9, 6], [4, 6, 10, 3, 8, 5, 9, 1, 2, 7]]
'''
# m_table = []
# for i in range(jobs):
#     a = list(range(1, 11))
#     random.shuffle(a)
#     m_table.append(a)  # 加工位置列表'''


# 处理以上数据为染色体形式，一行只含有工序不含机器
def com_tr(t_table):
    topo_order = []
    for j_num, o_num in enumerate(t_table):  # 根据时间表获取工件数索引和其工序列表
        # topo_order.append((np.ones([1, len(o_num)], int)*(j_num+1)).tolist())  # 将工件数转化为数值列表，长度为工序数
        topo_order = topo_order+(np.ones([1, len(o_num)], int) * (j_num + 1)).tolist()
        # topo_order.append(np.ones([1, len(o_num)], int) * (j_num + 1))
        # print([j_num,o_num])
    # print(topo_order)
    combin = []
    for li in topo_order:  # 将列表中各工件独立列表加起来
        combin = combin+li

    random.shuffle(combin)  # 随机打乱列表中元素
    return combin
# print(combin)


# order = []
# for j_num, o_num in enumerate(t_table):
#     for i in range(j_num):
#         order.append(i+1)

"""根据排序计算完工时间"""
'''1.首先对工序和机器以及时间分别初始化P，M，T'''
# P,M,T=[],[],[]
# Cmax = 0
# for i in range(len(combin)):  # 对合并后的列表进行遍历，
#     job = combin[i]
#     P.append(combin[i]*10+combin.count(job))  # 染色体上编码和工序对应
#     M.append(m_table[job-1][combin.count(job)-1])  # 染色体上每台机器
#     T.append(t_table[job-1][combin.count(job)-1])  # 每条染色体上对应工序的加工时间

# C = []
# for j in range(len(combin)):
#     x = int(P[j]/10)  # 工件号
#     y = P[j] % 10  # 工序号
#     if y == 1:
#         C
# 定义工作节点类 name为Cij:第i个工件在第j个机器上加工，StartTime为开始时间，LoadTime为加工时间，EndTime为加工结束时间
class Cij:
    def __init__(self, name, StartTime, LoadTime):
        self.name = name
        self.StartTime = StartTime
        self.LoadTime = LoadTime
        self.EndTime = StartTime + LoadTime

# 定义最大流程时间函数
def c_max(combin):  # n根据下文应该是单条染色体
    # 循环赋值函数，将工件数，机器数与加工时间进行绑定
    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
    for i in range(len(combin)):  # i为工件索引
        job = combin[i]  # 工件
        no_job = combin[:i+1].count(job)  # 工序
        machine = m_table[job-1][no_job-1]  # 机器号
        loadtime = t_table[job-1][no_job-1]  # 该工序加工时间
        locals()['c{}_{}_{}'.format(job, no_job, machine)] = Cij(name='c{}_{}_{}'.format(job, no_job, machine),
                                                                  StartTime=0, LoadTime=loadtime, )
            # "{1} {0} {1}".format("hello", "world")
            # 'world hello world'
    # Python的locals()函数会以dict类型  返回  当前位置的全部局部变量。
    # 加工流程记录表。
    load_time_tables = []
    # M_time = np.zeros(max(max(m_table)))  # 初始化所有机器当前加工时刻为0
    M_time=np.zeros(machines)
    for i in range(len(combin)):  # combin为数值编号，代表工件及其工序数量
        job = combin[i]  # 工件号
        no_job = combin[:i+1].count(job)  # 工序
        machine = m_table[job-1][no_job-1]  # 注意python索引
        if no_job == 1:  # 工序号，开始时间为机器完成上个工件任务的时间或0
            locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime = M_time[machine-1]  # 第一道工序开始时间为本机器此时时刻或0
            # locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime = locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime + \
            #                                                   locals()['c{}_{}_{}'.format(job, no_job, machine)].LoadTime
            # M_time[machine-1] = locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime
            # load_time_tables.append([locals()['c{}_{}_{}'.format(job, no_job, machine)].name, [
            #     locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime,
            #     locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime]])

        else:
            locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime = max(
                M_time[machine-1],  # 该工序所在加工位置机器的时间
                locals()['c{}_{}_{}'.format(job, no_job-1, m_table[job-1][no_job-2])].EndTime)  # 该工序的前道工序的完工时间
        locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime = locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime + \
                                                          locals()['c{}_{}_{}'.format(job, no_job, machine)].LoadTime
        M_time[machine - 1] = locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime
        load_time_tables.append([locals()['c{}_{}_{}'.format(job, no_job, machine)].name, [
            locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime,
            locals()['c{}_{}_{}'.format(job, no_job, machine)].EndTime]])
        T=[]
        for i in load_time_tables:
            T.append(i[-1][-1])

    return load_time_tables, max(T)  # load_time_tables 代表所有工件每个工序加工位置及其开始和结束加工时间

# c_max(combin)
# print(load_time_tables)
# load_time_tables, load_time_tables[-1][-1][-1] = c_max(combin)

# 进行遗传算法实现
pop_size = 10  # 种群规模
c_r = 0.2  # 交叉概率
variation_rate = 0.05  # 变异概率
iters = 800  # 迭代次数

target_points = [1,2,3]
# 种群初始化
def init_population(pop_size, chrom):
    pop = []
    for i in range(pop_size):
        c = copy.deepcopy(chrom)
        random.shuffle(c)
        pop.append(c)
    return pop


# 计算适应度
def fitness(combin):
    return 1 / (c_max(combin)[1])


class node:
    def __init__(self, state):
        self.state = state
        self.load_table = c_max(state)[0]  # 求出染色体上每个工序的负载表
        self.makespan = c_max(state)[1]  # 染色体的时间跨度
        self.fitness = fitness(state)

# node实现了完工时间的求解和适应度值

'''
出问题的地方,交叉错误
'''
def two_points_cross(chro1, chro2):
    # 不改变原始数据进行操作
    chro1_1 = copy.deepcopy(chro1)
    chro2_1 = copy.deepcopy(chro2)
    # 交叉位置，point1<point2
    point1 = random.randint(0, len(chro1_1))
    point2 = random.randint(0, len(chro1_1))
    while point1 > point2 or point1 == point2:
        point1 = random.randint(0, len(chro1_1))
        point2 = random.randint(0, len(chro1_1))

    # 记录交叉片段
    frag1 = chro1[point1:point2]
    frag2 = chro2[point1:point2]
    random.shuffle(frag1)
    random.shuffle(frag2)
    # 交叉
    chro1_1[point1:point2], chro2_1[point1:point2] = chro2_1[point1:point2], chro1_1[point1:point2]

    child1 = chro1_1[:point1] + frag1 + chro1_1[point2:]
    child2 = chro2_1[:point1] + frag2 + chro2_1[point2:]

    return child1, child2


# 交换变异
def gene_exchange(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2 or point1 > point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    n[point1], n[point2] = n[point2], n[point1]
    return n


# 插入变异
def gene_insertion(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    x = n.pop(point1)
    n.insert(point2, x)
    return n


# 局部逆序变异
def gene_reverse(n):
    point1 = random.randint(0, len(n) - 1)
    point2 = random.randint(0, len(n) - 1)
    while point1 == point2 or point1 > point2:
        point1 = random.randint(0, len(n) - 1)
        point2 = random.randint(0, len(n) - 1)
    ls_res = n[point1:point2]
    ls_res.reverse()
    l1 = n[:point1]
    l2 = n[point2:]
    n_res_end = l1 + ls_res + l2
    return n_res_end

def select(population):
    pop_fit=[]
    for i in population:
        pop_fit.append(fitness(i))
    best_chrom = min(pop_fit)

    return best_chrom

# 开始求解
combin = com_tr(t_table)
population = init_population(pop_size, combin)
solution_list = [node(i) for i in population]
solution_list.sort(key=lambda x: x.makespan)
best_fit, fit_ave = [], []
pops = [i.state for i in solution_list]  # 相当于把solution_list的染色体复制到pops中  pops = copy.deepcopy(population)
f_list = [i.makespan for i in solution_list]  # 计算该种群各适应度
Xb, fb, fave = pops[0], f_list[0], np.mean(f_list)  # 最佳个体与平均适应度
best_fit.append(fb)
fit_ave.append(fave)
# 开始循环
start = datetime.datetime.now()
for i in range(iters):

    pop_new = copy.deepcopy(pops)
    if i % 10 == 0:
        print('第{}次进化后的最优加工时间为{}'.format(i, fb))  # 首个染色体的结束时间, solution_list含makespan函数和方法
    pop_children1 = pop_new[1::2]  # 偶数解，偶数组成的种群
    pop_children2 = pop_new[::2]  # 奇数解
    # PMX两点交叉变异
    for i in range(len(pop_children1)):
        pop_children1[i], pop_children2[i] = two_points_cross(pop_children1[i], pop_children2[i])
    # 交叉后的子种群
    pop_new = pop_children1 + pop_children2
    # 变异
    for i in pop_new:
        mutation_rate = random.random()
        target = random.choice(target_points)  # ???三种编译策略
        # if mutation_rate > variation_rate:
        if target == 1:
            pop_new[pop_new.index(i)] = gene_exchange(i)
        elif target == 2:
            pop_new[pop_new.index(i)] = gene_insertion(i)
        else:
            pop_new[pop_new.index(i)] = gene_reverse(i)
    # print(cross_population)
    cross_solution = [node(i) for i in pop_new]
    solution_list = solution_list + cross_solution
    solution_list.sort(key=lambda x: x.makespan)  # 排序后首个染色体为最佳解
    del solution_list[pop_size:]
    pops = [i.state for i in solution_list]  # 相当于把solution_list的染色体复制到pops中  pops = copy.deepcopy(population)
    f_list = [i.makespan for i in solution_list]  # 计算该种群各适应度
    Xb, fb, fave = pops[0], f_list[0], np.mean(f_list)  # 最佳个体与平均适应度
    best_fit.append(fb)
    fit_ave.append(fave)
    # 选择
    P_list = np.array([fitness(c) for c in pop_new])
    for p in range(pop_size):
        P_list[p] = f_list[p] / sum(f_list)
    P_wheel = P_list.cumsum()
    for s in range(pop_size):
        if P_wheel[s] < random.random():
            random.shuffle(pop_new[s])  # 概率低的个体随机初始化
print('进化完成，最终最优加工时间为：', solution_list[0].makespan)
end = datetime.datetime.now()
print('耗时{}'.format(end - start))
print(solution_list[0].load_table)
config = {
            "font.family": 'serif',
            "font.size": 10,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)
plt.figure(1)
# 绘制甘特图
def color():# 甘特图颜色生成函数
    color_ls = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    col = ''
    for i in range(6):  # 6种颜色数字字母组合
        col += random.choice(color_ls)
    return '#'+col
colors = [color() for i in range(len(t_table))]  # 甘特图颜色列表,每个工件一个颜色
for i in solution_list[0].load_table:  # 对最佳染色体进行遍历，做出甘特图
    # print(i)  # 每个工件
    y = eval(re.findall('_(\d+)', i[0])[1])  # 正则表达式匹配工件数,找到_后面内部整数个数，机器号=工序号
    """
    i = ['c24_9', [1715, 1736]]  # 9
    # \d匹配任何十进制数，它相当于类[0-9]
    # \d+如果需要匹配一位或者多位数的数字时用
    a = re.search("(a4)+", "a4a4a4a4a4dg4g654gb")   # 匹配一个或多个a4
    a = re.findall(r"你|好", "a4a4a你4aabc4a4dgg好dg4g654g")   #|或，或就是前后其中一个符合就匹配  #打印出 ['你', '好']
    """
    # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
    label=re.findall(r'(\d*?)_', i[0])[0]  # 正则表达式匹配机器数
    plt.barh(y=y, left=i[1][0], width=i[1][-1] - i[1][0], height=0.5, color=colors[eval(label) - 1],
             label=f'job{label}')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.title('jsp最优调度甘特图')
plt.xlabel('加工时间')
plt.ylabel('加工机器')
handles, labels = plt.gca().get_legend_handles_labels()  # 标签去重
from collections import OrderedDict  # ：字典的子类，保留了他们被添加的顺序
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
config = {
            "font.family": 'serif',
            "font.size": 10,
            "mathtext.fontset": 'stix',
            "font.serif": ['SimSun'],
         }
rcParams.update(config)
plt.figure(2)
p1, = plt.plot(best_fit[:], label='best_fit')
p2, = plt.plot(fit_ave[:], label='fit_ave')
l1 = plt.legend([p1, p2], ["best_fit", "fit_ave"], loc='upper right')
# l1 = plt.legend([p2, p1], ["line 2", "line 1"], loc='upper left')
plt.title('variation of makespan with GA')
plt.xlabel('iteration')
plt.ylabel('working time')
plt.show()