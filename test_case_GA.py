# -*- coding: utf-8 -*-
# @Time : 2021/12/28 16:26
# @Author : hhq
# @File : test_case_GA.py
import copy
import datetime
import numpy as np
import random
import re
import matplotlib.pyplot as plt
import math

jobs = 6  # 工件数
machines = 6  # 机器数
# t_table = np.random.randint(1, 100, (jobs, machines))  # 加工时间列表随机初始化，6*6
# m_table = np.random.randint(1, machines+1, (jobs, machines))
t_table = [[1,3,6,7,3,6],[8,5,10,10,10,4],[5,4,8,9,1,7],[5,5,5,3,8,9],[9,3,5,4,3,1],[3,3,9,10,4,1]]
# 加工位置初始化，每个工序一个位置
m_table = []
m = list(range(1, machines + 1))
for i in range(jobs):
    m_table.append(random.shuffle(m))

m_table=[[3,1,2,4,6,5],[2,3,5,6,1,4],[3,4,6,1,2,5],[2,1,3,4,5,6],[3,2,5,6,1,4],[2,4,6,1,5,3]]

# 处理以上数据为染色体形式，一行只含有工序不含机器
def com_tr(t_table):
    topo_order = []
    for j_num, o_num in enumerate(t_table):  # 根据时间表获取工件数索引和其工序列表
        topo_order = topo_order+(np.ones([1, len(o_num)], int) * (j_num + 1)).tolist()
    combin = []
    for li in topo_order:  # 将列表中各工件独立列表加起来
        combin = combin+li

    random.shuffle(combin)  # 随机打乱列表中元素
    return combin

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
        # Python locals() 函数会以字典类型返回当前位置的全部局部变量。
    load_time_tables = []
    # M_time = np.zeros(max(max(m_table)))  # 初始化所有机器当前加工时刻为0
    M_time=np.zeros(machines)
    for i in range(len(combin)):  # combin为数值编号，代表工件及其工序数量
        job = combin[i]  # 工件号
        no_job = combin[:i+1].count(job)  # 工序
        machine = m_table[job-1][no_job-1]  # 注意python索引
        if no_job == 1:  # 工序号，开始时间为机器完成上个工件任务的时间或0
            locals()['c{}_{}_{}'.format(job, no_job, machine)].StartTime = M_time[machine-1]
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
    # print(load_time_tables)
    return load_time_tables, max(T)  # load_time_tables 代表所有工件每个工序加工位置及其开始和结束加工时间

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
    return 1/(c_max(combin)[1])

class node:
    def __init__(self, state):
        self.state = state
        self.load_table = c_max(state)[0]  # 求出染色体上每个工序所在机器的开始结束时间表
        self.makespan = c_max(state)[1]  # 染色体的时间跨度
        self.fitness = fitness(state)

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

def update_AK(A0, r0, K, t, c_r0, mu0):  # 更新参数
    A = K/(1+(K/A0-1)*np.exp(-r0*t))
    r = r0*(1-A/K)
    c_r = c_r0*(1-A/K)
    mu = mu0*(1-A/K)
    pop_size = math.ceil(math.log(K / A)) + 2  # 种群规模
    return A, r, c_r, mu, pop_size
def update_solution(population):
    solution_list = []
    # 可行解集，包含开始结束时间等信息
    for i in population:
        # locals()['solution{}'.format(population.index(i))] = node(i)  # i为染色体,node为类包含makespan属性
        solution_list.append(node(i))
    solution_list.sort(key=lambda x: x.makespan)  # 排序后首个染色体为最佳解
    pops = [i.state for i in solution_list]  # 相当于把solution_list的染色体复制到pops中
    f_list = [i.makespan for i in solution_list]
    Xb, fb, fave = pops[0], f_list[0], np.mean(f_list)  # 最佳个体与平均适应度
    return solution_list, Xb, fb, fave, f_list
# 进行遗传算法实现
# 入侵改进GA算法
A0 = 100
r0 = 0.01
K = 10000
t = 1
c_r0 = 0.8  # 交叉概率
mu0 = 0.9  # 变异概率
A,r,c_r,mu,pop_size = update_AK(A0,r0,K,t,c_r0,mu0)
target_points = [1, 2, 3]
# 开始求解
combin = com_tr(t_table)  # 转化成工件号编码
# population = init_population(pop_size, combin)
# 开始循环
start = datetime.datetime.now()
best_fit = []
# iters=100
population = init_population(pop_size, combin)
solution_list = []
for i in population:
    # locals()['solution{}'.format(population.index(i))] = node(i)  # i为染色体,node为类包含makespan属性
    solution_list.append(node(i))
solution_list.sort(key=lambda x: x.makespan)  # 排序后首个染色体为最佳解
pops = [i.state for i in solution_list]  # 相当于把solution_list的染色体复制到pops中
f_list = [i.makespan for i in solution_list]
Xb, fb, fave = pops[0], f_list[0], np.mean(f_list)  # 最佳个体与平均适应度
best_fit.append(fb)
# 也可以用argmin获取最小值索引
while A < K*0.98:
    if t % 10 == 0:
        print('第{}次进化后的最优加工时间为{}'.format(t, fb))  # 首个染色体的结束时间, solution_list含makespan函数和方法
    # pop_new = init_population(pop_size, combin)
    pop_new = copy.deepcopy(pops)
    for k in range(1, len(pop_new)):
        pk = np.exp((f_list[k]-fb)/(A*r))
        if pk < random.random():
            if mu > random.random():
                target = random.choice(target_points)  # ???三种编译策略
                if target == 1:
                    pop_new[k] = gene_exchange(pop_new[k])
                elif target == 2:
                    pop_new[k] = gene_insertion(pop_new[k])
                else:
                    pop_new[k] = gene_reverse(pop_new[k])
            elif c_r > random.random():
                pop_new[k], pop_new[k-1] = two_points_cross(pop_new[random.randint(0, int(pop_size/2))], pop_new[random.randint(math.ceil(pop_size/2), pop_size-1)])
            else:
                random.shuffle(pop_new[k])
        else:
            Xb, fb = pop_new[k], f_list[k]
    t += 1
    best_fit.append(fb)
    cross_population = pop_new
    cross_solution = [node(i) for i in cross_population]
    A, r, c_r, mu, pop_size = update_AK(A0, r0, K, t, c_r0, mu0)
    population = init_population(pop_size, combin)
    solution_list = [node(i) for i in population]
    solution_list = solution_list + cross_solution
    solution_list.sort(key=lambda x: x.makespan)  # 排序后首个染色体为最佳解
    del solution_list[pop_size:]  # 删除popsize后面的可行解，使其大小稳定
    pops = [i.state for i in solution_list]
    f_list = [i.makespan for i in solution_list]
    Xb, fb, fave = pops[0], f_list[0], np.mean(f_list)
print('进化完成，最终最优加工时间为：', fb)
end = datetime.datetime.now()
print('耗时{}'.format(end - start))
print(solution_list[0].load_table)


# 绘制甘特图
def color():# 甘特图颜色生成函数
    color_ls = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    col = ''
    for i in range(6):  # 6种颜色数字字母组合
        col += random.choice(color_ls)
    return '#'+col
colors = [color() for i in range(len(t_table))]  # 甘特图颜色列表,每个工件一个颜色
for i in node(Xb).load_table:  # 对最佳染色体进行遍历，做出甘特图
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
plt.show()