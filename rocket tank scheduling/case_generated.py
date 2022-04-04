# -*- coding: utf-8 -*-
# @Time : 2021/12/30 15:18
# @Author : hhq
# @File : case_generated.py
import numpy as np
import random
import copy
jobs = 6  # 工件数
machines = 10  # 机器数
t_table = np.random.randint(1, 100, (jobs, machines))  # 加工时间列表随机初始化，6*6
# m_table = np.random.randint(1, machines+1, (jobs, machines))
# 加工位置初始化，每个工序一个位置
m_table = []
m = list(range(1, machines + 1))
for i in range(jobs):
    c = copy.deepcopy(m)
    random.shuffle(c)
    m_table.append(c)
# print(t_table[2])
ll = []
for li in t_table:
    # line.replace("\s",",")
    # print(li)
    l=list(eval(",".join(str(i) for i in li)))
    # print(l)
    ll.append(l)
print(ll)
print(m_table)