# -*- coding: utf-8 -*-
# @Time : 2022/3/10 20:39
# @Author : hhq
# @File : Q_learning_scheduling.py
import numpy as np
from JSP import FT
import copy
import matplotlib.pyplot as plt
from draw_gantt import Node

PT = [[1,3,6,7,3,6],[8,5,10,10,10,4],[5,4,8,9,1,7],[5,5,5,3,8,9],[9,3,5,4,3,1],[3,3,9,10,4,1]]
Ma = [[3,1,2,4,6,5],[2,3,5,6,1,4],[3,4,6,1,2,5],[2,1,3,4,5,6],[3,2,5,6,1,4],[2,4,6,1,5,3]]

node = Node(PT, Ma)
Ft = FT(PT, Ma)
State_init, State_term = Ft.States_fun()
# print([State_term])
dimension = copy.copy(Ft.O_num)  # 各工件工序数集
for i in range(Ft.J_num):
    dimension[i] += 1
dimension.append(Ft.J_num)
Q = np.zeros(dimension)  # Q初始化为0列表,其维度为每个工件的工序数*工件数
# print(Q)
alpha = 0.1
gamma = 0.9
epsilon = 0.8
episode_num = 10000

C_plot = []
C_mean = []
min_C = []
for e in range(episode_num):
    # 初始化S
    S = State_init
    O_list = []
    C = []
    Ft.reset()
    start_list = []
    while True:
        # print(S)
        # print(Q)
        A = Ft.job_selection(S, Q, epsilon)
        O_list.append(A)  # 将加工工件添加到列表中
        # 计算A对应的工序，然后计算其对应加工时间
        O_sum = O_list.count(A)
        # print([A, O_sum])
        if O_sum == 1:
            Start = Ft.C_m[Ma[A][O_sum - 1] - 1]
        else:
            Start = max(Ft.C_m[Ma[A][O_sum-1]-1], Ft.C_J[A][O_sum-2])  # 工序最早开始时间
        start_list.append(Start)
        C.append(Ft.scheduling(Start, A, O_sum-1))  # 执行后的完工时间
        S_next = copy.copy(S)  #
        S_next[A] += 1
        # print([S, S_next])
        if len(C) > 1 and C[-1]-C[-2] > 0:
            R = 1/(C[-1]-C[-2])
        else:
            R = 10
        Q[S[0]][S[1]][S[2]][S[3]][S[4]][S[5]][A] += alpha*(R+gamma*np.max(Q[S_next[0]][S_next[1]][S_next[2]][S_next[3]][S_next[4]][S_next[5]])-Q[S[0]][S[1]][S[2]][S[3]][S[4]][S[5]][A])
        S = S_next

        if S == State_term:
            # print(Ft.C_J)
            break
    if e == episode_num - 1:
        plt.figure(1)
        C_J = Ft.C_J
        print("工件顺序列表:", O_list)  # 工件顺序列表
        print("各工序完工时间:", C_J)  # 各工序完工时间
        print("开始时间列表:", start_list)
        node.draw_gantt(start_list, O_list, C_J)


    C_plot.append(C[-1])
    C_mean.append(np.mean(C_plot))
    min_C.append(np.min(C_plot))
plt.figure(2)
plt.plot(C_plot[:], label="makespan of each episode")
plt.plot(C_mean[:], label="makespan of each episode with moving average")
plt.plot(min_C[:], label="min makespan of each episode")
plt.legend(loc="lower left")
plt.title('jsp-makespan')
plt.xlabel('episode')
plt.ylabel('time')
plt.show()