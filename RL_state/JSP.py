# -*- coding: utf-8 -*-
# @Time : 2022/3/11 10:23
# @Author : hhq
# @File : JSP.py
import numpy as np
import copy

class FT:
    def __init__(self, PT, Ma):
        self.PT = PT
        self.Ma = Ma
        self.J_num = len(self.PT)
        self.O_num = [len(self.PT[i]) for i in range(self.J_num)]
    # 环境重置‘’‘’‘’‘’‘’‘’‘
    def reset(self):
        self.C_m = [0 for i in range(self.J_num)]
        self.C_J = [[0 for i in range(len(self.PT[j]))] for j in range(self.J_num)]  # 每个工件的当前工序结束时间

    def States_fun(self):
        State_init = [0 for j in range(self.J_num)]
        State_term = self.O_num
        return State_init, State_term

    def scheduling(self, S, Job, O_sum):  # Job和O_num从0开始
        self.C_m[self.Ma[Job][O_sum]-1] = S + self.PT[Job][O_sum]  # 机器上的加工结束时间
        self.C_J[Job][O_sum] = S + self.PT[Job][O_sum]  # 每个工序的结束时间
        Cmax = max(self.C_m)
        return Cmax

    def job_selection(self, S, Q, epsilon):
        list_J = list(range(self.J_num))
        JJ = copy.copy(list_J)
        for i in list_J:  # 确定可以选择的工件
            if S[i] == self.O_num[i]:  # 工件i已经加工完成
                JJ.remove(i)
        # print("索引：", JJ)

        if np.random.random() < epsilon:
            '''# max_v = max(np.delete(Q[S[0]][S[1]][S[2]], [i]))
            # A = list(Q[S[0]][S[1]][S[2]]).index(max_v)  # 选择其他工件,但是在Q初始为0情况下，此方法失效，因为所有值相同'''
            A = JJ[0]
            if len(JJ) > 1:
                for j in JJ:
                    if Q[S[0]][S[1]][S[2]][S[3]][S[4]][S[5]][j] > Q[S[0]][S[1]][S[2]][S[3]][S[4]][S[5]][A]:
                        A = j
        else:
            A = np.random.choice(JJ)
        return A



