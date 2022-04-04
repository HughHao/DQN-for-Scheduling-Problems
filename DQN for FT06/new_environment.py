# -*- coding: utf-8 -*-
# @Time : 2022/3/15 21:52
# @Author : hhq
# @File : new_environment.py
import numpy as np
from numpy import *
from Object_for_JM import Object
class Situation:
    def __init__(self, NumOfMs, NumOfJs, machine_list, time_list):
        self.J_num = NumOfJs
        self.M_num = NumOfMs
        self.O_num = [0 for i in range(NumOfJs)]
        self.machine_list = machine_list
        for i in range(NumOfJs):
            self.O_num[i] = len(machine_list[i])
        self.Process_time = time_list
        JT = np.zeros(self.J_num)
        # print(J_num)
        for i in range(self.J_num):
            JT[i] = sum(time_list[i])

        TC = np.sum(self.Process_time)  # 所有工件所有工序加工时间之和
        self.mean_JT = np.mean(JT)
        self.TC = TC  # 所有工件所有工序加工时间之和
        self.JT = JT  # 每个工件的总加工时间
        self.CTK = [0 for i in range(self.M_num)]  # 各机器当前工序的结束时间
        self.CTI = [0 for i in range(self.J_num)]  # 各工件当前工序的结束时间
        self.OP = [0 for i in range(self.J_num)]  # 各工件已完工工序数
        self.UK = [0 for i in range(self.M_num)]  # 各设备利用率

        self.CRJ = [0 for i in range(self.J_num)]  # 各工件工序完成率


        # 定义或初始化每个工件的信息
        self.Jobs = []
        for i in range(self.J_num):
            obj = Object(i)
            self.Jobs.append(obj)

        # 定义或初始化每个机器的信息
        self.Machines = []
        for i in range(self.M_num):
            obj = Object(i)
            self.Machines.append(obj)

        # 初始化每台机器的加工列表
        self.PL = [[] for m in range(self.M_num)]  # 该机器加工工件集合
        for i in range(self.J_num):
            for m in range(self.M_num):
                if m in self.machine_list:
                    self.PL[m].append(i)  # 机器k所对应的工件
        self.pi = np.zeros(self.M_num)
        for m in range(self.M_num):
            self.pi[m] = np.mean(self.PL[m])
        self.Qi = [self.PL[k] for k in range(self.M_num)]  # 各设备等待队列

    def scheduling(self, Job, Machine):  # 根据行动（选择的工件，机器），将加工工序所在工件、加工位置、加工时间开始、结束均存储起来
        # 更新调度后的加工状态（原始状态）
        #  确定工件开始时间
        try:
            last_ot = self.CTI[Job]  # 上道工序加工结束时间
        except:
            last_ot = 0
        try:
            last_mt = self.CTK[Machine]  # 所在机器最后完工时间
        except:
            last_mt = 0

        Start_time = max(last_ot, last_mt)  # 开始时间
        PT = self.Process_time[Job][self.OP[Job]]  # 即将加工(目前工序号+1)的工序加工时间

        Idle = self.Machines[Machine].Idle  # 被选择的机器的闲置时间区间集合
        for i in range(len(Idle)):  # 多少个闲时间区间  Idle为空闲时段集合
            if Idle[i][1] - Idle[i][0] > PT:  # 大于该工序时间
                if Idle[i][0] > last_ot:
                    Start_time = Idle[i][0]
                    pass
                if Idle[i][0] < last_ot and Idle[i][1] - last_ot > PT:
                    Start_time = last_ot
                    pass
        end_time = Start_time + PT
        self.Machines[Machine]._add(Start_time, end_time, Job, PT)  # 机器对象更新
        self.Jobs[Job]._add(Start_time, end_time, Machine, PT)  # 工件对象更新
        self.Machines[Machine].idle_time()  # 计算机器加工两道连续工序的中间等待时间
        self.Jobs[Job].idle_time()  # 计算工件前后共需的中间间隔时间，可作为状态或奖励定义
        try:
            self.Qi[Machine] = self.PL[Machine].remove(Job)  # 队列更新,注意逐渐减少到空集
        except:
            None
        self.CTK[Machine] = max(self.Machines[Machine].End)
        self.CTI[Job] = max(self.Jobs[Job].End)  # 该工序加工结束的时间
        self.OP[Job] += 1  # 工件已加工的工序数量集合
        self.UK[Machine] = 0 if self.CTK[Machine] == 0 else sum(self.Machines[Machine].T) / self.CTK[
            Machine]  # 单台设备利用率 = 工件需要在机器加工时间之和/当前已结束工序的完工时间
        self.CRJ[Job] = sum(self.Jobs[Job].T) / self.JT[Job]  # 工件Job的完工率=已加工数/总工序数
        # self.Machines[Machine]._PL(Machine, Job)

    def Features(self):  # 同时反应局部信息和整体信息，定义机器特征，从中选择合适的工件

        # 1设备平均利用率
        U_ave = sum(self.UK) / self.M_num  #

        # 2设备使用率标准差
        K = 0
        for uk in self.UK:
            K += np.square(uk - U_ave)
        U_std = np.sqrt(K / self.M_num)

        # 3平均工序完成率
        Ct = 0
        for i in range(self.J_num):
            Ct += sum(self.Jobs[i].T)
        CRO_ave = Ct / self.TC  # 所有任务已加工时间/所有任务加工时间之和

        # 4平均工件工序完成率
        CRJ_ave = sum(self.CRJ) / self.J_num

        # 5工件工序完成率标准差
        K = 0
        for uk in self.CRJ:
            K += np.square(uk - CRJ_ave)
        CRJ_std = np.sqrt(K / self.J_num)

        # 机器i的等待队列中的工件数与总工件数比
        F1 = np.zeros(self.M_num)
        # 队列中所有工件平均加工时间与pi之比
        F2 = np.zeros(self.M_num)
        # 队列 Qi中所有工件在机器 Mi上加工时间均值与 pi 之比
        F3 = np.zeros(self.M_num)
        # 队列中所有工件在机器上加工时间最大值与pi之比
        F4 = np.zeros(self.M_num)
        # 队列中所有工件在机器上加工时间最小值与pi之比
        F5 = np.zeros(self.M_num)

        # 每台机器加工剩余时间占比
        F6 = np.zeros(self.M_num)

        # 队列 Qi 中所有工件在机器 Mi上与在下一台机器上加工时间比值最大值的归一化表示
        iQ_T = [[] for m in range(self.M_num)]  # 每台机器的加工工件工序时间列表
        for m in range(self.M_num):
            if self.Qi[m] is not None:
                mean_QI = np.mean(self.JT[self.Qi[m]])
                # print(self.Qi[m])
                for job in self.Qi[m]:  # 对于在该机器上加工的工件计算其加工时间
                    # 队列中在该机器上加工的工件工序的平均加工时间
                    if m in self.machine_list[job]:  # 该机器在工件的加工位置列表中，说明该机器加工该工件
                        index = self.machine_list[job].index(m)
                        iQ_T[m].append(self.Process_time[job][index])  # 工件对应位置工序的加工时间

                F1[m] = len(self.Qi[m])/len(self.PL[m])
                F2[m] = mean_QI/self.pi[m]  # Qi中工件在所有机器的加工时间均值
                F3[m] = np.mean(iQ_T[m])/self.pi[m]  # Qi中在该机器上的加工时间均值
                F4[m] = np.max(iQ_T[m])/self.pi[m]
                F5[m] = np.min(iQ_T[m])/self.pi[m]
                F6[m] = np.sum(iQ_T[m]) / np.sum(self.PL[m])


        return U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, F1[0],F1[1],F1[2],F1[3],F1[4],F1[5], F2[0],F2[1],F2[2],F2[3],F2[4],F2[5],F3[0],F3[1],F3[2],F3[3],F3[4],F3[5],F4[0],F4[1],F4[2],F4[3],F4[4],F4[5],F5[0],F5[1],F5[2],F5[3],F5[4],F5[5],F6[0],F6[1],F6[2],F6[3],F6[4],F6[5],


    def rule1(self):
        # 选出剩余加工时间最长的工件  MWKR
        remain_time = 0
        for i in range(self.J_num):
            if self.O_num[i] - self.OP[i] > 0:  # 存在剩余工件
                if self.JT[i] - sum(self.Jobs[i].T) > remain_time:
                    Job_i = i
                    remain_time = self.JT[i] - sum(self.Jobs[i].T)
        # print(Job_i, self.OP[Job_i])
                    Machine = self.machine_list[Job_i][self.OP[Job_i]]
        return Job_i, Machine

    def rule9(self):  # 选择加工工序最短的工件和其对应的机器
        job, machine = 0, 0
        for m in range(self.M_num):
            if self.Qi[m] != []:
                for jo in self.Qi[m]:
                    if self.Process_time[jo][self.OP[jo]][m] < self.Process_time[job][self.OP[job]][machine]:
                        job = jo
                        machine = m
        return job, machine
    def rule10(self):  # 选择加工工序最短的工件和其对应的机器
        job, machine = 0, 0
        for m in range(self.M_num):
            if self.Qi[m] != []:
                for jo in self.Qi[m]:
                    if self.Process_time[jo][self.OP[jo]][m] > self.Process_time[job][self.OP[job]][machine]:
                        job = jo
                        machine = m
        return job, machine

    # Composite dispatching rule 2
    # return Job,Machine
    def rule2(self):
        # 选出加工剩余工序最多的工件
        remain_ope = np.zeros(self.J_num)
        for i in range(self.J_num):
            remain_ope[i] = self.O_num[i] - self.OP[i]
        Job_i = np.argmax(remain_ope)
        for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
            if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                Machine = j

        return Job_i, Machine

    # Composite dispatching rule 3
    def rule3(self):
        # 选出机器负载最小的机器及其对应的工件,
        for i in range(self.J_num):
            if self.OP[i] < self.O_num[i]:
                Job_i = i
        # print(Job_i, self.OP[Job_i])
                for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                    if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                        Machine = j
        try:
            mac_load = np.zeros(self.M_num)
            for i in range(self.M_num):
                mac_load[i] = sum(self.Machines[i].T)
            Machine = np.argmin(mac_load)
            # 所有工件中在该机器加工的工序
            jobs_time = np.zeros(self.J_num)  # 所有在该台设备的加工时间工件集合
            index = []
            for j in range(self.J_num):
                # print((j, self.OP[j], Machine))
                if self.OP[j] < self.O_num[j]:
                    if self.Process_time[j][self.OP[j]][Machine] > 0:  # 在该台设备的加工时间大于0的工件
                        jobs_time[j] = self.CTI[j]  # 工件j的最近完工工序结束时间
                        index.append(j)  # 存储不为0的工件索引
            TIME = jobs_time[index[:]]  # 加工时间不为0的时间
            Job_i = index[np.argmin(TIME)]  # 最早结束工件索引
        except:  # 选出还未加工完全的机器
            for i in range(self.M_num):
                if len(self.Machines[i].T) < self.O_num[i]:
                    Machine = i
            for j in range(self.J_num):
                if self.OP[j]<self.O_num[j]:
                    if self.Process_time[j][self.OP[j]][Machine] > 0:
                        Job_i = j

        return Job_i, Machine

    # Composite dispatching rule 4
    def rule4(self):
        for i in range(self.J_num):
            if self.OP[i] < self.O_num[i]:
                Job_i = i
        # print(Job_i, self.OP[Job_i])
                for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                    if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                        Machine = j
        end = self.JT[0]
        for i in range(self.J_num):
            if self.OP[i]<self.O_num[i]:
                if self.CTI[i]<end:
                    Job_i = i
                    end = self.CTI[i]
        # print(Job_i, self.OP[Job_i])
                    for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                        if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                            Machine = j
        # 选出当前设备最先结束加工的工件
        return Job_i, Machine

    # Composite dispatching rule 5
    def rule5(self):
        for i in range(self.J_num):
            if self.OP[i] < self.O_num[i]:
                Job_i = i
        # print(Job_i, self.OP[Job_i])
                for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                    if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                        Machine = j
        # 选出加工时间最短的工件
        total = self.JT[0]
        # Job_i = 0
        for i in range(self.J_num):
            if self.OP[i] < self.O_num[i]:
                if self.JT[i] < total:
                    Job_i = i
                    total = self.JT[i]
                    for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                        if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                            Machine = j
        return Job_i, Machine

    # Composite dispatching rule 6
    # return Job,Machine
    def rule6(self):
        for i in range(self.J_num):
            if self.OP[i] < self.O_num[i]:
                Job_i = i
        # print(Job_i, self.OP[Job_i])
                for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                    if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                        Machine = j
        # 选出加工时间最长的工件
        total = self.JT[0]
        # Job_i = 0
        for i in range(self.J_num):
            if self.OP[i] < self.O_num[i]:
                if self.JT[i] > total:
                    Job_i = i
                    total = self.JT[i]
                    for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                        if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                            Machine = j
        return Job_i, Machine

    def rule7(self):
        for i in range(self.J_num):
            if self.OP[i] < self.O_num[i]:
                Job_i = i
        # print(Job_i, self.OP[Job_i])
                for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                    if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                        Machine = j
        # 选出剩余加工时间最短的工件
        remain_time = self.JT[0]
        # Job_i = 0
        for i in range(self.J_num):
            if self.O_num[i] - self.OP[i] > 0:  # 存在剩余工件
                if self.JT[i] - sum(self.Jobs[i].T) < remain_time:
                    Job_i = i
                    remain_time = self.JT[i] - sum(self.Jobs[i].T)
                    for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):  # 该工件的工序在不同位置对应加工时间
                        if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                            Machine = j

        return Job_i, Machine

    def rule8(self):
        for i in range(self.J_num):
            if self.OP[i] < self.O_num[i]:
                Job_i = i
        # print(Job_i, self.OP[Job_i])
                for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                    if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                        Machine = j
        # 选出加工剩余工序最少的工件
        remain_ope = self.O_num[0]
        # Job_i = 0
        for i in range(self.J_num):
            if self.O_num[i] - self.OP[i] > 0:  # 存在剩余工件
                if self.O_num[i] - self.OP[i] < remain_ope:
                    Job_i = i
                    remain_ope = self.O_num[i] - self.OP[i]
                    for j in range(len(self.Process_time[Job_i][self.OP[Job_i]])):
                        if self.Process_time[Job_i][self.OP[Job_i]][j] > 0:
                            Machine = j

        return Job_i, Machine



    def reward(self, obs, obs_t, episode):
        #  obs[0-6]:设备平均利用率，设备利用率标准差、所有工序完成率、每个工件工序完成率平均值、平均工件工序完成率标准差、预估延迟程度(下一状态更好-小)、当前已延迟数量比例

        # print([Ta_t, Te_t, Ta_t1, Te_t1, U_t, U_t1])
        # rt = np.exp(Te_t)/np.exp(Te_t1) + np.exp(Ta_t)/np.exp(Ta_t1) + np.exp(U_t1)/np.exp(U_t)
        # rt = (obs_t[0] - obs[0]) * (1 + np.exp(-1 * (obs_t[0] - obs[0]))) * 1 / (1 + np.exp(-1 * episode))
        rt = (obs_t[0] - obs[0]) * (1 + np.exp(-1 * (obs_t[0] - obs[0])))
        return rt