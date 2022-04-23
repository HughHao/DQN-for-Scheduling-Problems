# -*- coding: utf-8 -*-
# @Time : 2022/4/21 22:01
# @Author : hhq
# @File : DQN_torch.py
# 把一个大类分成多个小类


import numpy as np
import time
import random
from collections import deque  # 类似于list的容器，可以快速从头部或尾部添加、删除元素
import torch
import torch.nn as nn
import torch.optim as optim
from Environment import Situation
from Instance import Process_time, O_num, J_num, M_num
import matplotlib.pyplot as plt
import cupy as cp

# use_gpu = torch.cuda.is_available()
use_gpu = 0


def use_cupy():
    if use_gpu:
        pp = cp
    else:
        pp = np
    return pp


def use_GPU(v):
    if use_gpu:
        v = v.cuda()
    else:
        v = v
    return v

pp = use_cupy()


class QNetwork(nn.Module):
    def __init__(self, input_size, fc1_dims, output_size):
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc3 = nn.Linear(fc1_dims, 40)
        self.fc4 = nn.Linear(40, 30)
        self.fc2 = nn.Linear(30, output_size)
        self.relu = nn.ReLU()

        self.device = torch.device("cuda" if use_gpu else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = use_GPU(x)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)

        return out


# self.target_model = self.model
# print(self.target_model == self.model)
class ReplayBuffer(object):
    def __init__(self, batch_size, input_size, hid_size, out_size):
        # ---------------Replay Buffer---------------

        self.Batch_size = batch_size  # Batch Size of Samples to perform gradient descent  执行梯度下降的样本大小
        self.mse_loss = nn.CrossEntropyLoss()
        self.state_size = input_size
        self.hid_sie = hid_size
        self.action_size = out_size
        self.model = QNetwork(self.state_size, self.hid_sie, self.action_size)
        self.model = use_GPU(self.model)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def replay(self, buffer, action_size, model, target_model, global_step, gama):
        # 抽取样本，训练模型
        minibatch = random.sample(buffer, self.Batch_size)
        pre, rel = [], []
        for state, action, reward, next_state, done in minibatch:  # (obs, at, r_t, obs_t, done)
            # reward由state，action和next_state得出
            if not done:
                k = torch.argmax(
                    model.forward(next_state.to(torch.float32)))  # state：U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, Tard_e, Tard_a
                # k相当于action集合obs.to(torch.float32)
                # target = (reward + self.gama *
                #           np.argmax(self.target_model.predict(next_state)))
                target = (reward + gama *
                          target_model.forward(next_state.to(torch.float32))[k])  # 相当于y(DDQN)t，是实际的状态、动作并转移后的奖励和期望计算出来的
                # 动作k对应的奖励函数
            else:
                target = reward  # 如果最终状态
            target_f = model.forward(  # 目标值
                state.to(torch.float32))  # 模型输入s输出Q(s,a)  [[0.11677918  0.09287173 -0.3526993   0.0677374  0.12638253 -0.01366934]]
            tt = target_f.clone().detach()
            ta = target_f.clone().detach()
            ta[action] = target.clone().detach()
            pre.append(tt.reshape(1, action_size))
            rel.append(ta.reshape(1, action_size))
        pprree = torch.cat(pre[:])
        rreell = torch.cat(rel[:])
        # 是模型预测出来的，需要修正的
        '''target_f用于训练Q'''
        loss = self.mse_loss(pprree, rreell)
        loss.requires_grad_(True)
        '''target_f输出看看是什么，如何根据state得到预测值'''
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        target_model = model
        global_step += 1

        return model, target_model, global_step


class Agent:
    def __init__(self, gama, global_step, update_target_soft, e_greedy,
                 e_gre_de, L, makespan, input_size, hid_size, out_size, batch_size, maxlen):
        # ------------Q-network Parameters-------------
        # self.act_dim = [1, 2, 3, 4, 5, 6]  # 神经网络的输出节点  6种复合规则
        # self.obs_n = [0, 0, 0, 0, 0, 0]  # 神经网路的输入节点  6种状态指标
        self.gama = gama  # γ经验折损率
        # self.lr = 0.001  # 学习率
        self.global_step = global_step
        # self.update_target_steps = 200  # 更新目标函数的步长
        self.update_target_soft = update_target_soft
        # -------------------Agent-------------------
        self.e_greedy = e_greedy
        self.e_greedy_decrement = e_gre_de
        self.L = L  # Number of training episodes L
        self.makespan = makespan
        # for i in range(J_num):
        #     self.Total_tard += JT[i] - D[i]
        self.state_size = input_size
        self.hid_sie = hid_size
        self.action_size = out_size
        self.mu_increment = 1 / self.L
        self.model = QNetwork(self.state_size, self.hid_sie, self.action_size)
        self.target_model = QNetwork(self.state_size, self.hid_sie, self.action_size)
        self.batch_size = batch_size
        self.buffer = deque(maxlen=maxlen)  # 类似于list的容器，可以快速的在队列头部和尾部添加、删除元素  存储
        self.Replay = ReplayBuffer(self.batch_size, self.state_size, self.hid_sie, self.action_size)
        self.k = 0  # 批量的计数
        self.x = []  # 幕数
        self.make_max = []  # makespan
        self.TR = []  # 序列奖励
        self.mu = 1

    def Select_action(self, obs, mu):  # 根据observation进行决策，obs相当于state
        # obs=np.expand_dims(obs,0)
        Q_values = self.model.forward(obs.to(torch.float32))  # tensor
        A = Q_values.clone().detach()
        maxQ = max(A)
        # print(Q_values)
        # print(Q_values)
        ac_e = pp.zeros(len(A))  # 总长度为6 行动
        for i in range(len(A)):  #
            if maxQ != 0:
                A[i] = torch.div(A[i], maxQ)  # 归一化
                ac_e[i] = torch.exp(mu * A[i])  # 行动ac被选择的软计算处理
        # print(ac_e)
        Pi = np.zeros(len(A))  # 每个动作被选择的概率
        Qi = np.zeros(len(A))  # 动作选择的累计概率
        # if pp.random.random() > self.e_greedy:
        #     act = torch.argmax(A)  # 最大值的动作
        # else:
        #     act = np.random.choice(self.action_size)

        for i in range(len(A)):
            Pi[i] = ac_e[i] / sum(ac_e)
            Qi[i] = sum(Pi[:i + 1])
            if Qi[i] > np.random.random():
                act = i
                break

        return act

    def _append(self, exp):
        self.buffer.append(exp)

    def targer_fun(self, Sit):
        Jobb = Sit.Jobs  # 所有工件属性集合
        End = []
        for Ji in range(len(Jobb)):
            End.append(max(Jobb[Ji].End))
        Cmax = max(End)
        return Cmax

    def main(self, J_num, M_num, O_num, Process_time):
        '''
            :param J_num: 工件数
            :param M_num: 机器数
            :param O_num: 工序总数
            :param Process_time: 加工时间（包含同工序的不同机器）
            :return:
        '''
        # 重置

        for i in range(self.L):
            Total_reward = 0
            self.x.append(i + 1)  # episode_num i
            print('-----------------------开始第', i + 1, '次训练------------------------------')
            done = False  # 停止训练标记
            Sit = Situation(J_num, M_num, O_num, Process_time)  # Processing_time由案例生成得到
            # obs = [0 for i in range(7)]  # 7种观察-状态特征 np.expand_dims:用于扩展数组的形状
            obs = Sit.Features()  # 初始状态是否为0？？？
            obs = torch.tensor(obs)
            # 每个幕重置
            JOB_MACHINE = []
            sequence = []

            ''' 初始状态 '''
            for j in range(sum(O_num)):  # 对工序遍历，每次选出一个动作和机器

                # print(obs)
                '''选择动作'''
                # 根据状态选择动作，状态变化时动作空间也随之变化
                at = self.Select_action(obs, self.mu)  # 选择工件和机器  使用Q模型选出动作索引，根据softmax函数
                # Sit的属性self.OP
                # print(Sit.OP)
                # print(at)
                if at == 0:
                    Job, Machine = Sit.rule1()
                elif at == 1:
                    Job, Machine = Sit.rule2()
                elif at == 2:
                    Job, Machine = Sit.rule4()
                elif at == 3:
                    Job, Machine = Sit.rule4()
                elif at == 4:
                    Job, Machine = Sit.rule5()
                elif at == 5:
                    Job, Machine = Sit.rule6()
                elif at == 6:
                    Job, Machine = Sit.rule7()
                else:
                    Job, Machine = Sit.rule8()
                JOB_MACHINE.append(Job)
                '''每经过一次工序（包含工件及机器）更新学习状态和调度进程'''
                if j == sum(O_num) - 1:
                    done = True
                    print(JOB_MACHINE)
                # rb = self.targer_fun(Sit)
                Sit.scheduling(Job, Machine)  # 选择机器和工件后更新调度以及计算状态特征的数据
                obs_t = Sit.Features()  # 更新8个状态特征，U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, Tard_e
                # print(obs_t)
                obs_t = torch.tensor(obs_t)
                '''执行动作后，根据新状态获得奖励'''  # 奖励函数根据第6个前后状态进行判断
                # ra = self.targer_fun(Sit)
                r_t = Sit.reward(obs[6], obs_t[6], i)  # 根据新的状态获得奖励

                sequence.append([obs, at, r_t, obs_t, done])  # 将此次结果存储到暂存间
                obs = obs_t  # 状态更新
                Total_reward += r_t
            self.mu = min(2.0, self.mu + self.mu_increment)
            self.e_greedy += self.e_greedy_decrement

            Cmax = obs_t[6]

            for se in range(sum(O_num)):
                """sequence长度为所有工序数"""
                list = sequence[se]
                list[2] = list[2].clone().detach() + 1 / Cmax.clone().detach()  # 索引2表示奖励
                '''将此数据记录'''
            '''记录大小为Batch_size的数据,训练数据，更新Q'''
            self.k += 1  # 样本数
            if self.k > self.batch_size:
                # batch_obs, batch_action, batch_reward, batch_next_obs, done = self.sample()
                self.model, self.target_model, self.global_step = self.Replay.replay(self.buffer, self.action_size, self.model, self.target_model, self.global_step, self.gama)  # 更新^Q，即目标Q
            # self.target_model = self.model  # 更新目标Q的参数
            if self.makespan >= Cmax:
                self.makespan = Cmax
                for se in range(sum(O_num)):
                    """sequence长度为所有工序数"""
                    list = sequence[se]
                    self._append(list)  # 将词条数据（状态特征、动作选择、奖励、下个状态、结束标志）存储
            print('<<<<<<<<<-----------------makespan:', Cmax, '------------------->>>>>>>>>>')
            self.make_max.append(Cmax)
            print('<<<<<<<<<-----------------reward:', Total_reward, '------------------->>>>>>>>>>')
            self.TR.append(Total_reward)
            # plt.plot(K,End,color='y')
            # plt.plot(K,D,color='r')
            # plt.show()
        time_end = time.time()
        plt.figure(1)
        plt.plot(self.make_max[:])
        plt.figure(2)
        plt.plot(self.x, self.TR)
        plt.show()
        return time_end

time_start = time.time()

agent = Agent(0.95, 0, 0.2, 0.8, 0.001, 100000, 300, 7, 30, 8, 10, 1000)

time_end = agent.main(J_num, M_num, O_num, Process_time)
print("花费时间：", time_end - time_start)
