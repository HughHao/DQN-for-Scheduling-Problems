# -*- coding: utf-8 -*-
# @Time : 2022/3/15 22:12
# @Author : hhq
# @File : new_DQN.py
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
import random
from collections import deque  # 类似于list的容器，可以快速从头部或尾部添加、删除元素
from tensorflow.keras import layers, models
from Environment import Situation
from tensorflow.keras.optimizers import Adam
from loadDataSet import NumOfMs, NumOfJs, machine_list, time_list
import matplotlib.pyplot as plt


class DQN:
    def __init__(self, ):
        self.Hid_Size = 30

        # ------------Hidden layer=5   30 nodes each layer--------------
        model = models.Sequential()
        model.add(layers.Input(
            shape=(41,)))  # 此处表明其输入必须是二维数据,行为样本数，列为input_num，input_num为特征数，若shape=(input_num,2)，则特征向量维度为input_num，2
        model.add(layers.Dense(self.Hid_Size, name='l1'))  # dense是keras里的全连层函数
        model.add(layers.Dense(self.Hid_Size, name='l2'))
        model.add(layers.Dense(self.Hid_Size, name='l3'))
        model.add(layers.Dense(self.Hid_Size, name='l4'))
        model.add(layers.Dense(8, name='l5'))  # 最后一层表示输出层，输出维度和输入维度一致
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=0.001))
        # # model.summary()
        self.model = model
        # ------------Q-network Parameters-------------
        # self.act_dim = [1, 2, 3, 4, 5, 6]  # 神经网络的输出节点  6种复合规则
        # self.obs_n = [0, 0, 0, 0, 0, 0]  # 神经网路的输入节点  6种状态指标
        self.gama = 0.95  # γ经验折损率
        # self.lr = 0.001  # 学习率
        self.global_step = 0
        # self.update_target_steps = 200  # 更新目标函数的步长
        self.target_model = self.model  # 目标Q初始化为Q
        self.update_target_soft = 0.2

        # -------------------Agent-------------------
        self.e_greedy = 0.6
        self.e_greedy_decrement = 0.0001
        self.L = 1000  # Number of training episodes L

        self.makespan = 300
        # for i in range(J_num):
        #     self.Total_tard += JT[i] - D[i]
        self.mu_increment = 1 / self.L

        # ---------------Replay Buffer---------------
        self.buffer = deque(maxlen=1000)  # 类似于list的容器，可以快速的在队列头部和尾部添加、删除元素  存储
        self.Batch_size = 10  # Batch Size of Samples to perform gradient descent  执行梯度下降的样本大小

    # 目标Q更新
    def replace_target(self):  # 重置目标函数，将其按照
        self.target_model.get_layer(name='l1').set_weights(
            [(1 - self.update_target_soft) * i + self.update_target_soft * j
             for i, j in zip(self.target_model.get_layer(name='l1').get_weights(),
                             self.model.get_layer(name='l1').get_weights())])
        self.target_model.get_layer(name='l2').set_weights(
            [(1 - self.update_target_soft) * i + self.update_target_soft * j
             for i, j in zip(self.target_model.get_layer(name='l2').get_weights(),
                             self.model.get_layer(name='l2').get_weights())])
        self.target_model.get_layer(name='l3').set_weights(
            [(1 - self.update_target_soft) * i + self.update_target_soft * j
             for i, j in zip(self.target_model.get_layer(name='l3').get_weights(),
                             self.model.get_layer(name='l3').get_weights())])
        self.target_model.get_layer(name='l4').set_weights(
            [(1 - self.update_target_soft) * i + self.update_target_soft * j
             for i, j in zip(self.target_model.get_layer(name='l4').get_weights(),
                             self.model.get_layer(name='l4').get_weights())])
        self.target_model.get_layer(name='l5').set_weights(
            [(1 - self.update_target_soft) * i + self.update_target_soft * j
             for i, j in zip(self.target_model.get_layer(name='l5').get_weights(),
                             self.model.get_layer(name='l5').get_weights())])

        # self.target_model = self.model
        # print(self.target_model == self.model)

    def replay(self):
        # 抽取样本，训练模型
        minibatch = random.sample(self.buffer, self.Batch_size)
        for state, action, reward, next_state, done in minibatch:  # (obs, at, r_t, obs_t, done)
            # reward由state，action和next_state得出
            if not done:
                k = np.argmax(
                    self.model.predict(next_state)[0])  # state：U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, Tard_e, Tard_a
                # k为原模型预测出的动作index
                # target = (reward + self.gama *
                #           np.argmax(self.target_model.predict(next_state)))
                target = (reward + self.gama *  # 目标值为目标模型选择最大预测值
                          self.target_model.predict(next_state)[0][k])  # 相当于y(DDQN)t，是实际的状态、动作并转移后的奖励和期望计算出来的
                # 动作k对应的奖励函数
            else:
                target = reward  # 如果最终状态
            target_f = self.model.predict(
                state)  # 模型输入s输出Q(s,a)  [[0.11677918  0.09287173 -0.3526993   0.0677374  0.12638253 -0.01366934]]
            # 是模型预测出来的，需要修正的
            '''target_f用于训练Q'''
            # print(target_f)
            # print('aa')
            target_f[0][action] = target  # 更新yj  action=0,1,...,5
            '''target_f输出看看是什么，如何根据state得到预测值'''
            self.model.fit(state, target_f, epochs=1, verbose=0)  # 模型训练
        self.global_step += 1

    def Select_action(self, obs, mu):  # 根据observation进行决策，obs相当于state
        # obs=np.expand_dims(obs,0)
        Q_values = self.model.predict(obs)[0]
        maxQ = max(Q_values)
        # print(Q_values)
        ac_e = np.zeros(len(Q_values))  # 总长度为6 行动
        for i in range(len(Q_values)):  #
            if maxQ != 0:
                Q_values[i] /= maxQ
            ac_e[i] = np.exp(mu * Q_values[i])  # 行动ac被选择的软计算处理
        # print(ac_e)
        Pi = np.zeros(len(Q_values))  #
        Qi = np.zeros(len(Q_values))
        act = np.argmax(Q_values)  # 最大值的动作
        p = random.random()

        for i in range(len(Q_values)):
            Pi[i] = ac_e[i] / sum(ac_e)
            Qi[i] = sum(Pi[:i + 1])
            if Qi[i] > p:
                act = i
                break

        return act

    def _append(self, exp):
        self.buffer.append(exp)

    def main(self, NumOfMs, NumOfJs, machine_list, time_list):
        '''
        :param J_num: 工件数
        :param M_num: 机器数
        :param O_num: 工序总数
        :param Process_time: 加工时间（包含同工序的不同机器）
        :return:
        '''
        k = 0  # 批量的计数
        x = []
        make_max = []  # makespan
        TR = []  # 序列奖励
        mu = 1
        O_num = np.zeros(NumOfJs)
        for jo in range(NumOfJs):
            O_num[jo] = len(machine_list[jo])

        for i in range(self.L):
            Total_reward = 0
            x.append(i + 1)  # episode_num i
            print('-----------------------开始第', i + 1, '次训练------------------------------')
            done = False  # 停止训练标记
            Sit = Situation(NumOfMs, NumOfJs, machine_list, time_list)  # Processing_time由案例生成得到
            # obs = [0 for i in range(7)]  # 7种观察-状态特征 np.expand_dims:用于扩展数组的形状
            obs = Sit.Features()  # 初始状态是否为0？？？
            obs = np.expand_dims(obs, 0)  # 将obs扩展为二维array数组,存储特征数据,模型输入必须是二维数据
            JOB_MACHINE = []
            sequence = []

            ''' 初始状态 '''
            for j in range(sum(O_num)):  # 对工序遍历，每次选出一个动作和机器

                # print(obs)
                '''选择动作'''

                at = self.Select_action(obs, mu)  # 选择工件和机器  使用Q模型选出动作索引，根据softmax函数

                # print(at)
                if at == 0:
                    Job, Machine = Sit.rule1()
                if at == 1:
                    Job, Machine = Sit.rule2()
                if at == 2:
                    Job, Machine = Sit.rule3()
                if at == 3:
                    Job, Machine = Sit.rule4()
                if at == 4:
                    Job, Machine = Sit.rule5()
                if at == 5:
                    Job, Machine = Sit.rule6()
                if at == 6:
                    Job, Machine = Sit.rule7()
                if at == 7:
                    Job, Machine = Sit.rule8()
                # at_trans=self.act[at]
                # ？？？？？原有：？？？print('这是第', j, '道工序>>', '执行action:', at, ' ', '将工件', at_trans[0], '安排到机器', at_trans[1])
                JOB_MACHINE.append((Job, Machine))
                '''每经过一次工序（包含工件及机器）更新学习状态和调度进程'''
                if j == sum(O_num) - 1:
                    done = True
                    print(JOB_MACHINE)
                Sit.scheduling(Job, Machine)  # 选择机器和工件后更新调度以及计算状态特征的数据
                obs_t = Sit.Features()  # 更新8个状态特征，U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, Tard_e
                # print(obs_t)
                obs_t = np.expand_dims(obs_t,
                                       0)  # 用于扩展数组的形状,增加维度,将其变为2维1*7数据 U_ave, U_std, CRO_ave, CRJ_ave, CRJ_std, TARD, Tard_a
                # print(obs[0][5]-obs_t[0][5])
                # obs = obs_t
                # obs = np.expand_dims(obs, 0)
                # print(obs,obs_t)
                '''执行动作后，根据新状态获得奖励'''  # 奖励函数根据第6个前后状态进行判断
                r_t = Sit.reward(obs[0], obs_t[0], i)  # 根据新的状态获得奖励

                sequence.append([obs, at, r_t, obs_t, done])  # 将此次结果存储到暂存间
                # '''将此数据记录'''
                # self._append((obs, at, r_t, obs_t, done))  # 将词条数据（状态特征、动作选择、奖励、下个状态、结束标志）存储
                # '''记录大小为Batch_size的数据,训练数据，更新Q'''
                # if k > self.Batch_size:
                #     # batch_obs, batch_action, batch_reward, batch_next_obs, done = self.sample()
                #     self.replay()  # 更新^Q，即目标Q
                #     self.replace_target()  # 更新目标Q的参数
                # Total_reward += r_t
                obs = obs_t  # 状态更新
                Total_reward += r_t
            mu = min(2.0, mu + self.mu_increment)

            Job = Sit.Jobs
            # E = 0
            # K = [i for i in range(len(Job))]  # 工件遍历
            End = []
            for Ji in range(len(Job)):
                End.append(max(Job[Ji].End))
            Cmax = max(End)

            for se in range(sum(O_num)):
                """sequence长度为所有工序数"""
                list = sequence[se]
                # list[2] += 1/(1 + np.exp(-1 * self.Total_tard/total_tardiness))
                # list[2] *= self.Total_tard/total_tardiness
                list[2] += 1 / Cmax  # 索引2表示奖励
                '''将此数据记录'''
                # self._append(tuple(list))  # 将词条数据（状态特征、动作选择、奖励、下个状态、结束标志）存储

                # print(self.buffer)
            '''记录大小为Batch_size的数据,训练数据，更新Q'''
            k += 1  # 样本数
            if k > self.Batch_size:
                # batch_obs, batch_action, batch_reward, batch_next_obs, done = self.sample()
                self.replay()  # 更新^Q，即目标Q
            self.replace_target()  # 更新目标Q的参数
            if self.makespan >= Cmax:
                self.makespan = Cmax
                for se in range(sum(O_num)):
                    """sequence长度为所有工序数"""
                    list = sequence[se]
                    self._append(tuple(list))  # 将词条数据（状态特征、动作选择、奖励、下个状态、结束标志）存储
            print('<<<<<<<<<-----------------makespan:', Cmax, '------------------->>>>>>>>>>')
            make_max.append(Cmax)
            print('<<<<<<<<<-----------------reward:', Total_reward, '------------------->>>>>>>>>>')
            TR.append(Total_reward)
            # plt.plot(K,End,color='y')
            # plt.plot(K,D,color='r')
            # plt.show()
        plt.figure(1)
        plt.plot(make_max[:])
        plt.figure(2)
        plt.plot(x, TR)
        plt.show()
        return Total_reward


d = DQN()
d.main(NumOfMs, NumOfJs, machine_list, time_list)