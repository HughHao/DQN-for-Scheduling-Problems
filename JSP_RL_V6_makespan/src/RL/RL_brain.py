"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
#强化学习算法的定义，本例中是Q-learning
#每个agent都有一个Q表，ft06例子是构造6个QLearningTable实例
import numpy as np
import pandas as pd


class QLearningTable:
    #__init__方法：构造器，用于定义Q-learning中的各个参数
    def __init__(self, actions, learning_rate, reward_decay, e_greedy):
        self.actions = actions  # a list，动作集
        self.lr = learning_rate #学习率
        self.gamma = reward_decay #回报的衰减系数
        self.epsilon = e_greedy #贪婪系数
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64) #定义Q表

    #choose_action方法用于实现epsilon_greedy策略
    def choose_action(self, observation):
        #一般来讲，observation肯定不是0，是0就是终止状态，直接done了
        #if len(observation) != 0:
        self.check_state_exist(str(observation))  # 先检查是否是新状态
        self.availableActions = []
        for i in range(len(observation)):
            self.availableActions.append(observation[i][0])#将当前状态的所有工件序号加入

        # action selection
        randomNum = np.random.uniform()
        #print('randomNum',randomNum)
        #print('self.epsilon',self.epsilon)
        if randomNum < self.epsilon:
            # choose best action
            state_action = self.q_table.loc[str(observation), self.availableActions]#选择可选动作列表中的几列
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            #print('0000action',action)
        else:
            # choose random action
            action = np.random.choice(self.availableActions)
            #print('1111action', action)
        #防止工件序号变成字符串型，无法index
        action = int(action)

        return action

    #learning方法用于更新Q表，是Q-learning的学习过程
    def learn(self, s, a, r, s_):
        nextAvaiActions = []
        #状态和动作不一样，状态是s_
        if len(s_) != 0:
            for i in range(len(s_)):
                nextAvaiActions.append(s_[i][0])  # 将当前状态的所有工件序号加入
        self.check_state_exist(str(s_))#首先检查当前状态是否存在于状态集s中，不存在就加入状态集s
        q_predict = self.q_table.loc[str(s), a] #将Q表中的值当作q_predict
        if str(s_) != 'terminal':#如果当前状态不是最终状态，就用即时奖励r，加上衰减系数乘以Q表中状态s_的最大Q值作为q_target
            q_target = r + self.gamma * self.q_table.loc[str(s_), nextAvaiActions].max()  # next state is not terminal

        else:#如果当前状态是最终状态，q_target就是即时奖励
            q_target = r  # next state is terminal
        self.q_table.loc[str(s), a] += self.lr * (q_target - q_predict)  # update


    #check_state_exist方法用于检测状态是否存在于原有的状态集合中，如果不在，便将新的状态加入到状态集合中
    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),#可以计算一下所有工序加工时间总和
                    index=self.q_table.columns,
                    name=state, #name得是字符串类型str(state)
                )
            )