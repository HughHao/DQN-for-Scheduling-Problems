
#对强化学习运行环境的书写，定义了动作集a，状态集s，奖励集r，以及状态转移概率矩阵p

import numpy as np
import time
import copy


class Situation(object):#tk.Tk()是创建控件实例，这里不知道什么意思
    def __init__(self, NumOfMs, NumOfJs, dataSet, bestMakespan):
        super(Situation, self).__init__()
        self.NumOfMs = NumOfMs  # 机器总数
        self.NumOfJs = NumOfJs  # 工件总数
        self.dataSet = dataSet  # 加工信息表
        self.bestMakeSpan = bestMakespan  # 输入一个预设的makespan

        # 动作即为工件序号，固定了Q表纵向大小。不能是可选工序，否则Q表太大了，且随时变化
        self.actions = [i for i in range(self.NumOfJs)]
        #工件的总工序数列表
        self.totalOpsOfJs = []
        for i in range(len(dataSet)):
            self.totalOpsOfJs.append(len(dataSet[i])/2)

        self.TM = [0 for i in range(self.NumOfMs)]  # 各机器上最后一道工序的完工时间列表
        self.TJ = [0 for i in range(self.NumOfJs)]  # 各工件上最后一道工序的完工时间列表
        self.barList = [] # 甘特图条列表
        self.colors = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'aquamarine', \
                       'blanchedalmond', 'brown', 'cornsilk', 'darkseagreen', 'ghostwhite', 'gold', \
                       'ivory', 'lightblue', 'lightsalmon', 'lime', 'mediumblue', 'mistyrose', \
                       'moccasin', 'navajowhite', 'orchid', 'papayawhip', 'pink', 'powderblue']


    def reset(self):#重置工作环境
        # 单Agent，不区分机器
        self.iniOpList = []  # 等待的工序[1,2,3]列表，[1,2,3]表示O12加工时间为3
        for i in range(len(self.dataSet)):
            self.iniOpList.append([i, 0, self.dataSet[i][0], self.dataSet[i][1]]) #[工件序号，工序序号，机器序号，加工时间]
        #每一个episode后TM,TJ,barList都会清空
        self.TM = [0 for i in range(self.NumOfMs)]  # 各机器上最后一道工序的完工时间列表
        self.TJ = [0 for i in range(self.NumOfJs)]  # 各工件上最后一道工序的完工时间列表
        self.barList = []  # 甘特图条列表
        # return observation，即状态
        return self.iniOpList#返回初始可加工工序列表

    def step(self, observation, RL_Action):
        s = observation # 没对状态作更改
        #更新状态，动作
        s_ = copy.deepcopy(s)
        # 1删除状态中选择的工序
        for i in range(len(s_)):
            if s_[i][0] == RL_Action:
                ActionList = s_[i]
                s_.remove(s_[i])
                break

        # 2增加新工序
        job = ActionList[0]
        operation = ActionList[1]
        # 如果已经是最后一个工序了，那就没有新工序了
        if operation < self.totalOpsOfJs[job] - 1:
            newOperation = ActionList[1] + 1
            machine = self.dataSet[job][newOperation * 2]
            opTime = self.dataSet[job][newOperation * 2 + 1]
            s_.append([job, newOperation, machine, opTime])

        #3 生成bar，并添加到barList中，改变TM和TJ
        bar = []
        MIndex = ActionList[-2]
        JIndex = ActionList[0]
        bar.append(MIndex)  # 添加机器序号
        bar.append(ActionList[-1])  # 添加操作加工时间
        # 添加操作的起始时间
        # 取max(机器上的完工时间，工件的完工时间)
        maxValue = max(self.TM[MIndex], self.TJ[JIndex])
        bar.append(maxValue) #添加工序开始加工时间
        bar.append(JIndex)  # 添加工件序号
        self.barList.append(bar)

        # 相应的对应机器和工件的完工时间有所变化
        self.TM[MIndex] = maxValue + ActionList[-1]
        self.TJ[JIndex] = maxValue + ActionList[-1]


        # reward function
        NumOfWaitOps = len(s_)
        if NumOfWaitOps == 0:
            makespan = max(self.TM)
            if makespan < self.bestMakeSpan:
                reward = 1
                self.bestMakeSpan = makespan
            elif makespan == self.bestMakeSpan:
                reward = np.random.choice([-1,1])
            else:
                reward = -1
                #reward = self.bestMakeSpan-makespan
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward,  done

    def render(self):
        time.sleep(0.1)
