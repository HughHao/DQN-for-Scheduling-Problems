# -*- coding: utf-8 -*-
# @Time : 2022/3/10 14:47
# @Author : hhq
# @File : Object_for_JM.py
class Object:  # 对象为机器或工件
    def __init__(self, i):
        self.i = i
        self.Start = []
        self.End = []
        self.T = []  # 加工时间集合
        self.assign_for = []  # 选出的工件、机器集合
        self.Idle = []

    def _add(self, S, E, obs, t):
        # obs:安排的对象
        self.Start.append(S)  # 每个工件或机器加工的工序的开始时间集合，做甘特图
        self.End.append(E)  # 集合
        self.Start.sort()
        self.End.sort()
        self.T.append(t)  # 加工时间
        self.assign_for.insert(self.End.index(E), obs)  # 在某个结束时间对应位置插入新对象  insert() 函数用于将指定对象插入列表的指定位置，被插入的位置原先元素后移或新增




    def idle_time(self):
        # Idle = []  # 闲置时间起始，间隔时间段
        try:
            if self.Start[0] != 0 and len(self.Idle) == 0:  # 第一个开始时间不为0
                self.Idle.append([0, self.Start[0]])
            if self.Start[-1] - self.End[-2] > 0:
                K = [[self.End[-2], self.Start[-1]]]
                self.Idle.extend(K)  #
            # K = [[self.End[i], self.Start[i + 1]] for i in range(len(self.End)) if self.Start[i + 1] - self.End[i] > 0]
            # self.Idle.extend(K)  #
        except:
            pass