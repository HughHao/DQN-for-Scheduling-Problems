# -*- coding: utf-8 -*-
# @Time : 2022/3/12 11:23
# @Author : hhq
# @File : draw_gantt.py
import numpy as np
import copy
import matplotlib.pyplot as plt
class Node:
    def __init__(self, PT, Ma):
        self.PT = PT
        self.Ma = Ma
        self.J_num = len(self.PT)
        self.O_num = [len(self.PT[i]) for i in range(self.J_num)]


    def color(self):  # 甘特图颜色生成函数
        color_ls = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F']
        col = ''
        for i in range(6):  # 6种颜色数字字母组合
            col += np.random.choice(color_ls)
        return '#' + col

    def draw_gantt(self, Start_list, O_list, C_J):
        colors = [self.color() for i in range(self.J_num)]
        self.Start_list = Start_list
        num_list = []
        for i, job in enumerate(O_list):
            num_list.append(job)

            op = num_list.count(job)  # 工序
            machine = self.Ma[job][op-1]  # 位置
            # print([job, op, machine])
            plt.barh(y=machine,left=self.Start_list[i], width=self.PT[job][op-1],height=0.5,color=colors[job],label=f'job{job+1}')
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