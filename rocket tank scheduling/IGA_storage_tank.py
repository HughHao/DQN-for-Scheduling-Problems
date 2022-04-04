# -*- coding: utf-8 -*-
# @Time : 2022/1/2 16:30
# @Author : hhq
# @File : IGA_storage_tank.py
import copy
import scipy.stats as stats
'''实现截尾正态分布'''
import time
import matplotlib.pyplot as plt
import copy
import numpy as np
import xlwt
import xlrd
from funcs import BEST_ONE, best_one, c_max, cross, FIT, FULL_time, new_time_table, mutation, normfun, pop_init, stochastic, Sub_time, excel, update_Comp,ave_cycle, PLOT, draw_bar, JOB_NUM, jobs
import random
import re
from collections import OrderedDict  # ：字典的子类，保留了他们被添加的顺序
from matplotlib.pyplot import MultipleLocator
# 设备编号
Equip_no = {'Milling': [0], 'FW1': [1], 'CHECK': [2, 3], 'CNC_v_l': [4], 'FW2': [5, 6], 'TEST': [7], 'XX_4Dock': [8],
            'XX_5Dock': [9], 'XX_6Dock': [10]}
ma_num = 11

# 操作时间服从结尾正态分布，0.8~1.2倍的时间。单底焊接以5%的概率返工
Process = [['Milling', 'FW1', 'CHECK', 'CNC_v_l'], ['Milling', 'FW1', 'CHECK', 'CNC_v_l'],
           ['FW2', 'CHECK', 'CNC_v_l', 'TEST']]
Tube_section = {'Sid': 4}  # 筒段  壁板
Short_jar = {'Sid': 4}  # 短壳  壁板
Single_bottom = {'Mp': 6, 'Tc': 1, 'Pf': 1}  # 单底  瓜瓣，顶盖，型材框

start = time.time()
# new_com = update_Comp(Components_time_table)  # 将原先的加工时间表改为服从截尾正态分布的时间表
iter_max = 20
pop_size = 10  # 种群规模
mu = 0.8  # 变异概率
cr = 0.1  # 交叉概率
sr = 0.5
pop = pop_init(pop_size, JOB_NUM)  # 一个染色体表示贮箱，没有扩展
# print(pop)
best_fit, best_chrom, end_time, CT, sub_order, machine_LIST = BEST_ONE(pop)  # 找出最佳个体和及其适应度
BEST_FIT = np.zeros(iter_max)  # 最佳完工时间初始化
BEST_FIT[0] = best_fit  # 第一次最佳值
CT_LIST = [[] for i in range(iter_max)]
for iter in range(iter_max):
    # 交叉
    for k in range(pop_size):
        if np.random.rand() > cr:
            new_chrom = cross(pop[k], best_chrom)
            pop[k] = new_chrom
    # 变异
    for k in range(pop_size):
        if np.random.rand() > mu:
            new_chrom = mutation(pop[k])
            pop[k] = new_chrom
    new_fit, new_chrom, end_ti, ct, new_sub_order, new_list = BEST_ONE(pop)
    if new_fit < best_fit:
        best_chrom = new_chrom
        best_fit = new_fit
        sub_order = new_sub_order
        machine_LIST = new_list
        CT = ct  # 26单个贮箱的加工时间
        end_time = end_ti
    ave_list2 = ave_cycle(best_chrom, CT)  # 平均周期
    BEST_FIT[iter] = best_fit
    CT_LIST[iter] = ave_list2
CT_LIST = np.array(CT_LIST)

'''e = np.zeros(ma_num)  # 全程机器结束时间
s = copy.deepcopy(e)  # 全程机器开始时间
# sub_time = []  # 每种贮箱的前面子零件的加工时间跨度，即总对接前花费的时间
end_time = []
subsub = np.zeros(len(best_chrom))
for j in range(len(best_chrom)):
    order = sub_order[j]
    st, en, _ = c_max(order, s, e)'''
'''process_0 = order[0]  # 找出零件名称，0代表筒段，1代表短壳，2代表单底
    process_1 = order[-1]
    process_name_0 = Process[process_0][0]  # 找出该零件第一道工序对应设备名称，其跨度为最后一个工序的完工时间减去第一道工序的开始时间
    process_name_1 = Process[process_0][-1]
    first_equip_no = Equip_no[process_name_0]  # 找出当前贮箱的第一个加工的零件的首刀工序对应的设备号
    # (2) 利用序列设备起始时间差
    last_equip_no = Equip_no[process_name_1]'''
    # (1) # sub_time.append(max(c_max(order, s=np.zeros(ma_num), e=np.zeros(ma_num))[1]))  # 每个贮箱的子零件完工需要时间
    # sub_time.append(max(en[:8])-min(s[:8]))  # （3）前8台设备最早开始时间与最晚时间之差
'''subsub[j] = max(en[7]-e[7], en[4]-e[4])  # (4)后两道工序加工前有两台设备结束，计算两者加工前后的时间差最大值即为该序列的跨度
    print(subsub)  # 单个贮箱的零件前四道工序加工时间
    s, e, _ = FULL_time(best_chrom[j], st, en)
    end_time.append(max(e))  # 最大结束时间为该

st = np.zeros(ma_num)  # 全程机器开始时间
en = np.zeros(ma_num)  # 全程机器结束时间

'''
'''for j in range(len(sub_order)):
    order = sub_order[j]
    s, e, _ = c_max(order, st, en)
    sub_time.append(max(e[4], e[7])-min(st[0],st[5],st[6]))
    start_time.append(end_time[j]-sum(Components_time_table[-1])-(max(e[4], e[7])-min(st[0],st[5],st[6])))
    st, en = s, e'''
'''
v = list(map(lambda x: x[0]-x[1], zip(end_time, start_time)))  # 每个贮箱的完工时间列表
ave_list = ave_cycle(best_chrom, v)'''
start_time = []
for j in range(len(sub_order)):
    start_time.append(end_time[j]-CT[j])

ave_list = ave_cycle(best_chrom, CT)
print(best_chrom)
print(sub_order)
print(best_fit)
print('三种贮箱的平均生产周期为：', ave_list)
# for i in range(len(best_chrom)):
end = time.time()
print("完成时间: %f s" % (end - start))
# plt.figure()
# plt.plot(BEST_FIT[:])
# plt.show()

# 完工时间跨度
plt.figure(1)
plt.plot(BEST_FIT[:])
plt.title('各贮箱焊接最优完工时间(h)', fontsize=16)
plt.xlabel('迭代次数', fontsize=12)
plt.ylabel('完工时间', fontsize=12)
x_major_locator = MultipleLocator(1)
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 三种贮箱的平均生产周期
plt.figure(2)
x_axix = list(range(1, iter_max+1))
xx4 = CT_LIST[:, 0]
xx5 = CT_LIST[:, 1]
xx6 = CT_LIST[:, 2]
plt.plot(x_axix, xx4, color='green', label='贮箱XX-4')
plt.plot(x_axix, xx5, color='red', label='贮箱XX-5')
plt.plot(x_axix, xx6, color='blue', label='贮箱XX-6')
for a, b in zip(x_axix, xx4):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=14)
for a, b in zip(x_axix, xx5):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=14)
for a, b in zip(x_axix, xx6):
    plt.text(a, b, '%.2f' % b, ha='center', va='bottom', fontsize=14)
# plt.rcParams.update({'font.size': 15})
plt.legend()  # 显示图例
plt.title('三种贮箱平均生产周期变化', fontsize=16)
plt.xlabel('迭代次数', fontsize=16)
plt.ylabel('平均在制周期', fontsize=16)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.gcf().subplots_adjust(left=None,top=None,bottom=None, right=None)
x_major_locator = MultipleLocator(1)
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# print(load_time_tables)


# 绘制甘特图
for i, order in enumerate(sub_order):
    order += [3, 3]
    sub_order[i] = order

# fit, _, _, machine_LIST = FIT(c)  # 根据26个贮箱的顺序即可计算完整的加工时间
load_time_tables, T = PLOT(best_chrom, sub_order, machine_LIST)

#从pyplot导入MultipleLocator类，这个类用于设置刻度间隔
plt.figure(3)
def color():# 甘特图颜色生成函数
    color_ls = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    col = ''
    for i in range(6):  # 6种颜色数字字母组合
        col += random.choice(color_ls)
    return '#'+col
colors = [color() for i in range(26)]  # 甘特图颜色列表,每个工件一个颜色

# 定义每台设备的开始时间和结束时间
ma_start, ma_end = [[] for i in range(ma_num)], [[] for i in range(ma_num)]

for i in load_time_tables:  # 对最佳染色体进行遍历，做出甘特图
    # print(i)  # 每个工件
    y = eval(re.findall('_(\d+)', i[0])[2])+1  # 正则表达式匹配工件数,找到_后面内部整数个数，机器号=工序号
    ma_start[y-1].append(i[1][0])  # 开始
    ma_end[y-1].append(i[1][-1])  # 结束
    """
    i = ['c24_9', [1715, 1736]]  # 9
    # \d匹配任何十进制数，它相当于类[0-9]
    # \d+如果需要匹配一位或者多位数的数字时用
    a = re.search("(a4)+", "a4a4a4a4a4dg4g654gb")   # 匹配一个或多个a4
    a = re.findall(r"你|好", "a4a4a你4aabc4a4dgg好dg4g654g")   #|或，或就是前后其中一个符合就匹配  #打印出 ['你', '好']
    """
    # eval() 函数用来执行一个字符串表达式，并返回表达式的值。
    label = eval(re.findall(r'(\d*?)_', i[0])[0]) + 1  # 正则表达式匹配机器数,找到_前面内部整数个数
    plt.barh(y=y, left=i[1][0], width=i[1][-1] - i[1][0], height=0.5, color=colors[label - 1],
             label=f'贮箱{label}')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['font.serif'] = ['KaiTi']
plt.rcParams['axes.unicode_minus'] = False
plt.title('各贮箱焊接最优调度甘特图',fontsize=16)
plt.xlabel('加工时间(h)',fontsize=12)
plt.ylabel('加工位置',fontsize=12)
# 第一个参数是点的位置，第二个参数是点的文字提示。
plt.yticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [r'铣边机', r'立式纵缝搅拌摩擦焊设备', r'无损检测1', r'无损检测2', r'数控立车',
                                                 r'数控悬臂式搅拌摩擦焊设备', r'运载火箭箱底空间曲线搅拌摩擦焊', r'试验区', r'XX-4贮箱总对接设备',
                                                 r'XX-5贮箱总对接设备', r'XX-6贮箱总对接设备'])
x_major_locator = MultipleLocator(200)
# 把x轴的刻度间隔设置为1，并存在变量里
y_major_locator = MultipleLocator(1)
# 把y轴的刻度间隔设置为10，并存在变量里
# # $表示特殊的字体，这边如果后期有需要可以上网查，空格需要转译，数学alpha可以用\来实现
ax = plt.gca()
# ax为两条坐标轴的实例
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
ax.yaxis.set_major_locator(y_major_locator)
# # 把y轴的主刻度设置为10的倍数
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# # 设置坐标标签字体大小
# 设置图例字体大小
# plt.rcParams.update({'font.size': 15})
# plt.legend(loc='upper right')
font = {'family':'Arial'  #'serif',
#         ,'style':'italic'
        ,'weight':'bold'  # 'normal'
#         ,'color':'red'
        ,'size':20
       }
plt.legend(prop=font)
handles, labels = plt.gca().get_legend_handles_labels()  # 标签去重
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())



# 定义利用率1有效时长占总各自开机时长的比重
plt.figure(4)
ma_makespan = np.zeros(ma_num)  # 每台设备的时间跨度
fs_m_rate = np.zeros(ma_num)
ma_rate = np.zeros(ma_num)
for se in range(ma_num):
    ma_makespan[se] = max(ma_end[se])-min(ma_start[se])
    ma_each = list(map(lambda x: x[0] - x[1], zip(ma_end[se], ma_start[se])))  # 每台设备利用率
    ma_rate[se] = sum(ma_each)/ma_makespan[se]
    fs_m_rate[se] = sum(ma_each)/T

labels = ['铣边机', '立式纵缝搅拌摩擦焊设备', '无损检测1', '无损检测2', '数控立车', '数控悬臂式搅拌摩擦焊设备',
          '运载火箭箱底空间曲线搅拌摩擦焊', '试验区', 'XX-4贮箱总对接设备', 'XX-5贮箱总对接设备', 'XX-6贮箱总对接设备']
draw_bar(labels, ma_rate)

# FS设备利用率=每台设备工作时长/完工时间
plt.figure(5)
draw_bar(labels, fs_m_rate)

plt.show()
print(ma_start)
print(ma_end)
print(ma_rate)
# print(load_time_tables)



# 添加一个表
wb = xlwt.Workbook()
ws = wb.add_sheet('sheet0')
ws.write(0, 0, '零件类型')
ws.write(0, 1, '数量')
ws.write(0, 2, '投入时间')
j = 1
for i in range(len(best_chrom)):
    job = best_chrom[i]
    ti = start_time[i]  # 单位：小时
    day = int(ti/24)  # 单位：天
    hour = int(ti - day * 24)  # 单位：转化为天后的小时整数
    minutes = (ti - day * 24 - hour) * 60  # 单位：分
    minute = int(minutes)
    second = abs(int((minutes - minute) * 60))
    tim = str(day) + ':' + str(hour) + ':' + str(minute) + ':' + str(second)
    # if day > 0:
    #     time = str(day) + ':'+str(hour) + ':'+str(minute) + ':'+str(second)
    # elif hour>0:
    #     time = str(hour) + ':'+str(minute) + ':'+str(second)
    # elif minute>0:

    if job == 0:
        ws.write(j, 0, 'XX-4壁板1')  # 时间计算有误
        ws.write(j, 1, 4)
        ws.write(j, 2, tim)
        ws.write(j+1, 0, 'XX-4壁板2')
        ws.write(j+1, 1, 8)
        ws.write(j+1, 2, tim)
        ws.write(j+2, 0, 'XX-4瓜瓣')
        ws.write(j+2, 1, 18)
        ws.write(j+2, 2, tim)
        ws.write(j+3, 0, 'XX-4顶盖')
        ws.write(j+3, 1, 3)
        ws.write(j+3, 2, tim)
        ws.write(j+4, 0, 'XX-4型材框')
        ws.write(j+4, 1, 3)
        ws.write(j+4, 2, tim)
    elif job == 1:
        ws.write(j, 0, 'XX-5壁板1')
        ws.write(j, 1, 20)
        ws.write(j, 2, tim)
        ws.write(j + 1, 0, 'XX-5壁板2')
        ws.write(j + 1, 1, 8)
        ws.write(j + 1, 2, tim)
        ws.write(j + 2, 0, 'XX-5瓜瓣')
        ws.write(j + 2, 1, 12)
        ws.write(j + 2, 2, tim)
        ws.write(j + 3, 0, 'XX-5顶盖')
        ws.write(j + 3, 1, 2)
        ws.write(j + 3, 2, tim)
        ws.write(j + 4, 0, 'XX-5型材框')
        ws.write(j + 4, 1, 2)
        ws.write(j + 4, 2, tim)
    else:
        ws.write(j, 0, 'XX-6壁板1')
        ws.write(j, 1, 12)
        ws.write(j, 2, tim)
        ws.write(j + 1, 0, 'XX-6壁板2')
        ws.write(j + 1, 1, 8)
        ws.write(j + 1, 2, tim)
        ws.write(j + 2, 0, 'XX-6瓜瓣')
        ws.write(j + 2, 1, 24)
        ws.write(j + 2, 2, tim)
        ws.write(j + 3, 0, 'XX-6顶盖')
        ws.write(j + 3, 1, 4)
        ws.write(j + 3, 2, tim)
        ws.write(j + 4, 0, 'XX-6型材框')
        ws.write(j + 4, 1, 4)
        ws.write(j + 4, 2, tim)
    j += 5
# 保存文件
wb.save('加工列表.xls')


f = open("sub_order.txt", 'a')
for i in sub_order:
    f.write(str(i).replace('[','') + '\n')
f.close()


g = open("best_chrom.txt", 'a')
for i in best_chrom:
    g.write(str(i) + ' ')
g.close()

h = open("start_time.txt", 'a')
for i in start_time:
    h.write(str(i)+' ')
h.close()

#读取excel文件

a = excel()  # 返回整个函数的值
a = list(map(int, a))
b = []
c = []
for i in a:
    if i == 6:
        b.append(2)
    elif i == 5:
        b.append(1)
    else:
        b.append(0)
j = 0
while j < len(b)/5:
    c.append(b[j*5])
    j += 1
# fit, _, _, machine_LIST = FIT(c)  # 根据26个贮箱的顺序即可计算完整的加工时间
# print(len(machine_LIST)==len(c))  # True