import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.patches as mpatches

def Colourlist_Generator(n):
    #获得n个随机颜色
    '''有很小的几率颜色重复，后面再重新设置吧'''
    Rangelist = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    n = int(n)
    Colours = []             #空列表，用于插入n个表示颜色的字符串
    j = 1
    while j <= n:            #循环n次，每次在0到14间随机生成6个数，在加“#”符号，次次插入列表
        colour = ""          #空字符串，用于插入字符组成一个7位的表示颜色的字符串（第一位为#，可最后添加）
        for i in range(6):
            colour += Rangelist[np.random.randint(0,14)]    #用randint生成0到14的随机整数作为索引
        colour = "#"+colour                              #最后加上不变的部分“#”
        Colours.append(colour)
        j = j+1
    return Colours

def getGanttBars(total_operation, NumOfJs, NumOfMs, OSIndividual, MSIndividual, OStimeIndividual):
    # 获得所有工序的甘特条[[机器，操作加工时间，操作起始时间，工件序号]，...]
    # individual表示pop中的一个个体，例如pop[0]，对应加工时间IndiOsTime

    TM = np.zeros(NumOfMs)  # 每个机器上的完工时间
    TJ = np.zeros(NumOfJs)  # 每个工件的完工时间
    barList = []  # 所有甘特图条列表

    # 按顺序解析pop中的数据
    for j in range(total_operation):
        barList .append([])
        barList[j].append(MSIndividual[j])  # 添加机器序号
        barList[j].append(OStimeIndividual[j])  # 添加操作加工时间
        # 添加操作的起始时间
        # 取max(机器上的完工时间，工件的完工时间)
        MIndex = int(MSIndividual[j] - 1)
        JIndex = int(OSIndividual[j] - 1)
        maxValue = max(TM[MIndex], TJ[JIndex])
        barList[j].append(maxValue)
        barList[j].append(OSIndividual[j])  # 添加工件序号

        # 相应的对应机器和工件的完工时间有所变化
        TM[MIndex] = maxValue + OStimeIndividual[j]
        TJ[JIndex] = maxValue + OStimeIndividual[j]
    return barList

def draw_gantt(total_operation, NumOfMs, NumOfJs, barList):
    print("barList",barList)
    # 画布设置，大小与分辨率
    fig = plt.figure(figsize=(20, 8), dpi=80)
    ax = fig.add_subplot(111)

    x_major_locator = MultipleLocator(5)  # x轴刻度间隔设为5
    y_major_locator = MultipleLocator(1)  # y轴刻度间隔设为1
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(0, NumOfMs+1)  # y轴刻度总长为机器数
    # XY轴刻度标签大小
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=10)
    # XY轴标签
    ax.set_xlabel('process time', fontsize=20)
    ax.set_ylabel('machine', fontsize=20)

    colours = Colourlist_Generator(NumOfJs)
    for i in range(total_operation):
        ax.barh(barList[i][0], barList[i][1], height = 0.2, left = barList[i][2], color = colours[int(barList[i][3]-1)], edgecolor = "black")

    #图例标签
    labels = ['']*NumOfJs
    for job in range(NumOfJs):
        labels[job] = "job%d" % (job + 1)

    # 图例绘制
    patches = [mpatches.Patch(color=colours[i], label="{:s}".format(labels[i])) for i in range(NumOfJs)]
    plt.legend(handles=patches, loc=4)

    #plt.title("Flexible Job Shop Solution")
    #plt.savefig('gantt.svg')
    plt.show()
