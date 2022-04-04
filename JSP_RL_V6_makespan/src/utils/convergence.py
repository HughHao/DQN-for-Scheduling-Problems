import matplotlib.pyplot as plt

def getConvFig(makespanList):
    fig = plt.figure(figsize=(20, 8), dpi=80)
    ax1 = fig.add_subplot(111)
    ax1.set_title("makespan")
    ax1.plot(makespanList)
    plt.show()
    plt.savefig('MRL_Convergence-01')

def getMeanConvFig(episode, runNum, makespanTrails,rewardTrails):
    #输入：多个列表
    #多次实验结果的均值阴影图
    minM = []; maxM = []; meanM = []
    minR = []; maxR = []; meanR = []
    for i in range(episode):
        mList=[]
        rList=[]
        for j in range(runNum):
            mList.append(makespanTrails[j][i])
            rList.append(rewardTrails[j][i])
        minM.append(min(mList))
        maxM.append(max(mList))
        mean_makespan = sum(mList)/runNum
        meanM.append(mean_makespan)
        minR.append(min(rList))
        maxR.append(max(rList))
        mean_reward = sum(rList)/runNum
        meanR.append(mean_reward)
    #根据数据绘制图形
    fig = plt.figure(figsize=(20, 8), dpi=80)
    x = list(range(episode))
    plt.plot(x, meanM, c='blue', alpha=1)
    plt.fill_between(x, minM, maxM, facecolor='blue', alpha=0.1)
    plt.plot(x, meanR, c='orange', alpha=1)
    plt.fill_between(x, minR, maxR, facecolor='orange', alpha=0.1)
    plt.show()
