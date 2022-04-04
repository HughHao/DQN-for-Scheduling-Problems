
def getTxt(runNum, episode, valueTrails, prefixName):
    #将valueTrail以txt的形式进行存储，保存数据，随时可调
    for i in range(runNum):
        filename = prefixName+str(i)
        f = open('result/'+filename+'.txt','w')
        for j in range(episode):
            f.write(str(valueTrails[i][j])+'\n')
        f.close()

def readTxt(runNum, prefixName):
    #读取txt文件成valueTrails格式
    valueTrails = []
    for i in range(runNum):
        valueList=[]
        filename = prefixName + str(i)
        f = open('result/' + filename + '.txt')
        lines = f.readlines()
        for line in lines:
            curLine = line.strip().split()
            fltLine = int(curLine[0]) # map函数将数据映射成整数
            # 注意可能要改float，如果有数据集的加工工时不是整数
            valueList.append(fltLine)
        valueTrails.append(valueList)
    return valueTrails
