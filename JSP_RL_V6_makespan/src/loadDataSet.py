
def loadDataSet(fileName):
    '''加载txt/jsp文件，获得作业数、机器数、和数据集'''
    fr = open(fileName)
    lines = fr.readlines()

    #从第一行的信息获取作业数和机器数
    InfoLine = lines[0].strip().split()#以所有空字符分隔，包括空格、换行符、制表符
    NumOfJs = int(InfoLine[0])#得到作业数
    NumOfMs = int(InfoLine[1])#得到机器数

    #从剩余行的信息获取作业相关操作信息
    dataSet = []
    for line in lines[1:NumOfJs+1]:#不写成lines[1:]，是因为最后可能有空行
        curLine = line.strip().split()
        fltLine = list(map(int,curLine))#map函数将数据映射成整数
        #注意可能要改float，如果有数据集的加工工时不是整数
        dataSet.append(fltLine)

    return NumOfMs, NumOfJs, dataSet

NumOfMs, NumOfJs, dataSet = loadDataSet('test_data/ft06.txt')