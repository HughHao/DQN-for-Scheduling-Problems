#程序的入口，相当于main方法

from src.RL.train import train
from src.utils.convergence import getConvFig,getMeanConvFig
from src.utils.txt import getTxt,readTxt
import argparse


parser = argparse.ArgumentParser(description='JSP-RL')
parser.add_argument('--fileName', type=str, default='test_data/test.txt',
                    help='test data file name')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.1)')
parser.add_argument('--gamma', type=float, default=0.9,
                    help='discount factor for rewards (default: 0.9)')
parser.add_argument('--epsilon', type=float, default=0.9,
                    help='epsilon greedy (default: 0.1)')
parser.add_argument('--epsilon_increment', type=float, default=0.099,
                    help='epsilon cumulates epsilon_increment as the episode increases')
parser.add_argument('--episode', type=int, default=1000,
                    help='How many episode to train the RL algorithm')
parser.add_argument('--bestMakespan', type=int, default=100000,
                    help='give a self-defined bestMakespan')
parser.add_argument('--runNum', type=int, default=1,
                    help='How many run to get a figure')

def experiment(args):#运行多次均值
    makespanTrails=[]
    rewardTrails=[]
    for i in range(args.runNum):
        makespanList,rewardList = train(args)
        makespanTrails.append(makespanList)
        rewardTrails.append(rewardList)
    return makespanTrails,rewardTrails

def test(args):#只运行一次
    makespanList, rewardList = train(args)
    getConvFig(makespanList)

if __name__ == "__main__":
    args = parser.parse_args()
    #只运行一次的结果
    #makespanList, rewardList = train(args)
    #getConvFig(makespanList)
    #运行多次的结果
    makespanTrails,rewardTrails = experiment(args)
    getTxt(args.runNum, args.episode, makespanTrails, 'makespan_test')
    getTxt(args.runNum, args.episode, rewardTrails, 'reward_test')
    makespanTrails = readTxt(args.runNum, 'makespan_test')
    rewardTrails = readTxt(args.runNum, 'reward_test')
    getMeanConvFig(args.episode, args.runNum, makespanTrails,rewardTrails)