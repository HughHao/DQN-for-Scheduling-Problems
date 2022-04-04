from src.RL.job_env import Situation
from src.RL.RL_brain import QLearningTable
from src.loadDataSet import loadDataSet
from src.utils.excel import getExcel
import pandas as pd


def train(args):
    NumOfMs, NumOfJs, dataSet = loadDataSet(args.fileName)
    env = Situation(NumOfMs, NumOfJs, dataSet, args.bestMakespan)  # 实例化环境
    #实例化1个Agent，Agent的动作是固定的，均是所有工件序号，是环境赋予的
    RL = QLearningTable(actions=env.actions,
                        learning_rate=args.lr,
                        reward_decay=args.gamma,
                        e_greedy=args.epsilon)
    makespanList=[]
    rewardList=[]
    for episode in range(args.episode):
        # initial observation，observation是状态
        observation = env.reset()#初始化state的观测值，设置一个初始化的状态
        print('episode', episode)
        total_reward= 0
        RL.epsilon = args.epsilon
        while True:
            # fresh env
            #env.render()#用于更新可视化环境

            RL_Action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(observation, RL_Action)
            total_reward += reward

            # RL learn from this transition
            RL.learn(observation, RL_Action, reward, observation_)

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
        args.epsilon += args.epsilon_increment/args.episode
        makespan = max(env.TM)
        makespanList.append(makespan)
        rewardList.append(total_reward)
        print('makespan', makespan)
    return makespanList,rewardList

    # end of game
    print('game over')