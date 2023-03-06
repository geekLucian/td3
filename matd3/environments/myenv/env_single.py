import math
import random

import numpy as np

import gym
from gym import spaces

from os import path
from typing import List
from matd3.environments.utils import rank, generate_volume, select_clear_amount, calculate_total_amount

NUM_OF_SELLER = len(np.loadtxt("data/seller_data.txt"))
NUM_OF_BUYER = len(np.loadtxt("data/buyer_data.txt"))


class MyEnv(gym.Env):

    # todo: 以下函数必须补充实现
    def __init__(self):
        """初始化环境"""

        self.num_of_seller = NUM_OF_SELLER  # 卖方数量
        self.max_seller_volume = np.zeros(self.num_of_seller)  # 最大申报量
        self.min_seller_volume = np.zeros(self.num_of_seller)  # 最小申报量
        self.max_seller_price = np.zeros(self.num_of_seller)  # 最大申报价格
        self.min_seller_price = np.zeros(self.num_of_seller)  # 最小申报价格
        self.set_data_for_seller()  # 读取输入的售电方数据

        self.num_of_buyer = NUM_OF_BUYER  # 买方数量
        self.max_buyer_volume = np.zeros(self.num_of_buyer)  # 最大申报量
        self.min_buyer_volume = np.zeros(self.num_of_buyer)  # 最小申报量
        self.max_buyer_price = np.zeros(self.num_of_buyer)  # 最大申报价格
        self.min_buyer_price = np.zeros(self.num_of_buyer)  # 最小申报价格
        self.set_data_for_buyer()  # 读取输入的购电方数据

        self.seller_price_bound = self.max_seller_price - self.min_seller_price
        self.buyer_price_bound = self.max_buyer_price - self.min_buyer_price

        self.n = 1
        # 求动作空间的最大值，最小值
        act_high = np.zeros(2 * self.num_of_seller + 2 * self.num_of_buyer).flatten()
        act_low = np.zeros(2 * self.num_of_seller + 2 * self.num_of_buyer).flatten()
        i = 0
        for j in range(self.num_of_seller):
            act_high[i] = self.max_seller_price[j]
            act_high[i + 1] = self.max_seller_volume[j]

            act_low[i] = self.min_seller_price[j]
            act_low[i + 1] = self.min_seller_volume[j]
            i += 2
        for j in range(self.num_of_buyer):
            act_high[i] = self.max_buyer_price[j]
            act_high[i + 1] = self.max_buyer_volume[j]

            act_low[i] = self.min_buyer_price[j]
            act_low[i + 1] = self.min_buyer_volume[j]
            i += 2

        # print("act_high:", act_high)
        # print("act_low:", act_low)

        self.buyer_name = ["buyer_%d" % i for i in range(self.num_of_buyer)]  # 按顺序取买家的名字
        self.seller_name = ["seller_%d" % i for i in range(self.num_of_seller)]  # 按顺序取卖家的名字

        # 设定动作空间
        self.action_space = [spaces.Box(low=act_low[0:2], high=act_high[0:2],
                                        shape=(2,), dtype=np.float32)]

        # 设定观测空间
        self.observation_space = [spaces.Box(low=np.array([0, 0]), high=np.array([np.max(self.max_seller_price), np.max(self.max_seller_volume)]),
                                             shape=(2,),
                                             dtype=np.float32)]

    # 输入售电方的最大申报量、最小申报量、成本模型信息，strategy文件存储：最大申报价格、最小申报价格、低价区间采购量范围、中价区间采购量范围、高价区间采购量范围
    def set_data_for_seller(self, path="data/seller_data.txt", path_strategy="data/seller_strategy_data.txt"):
        """"
        设置卖方的成本函数，
        例子 [[76633,0.0016 , 102 , 4,560,000],[....],[....].. ] :
        第一个卖家 最大申报电量=76633, a_b = 0.0016 , b_g = 102 , c_g =4,560,000
        总共有 num_of_seller个 ， 成本函数默认从data/seller_data.txt中读取
        """
        seller_data = np.loadtxt(path)
        self.max_seller_volume = seller_data[:, 0]  # 读取各个卖方的最大申报电量
        self.min_seller_volume = seller_data[:, 1]  # 读取各个卖方的最小申报电量
        self.costfuncton_for_sellers = seller_data[:, 2:seller_data.shape[1]]  # 列切片，去掉第一列和第二列。

        seller_strategy = np.loadtxt(path_strategy)
        self.max_seller_price = seller_strategy[:, 0]
        self.min_seller_price = seller_strategy[:, 1]

        return True

    # 输入购电方的成本模型信息，strategy文件存储：最大申报价格、最小申报价格、低价区间采购量范围、中价区间采购量范围、高价区间采购量范围
    def set_data_for_buyer(self, path="data/buyer_data.txt", path_strategy="data/buyer_strategy_data.txt"):
        buyer_data = np.loadtxt(path)
        self.max_buyer_volume = buyer_data[:, 0]  # 读取各个卖方的最大申报电量
        self.min_buyer_volume = buyer_data[:, 1]  # 读取各个卖方的最小申报电量

        buyer_strategy = np.loadtxt(path_strategy)
        self.max_buyer_price = buyer_strategy[:, 0]
        self.min_buyer_price = buyer_strategy[:, 1]

        return True

    def reset(self):
        """
        重置场景，并返回状态观测

        返回格式： 列表，每一项是每个智能体的状态。每一项的numpy数组维度是1
        举例：
        return [np.zeros(shape=self.observation_space[i].shape) for i in range(self.n)]
        """

        seller_volume = []
        seller_price = []
        for i in range(self.num_of_seller):
            if i == 0:
                x = random.uniform(0, 1) * (self.max_seller_volume[i] - self.min_seller_volume[i]) \
                    + self.min_seller_volume[i]  # 随机生成售电方申报电量
                seller_volume.append(x)
                x = random.uniform(0, 1) * (self.max_seller_price[i] - self.min_seller_price[i]) \
                    + self.min_seller_price[i]  # 随机生成售电方申报电价
                seller_price.append(x)
            else:
                x = self.max_seller_volume[i]  # 随机生成售电方申报电量
                seller_volume.append(x)
                x = self.min_seller_price[i] # 随机生成售电方申报电价
                seller_price.append(x)

        buyer_volume = []
        buyer_price = []
        for i in range(self.num_of_buyer):
            x = self.max_buyer_volume[i]  # 随机生成购电方申报电量
            buyer_volume.append(x)
            x = self.max_buyer_price[i]  # 随机生成购电方申报电价
            buyer_price.append(x)

        # print("seller_volume:", seller_volume)  # 测试
        # print("buyer_volume:", buyer_volume)  # 测试

        # action组成[seller_price ,seller_volume , .........,buyer_price,buyer_volume,buyer_price,buyer_volume]
        action0 = []
        # 将售电方、购电方的申报电量、申报电价依次加入动作列表中
        for i in range(self.num_of_seller):
            action0.append(seller_price[i])
            action0.append(seller_volume[i])
        for i in range(self.num_of_buyer):
            action0.append(buyer_price[i])
            action0.append(buyer_volume[i])
        # print(action0)

        state = []
        match_result, clear_price = self.get_match_result(action0)  # 进行出清操作
        total_match_volume = calculate_total_amount(match_result)  # 计算总出清量
        seller_reported_volume = np.zeros(self.num_of_seller)  # seller_reported_volume存储了售电方各自的成交量
        for i in range(len(match_result)):
            seller_reported_volume[int(match_result[i][1][7:])] += match_result[i][2]

        buyer_reported_volume = np.zeros(self.num_of_buyer)  # buyer_reported_volume存储了购电方各自的成交量
        for i in range(len(match_result)):
            buyer_reported_volume[int(match_result[i][0][6:])] += match_result[i][2]

        state.append(clear_price)  # 得到初始状态
        state.append(seller_reported_volume[0])

        cost_result = self.calculate_cost(seller_reported_volume)  # 根据每个卖家出清量计算相应的各自的成本
        match_value = seller_reported_volume[0] * clear_price  # 发电商1收益

        reward = match_value - cost_result[0]  # 总利润=总收益-总成本

        return [np.array(state)], reward, clear_price

    def step(self, action: List[np.ndarray]):
        """
        执行动作。
        接收到的参数 action格式：列表，每一项是每个智能体的连续动作。每一项的numpy数组维度是1

        返回格式：
            下一状态（列表），每一项是每个智能体的状态。每一项的numpy数组维度是1
            奖励（标量）
            Done信号（标量）
            其他信息

        举例：
        return [np.zeros(shape=self.observation_space[i].shape) for i in range(self.n)], \
               np.zeros(shape=(self.n,)), \
               np.full(shape=(self.n,), fill_value=False), \
               None

        note:  因为神经网络策略的动作输出设置在-1到1之间，所以，如果智能体的动作值域不在[-1, 1]，则需要把动作做相应的坐标转换
        比如，将 x \in [-1, 1] 转到 [-2, 4]， 则是 x * (4-(-2))/2 + (4+(-2))/2
        """

        # 将list类型的action转换为ndarray类型的action_t
        action_t = self.action_strategy(np.array(action).flatten())
        for i in range(self.num_of_seller-1):
            action_t.append(self.min_seller_price[i + 1])
            action_t.append(self.max_seller_volume[i + 1])

        for i in range(self.num_of_buyer):
            action_t.append(self.max_buyer_price[i])
            action_t.append(self.max_buyer_volume[i])

        # action组成[seller_price ,seller_volume , .........,buyer_price,buyer_volume,buyer_price,buyer_volume]
        match_result, clear_price = self.get_match_result(action_t)  # 进行出清操作
        total_match_volume = calculate_total_amount(match_result)  # 计算总出清量

        seller_reported_volume = np.zeros(self.num_of_seller)  # seller_reported_volume存储了售电方各自的成交量
        for i in range(len(match_result)):
            seller_reported_volume[int(match_result[i][1][7:])] += match_result[i][2]

        buyer_reported_volume = np.zeros(self.num_of_buyer)  # buyer_reported_volume存储了购电方各自的成交量
        for i in range(len(match_result)):
            buyer_reported_volume[int(match_result[i][0][6:])] += match_result[i][2]

        state = []
        state.append(clear_price)  # 更新的状态
        state.append(seller_reported_volume[0])

        cost_result = self.calculate_cost(seller_reported_volume)  # 根据每个卖家出清量计算相应的各自的成本
        match_value = seller_reported_volume[0] * clear_price  # 发电商总收益

        reward = match_value - cost_result[0]  # 总利润=总收益-总成本

        seller_profit = np.zeros(self.num_of_seller)  # 存储售电方各自的利润
        for i in range(self.num_of_seller):  # 计算售电方各自利润
            seller_profit[i] = seller_reported_volume[i] * clear_price - cost_result[i]
        # buyer_pay = np.zeros(self.num_of_buyer)  # 存储购电方各自的购电花费
        # for i in range(self.num_of_buyer):  # 计算购电方各自购电花销
        #     buyer_pay[i] = buyer_reported_volume[i] * clear_price

        return [np.array(state)], reward, clear_price, total_match_volume, seller_reported_volume, buyer_reported_volume, match_result, seller_profit, False, None
        # 返回本轮状态，发电商总利润，发电量总成交量

    def render(self, mode='human'):
        """渲染环境，如果无法渲染，则不实现"""
        pass

    def get_match_result(self, action):  # 获取出清结果
        seller_data, buyer_data = [], []
        for i in range(0, self.num_of_seller * 2, 2):
            seller_data.append([action[i], action[i+1]])
        for i in range(self.num_of_seller * 2, self.num_of_seller * 2+self.num_of_buyer * 2, 2):
            buyer_data.append([action[i], action[i+1]])
        match_result, clear_price = rank(self.buyer_name, buyer_data, self.seller_name, seller_data)
        return match_result, clear_price

    # 售电侧成本
    def calculate_cost(self, reported_volume):
        """
        :param reported_volume: 申报的电量:list ，不是匹配后的电量
        :return: 长度为 num_of_seller的list，里面是float
        """
        result = np.zeros(self.num_of_seller)
        for i in range(self.num_of_seller):
            cost = self.costfuncton_for_sellers[i][0] * reported_volume[i] ** 2 + self.costfuncton_for_sellers[i][1] * \
                   reported_volume[i] + self.costfuncton_for_sellers[i][2]
            # cost = (self.costfuncton_for_sellers[i][1]+218.7)*(reported_volume[i]-self.costfuncton_for_sellers[i][0])+300*self.costfuncton_for_sellers[i][0]
            result[i] = cost
        return result

    # 根据价格生成策略
    def action_strategy(self, action):  # 输入未被归一化的action动作
        # 执行售电方策略
        act = []
        act.extend(action)
        for i in range(1):
            act[i*2] = act[i*2] * (self.max_seller_price[i] - self.min_seller_price[i]) / 2 + (self.max_seller_price[i] + self.min_seller_price[i]) / 2
            act[i*2+1] = act[i*2+1] * (self.max_seller_volume[i] - self.min_seller_volume[i]) / 2 + (self.max_seller_volume[i] + self.min_seller_volume[i]) / 2

        # for i in range(self.num_of_buyer):
        #     act[i*2+self.num_of_seller*2] = act[i*2+self.num_of_seller*2] *\
        #                                     (self.max_buyer_price[i] - self.min_buyer_price[i]) / 2 + (self.max_buyer_price[i] + self.min_buyer_price[i]) / 2
        #     act[i*2+1+self.num_of_seller*2] = act[i*2+1+self.num_of_seller*2] *\
        #                                       (self.max_buyer_volume[i] - self.min_buyer_volume[i]) / 2 + (self.max_buyer_volume[i] + self.min_buyer_volume[i]) / 2
        # print("unact:", unact)
        return act
    # # 根据上轮出清价格决定本轮购电方动作
