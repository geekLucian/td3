import math
import random
import pandas as pd
import numpy as np
import gym
from gym import spaces
from os import path
from typing import List
from matd3.environments.utils import rank, generate_volume, select_clear_amount, calculate_total_amount

NUM_OF_SELLER = len(np.loadtxt("data2/seller_data.txt"))
NUM_OF_BUYER = len(np.loadtxt("data2/buyer_data.txt"))


class RangeEnv(gym.Env):
    # TODO: 以下函数必须补充实现
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
        self.n = NUM_OF_SELLER + NUM_OF_BUYER
        # 求动作空间的最大值，最小值
        # act = [seller0price, seller0volume, seller1price, ...,
        #        buyer0price, buyer0volume, buyer1price, ...]
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

        self.buyer_name = ["buyer_%d" % i for i in range(self.num_of_buyer)]  # 按顺序取买家的名字
        self.seller_name = ["seller_%d" % i for i in range(self.num_of_seller)]  # 按顺序取卖家的名字

        self.observation_space = []
        self.action_space = []

        # 设定动作空间
        # action_space = [Box(price, volume), Box(price, volume), ...]
        # TODO: j += 2 ?
        for j in range(self.n):
            self.action_space.append(spaces.Box(low=act_low[j:j + 2], high=act_high[j:j + 2],
                                                shape=(2,), dtype=np.float32))

        # 设定观测空间
        # observation_space = [Box(volume, price), Box(volume, price), ...]
        # TODO: j += 2 ?
        for j in range(self.n):
            self.observation_space.append(spaces.Box(
                low=np.array([0, 0]),
                high=np.array([np.max(self.max_seller_volume),
                               max(np.max(self.max_seller_price), np.max(self.max_buyer_price))]),
                shape=(2,), dtype=np.float32))

    # 输入售电方的最大申报量、最小申报量、成本模型信息，strategy文件存储：最大申报价格、最小申报价格、低价区间采购量范围、中价区间采购量范围、高价区间采购量范围
    def set_data_for_seller(self, path="data2/seller_data.txt", path_strategy="data2/seller_strategy_data.txt"):
        """"
        设置卖方的数据：
            seller_data: [最大申报量, 最小申报量, 成本函数参数a_b, b_g, c_g]
            seller_strategy: [最大价格, 最小价格, 1, 0.00, 1.00, 0.00, 1.00] 低价区间采购量范围、中价区间采购量范围、高价区间采购量范围
        Eg. seller_data = [[76633, 38316, 0.0016, 102, 4560000], [...], ...]
        最大申报量=76633, a_b = 0.0016 , b_g = 102 , c_g =4,560,000
        总共有num_of_seller个， 成本函数默认从data/seller_data.txt中读取
        """
        seller_data = np.loadtxt(path)
        self.max_seller_volume = seller_data[:, 0]  # 读取各个卖方的最大申报电量
        self.min_seller_volume = seller_data[:, 1]  # 读取各个卖方的最小申报电量
        self.costfuncton_for_sellers = seller_data[:, 2:seller_data.shape[1]]  # 列切片，去掉第一列和第二列。

        seller_strategy = np.loadtxt(path_strategy)
        self.max_seller_price = seller_strategy[:, 0]
        self.min_seller_price = seller_strategy[:, 1]
        # for i in range(self.num_of_seller):
        #     for j in range(int(self.seller_strategy_num[i])*4):
        #         x.append(seller_strategy[i][2+j])  # 一组区间对应(价格低限，价格高限，电量低限，电量高限)
        #     # for y in x:
        #     #     self.seller_strategy[i].append(y)
        #     self.seller_strategy[i].append(x)

        return True

    # 输入购电方的成本模型信息，strategy文件存储：最大申报价格、最小申报价格、低价区间采购量范围、中价区间采购量范围、高价区间采购量范围
    def set_data_for_buyer(self, path="data2/buyer_data.txt", path_strategy="data2/buyer_strategy_data.txt"):
        buyer_data = np.loadtxt(path)
        self.max_buyer_volume = buyer_data[:, 0]  # 读取各个卖方的最大申报电量
        self.min_buyer_volume = buyer_data[:, 1]  # 读取各个卖方的最小申报电量

        buyer_strategy = np.loadtxt(path_strategy)
        self.max_buyer_price = buyer_strategy[:, 0]
        self.min_buyer_price = buyer_strategy[:, 1]
        # x = []
        # for i in range(self.num_of_buyer):
        #     for j in range(self.buyer_strategy_num[i] * 4):
        #         x.append(buyer_strategy[i][2 + j])  # 一组区间对应(价格低限，价格高限，电量低限，电量高限)
        #     for y in x:
        #         self.buyer_strategy[i].append(y)

        return True

    def reset(self):
        """
        重置场景，并返回状态观测

        返回格式： 列表，每一项是每个智能体的状态。每一项的numpy数组维度是1
        举例：
        return [np.zeros(shape=self.observation_space[i].shape) for i in range(self.n)]
        """
        # ======================== 报价报量 ==========================
        # 随机生成双方电量和报价，在[min, max]之间的均匀分布
        seller_volume = []
        seller_price = []
        # 卖方区间报价，下界seller_price_lower, 上界seller_price_upper
        #   seller_price_lower in [min[i], max[i]]
        #   seller_price_upper in [lower[i], max[i]]
        seller_price_lower = [random.uniform(self.min_seller_price[i], self.max_seller_price[i]) for i in
                              range(self.num_of_seller)]
        seller_price_upper = [random.uniform(seller_price_lower[i], self.max_seller_price[i]) for i in
                              range(self.num_of_seller)]

        # TODO: Refactor like above
        for i in range(self.num_of_seller):
            x = random.uniform(0, 1) * (self.max_seller_volume[i] - self.min_seller_volume[i]) \
                + self.min_seller_volume[i]  # 随机生成售电方申报电量
            seller_volume.append(x)
            x = random.uniform(0, 1) * (self.max_seller_price[i] - self.min_seller_price[i]) \
                + self.min_seller_price[i]  # 随机生成售电方申报电价
            seller_price.append(x)

        buyer_volume = []
        buyer_price = []
        for i in range(self.num_of_buyer):
            x = random.uniform(0, 1) * (self.max_buyer_volume[i] - self.min_buyer_volume[i]) \
                + self.min_buyer_volume[i]  # 随机生成购电方申报电量
            buyer_volume.append(x)
            x = random.uniform(0, 1) * (self.max_buyer_price[i] - self.min_buyer_price[i]) \
                + self.min_buyer_price[i]  # 随机生成购电方申报电价
            buyer_price.append(x)

        # _action data layout:
        #   [seller0_price_lower, seller0_price_upper, seller0_volume, ..., buyer0_price, buyer0_volume, ...]
        _seller_data = [seller_price_lower, seller_price_upper, seller_volume]
        _buyer_data = [buyer_price, buyer_volume]
        _seller_action = [val for tup in zip(*_seller_data) for val in tup]
        _buyer_action = [val for tup in zip(*_buyer_data) for val in tup]
        _action = _seller_action + _buyer_action

        # ========================= 出清 ===========================
        # TODO: here
        match_result, clear_price = self.get_match_result(_action)  # 进行出清操作
        total_match_volume = calculate_total_amount(match_result)  # 计算总出清量
        seller_reported_volume = np.zeros(self.num_of_seller)  # seller_reported_volume存储了售电方各自的成交量
        for i in range(len(match_result)):
            seller_reported_volume[int(match_result[i][1][7:])] += match_result[i][2]

        buyer_reported_volume = np.zeros(self.num_of_buyer)  # buyer_reported_volume存储了购电方各自的成交量
        for i in range(len(match_result)):
            buyer_reported_volume[int(match_result[i][0][6:])] += match_result[i][2]

        # ======================= 更新状态 ==========================
        state = []
        for i in range(self.num_of_seller):
            state.append(np.array([seller_reported_volume[i], clear_price]))
        for i in range(self.num_of_buyer):
            state.append(np.array([buyer_reported_volume[i], clear_price]))

        cost_result = self.calculate_cost(seller_reported_volume)  # 根据每个卖家出清量计算相应的各自的成本
        total_match_value = total_match_volume * clear_price  # 发电商总收益
        reward = total_match_value - sum(cost_result)  # 总利润 = 总收益 - 总成本
        return state, reward, clear_price

    def step(self, action: List[np.ndarray]):
        """
        执行一步动作。
        接收到的参数 action格式：列表，每一项是每个智能体的连续动作。每一项的数组维度是1

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
        for i in range(self.num_of_seller):
            state.append(np.array([seller_reported_volume[i], clear_price]))
        for i in range(self.num_of_buyer):
            state.append(np.array([buyer_reported_volume[i], clear_price]))

        cost_result = self.calculate_cost(seller_reported_volume)  # 根据每个卖家出清量计算相应的各自的成本
        total_match_value = total_match_volume * clear_price  # 发电商总收益

        reward = total_match_value - sum(cost_result)  # 总利润=总收益-总成本

        seller_profit = np.zeros(self.num_of_seller)  # 存储售电方各自的利润
        for i in range(self.num_of_seller):  # 计算售电方各自利润
            seller_profit[i] = seller_reported_volume[i] * clear_price - cost_result[i]
        # buyer_pay = np.zeros(self.num_of_buyer)  # 存储购电方各自的购电花费
        # for i in range(self.num_of_buyer):  # 计算购电方各自购电花销
        #     buyer_pay[i] = buyer_reported_volume[i] * clear_price

        return state, reward, clear_price, total_match_volume, seller_reported_volume, buyer_reported_volume, match_result, seller_profit, False, None
        # 返回本轮状态，发电商总利润，发电量总成交量

    def render(self, mode='human'):
        """渲染环境，如果无法渲染，则不实现"""
        pass

    # TODO: test
    def get_match_result(self, action):
        """
        获取匹配出清结果
        :param action: [seller0_price_lower, seller0_price_upper, seller0_volume, ..., buyer0_price, buyer0_volume, ...]
        :return: match_result = [[buyer_name, seller_name, match_volume, match_price], ...]
        """
        # ======================= Data Unpack ==========================
        seller_price_lower, seller_price_upper, seller_volume, buyer_price, buyer_volume = [], [], [], [], []
        seller_name, buyer_name = list(range(self.num_of_seller)), list(range(self.num_of_buyer))
        for idx in range(self.num_of_seller):
            seller_price_lower.append(action[3 * idx])
            seller_price_upper.append(action[3 * idx + 1])
            seller_volume.append(action[3 * idx + 2])
        for idx in range(self.num_of_buyer):
            buyer_price.append(action(self.num_of_seller + 2 * idx))
            buyer_volume.append(action(self.num_of_seller + 2 * idx + 1))

        # ======================= Price Rank ==========================
        seller_data = zip(seller_price_lower, seller_price_upper, seller_volume, seller_name)
        buyer_data = zip(buyer_price, buyer_volume, buyer_name)
        seller_data.sort()  # sort sellers' metadata according to their prices' lower bound increasingly
        buyer_data.sort(reverse=True)  # sort buyers' metadata according to their prices decreasingly
        seller_price_lower, seller_price_upper, seller_volume, seller_name = zip(*seller_data)
        buyer_price, buyer_volume, buyer_name = zip(*buyer_data)

        # ====================== Generate Ranges =======================
        seller_price = sorted(seller_price_lower + seller_price_upper)
        seller_clearance_price = []  # 卖方出清价格，为其 下限价格（区间下界） 和 最小的大于其下限价格（区间上界） 的平均
        for i in range(self.num_of_seller):
            lower_bound = seller_price_lower[i]
            upper_bound = next(price for price in seller_price if price > lower_bound)
            seller_clearance_price.append((lower_bound + upper_bound) / 2)

        # ==================== Matching Clearance ======================
        matching_result = []
        matching_seller_idx = 0
        matching_buyer_idx = 0
        matching_seller_price = seller_clearance_price[0]
        matching_buyer_price = buyer_price[0]
        # 终止条件： 当前匹配的买方价格低于卖方价格 or 卖方卖完 or 买方买完
        while matching_buyer_price > matching_seller_price \
                and matching_seller_idx < self.num_of_seller \
                and matching_buyer_idx < self.num_of_buyer:
            # name
            matching_seller_name = seller_name[matching_seller_idx]
            matching_buyer_name = buyer_name[matching_buyer_idx]
            # price
            matching_price = (matching_buyer_price + matching_seller_price) / 2
            # volume
            matching_seller_volume = seller_volume[matching_seller_idx]
            matching_buyer_volume = buyer_volume[matching_buyer_idx]
            matching_volume = min(matching_seller_volume, matching_buyer_volume)
            seller_volume[matching_seller_idx] -= matching_volume
            buyer_volume[matching_buyer_idx] -= matching_volume
            matching_result.append([matching_buyer_name, matching_seller_name, matching_price, matching_volume])
            # update index and price
            if seller_volume[matching_seller_idx] == 0:
                matching_seller_idx += 1
                matching_seller_price = seller_clearance_price[matching_seller_idx]
            if buyer_volume[matching_buyer_idx] == 0:
                matching_buyer_idx += 1
                matching_buyer_price = buyer_price[matching_buyer_idx]
        return matching_result

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
            result[i] = cost
        return result

    # 根据价格生成策略
    def action_strategy(self, action):  # 输入未被归一化的action动作
        act = []
        act.extend(action)
        # 还原动作值到为归一化数值
        for i in range(self.num_of_seller):
            act[i * 2] = act[i * 2] * (self.max_seller_price[i] - self.min_seller_price[i]) / 2 + (
                        self.max_seller_price[i] + self.min_seller_price[i]) / 2
            act[i * 2 + 1] = act[i * 2 + 1] * (self.max_seller_volume[i] - self.min_seller_volume[i]) / 2 + (
                        self.max_seller_volume[i] + self.min_seller_volume[i]) / 2

        for i in range(self.num_of_buyer):
            act[i * 2 + self.num_of_seller * 2] = act[i * 2 + self.num_of_seller * 2] * \
                                                  (self.max_buyer_price[i] - self.min_buyer_price[i]) / 2 + (
                                                              self.max_buyer_price[i] + self.min_buyer_price[i]) / 2
            act[i * 2 + 1 + self.num_of_seller * 2] = act[i * 2 + 1 + self.num_of_seller * 2] * \
                                                      (self.max_buyer_volume[i] - self.min_buyer_volume[i]) / 2 + (
                                                                  self.max_buyer_volume[i] + self.min_buyer_volume[
                                                              i]) / 2
        # print("unact:", unact)
        return act
    # # 根据上轮出清价格决定本轮购电方动作
