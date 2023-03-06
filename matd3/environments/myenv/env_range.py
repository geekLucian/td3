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
    def __init__(self):
        """初始化环境"""
        self.num_of_seller = NUM_OF_SELLER                      # 卖方数量
        self.max_seller_volume = np.zeros(self.num_of_seller)   # 最大申报量
        self.min_seller_volume = np.zeros(self.num_of_seller)   # 最小申报量
        self.max_seller_price = np.zeros(self.num_of_seller)    # 最大申报价格
        self.min_seller_price = np.zeros(self.num_of_seller)    # 最小申报价格
        self.costfunction_for_sellers = None                    # 卖方成本函数
        self.set_data_for_seller()                              # 读取输入的售电方数据

        self.num_of_buyer = NUM_OF_BUYER                        # 买方数量
        self.max_buyer_volume = np.zeros(self.num_of_buyer)     # 最大申报量
        self.min_buyer_volume = np.zeros(self.num_of_buyer)     # 最小申报量
        self.max_buyer_price = np.zeros(self.num_of_buyer)      # 最大申报价格
        self.min_buyer_price = np.zeros(self.num_of_buyer)      # 最小申报价格
        self.set_data_for_buyer()                               # 读取输入的购电方数据

        self.n = NUM_OF_SELLER + NUM_OF_BUYER
        self.seller_price_bound = self.max_seller_price - self.min_seller_price
        self.buyer_price_bound = self.max_buyer_price - self.min_buyer_price
        self.buyer_name = ["buyer_%d" % i for i in range(self.num_of_buyer)]
        self.seller_name = ["seller_%d" % i for i in range(self.num_of_seller)]

        self.action_space = self.observation_space = []
        # 设定动作空间
        # action_space = [Box(seller0_price_lower, seller0_price_range_factor, seller0_volume), ...,
        #                 Box(buyer0_price, placeholder, buyer0_volume), ...]
        # Notes:
        #   1. Not using seller_price_upper (should be >= seller_price_lower) since its lower bound is dynamic
        #   2. Price_Range_Factor(PRF) = (Price_Upper - Price_Lower) / (Max_Price - Min_Price), 0 <= PRF <= 1
        #   3. Require a placeholder in buyer's action to align its size with seller's
        seller_action_space = [spaces.Box(low=np.array([self.min_seller_price[i], 0, self.min_seller_volume[i]]),
                                          high=np.array([self.max_seller_price[i], 1, self.max_seller_volume[i]]),
                                          shape=(3,), dtype=np.float32) for i in range(self.num_of_seller)]
        buyer_action_space = [spaces.Box(low=np.array([self.min_buyer_price[i], 0, self.min_buyer_volume[i]]),
                                         high=np.array([self.max_buyer_price[i], 0, self.max_buyer_volume[i]]),
                                         shape=(3,), dtype=np.float32) for i in range(self.num_of_buyer)]
        self.action_space = seller_action_space + buyer_action_space
        max_volume = max(np.max(self.max_seller_volume), np.max(self.max_buyer_volume))
        max_price = max(np.max(self.max_seller_price), np.max(self.max_buyer_price))
        # 设定观测空间
        # observation_space = [Box(seller0_reported_volume, seller0_avg_price), ...,
        #                      Box(buyer0_reported_volume, buyer0_avg_price), ...]
        self.observation_space = [spaces.Box(low=np.array([0, 0]),
                                             high=np.array([max_volume, max_price]),
                                             shape=(2,), dtype=np.float32) for _ in range(self.n)]

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
        self.costfunction_for_sellers = seller_data[:, 2:seller_data.shape[1]]  # 列切片，去掉第一列和第二列。

        seller_strategy = np.loadtxt(path_strategy)
        self.max_seller_price = seller_strategy[:, 0]
        self.min_seller_price = seller_strategy[:, 1]

    # 输入购电方的成本模型信息，strategy文件存储：最大申报价格、最小申报价格、低价区间采购量范围、中价区间采购量范围、高价区间采购量范围
    def set_data_for_buyer(self, path="data2/buyer_data.txt", path_strategy="data2/buyer_strategy_data.txt"):
        buyer_data = np.loadtxt(path)
        self.max_buyer_volume = buyer_data[:, 0]  # 读取各个卖方的最大申报电量
        self.min_buyer_volume = buyer_data[:, 1]  # 读取各个卖方的最小申报电量

        buyer_strategy = np.loadtxt(path_strategy)
        self.max_buyer_price = buyer_strategy[:, 0]
        self.min_buyer_price = buyer_strategy[:, 1]

    def reset(self):
        """
        重置场景，并返回状态观测

        返回格式： 列表，每一项是每个智能体的状态。每一项的numpy数组维度是1
        举例：
        return [np.zeros(shape=self.observation_space[i].shape) for i in range(self.n)]
        """
        # ======================== 报价报量 ==========================
        # 随机生成双方电量和报价，在[min, max]之间的均匀分布
        # 卖方区间报价，下界seller_price_lower, 上界seller_price_upper
        #   seller_price_lower in [min[i], max[i]]
        #   seller_price_upper in [lower[i], max[i]]
        seller_price_lower = [random.uniform(self.min_seller_price[i], self.max_seller_price[i]) for i in
                              range(self.num_of_seller)]
        seller_price_upper = [random.uniform(seller_price_lower[i], self.max_seller_price[i]) for i in
                              range(self.num_of_seller)]
        buyer_price = [random.uniform(self.min_buyer_price[i], self.max_buyer_price[i]) for i in
                       range(self.num_of_buyer)]
        seller_volume = [random.uniform(self.min_seller_volume[i], self.max_seller_volume[i]) for i in
                         range(self.num_of_seller)]
        buyer_volume = [random.uniform(self.min_buyer_volume[i], self.max_buyer_volume[i]) for i in
                        range(self.num_of_buyer)]

        # Data Packing
        # _action data layout:
        #   [seller0_price_lower, seller0_price_upper, seller0_volume, ..., buyer0_price, 0, buyer0_volume, ...]
        _seller_data = [seller_price_lower, seller_price_upper, seller_volume]
        _buyer_data = [buyer_price, [0] * self.num_of_buyer, buyer_volume]
        _seller_action = [val for tup in zip(*_seller_data) for val in tup]
        _buyer_action = [val for tup in zip(*_buyer_data) for val in tup]
        _action = _seller_action + _buyer_action

        # ========================= 出清 ===========================
        # match_result := [[buyer_name, seller_name, match_volume, match_price], ...]
        match_result, end_reason = self.get_match_result(_action)
        seller_reported_volume, buyer_reported_volume, seller_avg_price, buyer_avg_price = self.statistics(match_result)

        # ======================= 更新状态 ==========================
        seller_state = [np.array([seller_reported_volume[i], seller_avg_price[i]]) for i in range(self.num_of_seller)]
        buyer_state = [np.array([buyer_reported_volume[i], buyer_avg_price[i]]) for i in range(self.num_of_buyer)]
        state = seller_state + buyer_state

        total_seller_cost = sum(self.calculate_cost(seller_reported_volume))  # 根据卖方成本函数计算成本
        total_seller_value = np.dot(seller_reported_volume, seller_avg_price)
        reward = total_seller_value - total_seller_cost  # 卖方回报

        clear_price = np.append(seller_avg_price, buyer_avg_price)
        # TODO: 1. reward 仅为卖方回报； 2. 更新所有调用此函数的clear_price（由标量变为了1d向量）
        return state, reward, clear_price, end_reason

    # TODO: test
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

        # Map NN output vector's component from [-1, 1] into real values
        _action = self.action_strategy(np.array(action).flatten())
        match_result, end_reason = self.get_match_result(_action)
        total_match_volume = calculate_total_amount(match_result)
        seller_reported_volume, buyer_reported_volume, seller_avg_price, buyer_avg_price = self.statistics(match_result)

        seller_state = [np.array([seller_reported_volume[i], seller_avg_price[i]]) for i in range(self.num_of_seller)]
        buyer_state = [np.array([buyer_reported_volume[i], buyer_avg_price[i]]) for i in range(self.num_of_buyer)]
        state = seller_state + buyer_state

        total_seller_cost = sum(self.calculate_cost(seller_reported_volume))  # 根据卖方成本函数计算成本
        total_seller_value = np.dot(seller_reported_volume, seller_avg_price)
        reward = total_seller_value - total_seller_cost  # 卖方回报

        clear_price = np.append(seller_avg_price, buyer_avg_price)

        cost_result = self.calculate_cost(seller_reported_volume)  # 根据每个卖家出清量计算相应的各自的成本

        seller_profit = np.array([seller_reported_volume[i] * clear_price[i] - cost_result[i]
                                  for i in range(self.num_of_seller)])

        # TODO: 1. update usage (clear_price->vector) 2. done = itr >= EPISODE_LENGTH
        return (state, reward, clear_price, total_match_volume, seller_reported_volume, buyer_reported_volume,
                match_result, seller_profit, end_reason, False, None)

    def render(self, mode='human'):
        """渲染环境，如果无法渲染，则不实现"""
        pass

    # TODO: test
    def get_match_result(self, _action):
        """
        获取匹配出清结果
        :param _action: [seller0_price_lower, seller0_price_upper, seller0_volume, ...,
                         buyer0_price, placeholder, buyer0_volume, ...]
        :return: match_result = [[buyer_name, seller_name, match_volume, match_price], ...]
        :return end_reason 出清结束的原因 := 剩下未成交的买方价格低于卖方价格 || 卖方卖完 || 买方买完
        """
        # ======================= Data Unpack ==========================
        seller_price_lower, seller_price_upper, seller_volume, buyer_price, buyer_volume = [], [], [], [], []
        seller_name, buyer_name = list(range(self.num_of_seller)), list(range(self.num_of_buyer))
        for idx in range(self.num_of_seller):
            seller_price_lower.append(_action[3 * idx])
            seller_price_upper.append(_action[3 * idx + 1])
            seller_volume.append(_action[3 * idx + 2])
        for idx in range(self.num_of_buyer):
            buyer_price.append(_action[(self.num_of_seller + idx) * 3])
            buyer_volume.append(_action[(self.num_of_seller + idx) * 3 + 2])

        # ======================= Price Rank ==========================
        seller_data = zip(seller_price_lower, seller_price_upper, seller_volume, seller_name)
        buyer_data = zip(buyer_price, buyer_volume, buyer_name)
        seller_data = sorted(seller_data)  # sort sellers' metadata according to their prices' lower bound increasingly
        buyer_data = sorted(buyer_data, reverse=True)  # sort buyers' metadata according to their prices decreasingly
        seller_price_lower, seller_price_upper, seller_volume, seller_name = map(list, zip(*seller_data))
        buyer_price, buyer_volume, buyer_name = map(list, zip(*buyer_data))

        # ====================== Generate Ranges =======================
        seller_price = sorted(seller_price_lower + seller_price_upper)
        seller_clearance_price = []  # 卖方出清价格，为其 下限价格（区间下界） 和 最小的大于其下限价格（区间上界） 的平均
        max_price = max(seller_price)
        for i in range(self.num_of_seller):
            lower_bound = seller_price_lower[i]
            upper_bound = next(_ for _ in seller_price if _ > lower_bound) if lower_bound < max_price else max_price
            seller_clearance_price.append((lower_bound + upper_bound) / 2)

        # ==================== Matching Clearance ======================
        matching_result = []
        matching_seller_idx = 0
        matching_buyer_idx = 0
        matching_seller_price = seller_clearance_price[0]
        matching_buyer_price = buyer_price[0]
        # 终止条件： 当前匹配的买方价格低于卖方价格 or 卖方卖完 or 买方买完
        while (matching_buyer_price > matching_seller_price
               and matching_seller_idx < self.num_of_seller
               and matching_buyer_idx < self.num_of_buyer):
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
            matching_result.append([matching_buyer_name, matching_seller_name, matching_volume, matching_price])
            # update index and price
            if seller_volume[matching_seller_idx] == 0:
                matching_seller_idx += 1
                try:
                    matching_seller_price = seller_clearance_price[matching_seller_idx]
                except IndexError:
                    pass  # do nothing, the loop will end by itself
            if buyer_volume[matching_buyer_idx] == 0:
                matching_buyer_idx += 1
                try:
                    matching_buyer_price = buyer_price[matching_buyer_idx]
                except IndexError:
                    pass  # do nothing, the loop will end by itself

        end_reason = ""
        if not matching_buyer_price > matching_seller_price:
            end_reason += "剩下未成交的买方价格低于卖方价格 "
        if not matching_seller_idx < self.num_of_seller:
            end_reason += "卖方卖完 "
        if not matching_buyer_idx < self.num_of_buyer:
            end_reason += "买方买完 "

        return matching_result, end_reason

    def statistics(self, match_result):
        seller_reported_volume = np.zeros(self.num_of_seller)   # 售电方各自的成交量
        buyer_reported_volume = np.zeros(self.num_of_buyer)     # 购电方各自的成交量
        seller_avg_price = np.zeros(self.num_of_seller)         # 售电方卖出的平均价格
        buyer_avg_price = np.zeros(self.num_of_seller)          # 购电方买入的平均价格
        for i in range(len(match_result)):
            buyer_name = int(match_result[i][0])
            seller_name = int(match_result[i][1])
            match_volume = int(match_result[i][2])
            match_price = int(match_result[i][3])
            match_value = match_volume * match_price
            seller_avg_price[seller_name] = (
                    (seller_avg_price[seller_name] * seller_reported_volume[seller_name] + match_value)
                    /
                    (seller_reported_volume[seller_name] + match_volume)
            )
            buyer_avg_price[buyer_name] = (
                    (buyer_avg_price[buyer_name] * buyer_reported_volume[buyer_name] + match_value)
                    /
                    (buyer_reported_volume[buyer_name] + match_volume)
            )
            seller_reported_volume[seller_name] += match_volume
            buyer_reported_volume[buyer_name] += match_volume
        return seller_reported_volume, buyer_reported_volume, seller_avg_price, buyer_avg_price

    def calculate_cost(self, reported_volume):
        """Caculate seller's cost accroding to its reported volume based on given cost function."""
        result = np.zeros(self.num_of_seller)
        for i in range(self.num_of_seller):
            cost = self.costfunction_for_sellers[i][0] * reported_volume[i] ** 2 + self.costfunction_for_sellers[i][1] * \
                   reported_volume[i] + self.costfunction_for_sellers[i][2]
            result[i] = cost
        return result

    def action_strategy(self, action):
        """Recover action vector from normalized output, i.e. mapping its components from [-1, 1] to [min, max].
        :param Normalized action: [seller0_price_lower, seller0_price_range_factor, seller0_volume, ...,
                        buyer0_price, placeholder, buyer0_volume, ...]
        :return: Real action vector.
                 [seller0_price_lower, seller0_price_upper, seller0_volume, ...,
                  buyer0_price, placeholder, buyer0_volume, ...]
        """
        _action = []
        _action.extend(action)

        for i in range(self.num_of_seller):
            max_price = self.max_seller_price[i]
            min_price = self.min_seller_price[i]
            max_volume = self.max_seller_volume[i]
            min_volume = self.min_seller_volume[i]
            # price_lower: [-1, 1] -> [min_price, max_price]
            price_lower = _action[3*i] = (_action[3*i] + 1) * (max_price - min_price) / 2 + min_price
            # PRF -> seller_price_upper: [-1, 1] -> [price_lower, max_price]
            price_upper = _action[3*i+1] = (_action[3*i+1] + 1) * (max_price - price_lower) / 2 + price_lower
            # volume: [-1, 1] -> [min_volume, max_volume]
            volume = _action[3*i+2] = (_action[3*i+2] + 1) * (max_volume - min_volume) / 2 + min_volume

        for i in range(self.num_of_seller, self.n):
            idx = i - self.num_of_seller
            max_price = self.max_buyer_price[idx]
            min_price = self.min_buyer_price[idx]
            max_volume = self.max_buyer_volume[idx]
            min_volume = self.min_buyer_volume[idx]
            # price: [-1, 1] -> [min_price, max_price]
            price = _action[3*i] = (_action[3*i] + 1) * (max_price - min_price) / 2 + min_price
            # placeholder: -> 0
            placeholder = _action[3*i+1] = 0
            # volume: [-1, 1] -> [min_volume, max_volume]
            volume = _action[3*i+2] = (_action[3*i+2] + 1) * (max_volume - min_volume) / 2 + min_volume

        return _action
