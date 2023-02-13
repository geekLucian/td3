#购电方队列
# buyer_name = ["a1","b1","c1","d1","e1"]
# buyer_num = [[350, 1000],[430,3000],[230, 4000],[300,5000],[430, 2000]]   [电价，电量]
# #售电方队列
# seller_name = ["f1","g1","h1","i1","j1"]
# seller_num = [[150, 4000],[250, 1500],[200, 2000],[400, 3000],[300, 3000]]  [电价，电量]
# 说明：buyer_name是购电方名称列表，buyer_num是购电方申报数据列表，名称与数据按照顺序一一对应。数据里前面的是电价，后面的是电量。对于售电方同理。
import numpy as np
from random import sample


def rank(buyer_name, buyer_num, seller_name, seller_num):
    """
    排序：买方电价降序，卖方电价升序
    调用匹配出清函数，返回匹配结果和出清价格
    """

    # ====================== 特殊处理 ======================
    # 存在特殊情况下出现神经网络输出的动作值全为Nan，导致程序出错，中断运行
    for i in range(len(buyer_num)):
        if np.isnan(buyer_num[i][0]):
            buyer_num[i][0] = 1
        else:
            pass
        if np.isnan(buyer_num[i][1]):
            buyer_num[i][1] = 1
        else:
            pass
    for i in range(len(seller_num)):
        if np.isnan(seller_num[i][0]):
            seller_num[i][0] = 1
        else:
            pass
        if np.isnan(seller_num[i][1]):
            seller_num[i][1] = 1
        else:
            pass

    # ====================== 买方降序 ======================
    buyer_amount = []  # 该列表存储了购电方申报电价
    for i in buyer_num:
        buyer_amount.append(i[0])
    buyer_amount_sort = sorted(buyer_amount, reverse=True)  # 购电方按照申报电价降序排列
    buyer_number_sort = []
    buyer_name_sort = []
    buyer_num_temp = buyer_num.copy()  # 复制购电方电价电量队列
    buyer_name_temp = buyer_name.copy()  # 复制购电方名称队列

    for j in range(len(buyer_num)):
        for i in range(len(buyer_num_temp)):
            if buyer_num_temp[i][0] == buyer_amount_sort[0]:  # 申报量最大值那个
                buyer_number_sort.append(buyer_num_temp[i])  # 将队列中最高电价申报电价电量存在数据排序列表里
                buyer_name_sort.append(buyer_name_temp[i])  # 将队列中最高电价的名称存在名称排序列表里
                buyer_amount_sort.pop(0)  # 去掉当前最大申报值
                buyer_num_temp.pop(i)  # 去掉复制队列中对应的成员
                buyer_name_temp.pop(i)  # 去掉复制队列中对应的成员
                break
    # 完成这部分循环，购电方名称和申报数据按电价从高到低排序

    # ====================== 卖方升序 ======================
    seller_amount = []
    for i in seller_num:
        seller_amount.append(i[0])
    seller_amount_sort = sorted(seller_amount)  # 售电方按照申报电价升序排列
    seller_number_sort = []
    seller_name_sort = []
    seller_num_temp = seller_num.copy()
    seller_name_temp = seller_name.copy()

    for j in range(len(seller_num)):
        for i in range(len(seller_num_temp)):
            if seller_num_temp[i][0] == seller_amount_sort[0]:
                seller_number_sort.append(seller_num_temp[i])
                seller_name_sort.append(seller_name_temp[i])
                seller_amount_sort.pop(0)
                seller_num_temp.pop(i)
                seller_name_temp.pop(i)
                break
    # 完成这部分循环，售电方名称和申报数据按电价从高到低排序

    # ====================== 匹配出清 ======================
    match_result, clear_price = match(buyer_number_sort, buyer_name_sort, seller_number_sort, seller_name_sort)
    # 进行出清操作

    return match_result, clear_price


def match(buyer_number_sort, buyer_name_sort, seller_number_sort, seller_name_sort):
    """
    匹配出清
    """
    buyer_number_sort_copy = buyer_number_sort.copy()  # 根据电价从高到低排的购电方申报数据列表
    seller_number_sort_copy = seller_number_sort.copy()  # 根据电价从高到低排的售电方申报数据列表
    buyer_name_sort_copy = buyer_name_sort.copy()  # 根据电价从高到低排的购电方名称列表
    seller_name_sort_copy = seller_name_sort.copy()  # 根据电价从高到低排的售电方名称列表

    # 记录匹配结果的列表,[[买方名字，卖方名字，匹配电量],[买方名字，卖方名字，匹配电量]，[……]]
    match_result = []
    # 当买方队列第一个申报电价大于卖方队列第一个申报电价，则满足匹配条件
    # 主循环 停止条件：1、不满足买方申报电价大于卖方申报电价 2、一方电量完全交易完 3、成交电量达到集中竞价交易电量规模
    while buyer_number_sort_copy[0][0] >= seller_number_sort_copy[0][0]:
        # 计算买方队列前方申报电价相同的申报电量总量
        m = 0  # 买方队列前方共有m+1个相同申报电价的成员
        buyer_amount = buyer_number_sort_copy[m][1]
        while m < (len(buyer_number_sort_copy) - 1):
            if buyer_number_sort_copy[m][0] == buyer_number_sort_copy[m + 1][0]:
                buyer_amount = buyer_amount + buyer_number_sort_copy[m + 1][1]
                m = m + 1
            else:
                break
            # 计算卖方队列前方申报电价相同的申报电量总量
        n = 0  # 卖方队列前方共有n+1个相同申报电价的成员
        seller_amount = seller_number_sort_copy[n][1]
        while n < (len(seller_number_sort_copy) - 1):
            if seller_number_sort_copy[n][0] == seller_number_sort_copy[n + 1][0]:
                seller_amount = seller_amount + seller_number_sort_copy[n + 1][1]
                n = n + 1
            else:
                break

        # 购电方申报电量大于售电方申报电量，则该电量下的售电方全部完成交易
        if buyer_amount > seller_amount:
            # 进行匹配
            for j in range(n + 1):
                for i in range(m + 1):
                    match_amount = (seller_number_sort_copy[j][1] / buyer_amount) * buyer_number_sort_copy[i][1]
                    # 按比例分配购电方申报电量
                    if np.isnan(match_amount):       # 可能是避免输入中Nan的影响
                        match_amount = 0
                    match_amount = int(match_amount + 0.5)  # 四舍五入，取整数
                    match_result.append([buyer_name_sort_copy[i], seller_name_sort_copy[j], match_amount])

            # 更新匹配过的买方信息
            for i in range(m + 1):
                trade_amount = (seller_amount / buyer_amount) * buyer_number_sort_copy[i][1]
                if np.isnan(trade_amount):  # 可能是避免输入中Nan的影响
                    trade_amount = 0
                trade_amount = int(trade_amount + 0.5)
                buyer_number_sort_copy[i][1] = buyer_number_sort_copy[i][1] - trade_amount
            # 删除配对完成的卖方
            for j in range(n + 1):
                seller_number_sort_copy.pop(0)
                seller_name_sort_copy.pop(0)

        # 购电方申报电量小于售电方申报电量，则该电量下的购电方全部完成交易
        if buyer_amount < seller_amount:
            # 进行匹配
            for i in range(m + 1):
                for j in range(n + 1):
                    match_amount = (buyer_number_sort_copy[i][1] / seller_amount) * seller_number_sort_copy[j][1]
                    # 按比例分配售电方申报电量
                    if np.isnan(match_amount):       # 可能是避免输入中Nan的影响
                        match_amount = 0
                    match_amount = int(match_amount + 0.5)  # 四舍五入，取整数
                    match_result.append([buyer_name_sort_copy[i], seller_name_sort_copy[j], match_amount])

            # 更新匹配过的卖方信息
            for j in range(n + 1):
                trade_amount = (buyer_amount / seller_amount) * seller_number_sort_copy[j][1]
                if np.isnan(trade_amount):  # 可能是避免输入中Nan的影响
                    trade_amount = 0
                trade_amount = int(trade_amount + 0.5)
                seller_number_sort_copy[j][1] = seller_number_sort_copy[j][1] - trade_amount

            # 删除配对完成的买方
            for i in range(m + 1):
                buyer_number_sort_copy.pop(0)
                buyer_name_sort_copy.pop(0)

        # 买方申报电量等于卖方申报电量，则该电量下的买卖双方均全部完成交易
        if buyer_amount == seller_amount:
            # 进行匹配
            for j in range(n + 1):
                for i in range(m + 1):
                    match_amount = (seller_number_sort_copy[j][1] / buyer_amount) * buyer_number_sort_copy[i][1]
                    # print(match_amount.type())
                    if np.isnan(match_amount):       # 可能是避免输入中Nan的影响
                        match_amount = 0
                    match_amount = int(match_amount + 0.5)  # 四舍五入，取整数
                    match_result.append([buyer_name_sort_copy[i], seller_name_sort_copy[j], match_amount])
            # 删除配对完成的买方
            for i in range(m + 1):
                buyer_number_sort_copy.pop(0)
                buyer_name_sort_copy.pop(0)
            # 删除配对完成的卖方
            for j in range(n + 1):
                seller_number_sort_copy.pop(0)
                seller_name_sort_copy.pop(0)

        # 判断买方交易队列和卖方交易队列，若一方为空，则一方交易完，跳出循环。
        # print("购电方名称：", buyer_name_sort_copy)
        # print("购电方数据：", buyer_number_sort_copy)
        # print("售电方名称：", seller_name_sort_copy)
        # print("售电方数据：", seller_number_sort_copy)
        if (len(buyer_number_sort_copy) == 0) or (len(seller_number_sort_copy) == 0):
            break

    if len(match_result) != 0:
        # 求统一匹配出清价格
        buyer_index = buyer_name_sort.index(match_result[-1][0])
        seller_index = seller_name_sort.index(match_result[-1][1])
        buyer_last_price = buyer_number_sort[buyer_index][0]
        seller_last_price = seller_number_sort[seller_index][0]
        clear_price = round((buyer_last_price + seller_last_price) / 2, 2)  # 出清电价保留两位小数
    else:
        clear_price = 0
    # print(buyer_number_sort_copy)
    # print(seller_number_sort_copy)
    return match_result, clear_price


def select_clear_amount(name, match_result):  # 选择相应名称name公司的出清电量
    amount = 0
    for i in match_result:
        if name in i:
            amount += i[2]
    return amount


def calculate_total_amount(match_result):  # 计算总的出清电量
    amount = 0
    for i in match_result:
        amount += i[2]
    return amount
# 程序使用示例：

# 传入购电方
# buyer_name = ['陕西秦电配售电有限责任公司', '大唐陕西能源营销有限公司','陕西榆林能源集团售电有限公司','陕西深电能售电有限公司','陕西洁能售电有限公司','郑州沃特节能科技股份有限公司','陕西盈智能源科技有限公司']
# buyer_num = [[354.9, 74559],[354.5, 58484],[353.8,31662],[353.3,25406],[352.8,20688],[351.9,20066],[350.5,17432]]
# # 传入售电方
# seller_name = ['陕西华电蒲城发电有限责任公司', '陕西渭河发电有限公司','陕西宝鸡第二发电有限责任公司','陕西华电杨凌热电有限公司','陕西清水川发电有限公司']
# seller_num = [[345.3, 63120], [346.2, 57021],[347.4,50117],[348.8,32016],[349.7,26163]]
# # 进行匹配
# match_result,clear_price = rank(buyer_name,buyer_num,seller_name,seller_num)
# print(match_result)
# # 计算每一个售电方的总成交电量
#
# for i in seller_name:
#     amount = select_clear_amount(i,match_result)
#     value = str(amount)
#     print(i+"  匹配电量  "+value)
# # 计算每一个购电方的总成交电量
# for i in buyer_name:
#     amount = select_clear_amount(i,match_result)
#     value = str(amount)
#     print(i+"  匹配电量  "+value)
# def set_data_for_seller( path="data/seller_data"):
#     """"
#     设置卖方的成本函数，
#     例子 [[76633,0.0016 , 102 , 4,560,000],[....],[....].. ] :
#     第一个卖家 最大申报电量=76633, a_b = 0.0016 , b_g = 102 , c_g =4,560,000
#     总共有 num_of_seller个 ， 成本函数默认从data/seller_data.txt中读取
#     """
#     result = np.loadtxt(path)
#     max_seller_volume = result[:, 0]  # 设置卖方的最大申报电量
#     costfuncton_for_sellers = result[:, 1:result.shape[1]]  # 列切片，去掉第一列。
#     return costfuncton_for_sellers
#
# buyer_name = ["s_a", "s_b", "s_c", "s_d", "s_e", "s_f", "s_g", "s_h", "s_i"]  # 随机取的名字
# seller_name = ["b_big", "b_mid", "b_mid2", "b_small", "b_small"]  # 随机取的名字
# seller_num = [[61.93, 20828.52], [50.64, 53530.98], [400.00, 0.00], [292.61, 34000.00], [383.57, 0.00]]
# buyer_num = [[399.75, 45324.00], [392.75, 31370.00], [366.00, 28219.00], [304.00, 25281.00], [363.75, 18291.00],[ 363.00, 7971.00], [371.00, 7330.00], [351.75, 6989.00], [371.50, 75.00]]
# match_result,clear_price = rank(buyer_name,buyer_num,seller_name,seller_num)
# print(rank(buyer_name,buyer_num,seller_name,seller_num))
# cost = set_data_for_seller()
# for i in seller_name:
#     print(select_clear_amount(i,match_result))
# total_cost =0
#
#
# for i in range(len(seller_num)):
#     cc = seller_num[i][1]**2*cost[i][0] +seller_num[i][1]*cost[i][1]+cost[i][2]
#     rr = select_clear_amount(seller_name[i],match_result) *clear_price
#     total_cost+= cc
#     print(cc,"profit:",rr-cc)
#
# r = calculate_total_amount(match_result)*clear_price
# print(r-total_cost)
def generate_volume(num_of_agent,total_volume):
    """
    生成 num_of_agent个总和为total_volume的数字
    >>> summa = 0
    >>> count =1
    >>> while True:
    >>>     summa += sum(generate_seller_volume(5,100))
    >>>     print(summa/count)
    >>>     count+=1
    >>>100
    >>>100
    >>>100
    .
    .
    .
    """
    dividers = sorted(sample(range(1, total_volume), num_of_agent-1))
    result = sorted([a-b for a, b in zip(dividers + [total_volume], [0]+dividers)])
    result.reverse()
    return np.array(result, dtype=np.float32)

# random_name = ["a", "b", "c"]  # 随机取的名字
# #
# seller_num = [[379.7390435728891, 13918.888088208376], [350.4500071985927, 46853.51455029531], [318.1118151155838, 23628.95238786419]]
# buyer_num = [[446.5236895613607, 27222.100235128702], [424.2412439023235, 19162.2246721079], [415.57281769852966, 16086.718740867243]]
# match_result,clear_price = rank(random_name,buyer_num,random_name,seller_num)

# print(match_result,clear_price)