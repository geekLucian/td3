import argparse
import os
import time

import numpy as np
import tensorflow as tf
from datetime import datetime

from matd3.MATD3Agent import MATD3Agent
from matd3.RangeEnv import RangeEnv
from common.RLLogger import RLLogger
from common.util import info

# 设置GPU显存动态分配
if tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

now = datetime.now().strftime("-%Y-%m-%d-%H-%M-%S")


def train(exp_name='range_pricing' + now,   # 指定本次实验的命名
          save_episode_rate=20,             # 指定多久保存一次模型，输出结果
          display=False,                    # 是否渲染环境
          restore_filepath=None,            # 指定加载已存在模型的路径，比如 "results/xxx/models"
          max_episode_len=10,               # 指定每个episode最长多少步
          num_episodes=1500,                # 指定最多训练多少episode，然后终止训练
          skip_log=False,                   # Whether disable logging
          mode="normal",                    # Specify which experiments to run
          svs=1.0,                          # seller volume scale
          bvs=1.0,                          # buyer volume scale
          rs=1.0                            # range size scale
          ):
    # 创建环境
    env = RangeEnv(mode, svs, bvs, rs)

    if skip_log:
        print("Log disabled")
        exp_name += "_nolog"
    logger = RLLogger(exp_name, env.n, save_episode_rate, skip_log)
    tf.summary.trace_on(graph=False)

    # 创建智能体
    agents = []
    for agent_idx in range(env.n):
        agents.append(MATD3Agent(env.observation_space,  # 所有智能体的状态空间
                                 env.action_space,  # 所有智能体的动作空间
                                 agent_idx,  # 当前智能体的序号
                                 logger=logger,  # 记录器
                                 batch_size=1024,  # 训练时采样的批大小
                                 buff_size=1e5,  # 经验池总大小
                                 policy_lr=1e-5,  # Policy网络学习率
                                 policy_num_layers=2,  # Policy网络隐藏层的数量
                                 policy_num_units=32,  # Policy网络单个隐藏层的的神经元数量
                                 critic_lr=1e-5,  # Critic网络学习率
                                 critic_num_layers=3,  # Critic网络隐藏层的数量
                                 critic_num_units=128,  # Critic网络单个隐藏层的的神经元数量
                                 gamma=0.1,  # 折扣因子
                                 tau=0.01,  # 软更新系数
                                 policy_update_freq=2,  # 策略延迟间隔
                                 target_policy_smoothing_eps=0.2  # 目标网络选动作时的噪声程度
                                 ))

    # 加载已有的训练模型
    if restore_filepath is not None:
        print('Loading previous state...')
        for ag_idx, agent in enumerate(agents):
            fp = os.path.join(restore_filepath, 'agent_{}'.format(ag_idx))
            agent.load(fp)

    # 重置环境
    obs_n, combine_reward0, clear_price, end_reason = env.reset()

    # 开始迭代收集数据，训练
    print('Starting iterations...')
    while True:
        if logger.episode_count > 180:
            action_n = [agent.action(obs.astype(np.float32), 0.2 / (logger.train_step / 1500 + 1)) for agent, obs in
                        zip(agents, obs_n)]
        else:
            action_n = [agent.policy.sample_action().numpy() for agent, obs in zip(agents, obs_n)]

        # 执行动作
        (new_obs_n, profit, clear_price, total_match_volume, seller_clear_volume, buyer_clear_volume, match_result,
         seller_profit, end_reason, action, done, _) = env.step(action_n)
        done |= (logger.episode_step >= max_episode_len)  # 判断当前episode的步长是否已经超过设定的最大步长限制
        rew = 0 * profit + 1 * total_match_volume
        logger.episode_step += 1
        logger.episode_rewards[-1] += rew

        # 保存经验到各个策略的经验池中
        for i, agent in enumerate(agents):
            agent.add_transition(obs_n, action_n, rew, new_obs_n, done)
        obs_n = new_obs_n  # 设定新状态

        # 判断episode是否结束
        if done:
            if logger.episode_count % 100 == 0:
                printlog(action_n, action, clear_price, match_result, end_reason)
            for i in range(env.num_of_seller):
                logger.log_scalar('seller_{}/profit'.format(i), seller_profit[i], step_type='episode')
                logger.log_scalar('seller_{}/volume'.format(i), seller_clear_volume[i], step_type='episode')
                logger.log_scalar('seller_{}/price'.format(i), clear_price[i], step_type='episode')
            for i in range(env.num_of_buyer):
                logger.log_scalar('buyer_{}/volume'.format(i), buyer_clear_volume[i], step_type='episode')
                logger.log_scalar('buyer_{}/price'.format(i), clear_price[i + env.num_of_seller], step_type='episode')
            # logger.log_scalar('training/clear_price', clear_price, step_type='episode')
            logger.log_scalar('training/total_profit', profit, step_type='episode')
            logger.log_scalar('training/total_match_volume', total_match_volume, step_type='episode')

            obs_n, combine_reward0, clear_price, end_reason = env.reset()  # 重置场景，拿到新的状态
            logger.record_episode_end(agents)  # 记录数据，重置某些变量，输出到tensorboard

        # 更新策略，经验充足
        _trained = False
        for agent in agents:
            if len(agent.replay_buffer) > agent.batch_size * max_episode_len / 4:
                agent.update(agents)
                _trained = True
        if _trained:
            logger.train_step += 1  # 训练步数增加1

        # 渲染界面
        if display:
            time.sleep(0.1)
            env.render()

        if logger.episode_count >= num_episodes:
            logger.experiment_end()
            break

def printlog(action_n, action, clear_price, match_result, end_reason):
    action_msg = ""
    for idx in range(len(action)):
        if idx % 3 == 0:
            action_msg += "\t["
        action_msg += "{}, ".format(action[idx])
        if idx % 3 == 2:
            action_msg += "\b\b]\n"
    info("神经网络输出数据:\n"
        "   [[seller0_price_lower, seller0_price_range_factor, seller0_volume], ...,\n"
            "   [buyer0_price, placeholder, buyer0_volume], ...]\n\t" + 
            '\n\t'.join(map(str, action_n)))
    print(action_n)
    info("实际报价报量:\n"
        "\t[seller0_price_lower, seller0_price_upper, seller0_volume, ...,\n"
        "\t buyer0_price, placeholder, buyer0_volume, ...]\n" + 
            action_msg)
    print(action)
    info("平均价格:\n\t[seller0_avg_price, seller1_avg_price, ..., buyer0_avg_price]\n" + str(clear_price))
    print(clear_price)
    info("交易记录:\n\t[[buyer_name, seller_name, match_volume, match_price]]\n\t" +
            '\n\t'.join(map(str, match_result)))
    print(match_result)
    info("共{}条，出清中止原因：{}".format(len(match_result), end_reason))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', action='store_true')
    parser.add_argument('-p', '--pretrained', action='store_true')
    parser.add_argument('-m', '--mode', type=str, action='store', required=True, choices=[
        "normal", "reverse", "volume", "felxible"
    ])
    parser.add_argument('-s', '--seller_volume_scale', type=float, action='store', default=1.0)
    parser.add_argument('-b', '--buyer_volume_scale', type=float, action='store', default=1.0)
    parser.add_argument('-r', '--range_scale', type=float, action='store', default=1.0)
    args = parser.parse_args()
    train(skip_log=(not args.log), 
          restore_filepath="results/range_pricing_pretrained/models" if args.pretrained else None,
          mode=args.mode,
          svs=args.seller_volume_scale,
          bvs=args.buyer_volume_scale,
          rs=args.range_scale)
