import os
import time
import pickle

import numpy as np
import tensorflow as tf


class RLLogger(object):

    def __init__(self, exp_name, n_agents, save_episode_rate):
        '''
        '''
        self.n_agents = n_agents  # 智能体数量

        # 实验结果保存目录
        self.ex_path = os.path.join('results', exp_name)
        os.makedirs(self.ex_path, exist_ok=True)
        # 实验模型保存目录
        self.model_path = os.path.join(self.ex_path, 'models')
        os.makedirs(self.model_path, exist_ok=True)
        # 实验Tensorboard数据保存目录
        self.tb_path = os.path.join(self.ex_path, 'tb_logs')
        os.makedirs(self.tb_path, exist_ok=True)
        self.tb_writer = tf.summary.create_file_writer(self.tb_path)
        self.tb_writer.set_as_default()

        self.episode_rewards = [0.0]  # 记录训练工程中各个episode的奖励
        self.train_step = 0  # 记录训练了多少步
        self.episode_step = 0  # 记录当前episode中环境已经执行的步数
        self.episode_count = 0  # 记录采样了多少个Episode
        self.t_start = time.time()  # 记录开始训练的时间
        self.t_last_print = time.time()  # 记录上一次打印信息的时间

        self.save_episode_rate = save_episode_rate  # 隔多少次保存模型

    def record_episode_end(self, agents):
        """
        记录当前episode结束
        """
        self.log_scalar('training/episode_reward', self.episode_rewards[-1], step_type='episode')

        self.episode_count += 1  # 当前episode数量加一
        self.episode_step = 0  # 新的episode步数归0
        self.episode_rewards.append(0.0)  # 新的episode总奖励置0

        if self.episode_count % self.save_episode_rate == 0:
            self.print_metrics()
            self.save_models(agents)

    def experiment_end(self):
        # 保存Trace信息到文件，写入日志，可视化网络模型
        with self.tb_writer.as_default():
            tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=self.tb_path)
        # 保存训练时每个episode的奖励得分到文件
        rew_file_name = os.path.join(self.ex_path, 'rewards.pkl')
        with open(rew_file_name, 'wb') as fp:
            pickle.dump(self.episode_rewards, fp)

        # print(np.array(self.episode_rewards))
        print('...Finished total of {} episodes in {} minutes.'.format(self.episode_count,
                                                                       (time.time() - self.t_start) / 60))

    def print_metrics(self):
        print('steps: {}, episodes: {}, mean episode reward: {}, time: {}'.format(
            self.train_step, self.episode_count, np.mean(self.episode_rewards[-self.save_episode_rate:-1]),
            round(time.time() - self.t_last_print, 3)))
        self.t_last_print = time.time()

    def save_models(self, agents):
        for idx, agent in enumerate(agents):
            agent.save(os.path.join(self.model_path, 'agent_{}'.format(idx)))

    def log_scalar(self, k, v, step_type='train_step'):
        if step_type == 'train_step':
            step = self.train_step
        elif step_type == 'episode':
            step = self.episode_count
        else:
            raise NotImplementedError

        with self.tb_writer.as_default():
            tf.summary.experimental.set_step(step)
            tf.summary.scalar(k, v)
