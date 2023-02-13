import numpy as np
import tensorflow as tf

from gym import Space

from matd3.common.replay_buffer import ReplayBuffer
from matd3.networks.actor import MAPolicyNetwork
from matd3.networks.critic import MACriticNetwork


class MATD3Agent:

    def __init__(self,
                 obs_space_n,
                 act_space_n,
                 agent_index,
                 logger,
                 batch_size,
                 buff_size,
                 policy_lr,
                 policy_num_layers,
                 policy_num_units,
                 critic_lr,
                 critic_num_layers,
                 critic_num_units,
                 gamma,
                 tau,
                 policy_update_freq=2,
                 target_policy_smoothing_eps=0.0):
        """
        An object containing critic, actor and training functions for Multi-Agent TD3.
        """
        assert isinstance(obs_space_n[0], Space)
        obs_shape_n = np.array([space.shape for space in obs_space_n])
        act_shape_n = np.array([space.shape for space in act_space_n])
        self.act_shape_n = act_shape_n
        self.obs_shape_n = obs_shape_n
        self.logger = logger
        # 生成经验池
        self.replay_buffer = ReplayBuffer(int(buff_size),
                                          len(obs_shape_n),
                                          obs_shape_n,
                                          act_shape_n)

        # 初始化Q值函数估计器
        self.critic_1 = MACriticNetwork(critic_num_layers,
                                        critic_num_units,
                                        critic_lr,
                                        obs_shape_n,
                                        act_shape_n,
                                        agent_index)
        # 初始化Q目标值函数估计器
        self.critic_1_target = MACriticNetwork(critic_num_layers,
                                               critic_num_units,
                                               critic_lr,
                                               obs_shape_n,
                                               act_shape_n,
                                               agent_index)
        self.critic_1_target.model.set_weights(self.critic_1.model.get_weights())

        # 初始化Q值函数估计器
        self.critic_2 = MACriticNetwork(critic_num_layers,
                                        critic_num_units,
                                        critic_lr,
                                        obs_shape_n,
                                        act_shape_n,
                                        agent_index)
        # 初始化Q目标值函数估计器
        self.critic_2_target = MACriticNetwork(critic_num_layers,
                                               critic_num_units,
                                               critic_lr,
                                               obs_shape_n,
                                               act_shape_n,
                                               agent_index)
        self.critic_2_target.model.set_weights(self.critic_2.model.get_weights())

        # 初始化策略
        self.policy = MAPolicyNetwork(policy_num_layers,
                                      policy_num_units,
                                      policy_lr,
                                      obs_shape_n,
                                      act_shape_n[agent_index],
                                      self.critic_1,
                                      agent_index)
        # 初始化目标策略
        self.policy_target = MAPolicyNetwork(policy_num_layers,
                                             policy_num_units,
                                             policy_lr,
                                             obs_shape_n,
                                             act_shape_n[agent_index],
                                             self.critic_1,
                                             agent_index)
        self.policy_target.model.set_weights(self.policy.model.get_weights())

        self.batch_size = batch_size  # 训练时的批大小
        self.gamma = gamma  # 折扣因子
        self.tau = tau  # 目标网络软更新系数
        self.policy_update_freq = policy_update_freq  # 策略网络延迟更新的步数
        self.target_policy_smoothing_eps = target_policy_smoothing_eps
        self.update_counter = 0  # 记录更新次数
        self.agent_index = agent_index  # 当前智能体的序号

    def action(self, obs, action_exploration_noise=0.2):
        """
        根据策略选择动作
        """
        action = self.policy.get_action(obs[None])[0].numpy()
        noise = np.random.normal(0, action_exploration_noise, size=action.shape)
        action = np.clip(action + noise, -1.0, 1.0)
        return action

    def add_transition(self, obs_n, act_n, rew, new_obs_n, done_n):
        """
        添加经验到经验池中，s,a,r,s_,done 五元组
        """
        self.replay_buffer.add(obs_n, act_n, rew, new_obs_n, float(done_n))

    def target_action(self, obs):
        """
        从目标策略中选择动作
        """
        return self.policy_target.get_action(obs)

    def update_target_networks(self, tau):
        """
        软更新源网络中的参数权重到目标网络
        """

        def update_target_network(net: tf.keras.Model, target_net: tf.keras.Model):
            net_weights = np.array(net.get_weights(), dtype=object)
            target_net_weights = np.array(target_net.get_weights(), dtype=object)
            new_weights = tau * net_weights + (1.0 - tau) * target_net_weights
            target_net.set_weights(new_weights)

        update_target_network(self.critic_1.model, self.critic_1_target.model)
        update_target_network(self.critic_2.model, self.critic_2_target.model)
        update_target_network(self.policy.model, self.policy_target.model)

    def update(self, agents):
        """
        Update the agent, by first updating the two critics and then the policy.
        Requires the list of the other agents as input, to determine the target actions.
        """
        # 确保要更新的智能体在智能体策略列表中位置是对的
        assert agents[self.agent_index] is self, "assert agents[self.agent_index] is self"
        # 更新次数加1
        self.update_counter += 1

        # 从经验池中采样经验
        obs_n, acts_n, rew_n, next_obs_n, done_n = self.replay_buffer.sample(self.batch_size)

        # Train the critic, using the target actions in the target critic network, to determine the
        # training target (i.e. target in MSE loss) for the critic update.
        target_act_next = [ag.target_action(obs) for ag, obs in zip(agents, next_obs_n)]
        # 为当前智能体产生的目标动作添加噪声
        noise = np.random.normal(0, self.target_policy_smoothing_eps, size=target_act_next[self.agent_index].shape)
        noise = np.clip(noise, -0.5, 0.5)
        target_act_next[self.agent_index] = np.clip(target_act_next[self.agent_index] + noise, -1.0, 1.0)

        critic_outputs = np.empty([2, self.batch_size], dtype=np.float32)
        critic_outputs[0] = self.critic_1_target.predict(next_obs_n, target_act_next)[:, 0]
        critic_outputs[1] = self.critic_2_target.predict(next_obs_n, target_act_next)[:, 0]
        target_q_next = np.min(critic_outputs, 0)[:, None]

        # 计算目标值
        q_train_target = rew_n[:, None] + self.gamma * target_q_next

        # 计算两个Critic网络的时序残差
        td_loss = np.empty([2, self.batch_size], dtype=np.float32)
        td_loss[0] = self.critic_1.train_step(obs_n, acts_n, q_train_target).numpy()[:, 0]
        td_loss[1] = self.critic_2.train_step(obs_n, acts_n, q_train_target).numpy()[:, 0]
        self.logger.log_scalar('agent_{}/q_loss0'.format(self.agent_index), np.mean(td_loss[0]))
        self.logger.log_scalar('agent_{}/q_loss1'.format(self.agent_index), np.mean(td_loss[1]))

        # 延迟更新Actor网络
        if self.update_counter % self.policy_update_freq == 0:  # delayed policy updates
            # 训练策略
            policy_loss = self.policy.train(obs_n, acts_n)
            self.logger.log_scalar('agent_{}/policy_loss'.format(self.agent_index), policy_loss.numpy())
            # 更新目标网络权重
            self.update_target_networks(self.tau)
        else:
            policy_loss = None

        return [td_loss, policy_loss]

    def save(self, fp):
        """保存模型到地址"""
        self.critic_1.model.save_weights(fp + 'critic_1.h5', )
        self.critic_1_target.model.save_weights(fp + 'critic_1_target.h5')
        self.critic_2.model.save_weights(fp + 'critic_2.h5', )
        self.critic_2_target.model.save_weights(fp + 'critic_2_target.h5')

        self.policy.model.save_weights(fp + 'policy.h5')
        self.policy_target.model.save_weights(fp + 'policy_target.h5')

    def load(self, fp):
        """从地址加载模型"""
        self.critic_1.model.load_weights(fp + 'critic_1.h5', )
        self.critic_1_target.model.load_weights(fp + 'critic_1_target.h5')
        self.critic_2.model.load_weights(fp + 'critic_2.h5', )
        self.critic_2_target.model.load_weights(fp + 'critic_2_target.h5')

        self.policy.model.load_weights(fp + 'policy.h5')
        self.policy_target.model.load_weights(fp + 'policy_target.h5')
