import tensorflow as tf

from matd3.util import clip_by_local_norm


class MAPolicyNetwork(object):
    def __init__(self,
                 num_layers,
                 units_per_layer,
                 lr,
                 obs_n_shape,
                 act_shape,
                 q_network,
                 agent_index):
        """
        Implementation of the policy network.
        """
        self.num_layers = num_layers
        self.lr = lr
        self.obs_n_shape = obs_n_shape
        self.act_shape = act_shape
        self.q_network = q_network  # Critic 1 网络
        self.agent_index = agent_index  # 当前智能体的序号
        self.clip_norm = 0.5

        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        ### set up network structure
        self.obs_input = tf.keras.layers.Input(shape=self.obs_n_shape[agent_index])

        self.hidden_layers = []
        for idx in range(num_layers):
            layer = tf.keras.layers.Dense(units_per_layer,
                                          activation='relu',
                                          name='ag{}pol_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(self.act_shape,
                                                  activation='tanh',
                                                  name='ag{}pol_out'.format(agent_index))

        # connect layers
        x = self.obs_input
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=[self.obs_input], outputs=[x])

    def forward_pass(self, obs):
        """
        根据状态输出动作
        """
        x = obs
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        outputs = self.output_layer(x)  # 值域-1到1
        return outputs

    def sample_action(self, ):
        """
        输出随机探索动作
        """
        return tf.random.uniform(self.act_shape, -1, 1)

    @tf.function
    def get_action(self, obs):
        return self.forward_pass(obs)

    @tf.function
    def train(self, obs_n, act_n):
        with tf.GradientTape() as tape:
            x = self.forward_pass(obs_n[self.agent_index])
            act_n = tf.unstack(act_n)
            act_n[self.agent_index] = x  # 替换动作序列中，当前智能体的动作为新采样的，其他智能体的动作保持从经验池中采样的
            q_value = self.q_network._predict_internal(obs_n + act_n)  # 预测Q值
            loss = -tf.math.reduce_mean(q_value)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        # gradients = tf.clip_by_global_norm(gradients, self.clip_norm)[0]
        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))
        return loss
