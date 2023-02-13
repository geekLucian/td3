import tensorflow as tf

from matd3.common.util import clip_by_local_norm


class MACriticNetwork(object):
    def __init__(self,
                 num_hidden_layers,
                 units_per_layer,
                 lr,
                 obs_n_shape,
                 act_shape_n,
                 agent_index):
        """
        Implementation of a critic to represent the Q-Values.
        """
        self.num_layers = num_hidden_layers
        self.lr = lr
        self.obs_shape_n = obs_n_shape
        self.act_shape_n = act_shape_n

        self.clip_norm = 0.5  # 梯度裁剪bound
        self.optimizer = tf.keras.optimizers.Adam(lr=self.lr)

        # set up layers
        # each agent's action and obs are treated as separate inputs
        self.obs_input_n = []
        for idx, shape in enumerate(self.obs_shape_n):
            self.obs_input_n.append(tf.keras.layers.Input(shape=shape, name='obs_in' + str(idx)))

        self.act_input_n = []
        for idx, shape in enumerate(self.act_shape_n):
            self.act_input_n.append(tf.keras.layers.Input(shape=shape, name='act_in' + str(idx)))

        self.input_concat_layer = tf.keras.layers.Concatenate()

        self.hidden_layers = []
        for idx in range(num_hidden_layers):
            layer = tf.keras.layers.Dense(units_per_layer, activation='relu',
                                          name='ag{}crit_hid{}'.format(agent_index, idx))
            self.hidden_layers.append(layer)

        self.output_layer = tf.keras.layers.Dense(1, activation='linear',
                                                  name='ag{}crit_out'.format(agent_index))

        # connect layers
        x = self.input_concat_layer(self.obs_input_n + self.act_input_n)  # 拼接所有智能体的状态和动作
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)

        self.model = tf.keras.Model(inputs=self.obs_input_n + self.act_input_n, outputs=[x])
        self.model.compile(self.optimizer, loss='mse')

    def predict(self, obs_n, act_n):
        """
        预测Q值
        """
        return self._predict_internal(obs_n + act_n)

    @tf.function
    def _predict_internal(self, concatenated_input):
        x = self.input_concat_layer(concatenated_input)
        for idx in range(self.num_layers):
            x = self.hidden_layers[idx](x)
        x = self.output_layer(x)
        return x

    def train_step(self, obs_n, act_n, target_q):
        """
        训练值网络
        """
        return self._train_step_internal(obs_n + act_n, target_q)

    @tf.function
    def _train_step_internal(self, concatenated_input, target_q):
        """
        由于连接Concat不能在tf.function内执行，所以把obs_n + act_n放在_train_step_internal之外
        """
        with tf.GradientTape() as tape:
            x = self.input_concat_layer(concatenated_input)
            for idx in range(self.num_layers):
                x = self.hidden_layers[idx](x)
            q_pred = self.output_layer(x)
            td_loss = tf.math.square(target_q - q_pred)
            loss = tf.reduce_mean(td_loss)

        gradients = tape.gradient(loss, self.model.trainable_variables)

        local_clipped = clip_by_local_norm(gradients, self.clip_norm)
        self.optimizer.apply_gradients(zip(local_clipped, self.model.trainable_variables))

        return td_loss
