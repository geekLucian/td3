import numpy as np


class ReplayBuffer(object):
    def __init__(self, size, n_agents, obs_shape_n, act_shape_n):
        """Create Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._obs_n = []  # 存储所有Agent的状态S
        self._acts_n = []  # 存储所有Agent的动作A
        self._obs_next_n = []  # 存储所有Agent的下一状态S'
        self._n_agents = n_agents  # 智能体数量
        for idx in range(n_agents):
            self._obs_n.append(np.empty([size, obs_shape_n[idx, 0]], dtype=np.float32))  # 存储单个Agent的状态S
            self._acts_n.append(np.empty([size, act_shape_n[idx][0]], dtype=np.float32))  # 存储单个Agent的动作A
            self._obs_next_n.append(np.empty([size, obs_shape_n[idx, 0]], dtype=np.float32))  # 存储单个Agent的下一状态S'
        self._done = np.empty([size], dtype=np.float32)  # 存储是否episode结束的标量
        self._reward = np.empty([size], dtype=np.float32)  # 存储奖励标量
        self._maxsize = size  # 经验池的大小
        self._next_idx = 0  # 下一个用于存储经验的序号地址
        self.full = False  # 判断经验池是否已经满了
        self.len = 0  # 当前经验池已存数据的数量

    def __len__(self):
        return self.len  # 当前经验池已存数据的数量

    def add(self, obs_t, action, reward, obs_next, done):
        """添加经验到经验池中"""
        for ag_idx in range(self._n_agents):
            self._obs_n[ag_idx][self._next_idx] = obs_t[ag_idx]
            self._acts_n[ag_idx][self._next_idx] = action[ag_idx]
            self._obs_next_n[ag_idx][self._next_idx] = obs_next[ag_idx]
        self._reward[self._next_idx] = reward
        self._done[self._next_idx] = done

        # 计算下一个要存储经验的地址
        if not self.full:
            self._next_idx = self._next_idx + 1
            if self._next_idx > self._maxsize - 1:
                self.full = True
                self.len = self._maxsize
                self._next_idx = self._next_idx % self._maxsize
            else:
                self.len = self._next_idx - 1
        else:
            self._next_idx = (self._next_idx + 1) % self._maxsize

    def sample(self, batch_size):
        """
        采样经验

        Parameters
        ----------
        batch_size: int 指定要采样多少经验

        Returns
        -------
        s, a, r, s_, d 五元组
        """
        if batch_size > self.len:  # 经验不足不可采样
            raise RuntimeError('Too few samples in buffer to generate batch.')

        indices = np.random.randint(self.len, size=[batch_size])

        obs_n = []
        acts_n = []
        next_obs_n = []
        for ag_idx in range(self._n_agents):
            obs_n.append(self._obs_n[ag_idx][indices])
            acts_n.append(self._acts_n[ag_idx][indices].copy())
            next_obs_n.append(self._obs_next_n[ag_idx][indices])

        rew = self._reward[indices]
        done = self._done[indices]
        return obs_n, acts_n, rew, next_obs_n, done
