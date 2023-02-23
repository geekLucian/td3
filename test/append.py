import numpy as np
from gym import spaces

if __name__ == "__main__":
    # ======================== TEST 1 ========================
    action_space = []
    for i in range(3):
        action_space.append(spaces.Box(low=np.array([0, 0, 0]),
                                       high=np.array([1, 1, 1]),
                                       shape=(3,), dtype=np.float32))
    for i in range(3):
        action_space.append(spaces.Box(low=np.array([0, 0, 0]),
                                       high=np.array([1, 1, 1]),
                                       shape=(3,), dtype=np.float32))

    action = [action_space[i].sample() for i in range(len(action_space))]
    res = np.array(action).flatten()
    print(res)

    # ======================== TEST 2 ========================
    l1 = []
    l1.extend(res)
    l2 = res
    print(type(l1))
    print(type(l2))

    # ======================== TEST 3 ========================
    ls = [1, 2, 3, 4]
    i = 2
    a = ls[i] = ls[i] *333
    print(a)
    print(ls[i])
    print(ls)

    # ======================== TEST 4 ========================
    res = [0] * 3
    assert res == [0, 0, 0]
    print("SUCCESSFUL")