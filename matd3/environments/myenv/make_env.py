# from matd3.environments.myenv.env_new import MyEnv
from matd3.environments.myenv.env_range import RangeEnv


def make_env():
    # todo：补全，传入必要参数去初始化训练场景
    return RangeEnv()
