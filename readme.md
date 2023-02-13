# TODO

- [ ] 改网络结构，对卖方增加一个维度的输出，卖方报价 -> [卖方报价下限， 卖方报价上限]
- [ ] 匹配出清改了action data layout，对应要改step的调用
- [ ] 改到了出清
- [ ] 测试

# MATD3 with TF2.x

## 代码结构

以 `_n` 为变量名后缀的表示列表, 比如 `state_n` 包含所有智能体的状态

## 用法

`python >= 3.6`

在运行环境中（如果使用`anaconda`，则可以先创建conda虚拟环境：`conda create -n xxx python=3.8`），安装依赖

```shell
pip install -r requirements.txt
```

开启训练：

```shell
python train.py
```

训练时的一些相关参数，在`train.py`中`train`函数默认值上修改即可，
智能体参数的配置在`train.py`中`MATD3Agent`调用时默认值上修改即可。

注意：每次训练最好起不同的训练名字，否则多次训练的实验结果会放到同一个文件夹里。

实验结果和模型存放在`results`文件夹内。

### 查看训练过程

```commandline
切换到实验目录

tensorboard --logdir results
```

在浏览器输入`http://localhost:6006`以查看训练情况

## 自定义环境

在`matd3/environments/myenv`文件夹下实现自己的训练环境。

其中，`env.py`用于撰写环境类，里边已经定义好的函数需要按需求重新实现，因为在训练时会有调用。
函数相应的输入格式、输出格式，均在文件中有注释说明，一定要确保函数返回格式符合定义，或者符合举例的形式。

`make_env.py`用于根据`env.py`定义的环境类生成训练场景，供`train.py`中的智能体交互训练。

撰写好这两个自定义文件后，在`train.py`头部将导入改为：

```python
# from matd3.environments.testenv.make_env import make_env

from matd3.environments.myenv.make_env import make_env
```

即可开启用自定义的场景训练。