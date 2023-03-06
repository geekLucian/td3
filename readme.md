# Range-based Electricity Trading with MATD3

## Project structure
```text
.
├── data                # legacy
├── data2               # seller/buyer price/volume limits and seller's cost function
├── matchTest.py        # matching method test (should be refactored into test directory)
├── matd3               # project src
├── readme.md           # this user guide
├── requirements.txt    # for building environments
├── results             # demonstrated below
├── test                # unit test
├── train.py            # legacy
└── train_range.py      # enter point, simulate the market with reinforcement learning
```

## Getting started

Create environments (`conda activate <env_name>` if already created):

```commandline
conda create --name <env_name> --file requirements.txt
```

Train:

```commandline
python ./train_env.py
```

This will create `./results` directory, which saves training log and trained model. To check the training process:

```commandline
cd ./results/<exp_name>
tensorboard --logdir results
```

Open `http://localhost:6006` in browser

## TODO

- [ ] test and debug
    - [ ] test `action_strategy` wich maps the NN's output to real values
    - [ ] refactor with `unittest`
    - [ ] automated regression test
- [ ] find better reward function
- [ ] make logs and results prettier
- [ ] remove legacies
- [ ] translation
