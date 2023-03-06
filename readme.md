# Range-based Electricity Trading with MATD3

## Project structure
```text
.
├── train_range.py      # enter point, simulate the market with reinforcement learning
├── common              # common utils (loggers)
├── data                # condition data (seller/buyer price/volume limits and seller's cost function)
├── matchTest.py        # matching method test (should be refactored into test directory)
├── matd3               # core MATD3 algorithm and trading environment
├── readme.md           # this user guide
├── requirements.txt    # package dependencies
├── results             # trained models and logs
└── test                # unit tests
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
- [ ] translation
