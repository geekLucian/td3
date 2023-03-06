# Range-based Electricity Trading with MATD3

## Project structure
```text
.
├── common              # common utils (loggers)
├── data                # condition data (seller/buyer price/volume limits and seller's cost function)
├── main.py             # simulate the market with multi-agent reinforcement learning
├── matd3               # core MATD3 algorithm, trading environment and unit tests
├── readme.md           # this user guide
├── requirements.txt    # package dependencies
└── results             # trained models and logs
```

## Getting started

Create environments (`conda activate <env_name>` if already created):

```commandline
conda create --name <env_name> --channel conda-forge --file requirements.txt
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

Then open `http://localhost:6006` in browser to see the training logs

To run unit test:
```commandline
python -m unittest matd3.test.<test_name>
```

## TODO

- [ ] test `action_strategy` wich maps the NN's output to real values
- [ ] automated regression test
- [ ] find better reward function
- [ ] make logs and results prettier
- [ ] translation
