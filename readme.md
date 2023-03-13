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
python main.py 2> err.log | tee out.log
```

This will create `./results` directory, which saves training log and trained model. To check the training process:

```commandline
tensorboard --logdir results
```

Then open `http://localhost:6006` in browser to see the training logs

To run unit test:
```commandline
python -m unittest matd3.test.<test_name>
```

## TODO

results naming: `model_ratio`

- model: normal, reverse, volume
- ratio: blance, buyer_more (10%~20%), seller_more, buyer_much_more (>50%), seller_much_more

- [ ] implement `reverse`
- [ ] understand & implement `volume`
- [ ] understand how to adjust ratio
- [ ] design expirments and batch testing

not urgent:

- [ ] test `action_strategy` wich maps the NN's output to real values
- [ ] fix `tf.function retracing` wanrings by converting data structures into tensors
- [ ] automated regression test
- [ ] find better reward function
- [ ] make logs and results prettier
- [ ] translation
