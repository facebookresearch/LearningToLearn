# LearningToLearn

## ML3 paper experiments and citation
To reproduce results of the ML3 paper follow the instructions.
All loss models are stored in 'experiments/data, all plots are stored in ./plots

#### Loss Learning for Regression
COMING SOON

#### Reward Learning for Model-based RL (MBRL) Reacher
For meta learning the loss run

```
python experiments/mbrl_reacher.py train
```

For testing the loss run

```
python experiments/mbrl_reacher.py test
```


#### Learning with extra information at meta-train time
##### For meta learning the loss with or without extra information on sine function run:
```
python experiments/shaped_sine.py train extra_info=True
```
To test the loss with or without extra information run:
```
python experiments/shaped_sine.py test extra_info=True
```
##### For meta learning the loss with or without additional goal in the mountain car experiment run:
```
python experiments/mountain_car.py train extra_info=True
```
To test the loss with or without extra goal run:
```
python experiments/mountain_car.py test extra_info=True
```