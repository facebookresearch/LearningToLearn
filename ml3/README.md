# LearningToLearn

## ML3 paper experiments and citation
To reproduce results of the ML3 paper follow the instructions.
All loss models are stored in 'experiments/data, all plots are stored in ./plots

#### Loss Learning for Regression (ML3 paper experiment section 4.1.1)
For meta learning the loss run

```
python experiments/run_sine_regression_exp.py
```

For visualizing the results run `jupyter notebook` and open `ml3_sine_regression_exp_viz`

#### Reward Learning for Model-based RL (MBRL) Reacher
For meta learning the loss run

```
python experiments/run_mbrl_reacher_exp.py train
```

For testing the loss run

```
python experiments/run_mbrl_reacher_exp.py test
```

#### Learning with extra information at meta-train time
The following scripts require two arguments, first one is `train\test`, the 2nd one 
indicates whether to use extra information by setting `True\False` (with\without extra info)
##### For meta learning the loss with extra information on sine function run:
```
python experiments/run_shaped_sine_exp.py train True
```
To test the loss with extra information run:
```
python experiments/run_shaped_sine_exp.py test True
```
The test script generates a gif of the final policy, and stores it in the experiment folder 
##### For meta learning the loss with additional goal in the mountain car experiment run:
```
python experiments/run_mountain_car_exp.py train True
```
To test the loss with extra goal run:
```
python experiments/run_mountain_car_exp.py True
```
The test script generates a gif of the final policy, and stores it in the experiment folder 