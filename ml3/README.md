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
For meta learning the reward, run

```
python experiments/run_mbrl_reacher_exp.py train
```

For testing the reward, run

```
python experiments/run_mbrl_reacher_exp.py test
```

#### Learning with extra information at meta-train time
The following scripts require two arguments, first one is `train\test`, the 2nd one 
indicates whether to use extra information by setting `True\False` (with\without extra info)

##### For meta learning the loss with extra information on sine function run:
In this experiment we show how the extra info can be used to shape the loss function for easier optimization.
```
python experiments/run_shaped_sine_exp.py train True
```
To test the loss with extra information run:
```
python experiments/run_shaped_sine_exp.py test True
```
To see how these results compare to not using the extra info, run the above scripts with the 2nd argument being `False`
To visualize the loss landscapes for this experiment run `jupyter notebook` and open `Loss shaping visualization.ipynb`

##### For meta learning the loss with additional goal in the mountain car experiment run:
In this experiment we show how the extra info can be used to guide exploration for an RL task.
```
python experiments/run_mountain_car_exp.py train True
```
To test the loss with extra goal run:
```
python experiments/run_mountain_car_exp.py test True
```
The test script generates a gif of the final policy, and stores it in the experiment folder 
To see how these results compare to not using the extra info, run the above scripts with the 2nd argument being `False`
