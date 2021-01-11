# LearningToLearn

## Setup
In the LearningToLearn folder, run:

```
conda create -n ml3 python=3.7
conda activate ml3
python setup.py develop
```

In the LearningToLearn folder, follow the commands to store plots and data:

```
mkdir plots
mkdir experiments/data
```

## ML3 paper experiments and citation
To reproduce results of the ML3 paper follow the instructions.
All loss models are stored in ./data, all plots are stored in ./plots

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
#### Citation
```
@inproceedings{ml3,
author    = {Sarah Bechtle and Artem Molchanov and Yevgen Chebotar and Edward Grefenstette and Ludovic Righetti and Gaurav Sukhatme and Franziska Meier},
title     = {Meta Learning via Learned Loss},
booktitle = {International Conference on Pattern Recognition, {ICPR}, Italy, January 10-15, 2021},
year      = {2021} }
```

## MBIRL - Model Based Inverse Reinforcement Learning

### Simulation
#### Dependencies:
```python 3.6
pytorch
math
numpy
higher
cvxpy
urdf_parser_py
```
#### Generate Data
1. ```python mbirl/generate_expert_demo_interpolate.py``` 
2. Check the data and visualizations of the demonstration in 'mbirl/traj_data'

#### Run Our Method
1. ```python mbirl/example_irl_cost_learning_our_method.py```
2. Check the trajectories predicted during training in model_data/placing/<cost_type>

#### Plot the losses, evaluate our method
1. ```jupyter notebook```
2. Access the notebook in the browser in 'mbirl/plot_mbirl_training_and_eval.ipynb'

#### Citation
```
@InProceedings{mbirl,
  author    = {Neha Das, Sarah Bechtle, Todor Davchev, Dinesh Jayaraman, Akshara Rai and Franziska Meier},
  booktitle = {Conference on Robot Learning (CoRL)},
  title     = {Model Based Inverse Reinforcement Learning from Visual Demonstration},
  year      = {2020},
  video     = {https://www.youtube.com/watch?v=sRrNhtLk12M&t=52s},
}
```

## License

`LearningToLearn` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
See also our [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
