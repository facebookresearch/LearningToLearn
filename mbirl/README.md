## MBIRL - Model Based Inverse Reinforcement Learning

### Simulation with ground truth keypoint predictions

#### Generate expert demonstrations
1. ```python mbirl/generate_expert_demo.py``` 
2. Check the data and visualizations of the demonstration in 'mbirl/traj_data'

#### Run Our Method
1. ```python mbirl/experiments/run_model_based_irl.py```
2. Check the trajectories predicted during training in model_data/placing/<cost_type>

#### Plot the losses, evaluate our method
1. ```jupyter notebook```
2. Access the notebook in the browser in 'mbirl/experiments/plot_mbirl_training_and_eval.ipynb'

### Simulation with learned keypoint representation and dynamics
COMING SOON
