# LearningToLearn
This repository contains code for 
* ML3: Meta-Learning via Learned Losses ([pdf](https://arxiv.org/pdf/1906.05374.pdf))
* MBIRL: Model-Based Inverse Reinforcement Learning from Visual Demonstrations ([pdf](https://arxiv.org/pdf/2010.09034.pdf))

## Setup
In the LearningToLearn folder, run:

```
conda create -n l2l python=3.7
conda activate l2l 
python setup.py develop
```

## ML3 paper experiments and citation
To reproduce results of the ML3 paper follow the README instructions in the `ml3` folder

#### Citation
```
@inproceedings{ml3,
author    = {Sarah Bechtle and Artem Molchanov and Yevgen Chebotar and Edward Grefenstette and Ludovic Righetti and Gaurav Sukhatme and Franziska Meier},
title     = {Meta Learning via Learned Loss},
booktitle = {International Conference on Pattern Recognition, {ICPR}, Italy, January 10-15, 2021},
year      = {2021} }
```

## MBIRL paper experiments and citation
To test our MBIRL algorithm follow the README instructions in the `mbirl` folder

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
