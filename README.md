# Roboro

**Roboro**: to strengthen, reinforce

The aim of this library is to implement modular deep reinforcement learning algorithms (RL). It is based on essential libraries such as pytorch-lightning for training organization, hydra for command line input and configuartions, and mlflow for experiment logging.

The modularity of this library is supposed to provide a taxonomy over algorithms and their extensions. Furthermore, orthogonal improvements to e.g. DQN can be combined on the fly, while keeping their implementations enclosed in classes.

Another focus of this library is offline RL (also called batch RL). Not only environments can be used to train agents, but also expert-data datasets (or both in combination). Evaluation of agents can be conducted on datasets and within environments.

## Welcome!
Please install the `requirements.txt` first

To run cartpole with predefined settings, enter: ```python3 train.py env=cart```

For Pong: ```python3 train.py env=pong```

For any other gym-registered env e.g.: ```python3 train.py learner.train_env=PongNoFrameskip-v4```

Check out `configs/main.py` for adjustable hyperparameters. E.g. you can force the use of frameskipping and change the learning rate by calling: ```python3 train.py learner.train_env=PongNoFrameskip-v4 opt.lr=0.001 learner.frameskip=2```
