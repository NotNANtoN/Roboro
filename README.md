# Roboro

**Roboro**: to strengthen, reinforce

## Welcome!
Please install the `requirements.txt` first

To run cartpole with predefined settings, enter: ```python3 train.py env=cart```

For Pong: ```python3 train.py env=pong```

For any other gym-registered env e.g.: ```python3 train.py train_env=PongNoFrameskip-v4```

Check out `configs/main.py` for adjustable hyperparameters. E.g. you can force the use of frameskipping and change the learning rate by calling: ```python3 train.py train_env=PongNoFrameskip-v4 learner.learning_rate=0.001 learner.frameskip=2```
