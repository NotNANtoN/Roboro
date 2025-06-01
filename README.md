# Roboro

**Roboro**: to strengthen, reinforce

The aim of this library is to implement modular deep reinforcement learning algorithms (RL). It is based on essential libraries such as [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training organization, [hydra](https://hydra.cc/) for command line input and configuartions, and [mlflow](https://github.com/mlflow/mlflow) for experiment logging.

The modularity of this library is supposed to provide a taxonomy over algorithms and their extensions. Furthermore, orthogonal improvements to e.g. DQN can be combined on the fly, while keeping their implementations enclosed in classes.

Another focus of this library is offline RL (also called batch RL). Not only environments can be used to train agents, but also expert-data datasets (or both in combination). Evaluation of agents can be conducted on datasets and within environments.

## Welcome!
Please install the `requirements.txt` first

To run cartpole with predefined settings, enter: ```python3 train.py env=cart```

For Pong: ```python3 train.py env=pong```

For any other gym-registered env e.g.: ```python3 train.py learner.train_env=PongNoFrameskip-v4```

Check out `configs/main.py` for adjustable hyperparameters. E.g. you can force the use of frameskipping and change the learning rate by calling: ```python3 train.py learner.train_env=PongNoFrameskip-v4 opt.lr=0.001 learner.frameskip=2```.

You can combine algorithms as you want. E.g. you can combine IQN with PER, CER and M-Rl to train on Pong like this:
```python3 train.py env=pong agent.iqn=1 agent.munch_q=1 learner.per=1 learner.cer=1```.

## Supported algorithms

- [x] [Uniform Experience Replay](http://www.incompleteideas.net/lin-92.pdf) and [Prioritied Experience Replay](https://arxiv.org/abs/1511.05952). Defaults to uniform exp replay.
- [x] Corrected Experience Replay, [CER](https://arxiv.org/abs/1712.01275). Can be combined either with uniform or prioritized experience replay.
- [x] Double Q-learning [DDQN](https://arxiv.org/abs/1509.06461). To avoid the Q-learning maximization bias, the online network is used in the action-selection of the Bellman update, whereas the target network is used for the evaluation of this selected action.
- [x] Use of a target net that is updated every N steps or of a Polyak-averaged target network, as seen in [DDPG](https://arxiv.org/abs/1509.02971). Defaults to Polyak-averaging.
- [x] [QV](https://www.researchgate.net/publication/224446250_The_QV_family_compared_to_other_reinforcement_learning_algorithms) and [QVMax](https://arxiv.org/abs/1909.01779v1) learning. Next to a Q-network, a V-network (state-value network) is trained. In QV-learning, the Q-network is trained using the target of the V-network. In QVMAX, additionally, the V-network ist trained using the target of the Q-network (making this an off-policy algorithm).
- [x] Observation Z-standardization. The mean and std are collected during an initial rollout of random actions. Turned on by default.
- [x] Random Ensemble Mixture, [REM](https://arxiv.org/abs/1907.04543). During the value net optimization, a mixture of a randomly sampled categorical distribution of N value networks is used.
- [x] Implicit Quantile Networks [IQN](https://arxiv.org/abs/1806.06923). The value network is trained to predict N quantiles of the return.
- [x] Munchausen RL [M-RL](https://arxiv.org/abs/2007.14430). A form of maximum-entropy RL that focuses on optimizing for the optimal policy, next to the optimal value function.


## Coming soon

- [ ] Training on offline data
- [ ] Evaluating agents using offline data
- [ ] [Efficient Eligibility traces](https://arxiv.org/abs/1810.09967) - as described in v1 of the arXiv paper.
- [ ] [MuZero](https://arxiv.org/abs/1911.08265)

See the [Trello board](https://trello.com/b/8Jaoco2O/roboro) for details.




# TODOs

- [ ] Debug HER buffer
- [ ] Add monitoring of what the agent is doing. E.g. track what is the gradient towards inputs, are they actually used?
- [ ] **Continuous Control Progression (toward Hugging Face S100 robot arm):**
  - [ ] Test on FetchReach-v3 (7-DOF arm reaching - foundational manipulation)
  - [ ] Test on FetchPush-v3 (object interaction - pushing tasks)
  - [ ] Test on FetchSlide-v3 (dynamic object control - sliding on low friction)
  - [ ] Test on FetchPickAndPlace-v3 (full manipulation pipeline)
  - [ ] Implement multi-goal API support for goal-conditioned RL
  - [ ] Test algorithm combinations (IQN+PER+CER+M-RL) on manipulation tasks
  - [ ] Final target: Hugging Face S100/SO-101 robot arm integration
- [ ] Add actor-critic algorithms (DDPG, SPG).
