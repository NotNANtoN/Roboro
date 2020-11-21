import time
from argparse import ArgumentParser

import gym
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from roboro.agent import Agent
from roboro.learner import Learner


def test_agent(agent, env):
    state = env.reset()
    done = False
    total_return = 0
    while not done:
        action = agent(state)
        next_state, reward, _ = env.step(action)
        state = next_state
        env.render()
        total_return += reward
    return total_return


parser = ArgumentParser()
# Add PROGRAM level args
parser.add_argument('--train_env', type=str, default='CartPole-v0')
parser.add_argument('--path', type=str)
# Add model specific args
parser = Agent.add_model_specific_args(parser)
#parser = Learner.add_model_specific_args(parser)
# Parse
args = parser.parse_args()
# Get env str
env_str = args.train_env
# Create agent and learner
if args.path is not None:
    # load from checkpoint
    learner = Learner.load_from_checkpoint(args.path)
    agent = learner.agent
else:
    # create from scratch
    learner = Learner(train_env=env_str)
    agent = learner.agent
    # Do the training!
    time = time.strftime('%d-%h_%H:%M:%S', time.gmtime())
    checkpoint_callback = ModelCheckpoint(
        monitor='val_return',
        dirpath=f'checkpoints/{time}',
        filename='{epoch:02d}-{val_return:.1f}',
        save_top_k=3,
        mode='max')
    args.max_steps = 1000
    args.early_stopping_callback = []
    args.gpus = 0
    args.callbacks = [checkpoint_callback]
    trainer = Trainer.from_argparse_args(
            args
    )
    trainer.fit(learner)
    trainer.save_checkpoint("after_training.ckpt")
# Get train env:
env = learner.train_env
# Test the agent after training:
total_return = test_agent(learner, env)
print("Return of learner: ", total_return)
#total_return = test_agent(agent, env)
#print("Return of learner: ", total_return)

# Test agent using internal function:
total_return = learner.run(env, n_steps=0, n_eps=1) #epsilon = 1.0, store=False)

