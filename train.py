import time
from argparse import ArgumentParser

import gym
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from roboro.agent import Agent
from roboro.learner import Learner
from roboro.data import RLDataModule


def test_agent(agent, env):
    state = env.reset()
    done = False
    total_return = 0
    while not done:
        action = learner(state)
        next_state, reward, _ = env.step(action)
        state = next_state
        env.render()
        total_return += reward
    return total_return


parser = ArgumentParser()
# Add PROGRAM level args
parser.add_argument('--conda_env', type=str, default='some_name')
parser.add_argument('--notification_email', type=str, default='will@email.com')
# Add model specific args
parser = Agent.add_model_specific_args(parser)
parser = Learner.add_model_specific_args(parser)
# Parse
args = parser.parse_args()
# Create env
env_str = args.env
env = gym.make(env_str)
env_data_module = RLDataModule(env)
# Create agent and learner
if args.path is not None:
    learner = Learner.load_from_checkpoint(args.path)
    agent = learner.agent
else:
    agent = Agent(env.observation_space, env.action_space)
    learner = Learner(agent, env)
    # Test agent before training:
    total_return = test_agent(agent, env)
    print("Return of learner: ", total_return)
    # Do the training!
    time = time.strftime('%d-%h_%H:%M:%S', time.gmtime())
    checkpoint_callback = ModelCheckpoint(
        monitor='val_return',
        dirpath=f'checkpoints/{time}',
        filename='{epoch:02d}-{val_return:.1f}',
        save_top_k=3,
        mode='max')
    trainer = Trainer.from_argparse_args(
            args,
            max_steps=10000,
            early_stopping_callback=[],
            gpus=0,
            callbacks=[checkpoint_callback]
    )
    trainer.fit(Learner)
    trainer.save_checkpoint("after_training.ckpt")
# Test the agent after training:
total_return = test_agent(learner, env)
print("Return of learner: ", total_return)
total_return = test_agent(agent, env)
print("Return of learner: ", total_return)

# Test agent using internal function:
total_return = learner.run(env, n_steps=0, n_eps=1) #epsilon = 1.0, store=False)

