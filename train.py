import time
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from roboro.agent import Agent
from roboro.learner import Learner


def test_agent(agent, env):
    state = env.reset()
    done = False
    total_return = 0
    while not done:
        action = agent(state)
        print("Action: ", action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        env.render()
        total_return += reward
    env.close()
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
    mlf_logger = MLFlowLogger(
            experiment_name="default",
            tracking_uri="file:./ml-runs"
    )
    args.max_steps = 200
    args.early_stopping_callback = []
    args.gpus = 0
    args.callbacks = [checkpoint_callback]
    args.logger = mlf_logger
    trainer = Trainer.from_argparse_args(
            args
    )
    trainer.fit(learner)
    trainer.save_checkpoint("checkpoints/after_training.ckpt")
# Get train env:
env = learner.train_env
# Test the agent after training:
total_return = test_agent(learner, env)
print("Return of learner: ", total_return)

# Test agent using internal function:
total_return = learner.run(env, n_steps=0, n_eps=1, render=True) #epsilon = 1.0, store=False)
print("Return from internal function: ", sum(total_return))

