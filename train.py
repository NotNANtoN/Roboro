import time
from argparse import ArgumentParser

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger

from roboro.agent import Agent
from roboro.learner import Learner


def test_agent(agent, env):
    state = env.reset()
    done = False
    total_return = 0
    while not done:
        action = agent(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        #env.render()
        total_return += reward
    env.close()
    return total_return


parser = ArgumentParser()
# Add PROGRAM level args
parser.add_argument('--train_env', type=str, default='CartPole-v0')
parser.add_argument('--path', type=str)
parser.add_argument('--steps', type=int, default=10000, help="max env steps")
parser.add_argument('--frame_stack', type=int, default=0, help="How many frames to sack")
parser.add_argument('--frameskip', type=int, default=2, help="frameskip")
parser.add_argument('--steps_per_batch', type=float, default=1, help="how many env steps are taken per training batch")

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
    learner = Learner(max_steps=args.steps,
                      train_env=env_str,
                      frameskip=args.frameskip,
                      steps_per_batch=args.steps_per_batch,
                      frame_stack=args.frame_stack,
                      )
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
            #tracking_uri="file:./ml-runs"
    )
    # early_stop_callback = EarlyStopping(
    #         monitor='steps',
    #         min_delta=0.00,
    #         patience=3,
    #         verbose=False,
    # )
    frameskip = args.frameskip if args.frameskip > 0 else 1
    max_batches = args.steps / frameskip / args.steps_per_batch
    args.max_steps = max_batches
    print("Number of env steps to train on: ", args.steps)
    print("Number of batches to train on: ", args.max_steps)
    args.early_stopping_callback = []
    args.gpus = 1 if torch.cuda.is_available() else 0
    # TODO: investigate why 16 precision is slower than 32
    args.precision = 32 # 16 if args.gpus else 32
    args.callbacks = [checkpoint_callback]
    args.logger = mlf_logger
    args.weight_summary = "full"
    args.terminate_on_nan = True
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(learner)
    trainer.save_checkpoint("checkpoints/after_training.ckpt")
# Get train env:
env = learner.train_env
# Test the agent after training:
total_return = test_agent(learner, env)
print("Return of learner: ", total_return)

# Test agent using internal function:
total_return = learner.run(env, n_steps=0, n_eps=1, render=False) #epsilon = 1.0, store=False)
print("Return from internal function: ", sum(total_return))

