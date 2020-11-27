import os
import time

import hydra
import mlflow
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from roboro.learner import Learner


def test_agent(agent, env):
    """Test agent using normal gym style outside of learner"""
    obs = env.reset()
    done = False
    total_return = 0
    while not done:
        action = agent(obs)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        #env.render()
        total_return += reward
    env.close()
    return total_return


@hydra.main(config_name="main", config_path="configs")
def main(args: DictConfig):
    # Deal with args
    print("Args:")
    print(OmegaConf.to_yaml(args))
    # keep original working directory for mlflow etc
    os.chdir(hydra.utils.get_original_cwd())
    learner_args = args.learner
    trainer_args = args.trainer
    # Create agent and learner
    if args.path is not None:
        # load from checkpoint
        learner = Learner.load_from_checkpoint(args.path)
    else:
        # create from scratch
        learner = Learner(steps=args.env_steps, agent_args=args.agent, **learner_args)
        # Do the training!
        current_time = time.strftime('%d-%h_%H:%M:%S', time.gmtime())
        checkpoint_callback = ModelCheckpoint(
            monitor='val_return',
            dirpath=f'checkpoints/{current_time}',
            filename='{epoch:02d}-{val_return:.1f}',
            save_top_k=3,
            mode='max')
        # Set up mlflowlogger
        mlf_logger = MLFlowLogger(experiment_name="Default", tags={"mlflow.runName": args.override_args})

        # early_stop_callback = EarlyStopping(
        #         monitor='steps',
        #         min_delta=0.00,
        #         patience=3,
        #         verbose=False,
        # )

        # TODO: investigate why 16 precision is slower than 32

        # Apply seed if wanted
        deterministic = False
        if args.seed is not None:
            seed_everything(args.seed)
            deterministic = True
        # Calculate number of training batches based off maximal number of env steps
        frameskip = learner_args.frameskip if learner_args.frameskip > 0 else 1
        max_batches = args.env_steps / frameskip / learner_args.steps_per_batch
        print("Number of env steps to train on: ", args.env_steps)
        print("Number of batches to train on: ",max_batches)
        trainer = Trainer(max_steps=max_batches,
                          gpus=1 if torch.cuda.is_available() else 0,
                          callbacks=[checkpoint_callback],
                          logger=mlf_logger,
                          deterministic=deterministic,
                          **trainer_args,
                          )
        trainer.fit(learner)
        trainer.save_checkpoint("checkpoints/latest.ckpt")
    # Send explicitly to correct device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    learner.to(device)
    # Get train env:
    env = learner.train_env
    # Test agent using internal function:
    # TODO: re-enable rendering once it works on my machine
    total_return = learner.run(env, n_steps=0, n_eps=10, render=False)
    print("Avg return from internal run function: ", sum(total_return) / len(total_return))
    # Test the agent after training:
    total_return = test_agent(learner, env)
    print("Return of learner: ", total_return)
    # TODO: investigate embeddings of feature net of agent by applying UMAP and coloring by value!


if __name__ == "__main__":
    main()

