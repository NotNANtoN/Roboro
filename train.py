import os
import time

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from roboro.learner import Learner


def filter_out_env_override(ovargs):
    env_start_idx = ovargs.find("env=")
    env_end_idx = ovargs[env_start_idx:].find(",")
    if env_end_idx == -1:
        env_end_idx = len(ovargs)
    else:
        env_end_idx += env_start_idx
    # to remove starting or trailing comma
    if env_start_idx > 0:
        env_start_idx -= 1
    else:
        env_end_idx += 1
    ovargs = ovargs[:env_start_idx] + ovargs[env_end_idx:]
    return ovargs


def test_agent(agent, env, render=False):
    """Test agent using normal gym style outside of learner"""
    obs = env.reset()[0]  # gymnasium reset returns (obs, info)
    done = False
    total_return = 0
    while not done:
        action = agent(obs)
        # Move action to CPU before converting to numpy
        if torch.is_tensor(action):
            action = action.cpu()
        next_obs, reward, terminated, truncated, _ = env.step(action)
        obs = next_obs
        if render:
            env.render()
        total_return += reward
        done = terminated or truncated
    env.close()
    return total_return


@hydra.main(config_name="main", config_path="configs")
def main(conf: DictConfig):
    # Deal with args
    print("Args:")
    print(OmegaConf.to_yaml(conf))
    # keep original working directory for wandb etc
    os.chdir(hydra.utils.get_original_cwd())
    learner_args = conf.learner
    trainer_args = conf.trainer
    # Apply seed if wanted
    deterministic = False
    if conf.seed is not None:
        seed_everything(conf.seed)
        deterministic = True

    # Create agent and learner
    if conf.path is not None:
        # load from checkpoint
        learner = Learner.load_from_checkpoint(conf.path)
    else:
        # create from scratch
        render_mode = "human" if conf.render else None
        learner = Learner(
            steps=conf.env_steps,
            agent_conf=conf.agent,
            opt_conf=conf.opt,
            buffer_conf=conf.buffer,
            render_mode=render_mode,
            seed=conf.seed,
            **learner_args,
        )
        # Do the training!
        current_time = time.strftime("%d-%h_%H:%M:%S", time.gmtime())
        checkpoint_callback = ModelCheckpoint(
            monitor="val_ret",
            dirpath=f"checkpoints/{current_time}",
            filename="{epoch:02d}-{val_ret:.1f}",
            save_top_k=3,
            mode="max",
        )

        # Set up callbacks
        callbacks: list[Callback] = [checkpoint_callback]

        # Set up wandb logger
        exp_name = (
            conf.learner.train_env
            if conf.learner.train_env is not None
            else conf.learner.train_ds
        )
        ovargs = conf.override_args
        ovargs = filter_out_env_override(ovargs)
        print(exp_name, ovargs)
        wandb_logger = WandbLogger(project=exp_name, name=ovargs, save_dir="wandb_logs")

        # early_stop_callback = EarlyStopping(
        #         monitor='steps',
        #         min_delta=0.00,
        #         patience=3,
        #         verbose=False,
        # )

        # Calculate number of training batches based off maximal number of env steps
        frameskip = learner_args.frameskip if learner_args.frameskip > 0 else 1
        max_batches = conf.env_steps / frameskip / learner_args.steps_per_batch
        print("Number of env steps to train on: ", conf.env_steps)
        print("Number of batches to train on: ", max_batches)

        # Determine the best accelerator
        if torch.cuda.is_available():
            accelerator = "cuda"
            devices = 1
        elif torch.backends.mps.is_available():
            accelerator = "mps"
            devices = 1
        else:
            accelerator = "cpu"
            devices = "auto"

        trainer = Trainer(
            max_steps=max_batches,
            accelerator=accelerator,
            devices=devices,
            callbacks=callbacks,
            logger=wandb_logger,
            deterministic=deterministic,
            **trainer_args,
        )
        trainer.fit(learner, datamodule=learner.datamodule)
        trainer.save_checkpoint("checkpoints/latest.ckpt")
    # Send explicitly to correct device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    learner.to(device)
    # Get train env:
    env = learner.train_env
    # Test agent using internal function:
    total_return = learner.run(env, n_steps=0, n_eps=10, render=conf.render)
    print(
        "Avg return from internal run function: ", sum(total_return) / len(total_return)
    )
    # Test the agent after training:
    total_return = test_agent(learner, env, render=conf.render)
    print("Return of learner: ", total_return)


if __name__ == "__main__":
    main()
