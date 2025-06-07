import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from roboro.env_wrappers import create_env


class RLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        buffer,
        train_env=None,
        train_ds=None,
        val_env=None,
        val_ds=None,
        test_env=None,
        test_ds=None,
        batch_size=16,
        num_workers=0,
        discretize_actions: bool = False,
        num_bins_per_dim: int = 5,
        render_mode: str | None = None,
        seed: int | None = None,
        **env_kwargs,
    ):
        super().__init__()
        assert (
            train_env is not None or train_ds is not None
        ), "Can't fit agent without training data!"
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_env, self.train_dl = None, None
        if train_env is not None:
            self.train_env, self.train_obs = create_env(
                train_env,
                discretize_actions=discretize_actions,
                num_bins_per_dim=num_bins_per_dim,
                render_mode=render_mode,
                seed=seed,
                **env_kwargs,
            )
            print(self.train_env)
        if train_ds is not None:
            # self.dev_dataset = create_dl(train_ds)
            # self.train_data, self.val_data = random_split(dev_data, [55000, 5000])
            pass
        # init val loader
        self.val_env, self.val_dl = self.train_env, self.train_dl
        if val_env is not None:
            self.val_env, self.val_obs = create_env(
                val_env,
                discretize_actions=discretize_actions,
                num_bins_per_dim=num_bins_per_dim,
                render_mode=render_mode,
                seed=seed,
                **env_kwargs,
            )
        if self.val_env:
            self.val_obs = self.val_env.reset()
        if val_ds is not None:
            pass
            # self.val_dl = create_dl(val_ds)
        # init test_env
        self.test_env, self.test_dl = self.val_env, self.val_dl
        if test_env is not None:
            self.test_env, self.test_obs = create_env(
                test_env,
                discretize_actions=discretize_actions,
                num_bins_per_dim=num_bins_per_dim,
                render_mode=render_mode,
                seed=seed,
                **env_kwargs,
            )
        if self.test_env:
            self.test_obs = self.test_env.reset()
        else:
            self.test_obs = None

        # if val_ds is not None:
        #    self.val_dl = create_dl(val_ds)

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader(self.buffer)

    def val_dataloader(self) -> DataLoader:
        if self.val_dl is not None:
            return self.val_dl
        # If not using a specific validation dataset,
        # default to using the main buffer for PL's validation loop / sanity check.
        # Actual environment validation is in Learner's on_train_epoch_end.
        return self._dataloader(self.buffer)

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        if self.test_dl is not None:
            return self.test_dl
        # If not using a specific test dataset,
        # default to using the main buffer for PL's test loop / sanity check.
        # Actual environment testing is in Learner's test_epoch_end.
        return self._dataloader(self.buffer)

    def collate(self, batch):
        print(batch)
        print(len(batch))
        quit()

    def _dataloader(self, ds) -> DataLoader:
        return torch.utils.data.DataLoader(
            ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
        # , collate_fn=self.collate)

    def get_train_env(self):
        return self.train_env, self.train_obs

    def get_val_env(self):
        return self.val_env, self.val_obs

    def get_test_env(self):
        return self.test_env, self.test_obs
        # return DataLoader(self.test, batch_size=32)
