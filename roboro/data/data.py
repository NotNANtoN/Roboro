import pytorch_lightning as pl
import torch
from torch.utils.data import random_split, DataLoader

from roboro.env_wrappers import create_env


class RLDataModule(pl.LightningDataModule):
    def __init__(self, buffer, train_env=None, train_ds=None, val_env=None, val_ds=None, test_env=None, test_ds=None,
                 batch_size=16, num_workers=0,
                 **env_kwargs):
        super().__init__()
        assert train_env is not None or train_ds is not None, "Can't fit agent without training data!"
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_env, self.train_dl = None, None
        if train_env is not None:
            self.train_env, self.train_obs = create_env(train_env, **env_kwargs)
        if train_ds is not None:
            #self.dev_dataset = create_dl(train_ds)
            #self.train_data, self.val_data = random_split(dev_data, [55000, 5000])
            # TODO: make train/val/test split and use it
            # TODO: somehow extract obs wrapper from env and use it in dataloader (what if there is no env?)
            pass
        # init val loader
        self.val_env, self.val_dl = self.train_env, self.train_dl
        if val_env is not None:
            self.val_env = create_env(val_env, **env_kwargs)
        self.val_obs = self.val_env.reset()
        if val_ds is not None:
            pass
            #self.val_dl = create_dl(val_ds)
        # init test_env
        self.test_env, self.test_dl = self.val_env, self.val_dl
        if test_env is not None:
            self.test_env = create_env(test_env, **env_kwargs)
        self.test_obs = self.test_env.reset()

        #if val_ds is not None:
        #    self.val_dl = create_dl(val_ds)

    def collate(self, batch):
        print(batch)
        print(len(batch))
        quit()

    def _dataloader(self, ds) -> DataLoader:
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers)
        #, collate_fn=self.collate)

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        # TODO: combine replay buffer dataloader with expert data dataloader
        return self._dataloader(self.buffer)

    def val_dataloader(self) -> DataLoader:
        return self.val_dl
        #return self.train_dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self.test_dl

    def get_train_env(self):
        return self.train_env, self.train_obs

    def get_val_env(self):
        return self.val_env, self.val_obs

    def get_test_env(self):
        return self.test_env, self.test_obs
        #return DataLoader(self.test, batch_size=32)
