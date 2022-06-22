import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dataloader.pose_dataset import DatasetWrapper, random_pose_obj_dp, rendered_pose_bop_obj_dp
from dataloader.sample import Sample
from utils.config import Config


class LitDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg: Config = cfg
        self.batch_size: int = self.cfg.dataloader.batch_size
        self.train_epoch_len: int = self.cfg.dataloader.train_epoch_len
        self.val_epoch_len: int = self.cfg.dataloader.val_epoch_len

        self.dataset: torch.utils.data.IterableDataset = random_pose_obj_dp(
            dtype=self.cfg.dtype, device=self.cfg.device,
            crop_out_size=self.cfg.model.img_input_size,
            # img_input_size=self.cfg.model.img_input_size,
            # pnp_input_size=self.cfg.model.pnp_input_size,
            **self.cfg.dataset
        )

    def setup(self, stage: str = None):
        pass

    def train_dataloader(self):
        return DataLoader(DatasetWrapper(self.dataset, self.train_epoch_len),
            batch_size=self.batch_size, drop_last=True, collate_fn=Sample.collate, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(DatasetWrapper(self.dataset, self.val_epoch_len),
            batch_size=self.batch_size, collate_fn=Sample.collate, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()

    def predict_dataloader(self):
        return self.val_dataloader()

    def teardown(self, stage: str = None):
        pass
