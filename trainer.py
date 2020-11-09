from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import PhotoArrayDataset
from model import MLP


log = logging.getLogger(__name__)


class MLPReconstructionTrainer(object):

  def __init__(self, dataset_args, checkpoint_dir=None, hidden_units=[256, 256], batch_size=128,
               use_gpu=False):
    self.dataset = PhotoArrayDataset(**dataset_args)
    model_args = dict(in_units=self.dataset.tensors["features"].shape[1],
                      hidden_units=hidden_units)
    self.model_args = model_args
    self.model = MLP(**model_args)
    self.use_gpu = use_gpu
    self.checkpoint_dir = checkpoint_dir
    self.batch_size = batch_size
    self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                 num_workers=8)
    if self.use_gpu:
      self.model = self.model.cuda()
    self.opt = torch.optim.Adam(self.model.parameters())
    self.loss_func = torch.nn.MSELoss()
    if self.checkpoint_dir is not None:
      self.checkpoint_dir = Path(self.checkpoint_dir)
      self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

  def save_checkpoint(self, ind, epoch_loss):
    basepath = f"checkpoint_{ind:05d}.pt"
    checkpoint_path = self.checkpoint_dir.joinpath(basepath)
    checkpoint_data = dict(
      output_mode = self.dataset.mode,
      source_image_path = self.dataset.path,
      source_image_size = (self.dataset.h, self.dataset.w),
      feature_encoding_config = self.dataset.encoding_config,
      model_args = self.model_args,
      model_state = self.model.state_dict(),
      opt_state = self.opt.state_dict(),
      loss = epoch_loss
    )
    torch.save(checkpoint_data, checkpoint_path)
    log.info(f"Checkpoint saved at {checkpoint_path}")

  def train(self, num_epochs):
    for ind_epoch in range(num_epochs):
      losses = []
      for batch in self.dataloader:
        if self.use_gpu:
          features = batch["features"].to("cuda:0")
          labels = batch["pixels"].to("cuda:0")
        else:
          features = batch["features"]
          labels = batch["pixels"]
        y_hat = self.model(features)
        loss = self.loss_func(y_hat, labels)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        losses.append(loss.mean().item())
      mean_loss = np.array(losses).mean()
      log.info(f"Epoch {ind_epoch:02d} complete - loss: {mean_loss:.2f}")
      if self.checkpoint_dir is not None:
        self.save_checkpoint(ind_epoch, mean_loss)

