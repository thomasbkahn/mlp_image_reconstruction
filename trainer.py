from pathlib import Path
import logging

import torch
from torch.utils.data import DataLoader
import numpy as np

from dataset import MultiPhotoArrayDataset
from model import MultiHeadMLP


log = logging.getLogger(__name__)


class MLPReconstructionTrainer(object):

  def __init__(self, dataset_args, checkpoint_dir=None, hidden_units_heads=[256],
               hidden_units_core=[256, 256], batch_size=128, use_gpu=False):
    # new design seamlessly support multi head / multi image (e.g. for interpolation)
    # if there is only 1 path provided in dataset args, this becomes the older single
    # image style, in which case the old hidden_units arg is now hidden_units_heads + hidden_units_core
    # and it doesn't matter which values are in head or core. but note that the
    # head final layer is not automatically added
    # example - you want an mlp for a single image with hidden units (256, 256, 256, 256)
    # equivalent options are: hidden_unit_heads = [256, 256, 256], hidden_units_core = [256]
    # and hidden_unit_heads = [256, 256], hidden_units_core = [256, 256]
    self.dataset = MultiPhotoArrayDataset(**dataset_args)
    n_heads = len(self.dataset.paths)
    model_args = dict(in_units=self.dataset.tensors["features"].shape[1],
                      n_heads=n_heads, hidden_units_heads=hidden_units_heads,
                      hidden_units_core=hidden_units_core,
                      output_targets=self.dataset.tensors["pixels"].shape[1])
    self.model_args = model_args
    self.model = MultiHeadMLP(**model_args)
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
      source_image_paths = self.dataset.paths,
      source_image_sizes = self.dataset.image_sizes,
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
          weights = batch["weights"].to("cuda:0")
        else:
          features = batch["features"]
          labels = batch["pixels"]
          weights = batch["weights"]
        y_hat = self.model(features, weights)
        loss = self.loss_func(y_hat, labels)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        losses.append(loss.mean().item())
      mean_loss = np.array(losses).mean()
      log.info(f"Epoch {ind_epoch:02d} complete - loss: {mean_loss:.2f}")
      if self.checkpoint_dir is not None:
        self.save_checkpoint(ind_epoch, mean_loss)

