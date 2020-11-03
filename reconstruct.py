from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2

from model import MLP
from dataset import CoordinateArrayDataset
from utils import rectify_pixel_values

def evaluate_model(checkpoint, n_x, n_y, x_lims, y_lims, batch_size=256):
  dataset = CoordinateArrayDataset(n_x, n_y, x_lims, y_lims,
                                   encoding_config=checkpoint["feature_encoding_config"])
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
  model = MLP(**checkpoint["model_args"])
  model.load_state_dict(checkpoint["model_state"])
  out_shape = (n_y, n_x)
  collect = []
  for batch in dataloader:
    with torch.no_grad():
        collect.append(model(batch["features"]).detach().numpy())

  collect = np.concatenate(collect, axis=0)
  output = rectify_pixel_values(collect, mode=checkpoint["output_mode"], target_shape=out_shape)
  return output


def reconstruct_single(checkpoint, hallucination_buffer=None, scale_factor=None):
  if isinstance(checkpoint, Path) or isinstance(checkpoint, str):
    checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
  if hallucination_buffer is None:
    hallucination_buffer = 0
  n_y, n_x = checkpoint["source_image_size"]
  if scale_factor is not None:
      n_y = int(round(n_y * scale_factor))
      n_x = int(round(n_x * scale_factor))
  # get the amount of original image range per pixel
  fac_y = 1.0 / n_y
  fac_x = 1.0 / n_x
  # extend in all directions by hallucination_buffer
  y_lims = (0.0 - fac_y * hallucination_buffer, 1.0 + fac_y * hallucination_buffer)
  x_lims = (0.0 - fac_x * hallucination_buffer, 1.0 + fac_x * hallucination_buffer)
  n_y += 2 * hallucination_buffer
  n_x += 2 * hallucination_buffer
  return evaluate_model(checkpoint, n_x, n_y, x_lims, y_lims)


def reconstruct_directory(checkpoint_dir, output_dir, hallucination_buffer=None,
                          glob_pattern="./*.pt"):
  checkpoint_dir = Path(checkpoint_dir)
  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  checkpoint_paths = sorted(checkpoint_dir.glob(glob_pattern))
  for checkpoint_path in checkpoint_paths:
    img = reconstruct_single(checkpoint_path, hallucination_buffer)
    output_path = output_dir.joinpath(f"{checkpoint_path.stem}.png")
    cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
  checkpoint_dirs = [
    # "/home/tk/data/mlp_image_reconstruction_checkpoints/mabel_ithaca_experiment_big_run/mode_HSV",
    # "/home/tk/data/mlp_image_reconstruction_checkpoints/mabel_ithaca_experiment_big_run/mode_RGB",
    # "/shed/data/mlp_image_reconstruction_checkpoints/tom_cykana_pic/mode_RGB",
    # "/shed/data/mlp_image_reconstruction_checkpoints/tom_cykana_pic/mode_HSV",
    "/shed/data/mlp_image_reconstruction_checkpoints/tom_cykana_pic_256x5",
  ]
  for checkpoint_dir in checkpoint_dirs:
    checkpoint_dir = Path(checkpoint_dir)
    output_dir_parts = []
    for part in checkpoint_dir.parts:
      output_dir_parts.append(part.replace("reconstruction_checkpoints", "reconstruction_output"))

    output_dir = Path().joinpath(*output_dir_parts)

    reconstruct_directory(checkpoint_dir, output_dir, hallucination_buffer=100)
