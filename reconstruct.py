from pathlib import Path
import multiprocessing
from functools import partial

import torch
import numpy as np
from torch.utils.data import DataLoader
import cv2

from model import MultiHeadMLP
from dataset import CoordinateArrayDataset
from utils import rectify_pixel_values

def evaluate_model(checkpoint, n_x, n_y, x_lims, y_lims, batch_size=256, image_weights=None,
                   model=None):
  dataset = CoordinateArrayDataset(n_x, n_y, x_lims, y_lims,
                                   encoding_config=checkpoint["feature_encoding_config"],
                                   head_weights=image_weights)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
  if model is None:
      model = MultiHeadMLP(**checkpoint["model_args"])
      model.load_state_dict(checkpoint["model_state"])
  out_shape = (n_y, n_x)
  collect = []
  for batch in dataloader:
    with torch.no_grad():
        collect.append(model(batch["features"], batch["weights"]).detach().numpy())

  collect = np.concatenate(collect, axis=0)
  output = rectify_pixel_values(collect, mode=checkpoint["output_mode"], target_shape=out_shape)
  return output


def reconstruct_single(checkpoint, hallucination_buffer=None, scale_factor=None,
                       image_weights=None, size_index=0):
  # TODO make GPU flag, if present don't map location and force gpu usage (see corruption notebook)
  if isinstance(checkpoint, Path) or isinstance(checkpoint, str):
    checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
  if hallucination_buffer is None:
    hallucination_buffer = 0
  n_y, n_x = checkpoint["source_image_sizes"][size_index]
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
  return evaluate_model(checkpoint, n_x, n_y, x_lims, y_lims, image_weights=image_weights)


def reconstruct_with_image_interpolation(checkpoint, output_dir, interpolation_steps=20,
                                         hallucination_buffer=None, size_index=0, scale_factor=None, cycle=False):
  if isinstance(checkpoint, Path) or isinstance(checkpoint, str):
    checkpoint = torch.load(checkpoint, map_location=torch.device("cpu"))
  if hallucination_buffer is None:
    hallucination_buffer = 0
  n_y, n_x = checkpoint["source_image_sizes"][size_index]
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
  model = MultiHeadMLP(**checkpoint["model_args"])
  model.load_state_dict(checkpoint["model_state"])

  n_paths = len(checkpoint["source_image_paths"])
  anchors = []
  for ind in range(n_paths):
    anchor = np.zeros((1, n_paths))
    anchor[:, ind] = 1.0
    anchors.append(anchor)
  if cycle:
    anchors.append(anchors[0].copy())
  step_factor = np.linspace(0.0, 1.0, interpolation_steps)

  output_dir = Path(output_dir)
  output_dir.mkdir(parents=True, exist_ok=True)
  for stage_ind, (anchor_a, anchor_b) in enumerate(zip(anchors, anchors[1:])):
    for step_ind, fac in enumerate(step_factor):
      image_weights = anchor_a * (1 - fac) + anchor_b * fac

      result = evaluate_model(checkpoint, n_x, n_y, x_lims, y_lims, image_weights=image_weights, model=model)
      output_path = output_dir.joinpath(f"stage_{stage_ind:03d}_step_{step_ind:05d}.png")
      cv2.imwrite(str(output_path), cv2.cvtColor(result, cv2.COLOR_RGB2BGR))



    

# make reconstruct image interpolation, load model once, compute n_x n_y etc once, step over weights write to file
# use multiprocessing


def _reconstruct_all_worker(paths, hallucination_buffer=None, scale_factor=None):
  checkpoint_path, output_path = paths
  img = reconstruct_single(checkpoint_path, hallucination_buffer, scale_factor)
  output_path.parent.mkdir(parents=True, exist_ok=True)
  print(checkpoint_path, output_path)
  cv2.imwrite(str(output_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

# deprecated, use multiprocessing form and reconstuct_all
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


def reconstruct_all(checkpoint_toplevel, replace_source="reconstruction_checkpoints",
                    replace_target="reconstruction_output", hallucination_buffer=None,
                    glob_pattern="**/*.pt", n_procs=6):
  checkpoint_toplevel = Path(checkpoint_toplevel)
  checkpoint_paths = list(checkpoint_toplevel.glob(glob_pattern))
  paths = []
  for checkpoint_path in checkpoint_paths:
    output_path_parts = [part.replace(replace_source, replace_target) for part in
                         checkpoint_path.parts]
    output_path = Path().joinpath(*output_path_parts).with_suffix(".png")
    if output_path.exists():
      continue
    paths.append((checkpoint_path, output_path))
  map_func = partial(_reconstruct_all_worker, hallucination_buffer=hallucination_buffer)
  pool = multiprocessing.Pool(n_procs)
  pool.map(map_func, paths)




if __name__ == "__main__":
  # checkpoint_dirs = [
    # "/home/tk/data/mlp_image_reconstruction_checkpoints/mabel_ithaca_experiment_big_run/mode_HSV",
    # "/home/tk/data/mlp_image_reconstruction_checkpoints/mabel_ithaca_experiment_big_run/mode_RGB",
    # "/shed/data/mlp_image_reconstruction_checkpoints/tom_cykana_pic/mode_RGB",
    # "/shed/data/mlp_image_reconstruction_checkpoints/tom_cykana_pic/mode_HSV",
    # "/shed/data/mlp_image_reconstruction_checkpoints/tom_cykana_pic_256x5",
  # ]
  # for checkpoint_dir in checkpoint_dirs:
    # checkpoint_dir = Path(checkpoint_dir)
    # output_dir_parts = []
    # for part in checkpoint_dir.parts:
    #   output_dir_parts.append(part.replace("reconstruction_checkpoints", "reconstruction_output"))

    # output_dir = Path().joinpath(*output_dir_parts)

    # reconstruct_directory(checkpoint_dir, output_dir, hallucination_buffer=100)
  reconstruct_all("/shed/data/mlp_image_reconstruction_checkpoints", hallucination_buffer=100)
