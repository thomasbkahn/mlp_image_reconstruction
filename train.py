import logging
import sys
from pathlib import Path

from trainer import MLPReconstructionTrainer
from trainer import log as trainer_log


def main(dataset_args, checkpoint_dir, hidden_units, num_epochs=200):
  trainer = MLPReconstructionTrainer(dataset_args, checkpoint_dir, hidden_units)
  trainer.train(num_epochs)


if __name__ == "__main__":
  trainer_log.setLevel(logging.INFO)
  format = '[%(asctime)s - %(name)s] %(message)s'
  logFormatter = logging.Formatter(format)
  consoleHandler = logging.StreamHandler(sys.stdout)
  consoleHandler.setFormatter(logFormatter)
  trainer_log.addHandler(consoleHandler)

  base_checkpoint_dir = Path("/home/tk/data/mlp_image_reconstruction_checkpoints/")
  # base_checkpoint_dir = base_checkpoint_dir.joinpath("mabel_ithaca_experiment_tiny_mlp")
  base_checkpoint_dir = base_checkpoint_dir.joinpath("mabel_ithaca_experiment_64_mlp")


  # hidden_units = [256] * 4
  # hidden_units = [16] * 4
  hidden_units = [64] * 4
  img_path = "/shed/photo/export/watkins_glen_camping_aug2020/instagram/0003_DSC07470.png"

  dataset_args_common = dict(
    path = img_path,
    scale_factor = 0.1
  )
  modes = ["HSV", "RGB"]



  for mode in modes:
    dataset_args = dataset_args_common.copy()
    dataset_args["mode"] = mode
    checkpoint_dir = base_checkpoint_dir.joinpath(f"mode_{mode}")
    main(dataset_args, checkpoint_dir, hidden_units, num_epochs=200)

