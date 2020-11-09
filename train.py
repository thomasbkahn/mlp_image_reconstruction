import logging
import sys
from pathlib import Path

from trainer import MLPReconstructionTrainer
from trainer import log as trainer_log


def main(dataset_args, checkpoint_dir, hidden_units, num_epochs=200, use_gpu=False):
  trainer = MLPReconstructionTrainer(dataset_args, checkpoint_dir, hidden_units, use_gpu=use_gpu)
  trainer.train(num_epochs)


if __name__ == "__main__":
  trainer_log.setLevel(logging.INFO)
  format = '[%(asctime)s - %(name)s] %(message)s'
  logFormatter = logging.Formatter(format)
  consoleHandler = logging.StreamHandler(sys.stdout)
  consoleHandler.setFormatter(logFormatter)
  trainer_log.addHandler(consoleHandler)

  base_checkpoint_dir = Path("/shed/data/mlp_image_reconstruction_checkpoints/")
  # base_checkpoint_dir = base_checkpoint_dir.joinpath("mabel_ithaca_experiment_tiny_mlp")
  # base_checkpoint_dir = base_checkpoint_dir.joinpath("tom_cykana_pic")


  # hidden_units = [256] * 4
  # hidden_units = [16] * 4
  # hidden_units = [64] * 4
  # img_path = "/shed/photo/export/watkins_glen_camping_aug2020/instagram/0003_DSC07470.png"
  # img_path = "/home/tk/Downloads/IMG_20201102_004831_183.jpg"

  dataset_args_all = {
    # "tom_cykana_pic_256x5": {"path": "/home/tk/Downloads/IMG_20201102_004831_183.jpg", "scale_factor": 0.3, "mode":"RGB"},
    # "mabel_ithaca_64x10": {"path": "/shed/photo/export/watkins_glen_camping_aug2020/instagram/0003_DSC07470.png", "scale_factor":0.1, "mode":"RGB"},
    # "mabel_ithaca_256x5_scale02": {"path": "/shed/photo/export/watkins_glen_camping_aug2020/instagram/0003_DSC07470.png", "scale_factor":0.2, "mode":"RGB"},
    # "goya_saturn": {"path": "/home/tk/Downloads/goya_saturn.jpg", "scale_factor":0.2, "mode":"RGB"},
    "mabel_ithaca_16x30": {"path": "/shed/photo/export/watkins_glen_camping_aug2020/instagram/0003_DSC07470.png", "scale_factor":0.1, "mode":"RGB"},
    "mabel_ithaca_16x20": {"path": "/shed/photo/export/watkins_glen_camping_aug2020/instagram/0003_DSC07470.png", "scale_factor":0.1, "mode":"RGB"},
    "mabel_ithaca_32x20": {"path": "/shed/photo/export/watkins_glen_camping_aug2020/instagram/0003_DSC07470.png", "scale_factor":0.1, "mode":"RGB"},
    "mabel_ithaca_32x10": {"path": "/shed/photo/export/watkins_glen_camping_aug2020/instagram/0003_DSC07470.png", "scale_factor":0.1, "mode":"RGB"},
  }

  hidden_unit_dict = {
    # "tom_cykana_pic_256x5": [256] * 5,
    # "mabel_ithaca_64x10": [64] * 10,
    # "mabel_ithaca_256x5_scale02": [256] * 5,
    # "goya_saturn": [256] * 5
    "mabel_ithaca_16x30": [16] * 30,
    "mabel_ithaca_16x20": [16] * 20,
    "mabel_ithaca_32x20": [32] * 20,
    "mabel_ithaca_32x10": [32] * 10
  }


  for expt_name, hidden_units in hidden_unit_dict.items():
    checkpoint_dir = base_checkpoint_dir.joinpath(expt_name)
    main(dataset_args_all[expt_name], checkpoint_dir, hidden_units, num_epochs=300, use_gpu=True)

