import logging
import sys
from pathlib import Path

from trainer import MLPReconstructionTrainer
from trainer import log as trainer_log


def main(dataset_args, checkpoint_dir, hidden_units_heads, hidden_units_core,
         num_epochs=200, use_gpu=False):
  trainer = MLPReconstructionTrainer(dataset_args, checkpoint_dir, hidden_units_heads,
                                     hidden_units_core, use_gpu=use_gpu)
  trainer.train(num_epochs)


if __name__ == "__main__":
  trainer_log.setLevel(logging.INFO)
  format = '[%(asctime)s - %(name)s] %(message)s'
  logFormatter = logging.Formatter(format)
  consoleHandler = logging.StreamHandler(sys.stdout)
  consoleHandler.setFormatter(logFormatter)
  trainer_log.addHandler(consoleHandler)

  base_checkpoint_dir = Path("/shed/data/mlp_image_reconstruction_checkpoints/")

  dataset_args_all = {
    "ac_beach_the_family_duplex": {
      "paths": [
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0012_DSC09130.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0013_DSC09148.png'
      ],
      "mode": "RGB",
      "scale_factor": 0.1,
      "force_common_size": True
    },
    "ac_beach_the_family_single_test": {
      "paths": [
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0012_DSC09130.png'
      ],
      "mode": "RGB",
      "scale_factor": 0.1,
      "force_common_size": True
    },
  }

  dataset_args_all["ac_beach_the_family_duplex_bigger_head"] = dataset_args_all["ac_beach_the_family_duplex"].copy()

  hidden_unit_dict = {
    "ac_beach_the_family_duplex": ([256], [256] * 4),
    "ac_beach_the_family_duplex_bigger_head": ([256] * 2, [256] * 3),
    "ac_beach_the_family_single_test": ([256], [256] * 4)
  }



  for expt_name, hidden_units in hidden_unit_dict.items():
    checkpoint_dir = base_checkpoint_dir.joinpath(expt_name)
    hidden_units_heads, hidden_units_core = hidden_units
    main(dataset_args_all[expt_name], checkpoint_dir, hidden_units_heads, hidden_units_core,
         num_epochs=300, use_gpu=True)

