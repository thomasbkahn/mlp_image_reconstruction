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
    "ac_beach_set_of_eight": {
      "paths": [
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0008_DSC08949.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0009_DSC08976.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0010_DSC09041.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0011_DSC09073.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0012_DSC09130.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0013_DSC09148.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0014_DSC09477.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0015_DSC09503.png'
      ],
      "mode": "RGB",
      "scale_factor": 0.1,
      "force_common_size": True
    },
    "ac_beach_set_of_three": {
      "paths": [
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0008_DSC08949.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0010_DSC09041.png',
        '/shed/photo/export/atlantic_city_thanksgiving_nov2020/album/0011_DSC09073.png',
      ],
      "mode": "RGB",
      "scale_factor": 0.1,
      "force_common_size": True
    },
  }

  copy_keys = [
      ("ac_beach_set_of_eight", "ac_beach_set_of_eight_bigger_head"),
      ("ac_beach_set_of_eight", "ac_beach_set_of_eight_bigger_model"),
  ]

  for src_key, dest_key in copy_keys:
    dataset_args_all[dest_key] = dataset_args_all[src_key].copy()

  hidden_unit_dict = {
    "ac_beach_set_of_eight": ([256] * 2, [256] * 3),
    "ac_beach_set_of_eight_bigger_head": ([256] * 3, [256] * 2),
    "ac_beach_set_of_eight_bigger_model": ([256] * 3, [256] * 4),
    "ac_beach_set_of_three": ([256] * 2, [256] * 3),
    
  }


  for expt_name, hidden_units in hidden_unit_dict.items():
    checkpoint_dir = base_checkpoint_dir.joinpath(expt_name)
    hidden_units_heads, hidden_units_core = hidden_units
    main(dataset_args_all[expt_name], checkpoint_dir, hidden_units_heads, hidden_units_core,
         num_epochs=300, use_gpu=True)

