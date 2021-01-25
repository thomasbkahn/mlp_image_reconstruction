import numpy as np
import cv2

BASIC_CONFIG = dict(
  b_vals = np.eye(2),
  a_vals = np.ones(2) 
)


def apply_encoding(X_in, config):
  if config is None:
    return X_in
  a = config["a_vals"]
  b = config["b_vals"]
  # X = X_in.reshape(-1, 2)
  return  np.concatenate([a * np.sin((2.*np.pi*X_in) @ b.T), a * np.cos((2.*np.pi*X_in) @ b.T)], axis=-1)


def build_coordinate_array(x_lims, y_lims, n_x, n_y, encoding_config=None):
  y = np.linspace(y_lims[0], y_lims[1], n_y)
  x = np.linspace(x_lims[0], x_lims[1], n_x)
  xx, yy = np.meshgrid(x, y)
  X = np.stack([xx, yy], axis=-1)
  return apply_encoding(X, encoding_config)


def rectify_pixel_values(values, mode="RGB", target_shape=None):
  if mode == "RGB":
    values = np.clip(values, 0, 255)
  elif mode == "HSV":
    values[:, 0] = np.clip(values[:, 0], 0, 180)
    values[:, 1] = np.clip(values[:, 1], 0, 255)
    values[:, 2] = np.clip(values[:, 2], 0, 255)
  else:
    raise NotImplementedError
  values = np.round(values).astype(np.uint8)
  if target_shape is not None:
    values = values.reshape(target_shape[0], target_shape[1], 3)
  if mode == "HSV":
    values = cv2.cvtColor(values, cv2.COLOR_HSV2RGB)
  return values



