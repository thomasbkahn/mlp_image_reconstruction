import torch
from torch.utils.data import Dataset
import cv2
import numpy as np


from utils import build_coordinate_array


class CoordinateArrayDataset(Dataset):

  def __init__(self, n_x, n_y, x_lims=(0,1), y_lims=(0,1), encoding_config=None):
    self.encoding_config = encoding_config
    self.n_x = n_x
    self.n_y = n_y
    self.x_lims = x_lims
    self.y_lims = y_lims
    X = build_coordinate_array(x_lims, y_lims, n_x, n_y, encoding_config)
    self.X = torch.tensor(X.reshape(n_x * n_y, -1), dtype=torch.float32)

  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, ind):
    return {"features": self.X[ind]}



class PhotoArrayDataset(Dataset):

  def __init__(self, path, mode="RGB", encoding_config=None, scale_factor=0.1):
    self.mode = mode
    self.scale_factor = scale_factor
    self.encoding_config = encoding_config
    self._load_image(path)
    self._build_tensors()

  def _load_image(self, path):
    self.path = path
    img = cv2.imread(path)
    h, w, _ = img.shape
    new_h = int(h * self.scale_factor)
    new_w = int(w * self.scale_factor)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if self.mode == "HSV":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif self.mode == "RGB":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      raise NotImplementedError
    self.img_array = img

  def _build_tensors(self):
    h, w, _ = self.img_array.shape
    self.h = h
    self.w = w
    X = build_coordinate_array((0, 1), (0, 1), w, h)
    # y = np.linspace(0, 1, h)
    # x = np.linspace(0, 1, w)
    # xx, yy = np.meshgrid(x, y)
    # X = np.stack([xx, yy], axis=-1)
    # if self.encoding_config is not None:
    #   a = self.encoding_config["a_vals"]
    #   b = self.encoding_config["b_vals"]
    #   X = np.concatenate([a * np.sin((2.*np.pi*X) @ b.T), a * np.cos((2.*np.pi*X) @ b.T)], axis=-1)
    tensors = {}
    tensors["features"] = torch.tensor(X.reshape(h * w, -1), dtype=torch.float32)
    tensors["pixels"] = torch.tensor(self.img_array.reshape(h * w, -1), dtype=torch.float32)
    self.tensors = tensors

  def __len__(self):
    return self.tensors["features"].shape[0]

  def __getitem__(self, ind):
    return {key: tensor[ind] for key, tensor in self.tensors.items()}
