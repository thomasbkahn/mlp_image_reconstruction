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



class MultiPhotoArrayDataset(Dataset):
  
  def __init__(self, paths, mode="RGB", encoding_config=None, scale_factor=0.1, force_common_size=True):
    self.mode = mode
    self.paths = paths
    self.scale_factor = scale_factor
    self.encoding_config = encoding_config
    self.force_common_size = force_common_size
    self._load_images(paths)
    self._build_tensors()

  def _load_image(self, path):
    img = cv2.imread(path)
    if self.force_common_size and self.img_arrays:
      # we want to force common size AND have already loaded at least one image
      h, w, _ = self.img_arrays[0].shape
      img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    else:
      # either we don't want to force common size or this is the
      # first image (which defines the size of the rest)
      img = cv2.resize(img, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_AREA)
    if self.mode == "HSV":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif self.mode == "RGB":
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
      raise NotImplementedError
    return img

  def _load_images(self, paths):
    self.img_arrays = []
    for path in paths:
        self.img_arrays.append(self._load_image(path))

  def _concat_and_make_tensor(self, arrays):
    return torch.tensor(np.concatenate(arrays, axis=0), dtype=torch.float32)

  def _build_tensors(self):
    pixels = []
    features = []
    weights = []
    img_sizes = []
    n_images = len(self.img_arrays)
    for img_ind, img_array in enumerate(self.img_arrays):
      h, w, _ = img_array.shape
      img_sizes.append((h, w))
      X = build_coordinate_array((0, 1), (0, 1), w, h)
      features.append(X.reshape(h * w, -1))
      pixels.append(img_array.reshape(h * w, -1))
      weights_img = np.zeros((features[-1].shape[0], n_images), dtype=np.float64)
      weights_img[:, img_ind] = 1.0
      weights.append(weights_img)
    tensors = {}
    tensors["features"] = self._concat_and_make_tensor(features)
    tensors["pixels"] = self._concat_and_make_tensor(pixels)
    tensors["weights"] = self._concat_and_make_tensor(weights)
    self.tensors = tensors
    self.image_sizes = img_sizes

  def __len__(self):
    return self.tensors["features"].shape[0]

  def __getitem__(self, ind):
    return {key: tensor[ind] for key, tensor in self.tensors.items()}

    
class PhotoArrayDataset(MultiPhotoArrayDataset):
  # TODO change this to subclass multi case, just drop weight key

  def __init__(self, path, mode="RGB", encoding_config=None, scale_factor=0.1):
    paths = [path]
    super(PhotoArrayDataset, self).__init__(paths, mode, encoding_config, scale_factor)
