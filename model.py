from torch import nn
import torch

class MLP(nn.Module):

  def __init__(self, in_units, hidden_units, output_targets=3, drop_last_activation=True):
    super(MLP, self).__init__()
    layers = []
    units = [in_units] + hidden_units
    if output_targets is not None:
      units += [output_targets]
    for in_units, out_units in zip(units, units[1:]):
      layers.append(nn.Linear(in_units, out_units))
      layers.append(nn.ReLU())
    if drop_last_activation:
        layers.pop(-1)  # remove last relu
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)

class MultiHeadMLP(nn.Module):

  def __init__(self, in_units, n_heads, hidden_units_heads, hidden_units_core,
               output_targets=3):
    super(MultiHeadMLP, self).__init__()
    assert hidden_units_heads[-1] == hidden_units_core[0]
    heads = []
    for head_ind in range(n_heads):
      head = MLP(in_units, hidden_units=hidden_units_heads,
                 output_targets=None, drop_last_activation=False)
      heads.append(head)
    self.heads = nn.ModuleList(heads)
    self.n_heads = n_heads

    core = MLP(in_units=hidden_units_core[0], hidden_units=hidden_units_core,
               output_targets=output_targets, drop_last_activation=True)
    self.core = core

  def forward(self, x, head_weight):
    h_heads = torch.stack([head(x) for head in self.heads]) # stacks, adding new dimension in axis 0
    head_weight = head_weight.transpose(1, 0).view(self.n_heads, -1, 1) # get in same form as h_heads for broadcast
    h = torch.sum(h_heads * head_weight, dim=0) # now back to single-head-like dimensions, dropped new axis
    return self.core(h)

