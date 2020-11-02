from torch import nn

class MLP(nn.Module):

  def __init__(self, in_units, hidden_units, output_targets=3):
    super(MLP, self).__init__()
    layers = []
    units = [in_units] + hidden_units + [output_targets]
    for in_units, out_units in zip(units, units[1:]):
      layers.append(nn.Linear(in_units, out_units))
      layers.append(nn.ReLU())
    layers.pop(-1)  # remove last relu
    self.net = nn.Sequential(*layers)

  def forward(self, x):
    return self.net(x)
