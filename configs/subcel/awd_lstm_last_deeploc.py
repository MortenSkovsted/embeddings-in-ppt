import torch.nn as nn

from configs.subcel.base import Config as BaseConfig

from models.encoders.awd_lstm import Encoder
from models.decoders.deeploc_subcel import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args, awd_layer="second")
    self.decoder = Decoder(args, in_size=args.n_hid*2)

  def forward(self, inp, seq_len):
    output = self.encoder(inp, seq_len)
    output = self.decoder(output, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)