import torch.nn as nn

from configs.subcel.base import Config as BaseConfig

from models.encoders.awd import Encoder
from models.decoders.attention_subcel import Decoder

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()

    self.args = args
    self.encoder = Encoder(args)
    self.decoder = Decoder(args, in_size=320)

  def forward(self, inp, seq_len):
    output, _, _ = self.encoder(inp, seq_len)

    #perform a permutation of the output since the decoder needs it batch first
    output = output[2].permute(1,0,2) # (bs, seq_len, 320)

    output = self.decoder(output, seq_len)
    return output

class Config(BaseConfig):
  def __init__(self, args):
    super().__init__(args, Model)