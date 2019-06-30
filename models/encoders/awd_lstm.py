import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils.awd_model import AWD_Embedding
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(nn.Module):
  """
  Encoder with elmo concatenated to the LSTM output

  Parameters:
    -- project_size: size of projection layer from elmo to lstm

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2)
  """
  def __init__(self, args, project_size=None):
    super().__init__()
    self.args = args
    self.project_size = project_size
    self.drop = nn.Dropout(args.hid_dropout)

    if project_size is not None:
      self.project = nn.Linear(1280, project_size, bias=False)

    self.lstm = nn.LSTM(project_size if project_size is not None else 1280, args.n_hid, bidirectional=True, batch_first=True)
    self.awd = AWD_Embedding(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    # load pretrained awd
    with open("pretrained_models/awd_lstm/test_v2_statedict.pt", 'rb') as f:
        state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    self.awd.load_state_dict(state_dict)

  def forward(self, inp, seq_lengths):
    #### AWD 
    with torch.no_grad():
      all_hid, _, _ = self.awd(input=inp, seq_lengths=seq_lengths)

    awd_hid = all_hid[1].permute(1,0,2) # (bs, seq_len, 1280)

    if self.project_size is not None:
      awd_hid = self.project(awd_hid) # (bs, seq_len, project_size) 
    ### End AWD

    awd_hid = self.drop(awd_hid) #( batch_size, seq_len, project_size or 1280)
    
    pack = nn.utils.rnn.pack_padded_sequence(awd_hid, seq_lengths, batch_first=True)
    packed_output, _ = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)
  
    return output