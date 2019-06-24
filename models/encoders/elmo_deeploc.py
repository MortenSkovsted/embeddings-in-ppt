import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import rename_state_dict_keys
from models.utils.elmo_model import Elmo, key_transformation
from models.encoders.deeploc_raw import Encoder as BaseEncoder


class Encoder(BaseEncoder):
  """
  Encoder with elmo concatenated to the LSTM output

  Inputs: input, seq_len
    - **input** of shape
  Outputs: output
    - **output** of shape (batch_size, seq_len, hidden_size*2+320*2)
  """
  def __init__(self, args):
    super().__init__(args)

    self.elmo = Elmo(ntoken=21, ninp=320, nhid=1280, nlayers=3, tie_weights=True)

    # load pretrained elmo
    with open("pretrained_models/elmo/elmo_parameters_statedict.pt", 'rb') as f:
      state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = rename_state_dict_keys(state_dict, key_transformation)
    self.elmo.load_state_dict(state_dict, strict=False)

  def forward(self, inp, seq_lengths):
    #### Elmo 
    with torch.no_grad():
      (elmo_hid, elmo_hid_rev), last_hid, raw_all_hid, dropped_all_hid, emb = self.elmo(input=inp, seq_lengths=seq_lengths)
    
    elmo_hid = elmo_hid.permute(1,0,2) # (bs, seq_len, emb_size) 
    elmo_hid_rev = elmo_hid_rev.permute(1,0,2) # (bs, seq_len, emb_size) 
    ### End Elmo 
    
    inp = self.embed(inp) # (batch_size, seq_len, emb_size)

    inp = self.in_drop1d(inp) # feature dropout
    x = self.in_drop2d(inp)  # (batch_size, seq_len, emb_size) - 2d dropout

    x = x.permute(0, 2, 1)  # (batch_size, emb_size, seq_len)
    conv_cat = torch.cat([self.relu(conv(x)) for conv in self.convs], dim=1) # (batch_size, emb_size*len(convs), seq_len)
    x = self.relu(self.cnn_final(conv_cat)) #(batch_size, out_channels=128, seq_len)

    x = x.permute(0, 2, 1) #(batch_size, seq_len, out_channels=128)
    x = self.drop(x) #( batch_size, seq_len, lstm_input_size)
    
    pack = nn.utils.rnn.pack_padded_sequence(x, seq_lengths, batch_first=True)
    packed_output, (h, c) = self.lstm(pack) #h = (2, batch_size, hidden_size)
    output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True) #(batch_size, seq_len, hidden_size*2)

    ### Concat 320 hidden layer to BiLSTM
    output = torch.cat((output, elmo_hid, elmo_hid_rev),dim=2) # (batch_size, seq_len, hidden_size*2+320*2)
    ### End Concat
  
    return output