import torch
import torch.nn as nn

from utils.utils import rename_state_dict_keys

from pretrained_models.bi_awd_lstm.weight_drop import WeightDrop
from pretrained_models.bi_awd_lstm.embed_regularize import embedded_dropout
from pretrained_models.bi_awd_lstm.locked_dropout import LockedDropout

# Used in a configuration by rename_state_dict_keys function to adapt from 2xlstm to 1xlstm 1xbilstm
def key_transformation(old_key: str):
    if "rnns_rev" in old_key:
        old_key = old_key + "_reverse"

    if "_raw_reverse" in old_key:
        old_key = old_key.split("_raw_reverse")[0] + "_reverse_raw"

    return old_key

class BiAWDEmbedding(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.1, dropouth=0.1, dropouti=0.1, dropoute=0.1, wdrop=0.1, tie_weights=False):
        super(BiAWDEmbedding, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
        self.rnns_rev = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0, bidirectional=True) for l in range(nlayers)]
        
        self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns_rev = [WeightDrop(rnn, ['weight_hh_l0_reverse'], dropout=wdrop) for rnn in self.rnns_rev]

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.rnns_rev = torch.nn.ModuleList(self.rnns_rev)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def load_pretrained(self, path="pretrained_models/bi_awd_lstm/elmo_parameters_statedict.pt"):
        with open(path, 'rb') as f:
            state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = rename_state_dict_keys(state_dict, key_transformation)
        self.load_state_dict(state_dict, strict=False)

    def forward(self, input, seq_lengths):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0) # (bs, seq_len, emb_size)
        emb = self.lockdrop(emb, self.dropouti) #(bs, seq_len, emb_size)
        emb = emb.permute(1,0,2) # (seq_len, bs, emb_size)

        raw_output = emb
        raw_output_rev = emb

        hidden = []
        hidden_rev = []
        raw_outputs = []
        raw_outputs_rev = []
        outputs = []
        outputs_rev = []
        
        for l, _ in enumerate(self.rnns):
            # current_input = raw_output
            raw_output = nn.utils.rnn.pack_padded_sequence(raw_output, seq_lengths)
            packed_output, new_h = self.rnns[l](raw_output)
            raw_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output) # (seq_len, bs, hid)

            raw_output_rev = nn.utils.rnn.pack_padded_sequence(raw_output_rev, seq_lengths)
            packed_output_rev, new_h_rev = self.rnns_rev[l](raw_output_rev)
            raw_output_rev, _ = nn.utils.rnn.pad_packed_sequence(packed_output_rev) # (seq_len, bs, hid*2)
            raw_output_rev = raw_output_rev[:, :, raw_output_rev.size(-1)//2:] # (seq_len, bs, hid) - take the backwards half of output 

            hidden.append(new_h)
            hidden_rev.append(new_h_rev[1])

            raw_outputs.append(raw_output)
            raw_outputs_rev.append(raw_output_rev)

            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
                raw_output_rev = self.lockdrop(raw_output_rev, self.dropouth)
                outputs_rev.append(raw_output_rev)

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
        output_rev = self.lockdrop(raw_output_rev, self.dropout)
        outputs_rev.append(output_rev)
        
        return (outputs, outputs_rev), (hidden, hidden_rev), emb