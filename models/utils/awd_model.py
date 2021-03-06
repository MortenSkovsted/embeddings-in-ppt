import torch
import torch.nn as nn

from pretrained_models.awd_lstm.weight_drop import WeightDrop
from pretrained_models.awd_lstm.embed_regularize import embedded_dropout
from pretrained_models.awd_lstm.locked_dropout import LockedDropout

class AWDEmbedding(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, ntoken, ninp, nhid, nlayers, dropout=0.1, dropouth=0.1, dropouti=0.1, dropoute=0.1, wdrop=0.1, tie_weights=False):
        super(AWDEmbedding, self).__init__()
        
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        # maybe add padding_idx=20
        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]

        self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        self.rnns = torch.nn.ModuleList(self.rnns)
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

    def load_pretrained(self, path="pretrained_models/awd_lstm/test_v2_statedict.pt"):
        with open(path, 'rb') as f:
            state_dict = torch.load(f, map_location='cuda' if torch.cuda.is_available() else 'cpu')

        self.load_state_dict(state_dict)

    def forward(self, input, seq_lengths):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0) # (bs, seq_len, emb_size)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti) #(bs, seq_len, emb_size)
        emb = emb.permute(1,0,2) # (seq_len, bs, emb_size)

        raw_output = emb
        new_hidden = []
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output = nn.utils.rnn.pack_padded_sequence(raw_output, seq_lengths)
            packed_output, new_h = rnn(raw_output)
            raw_output, _ = nn.utils.rnn.pad_packed_sequence(packed_output) # (seq_len, bs, hid)
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)

        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)

        return outputs, hidden, emb


    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                for l in range(self.nlayers)]