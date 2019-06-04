import math
import numpy as np
import torch
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from datautils.dataloader import tokenize_sequence
from model_utils.attention import Attention, MultiStepAttention
from utils import iterate_minibatches, length_to_mask, do_layer_norm, ResultsContainer
from confusionmatrix import ConfusionMatrix
from metrics_mc import gorodkin, IC

class Model(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.in_drop1d = nn.Dropout(args.in_dropout1d)
    self.in_drop2d = nn.Dropout2d(args.in_dropout2d)
    self.drop = nn.Dropout(args.hid_dropout)

    self.embed = nn.Embedding(num_embeddings=21, embedding_dim=args.n_features, padding_idx=20)

    self.convs = nn.ModuleList([nn.Conv1d(in_channels=args.n_features, out_channels=args.n_filters, kernel_size=i, padding=i//2) for i in args.conv_kernels])
    self.cnn_final = nn.Conv1d(in_channels=len(self.convs)*args.n_filters, out_channels=128, kernel_size=3, padding= 3//2)
    self.relu = nn.ReLU()

    self.lstm = nn.LSTM(128, args.n_hid, bidirectional=True, batch_first=True)
    self.attn = Attention(in_size=args.n_hid*2, att_size=args.att_size)

    self.dense = nn.Linear(args.n_hid*2, args.n_hid*2)
    self.label = nn.Linear(args.n_hid*2, args.num_classes)
    self.mem = nn.Linear(args.n_hid*2, 1)
 
    self.init_weights()
    
    
  def init_weights(self):
    self.dense.bias.data.zero_()
    torch.nn.init.orthogonal_(self.dense.weight.data, gain=math.sqrt(2))
    
    self.label.bias.data.zero_()
    torch.nn.init.orthogonal_(self.label.weight.data, gain=math.sqrt(2))
    
    for m in self.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
          for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data, gain=1)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data, gain=1)
            elif 'bias_ih' in name:
              param.data.zero_()
            elif 'bias_hh' in name:
              param.data.zero_()
        elif type(m) in [nn.Conv1d]:
          for name, param in m.named_parameters():
            if 'weight' in name:
              torch.nn.init.orthogonal_(param.data, gain=math.sqrt(2))           
            if 'bias' in name:
              param.data.zero_()
    
  def forward(self, inp, seq_lengths): # inp: (batch_size, seq_len)
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
  
    attn_output, alpha = self.attn(x_in=output, seq_lengths=seq_lengths) #(batch_size, hidden_size*2) alpha = (batch_size, seq_len)
    output = self.drop(attn_output)
    
    output = self.relu(self.dense(output)) # (batch_size, hidden_size*2)
    output = self.drop(output)
    
    out = self.label(output) #(batch_size, num_classes)
    out_mem = torch.sigmoid(self.mem(output)) #(batch_size, 1)

    return (out, out_mem), alpha # alpha only used for visualization in notebooks


################################
#            Config
################################

class Config:
  def __init__(self, args):
    self.results = ResultsContainer()
    self.args = args
    self.Model = Model

    self.traindata, self.testdata = self._load_data()

  def _load_data(self):
    # Load data
    print("Loading data...")
    test_data = np.load(self.args.testset)
    train_data = np.load(self.args.trainset)

    # Test set
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    mask_test = test_data['mask_test']
    mem_test = test_data['mem_test'].astype(np.int32)
    unk_test = test_data['unk_test'].astype(np.int32)

    # Training set
    X_train = train_data['X_train']
    y_train = train_data['y_train']
    mask_train = train_data['mask_train']
    partition = train_data['partition']
    mem_train = train_data['mem_train']
    unk_train = train_data['unk_train']

    print("X_train.shape", X_train.shape)
    print("X_test.shape", X_test.shape)

    # Tokenize and remove invalid sequenzes
    if (True): # is_raw
      X_train, mask = tokenize_sequence(X_train)
      X_train = np.asarray(X_train)
      y_train = y_train[mask]
      mask_train = mask_train[mask]
      partition = partition[mask]
      mem_train = mem_train[mask]
      unk_train = unk_train[mask]

      X_test, mask = tokenize_sequence(X_test)
      X_test = np.asarray(X_test)
      y_test = y_test[mask]
      mask_test = mask_test[mask]
      mem_test = mem_test[mask]
      unk_test = unk_test[mask]

    print("Loading complete!")
    return (X_train, y_train, mask_train, partition, mem_train, unk_train), (X_test, y_test, mask_test, mem_test, unk_test) 

  def _prepare_tensors(self, batch):
    inputs, targets, in_masks, targets_mem, unk_mem = batch

    seq_lengths = in_masks.sum(1).astype(np.int32)

    #sort to be in decending order for pad packed to work
    perm_idx = np.argsort(-seq_lengths)
    in_masks = in_masks[perm_idx]
    seq_lengths = seq_lengths[perm_idx]
    inputs = inputs[perm_idx]
    targets = targets[perm_idx]
    targets_mem = targets_mem[perm_idx]
    unk_mem = unk_mem[perm_idx]

    #convert to tensors
    inputs = Variable(torch.from_numpy(inputs)).type(torch.long).to(self.args.device) # (batch_size, seq_len)
    in_masks = torch.from_numpy(in_masks).to(self.args.device)
    seq_lengths = torch.from_numpy(seq_lengths).to(self.args.device)

    return inputs, seq_lengths, in_masks, targets, targets_mem, unk_mem

  def _calculate_loss_and_accuracy(self, output, output_mem, targets, targets_mem, unk_mem, confusion, confusion_mem):
    #Confusion Matrix
    preds = np.argmax(output.cpu().detach().numpy(), axis=-1)
    
    mem_preds = torch.round(output_mem).type(torch.int).cpu().detach().numpy()
    confusion.batch_add(targets, preds)
    confusion_mem.batch_add(targets_mem[np.where(unk_mem == 1)], mem_preds[np.where(unk_mem == 1)])
    
    unk_mem = Variable(torch.from_numpy(unk_mem)).type(torch.float).to(self.args.device)
    targets = Variable(torch.from_numpy(targets)).type(torch.long).to(self.args.device)
    targets_mem = Variable(torch.from_numpy(targets_mem)).type(torch.float).to(self.args.device)

    # squeeze from [batch_size,1] -> [batch_size] such that it matches weight matrix for BCE
    output_mem = output_mem.squeeze(1)
    targets_mem = targets_mem.squeeze(1)

    # calculate loss
    loss = F.cross_entropy(input=output, target=targets)
    loss_mem = F.binary_cross_entropy(input=output_mem, target=targets_mem, weight=unk_mem, reduction="sum")
    loss_mem = loss_mem / sum(unk_mem)
    combined_loss = loss + 0.5 * loss_mem
    
    return combined_loss

  def run_train(self, model, X, y, mask, mem, unk):
    optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
    model.train()

    train_err = 0
    train_batches = 0
    confusion_train = ConfusionMatrix(num_classes=10)
    confusion_mem_train = ConfusionMatrix(num_classes=2)

    # Generate minibatches and train on each one of them	
    for batch in iterate_minibatches(X, y, mask, mem, unk, self.args.batch_size):
      inputs, seq_lengths, in_masks, targets, targets_mem, unk_mem = self._prepare_tensors(batch)
    
      optimizer.zero_grad()
      (output, output_mem), alphas = model(inputs, seq_lengths)

      loss = self._calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion_train, confusion_mem_train)
      loss.backward()

      torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
      optimizer.step()

      train_err += loss.item()
      train_batches += 1 

    train_loss = train_err / train_batches
    return train_loss, confusion_train, confusion_mem_train

  def run_eval(self, model, X, y, mask, mem, unk):
    model.eval()

    val_err = 0
    val_batches = 0
    confusion_valid = ConfusionMatrix(num_classes=10)
    confusion_mem_valid = ConfusionMatrix(num_classes=2)

    with torch.no_grad():
      # Generate minibatches and train on each one of them	
      for batch in iterate_minibatches(X, y, mask, mem, unk, self.args.batch_size, sort_len=False, shuffle=False, sample_last_batch=False):
        inputs, seq_lengths, in_masks, targets, targets_mem, unk_mem = self._prepare_tensors(batch)

        (output, output_mem), alphas = model(inputs, seq_lengths)

        loss = self._calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion_valid, confusion_mem_valid)

        val_err += loss.item()
        val_batches += 1

    val_loss = val_err / val_batches
    return val_loss, confusion_valid, confusion_mem_valid, (alphas, targets, seq_lengths)

  def run_test(self, models, X, y, mask, mem, unk):
    for i in range(len(models)):
      models[i].eval()

    val_err = 0
    val_batches = 0
    confusion_valid = ConfusionMatrix(num_classes=10)
    confusion_mem_valid = ConfusionMatrix(num_classes=2)

    with torch.no_grad():
      # Generate minibatches and train on each one of them	
      for batch in iterate_minibatches(X, y, mask, mem, unk, self.args.batch_size, sort_len=False, shuffle=False, sample_last_batch=False):
        inputs, seq_lengths, in_masks, targets, targets_mem, unk_mem = self._prepare_tensors(batch)

        (output, output_mem), alphas = models[0](inputs, seq_lengths)
        #When multiple models are given, perform ensambling
        for i in range(1,len(models)):
          (out, out_mem), alphas  = models[i](inputs, seq_lengths)
          output = output + out
          output_mem = output_mem + out_mem

        #divide by number of models
        output = torch.div(output,len(models))
        output_mem = torch.div(output_mem,len(models))

        loss = self._calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion_valid, confusion_mem_valid)

        val_err += loss.item()
        val_batches += 1

    val_loss = val_err / val_batches
    return val_loss, confusion_valid, confusion_mem_valid, (alphas, targets, seq_lengths)

  def trainer(self):
    (X_train, y_train, mask_train, partition, mem_train, unk_train) = self.traindata
    best_model = None
    best_val_accs = []
    best_val_models = []

    for i in range(1,5):
      best_val_acc = 0
      best_val_model = None
      # Network compilation
      print("Compilation model {}".format(i))
      model = self.Model(self.args).to(self.args.device)
      print("Model: ", model)

      # Train and validation sets
      train_index = np.where(partition != i)
      val_index = np.where(partition == i)
      X_tr = X_train[train_index].astype(np.float32)
      X_val = X_train[val_index].astype(np.float32)
      y_tr = y_train[train_index].astype(np.int32)
      y_val = y_train[val_index].astype(np.int32)
      mask_tr = mask_train[train_index].astype(np.float32)
      mask_val = mask_train[val_index].astype(np.float32)
      mem_tr = mem_train[train_index].astype(np.int32)
      mem_val = mem_train[val_index].astype(np.int32)
      unk_tr = unk_train[train_index].astype(np.int32)
      unk_val = unk_train[val_index].astype(np.int32)

      print("Validation shape: {}".format(X_val.shape))
      print("Training shape: {}".format(X_tr.shape))
      
      for epoch in range(self.args.epochs):
        start_time = time.time()

        train_loss, confusion_train, confusion_mem_train = self.run_train(model, X_tr, y_tr, mask_tr, mem_tr, unk_tr)
        val_loss, confusion_valid, confusion_mem_valid, (alphas, targets, seq_lengths) = self.run_eval(model, X_val, y_val, mask_val, mem_val, unk_val)
        
        self.results.append_epoch(train_loss, val_loss, confusion_train.accuracy(), confusion_valid.accuracy()) 
        
        if confusion_valid.accuracy() > best_val_acc:
          best_val_acc = confusion_valid.accuracy()
          best_val_model = model

        if best_val_acc > self.results.best_val_acc:
          self.results.best_val_acc = best_val_acc
          best_model = model
        
        print('-' * 22, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, self.args.epochs-1, time.time() - start_time), '-' * 22 )
        print('| Train | loss {:.4f} | acc {:.2f}% | mem_acc {:.2f}% | Gorodkin {:2.2f} | IC {:2.2f}' 
              ' |'.format(train_loss, confusion_train.accuracy()*100, confusion_mem_train.accuracy()*100, gorodkin(confusion_train.ret_mat()), IC(confusion_train.ret_mat())))
        print('| Valid | loss {:.4f} | acc {:.2f}% | mem_acc {:.2f}% | Gorodkin {:2.2f} | IC {:2.2f}' 
              ' |'.format(val_loss, confusion_valid.accuracy()*100, confusion_mem_valid.accuracy()*100, gorodkin(confusion_valid.ret_mat()), IC(confusion_valid.ret_mat())))
        print('-' * 79)
        
        sys.stdout.flush()

      best_val_accs.append(best_val_acc)
      best_val_models.append(best_val_model)

    for i, acc in enumerate(best_val_accs):
      print("Partion {:1d} : acc {:.2f}%".format(i, acc*100))
    print("Average validation accuracy {:.2f}% \n".format((sum(best_val_accs)/len(best_val_accs))*100))

    return best_val_accs, best_val_models


  def tester(self, best_val_models):
    (X_test, y_test, mask_test, mem_test, unk_test) = self.testdata
    test_loss, confusion_test, confusion_mem_test, (alphas, targets, seq_lengths) = self.run_test(best_val_models, X_test, y_test, mask_test, mem_test, unk_test)
    
    print("ENSAMBLE TEST RESULTS")
    print(confusion_test)
    print(confusion_mem_test)
    print("test accuracy:\t\t{:.2f} %".format(confusion_test.accuracy() * 100))
    print("test mem accuracy:\t{:.2f} %".format(confusion_mem_test.accuracy() * 100))
    print("test Gorodkin:\t\t{:.2f}".format(gorodkin(confusion_test.ret_mat())))
    print("test IC:\t\t{:.2f}".format(IC(confusion_test.ret_mat())))

    self.results.set_final(
      alph = alphas.cpu().detach().numpy(), 
      seq_len = seq_lengths.cpu().detach().numpy(), 
      targets = targets, 
      cf = confusion_test.ret_mat(),
      cf_mem = confusion_mem_test.ret_mat(), 
      acc = confusion_test.accuracy(), 
      acc_mem = confusion_mem_test.accuracy())