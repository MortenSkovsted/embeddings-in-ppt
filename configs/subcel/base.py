import math
import copy
import numpy as np
import torch
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


from dataloaders.subcel import iterate_minibatches, load_data
from models.utils.attention import Attention, MultiStepAttention
from utils.utils import do_layer_norm, save_model, load_model, save_results, ResultsContainer
from utils.confusionmatrix import ConfusionMatrix
from utils.metrics_mc import gorodkin, IC

from configs.config_base import Config as ConfigBase

class Config(ConfigBase):
  """
  THIS IS A BASE CLASS AND CAN ONLY RUN IF GIVEN AN ENCODER AND DECODER
  """
  def __init__(self, args, Model, raw=True):
    self.results = ResultsContainer()
    self.args = args
    self.Model = Model
    self.raw = raw

    if raw:
      self.trainset = "data/Deeploc_raw/train.npz"
      self.testset = "data/Deeploc_raw/test.npz"
    else:
      self.trainset = "data/Deeploc/train.npz"
      self.testset = "data/Deeploc/test.npz"

    self.traindata, self.testdata = load_data(train_path=self.trainset, test_path=self.testset, is_raw=raw)

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
    inputs = torch.from_numpy(inputs).to(self.args.device) # (batch_size, seq_len)
    if self.raw:
      inputs = inputs.long()
    in_masks = torch.from_numpy(in_masks).to(self.args.device)
    seq_lengths = torch.from_numpy(seq_lengths).to(self.args.device)

    return inputs, seq_lengths, in_masks, targets, targets_mem, unk_mem

  def _calculate_loss_and_accuracy(self, output, output_mem, targets, targets_mem, unk_mem, confusion, confusion_mem):
    #Confusion Matrix
    preds = torch.round(output).type(torch.int).cpu().detach().numpy()
    print(f'preds {preds.shape}')
    #np.argmax(output.cpu().detach().numpy(), axis=-1)
    
    mem_preds = torch.round(output_mem).type(torch.int).cpu().detach().numpy()
    #confusion.batch_add(targets, preds)
    #confusion_mem.batch_add(targets_mem[np.where(unk_mem == 1)], mem_preds[np.where(unk_mem == 1)])
    
    unk_mem = Variable(torch.from_numpy(unk_mem)).type(torch.float).to(self.args.device)
    targets = Variable(torch.from_numpy(targets)).type(torch.long).to(self.args.device)
    targets_mem = Variable(torch.from_numpy(targets_mem)).type(torch.float).to(self.args.device)

    # squeeze from [batch_size,1] -> [batch_size] such that it matches weight matrix for BCE
    output_mem = output_mem.squeeze(1)
    targets_mem = targets_mem.squeeze(1)

    # calculate loss
<<<<<<< HEAD

    # change from cross_entropy to Multi_label function
    # loss = F.cross_entropy(input=output, target=targets)
    # torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
    # torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    # torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
    print(f'output.size() ={output[10]}')
    print(f'targets ={targets[0:10]}')
    print(f'targets.size() ={targets.size()}')
    print(f'output.size() ={output.size()}')
    criterion = nn.BCEWithLogitsLoss()
    
    loss = criterion(output, targets)
    
    #loss_mem = F.binary_cross_entropy(input=output_mem, target=targets_mem, weight=unk_mem, reduction="sum")
    
    criterion_mem = nn.BCEWithLogitsLoss(weight=unk_mem,reduce="sum")
    loss_mem = criterion_mem(output_mem, targets_mem)
    
=======
    loss = F.cross_entropy(input=output, target=targets)
    loss_mem = F.binary_cross_entropy(input=output_mem, target=targets_mem, weight=unk_mem, reduction="sum")
>>>>>>> parent of dcb3bf7... Update base.py
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
    val_err = 0
    val_batches = 0
    confusion_valid = ConfusionMatrix(num_classes=10)
    confusion_mem_valid = ConfusionMatrix(num_classes=2)

    with torch.no_grad():
      # Generate minibatches and train on each one of them	
      # 71 is a hack to hit the an batch size that goes up in 2769 which is the amount of test data in raw dataset
      for batch in iterate_minibatches(X, y, mask, mem, unk, 71, sort_len=False, shuffle=False, sample_last_batch=False):
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
    best_val_accs = []
    best_val_mem_accs = []
    best_val_mccs = []
    best_val_gorodkins = []

    for i in range(1,5):
      best_val_acc = 0
      best_val_mem_acc = 0
      best_val_mcc = -1
      best_val_gorodkin = -1
      best_val_epoch = 0
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
          best_val_epoch = epoch
          best_val_acc = confusion_valid.accuracy()
          best_val_mem_acc = confusion_mem_valid.accuracy()
          best_val_gorodkin = gorodkin(confusion_valid.ret_mat())
          best_val_mcc = confusion_mem_valid.MCC()
          save_model(model, self.args, index=i)

        if best_val_acc > self.results.best_val_acc:
          self.results.best_val_acc = best_val_acc
        
        print('-' * 24, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, self.args.epochs-1, time.time() - start_time), '-' * 25 )
        print('| Train | loss {:.4f} | acc {:.2f}% | mem_acc {:.2f}% | Gorodkin {:2.4f} | MCC {:2.4f}' 
              ' |'.format(train_loss, confusion_train.accuracy()*100, confusion_mem_train.accuracy()*100, gorodkin(confusion_train.ret_mat()), confusion_mem_train.MCC()))
        print('| Valid | loss {:.4f} | acc {:.2f}% | mem_acc {:.2f}% | Gorodkin {:2.4f} | MCC {:2.4f}' 
              ' |'.format(val_loss, confusion_valid.accuracy()*100, confusion_mem_valid.accuracy()*100, gorodkin(confusion_valid.ret_mat()), confusion_mem_valid.MCC()))
        print('-' * 84)
        
        sys.stdout.flush()
        torch.cuda.empty_cache()

      print('|', ' ' * 17, 'Best accuracy: {:.2f}% found after {:3d} epochs'.format(best_val_acc*100, best_val_epoch), ' ' * 17, '|')
      print('|', ' '* 17, 'Mem acc {:.2f}% | Gorodkin {:2.4f} | MCC {:2.4f}'.format(best_val_mem_acc*100, best_val_gorodkin, best_val_mcc) , ' '*16, '|' )
      print('-' * 84)
      best_val_accs.append(best_val_acc)
      best_val_mem_accs.append(best_val_mem_acc)
      best_val_gorodkins.append(best_val_gorodkin)
      best_val_mccs.append(best_val_mcc)

    for i, _ in enumerate(best_val_accs):
      print("Partion {:1d} : acc {:.2f}% | mem_acc {:.2f}% | Gorodkin {:2.4f} | MCC {:2.4f}".format(
        i, best_val_accs[i]*100, best_val_mem_accs[i]*100, best_val_gorodkins[i], best_val_mccs[i]))
    print("\nAverage is: acc {:.2f}% | mem_acc {:.2f}% | Gorodkin {:2.4f} | MCC {:2.4f} \n".format(
      (sum(best_val_accs)/len(best_val_accs))*100, sum(best_val_mem_accs)/len(best_val_accs)*100, sum(best_val_gorodkins)/len(best_val_accs), sum(best_val_mccs)/len(best_val_accs)))


  def tester(self):
    (X_test, y_test, mask_test, mem_test, unk_test) = self.testdata

    best_models = []
    for i in range(1,5):
      model = self.Model(self.args).to(self.args.device)
      load_model(model, self.args, index=i)
      model.eval()
      best_models.append(model)

    test_loss, confusion_test, confusion_mem_test, (alphas, targets, seq_lengths) = self.run_test(best_models, X_test, y_test, mask_test, mem_test, unk_test)
    
    print("ENSAMBLE TEST RESULTS")
    print(confusion_test)
    print(confusion_mem_test)
    print("test accuracy:\t\t{:.2f} %".format(confusion_test.accuracy() * 100))
    print("test Gorodkin:\t\t{:.4f}".format(gorodkin(confusion_test.ret_mat())))
    print("test mem accuracy:\t{:.2f} %".format(confusion_mem_test.accuracy() * 100))
    print("test mem MCC:\t\t{:.4f}".format(confusion_mem_test.MCC()))

    self.results.set_final(
      alph = alphas.cpu().detach().numpy(), 
      seq_len = seq_lengths.cpu().detach().numpy(), 
      targets = targets, 
      cf = confusion_test.ret_mat(),
      cf_mem = confusion_mem_test.ret_mat(), 
      acc = confusion_test.accuracy(), 
      acc_mem = confusion_mem_test.accuracy())
    
    save_results(self.results,self.args)