import math
import copy
import numpy as np
import torch
import sys
import time
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, hamming_loss, f1_score
from sklearn.covariance import EmpiricalCovariance
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
  
  def NBprediction(self,np_row):
    #print(WTF)
    if np.around(np_row).sum() == 0:
      prediction = np.array([0,0,0,0,0,0,0,0,0,0])
      prediction[np.argmax(np_row)] = 1
    else:
      prediction = np.around(np_row)
    return prediction
  def simple_cov_adjustment(self, np_row,cov_row,position):
    prediction = np_row + cov_row
    prediction[position] = 1
    return prediction

  def _calculate_loss_and_accuracy(self, output, output_mem, targets, targets_mem, unk_mem, confusion, confusion_mem, cov):
    #Confusion Matrix
    m = nn.Sigmoid()

    #Previuse code
    #np.argmax(output.cpu().detach().numpy(), axis=-1)

    #Choose prediction algorim
    choose = 1

    #print(cov)

    model_output = m(output).type(torch.FloatTensor).cpu().detach().numpy()

    #navive basian prediction
    if choose == 0:
      preds = torch.round(m(output)).type(torch.int).cpu().detach().numpy()
    
    
    # Naive basian aka propabilityes higher than 0.5 is converted to 1
    # but if no sublocation is predicted (0,0,0,0,0,0,0,0,0), then the highest
    # properbility will be changed to a 1

    elif choose == 1:
      preds = m(output).cpu().detach().numpy()
      preds = np.apply_along_axis(self.NBprediction,axis=1,arr=preds)

    # Pick highest likely sublocation, then adjust the rest with variance covariance matrix
    # If any location is higher than 0.5 pick the highest, adjust with var-cov matrix etc.
    # This is part does not work and may not be needed.
    elif choose == 2:
      # find max value in each row
      max_pos = np.argmax(output.cpu().detach().numpy(),axis=1)
      preds = np.apply_along_axis(self.simple_cov_adjustment,axis=1,arr=output,)
      pass


    #exact match algo from sklearn
    exact_match = accuracy_score(targets,preds)
    #print(exact_match)
    #Hamming loss algo from sklearn
    hamming_los = hamming_loss(targets,preds)
    #print(hamming_los)

    mem_preds = torch.round(output_mem).type(torch.int).cpu().detach().numpy()
    #confusion.batch_add(targets, preds)
    try:
      confusion_mem.batch_add(targets_mem[np.where(unk_mem == 1)], mem_preds[np.where(unk_mem == 1)])
    except Exception as e:
      print("type error: " + str(e))
    unk_mem = Variable(torch.from_numpy(unk_mem)).type(torch.float).to(self.args.device)
    targets = Variable(torch.from_numpy(targets)).type(torch.torch.double).to(self.args.device)

    targets_mem = Variable(torch.from_numpy(targets_mem)).type(torch.float).to(self.args.device)

    # squeeze from [batch_size,1] -> [batch_size] such that it matches weight matrix for BCE
    output_mem = output_mem.squeeze(1)
    targets_mem = targets_mem.squeeze(1)

    # calculate loss
    #loss = F.cross_entropy(input=output, target=targets) #Old loss function

    # change from cross_entropy to Multi_label function
    # The following is possible loss functions:
    # torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='mean', pos_weight=None)
    # torch.nn.BCELoss(weight=None, size_average=None, reduce=None, reduction='mean')
    # torch.nn.MultiLabelMarginLoss(size_average=None, reduce=None, reduction='mean')
    # torch.nn.MultiLabelSoftMarginLoss(weight=None, size_average=True)

    #Ready? Choose your loss function!!!
    #weights = torch.uint8()
    total_proteins_labels = 27319
    tpl = total_proteins_labels
    weights = torch.DoubleTensor([tpl/7397, tpl/7662, tpl/2615, tpl/1942, tpl/3117, tpl/1469, tpl/821, tpl/965, tpl/1120, tpl/211]).to(self.args.device)
    criterion = nn.MultiLabelSoftMarginLoss(weights)
    #criterion = nn.BCEWithLogitsLoss(weights)
    
    #print(torch.tensor([tpl/7397, tpl/7662, tpl/2615, tpl/1942, tpl/3117, tpl/1469, tpl/821, tpl/965, tpl/1120, tpl/211],dtype=torch.double))
    #output = output.type(torch.cuda.DoubleTensor)
    output = output.type(torch.double).to(self.args.device)
    loss = criterion(output, targets)
    loss = loss.type(torch.float)
    #loss_mem = F.binary_cross_entropy(input=output_mem, target=targets_mem, weight=unk_mem, reduction="sum")

    criterion_mem = nn.BCEWithLogitsLoss(weight=unk_mem,reduce="sum")
    loss_mem = criterion_mem(output_mem, targets_mem)


    #loss = F.cross_entropy(input=output, target=targets)
    #loss_mem = F.binary_cross_entropy(input=output_mem, target=targets_mem, weight=unk_mem, reduction="sum")

    loss_mem = loss_mem / sum(unk_mem)
    combined_loss = loss + 0.5 * loss_mem

    return combined_loss, exact_match, hamming_los, preds, targets, model_output

  def run_train(self, model, X, y, mask, mem, unk, cov):
    optimizer = torch.optim.Adam(model.parameters(), lr=self.args.learning_rate)
    model.train()

    train_err = 0
    train_batches = 0
    total_N = 0
    #Exact match
    total_EM_avg = 0
    total_HL_avg = 0
    total_F1_avg = 0
    confusion_train = ConfusionMatrix(num_classes=10)
    confusion_mem_train = ConfusionMatrix(num_classes=2)
    first = True
    
    # Generate minibatches and train on each one of them
    for batch in iterate_minibatches(X, y, mask, mem, unk, self.args.batch_size):
      inputs, seq_lengths, in_masks, targets, targets_mem, unk_mem = self._prepare_tensors(batch)
      optimizer.zero_grad()
      (output, output_mem), alphas = model(inputs, seq_lengths)
      loss , exact_match, HammingLoss, preds, targets, model_output = self._calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion_train, confusion_mem_train, cov)
      loss.backward()
      if first:
        total_predicted = torch.IntTensor(preds)
        total_targets = torch.IntTensor(targets.type('torch.IntTensor'))
        model_pre_output = torch.FloatTensor(model_output)
        first = False
      else:
        total_predicted = torch.cat((total_predicted, torch.IntTensor(preds)),dim=0)
        total_targets = torch.cat((total_targets, targets.type('torch.IntTensor')),dim=0)
        model_pre_output = torch.cat((model_pre_output, torch.FloatTensor(model_output)),dim=0)
      torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
      optimizer.step()
      
      train_err += loss.item()
      train_batches += 1

      count = targets.shape[0]
      

      #Algotimen crates total average of Exact match or hammingLoss
      
      total_EM_avg = (total_EM_avg * total_N + exact_match * count)/(total_N + count)
      total_HL_avg = (total_HL_avg * total_N + HammingLoss * count)/(total_N + count)
      #print(f'EM {total_EM_avg} and Hamming Loss {total_HL_avg}')
      total_N += count
    #run on same metric as fairSeq
    #f1scores = f1_score(total_targets, total_predicted,average=None)
    #f1average = f1scores.sum()/len(f1scores)
    f1average = f1_score(total_targets, total_predicted,average='macro')
    #print(f1average)


    #print(f'EM {total_EM_avg} and Hamming Loss {total_HL_avg} end')
    #print(total_predicted)
    #print(total_targets)
    train_loss = train_err / train_batches
    return train_loss, confusion_train, confusion_mem_train, total_EM_avg, total_HL_avg, f1average, total_predicted, total_targets, model_pre_output

  def run_eval(self, model, X, y, mask, mem, unk, cov):
    model.eval()

    val_err = 0
    val_batches = 0
    total_N = 0
    #Exact match
    total_EM_avg = 0
    total_HL_avg = 0
    total_F1_avg = 0
    confusion_valid = ConfusionMatrix(num_classes=10)
    confusion_mem_valid = ConfusionMatrix(num_classes=2)

    #train_loss, confusion_train, confusion_mem_train, EM_train, Hamming_loss_train, total_predicted, total_targets

    with torch.no_grad():
      # Generate minibatches and train on each one of them
      first = True
      for batch in iterate_minibatches(X, y, mask, mem, unk, self.args.batch_size, sort_len=False, shuffle=False, sample_last_batch=False):
        inputs, seq_lengths, in_masks, targets, targets_mem, unk_mem = self._prepare_tensors(batch)

        (output, output_mem), alphas = model(inputs, seq_lengths)
        
        loss, exact_match, HammingLoss, preds, targets, model_output = self._calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion_valid, confusion_mem_valid, cov)
        if first:
          total_predicted = torch.IntTensor(preds)
          total_targets = torch.IntTensor(targets.type('torch.IntTensor'))
          total_modeloutput = torch.FloatTensor(model_output)
          first = False
        else:
          total_predicted = torch.cat((total_predicted, torch.IntTensor(preds)),dim=0)
          total_targets = torch.cat((total_targets, targets.type('torch.IntTensor')),dim=0)
          total_modeloutput = torch.cat((total_modeloutput, torch.FloatTensor(model_output)),dim=0)
          
        val_err += loss.item()
        val_batches += 1

        ####Do F1 calculation
        #go through all predictions and targest and classify them as:
        #True positive, False positive, True negative or False negative
        #sklearn.metrics.f1_score
        #print(targets)
        #print(preds)
        count = targets.shape[0]


        # This is not the same F1 as used in fairseq
        #f1scores = f1_score(targets.cpu(), preds,average=None)
        #f1average = f1scores.sum()/len(f1scores)

        f1average = f1_score(targets.cpu(), preds,average='macro')

        #print(f1average)
        total_F1_avg = (total_F1_avg * total_N + f1average * count)/(total_N + count)

        #Algotimen crates total average of Exact match or hammingLoss
        
        total_EM_avg = (total_EM_avg * total_N + exact_match * count)/(total_N + count)
        total_HL_avg = (total_HL_avg * total_N + HammingLoss * count)/(total_N + count)
        #print(f'EM {total_EM_avg} and Hamming Loss {total_HL_avg}')
        total_N += count
    #print(total_F1_avg)
    val_loss = val_err / val_batches
    return val_loss, confusion_valid, confusion_mem_valid, (alphas, total_targets, total_predicted, seq_lengths, total_EM_avg, total_HL_avg, total_F1_avg, total_modeloutput)

  def run_test(self, models, X, y, mask, mem, unk, cov):
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
        
        loss, exact_match, HammingLoss, preds, targets, model_output = self._calculate_loss_and_accuracy(output, output_mem, targets, targets_mem, unk_mem, confusion_valid, confusion_mem_valid, cov)
        val_err += loss.item()
        val_batches += 1

    val_loss = val_err / val_batches
    return val_loss, confusion_valid, confusion_mem_valid, (alphas, targets, seq_lengths, exact_match, HammingLoss, preds, targets)

  def trainer(self):
    (X_train, y_train, mask_train, partition, mem_train, unk_train) = self.traindata
    best_val_accs = []
    best_val_mem_accs = []
    best_val_mccs = []
    best_val_gorodkins = []
    best_val_F1 = []
    uniq_run_number = time.clock()

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

      #### Make a function that makes a variance covariance matrix of all the training labels
      #### This is done to be able to use the covariances in the prediction of mutilabeled proteins
      #print(y_tr)
      #print(y_tr.shape)
      cov = EmpiricalCovariance().fit(y_tr)
      #print(cov)
      #print(cov.covariance_.shape)
      #print(cov.covariance_)


      for epoch in range(self.args.epochs):
        start_time = time.time()
        
        train_loss, confusion_train, confusion_mem_train, EM_train, Hamming_loss_train, f1_train, \
          predicted_train, targets_train, model_output = self.run_train(model, X_tr, y_tr, mask_tr, mem_tr, unk_tr, cov)
        
        val_loss, confusion_valid, confusion_mem_valid, \
          (alphas, targets_test, predicted_test, seq_lengths, EM_test, Hamming_loss_test, F1_value, total_modeloutput)\
            = self.run_eval(model, X_val, y_val, mask_val, mem_val, unk_val, cov)

        self.results.append_epoch(train_loss, val_loss, EM_train, EM_test)

        if F1_value > best_val_acc:
          best_val_epoch = epoch
          best_val_acc = EM_test
          best_val_mem_acc = confusion_mem_valid.accuracy()
          best_val_gorodkin = Hamming_loss_test
          best_val_mcc = confusion_mem_valid.MCC()
          best_prediction = predicted_test
          related_targets = targets_test
          best_F1_value = F1_value
          best_postpredictions = total_modeloutput
          save_model(model, self.args, index=i)

        if best_val_acc > self.results.best_val_acc:
          self.results.best_val_acc = best_val_acc

        print('-' * 24, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, self.args.epochs-1, time.time() - start_time), '-' * 25 )
        print('| Train | loss {:.4f} | F1 Score {:.2f}% | Exact Match {:.2f}% | mem_acc {:.2f}% | Hamming Loss {:2.4f} | MCC {:2.4f}'
              ' |'.format(train_loss, f1_train*100, EM_train*100, confusion_mem_train.accuracy()*100, Hamming_loss_train, confusion_mem_train.MCC()))
        print('| Valid | loss {:.4f} | F1 Score {:.2f}% | Exact Match {:.2f}% | mem_acc {:.2f}% | Hamming Loss {:2.4f} | MCC {:2.4f}'
              ' |'.format(val_loss, F1_value*100, EM_test*100, confusion_mem_valid.accuracy()*100, Hamming_loss_test, confusion_mem_valid.MCC()))
        print('-' * 84)
      
        sys.stdout.flush()
        torch.cuda.empty_cache()

      
      print(f'best_prediction.size() {best_prediction.size()}')
      print(f'best_postpredictions.size() {best_postpredictions.size()}')
      
      torch.save(best_prediction,f'./hpc-tensors/Best_prediction_{i}_{uniq_run_number}.pt')
      torch.save(related_targets,f'./hpc-tensors/Targes_{i}_{uniq_run_number}.pt')
      torch.save(best_postpredictions,f'./hpc-tensors/Best_postpredictions_{i}_{uniq_run_number}.pt')

      print('|', ' ' * 17, 'Best F1 score : {:.2f}% found after {:3d} epochs'.format(best_F1_value*100, best_val_epoch), ' ' * 17, '|')
      print('|', ' ' * 17, 'Exact Match accuracy : {:.2f}% found after {:3d} epochs'.format(best_val_acc*100, best_val_epoch), ' ' * 17, '|')
      print('|', ' '* 17, 'Mem acc {:.2f}% | Hamming Loss {:2.4f} | MCC {:2.4f}'.format(best_val_mem_acc*100, best_val_gorodkin, best_val_mcc) , ' '*16, '|' )
      print('-' * 84)
      best_val_accs.append(best_val_acc)
      best_val_mem_accs.append(best_val_mem_acc)
      best_val_gorodkins.append(best_val_gorodkin)
      best_val_mccs.append(best_val_mcc)
      best_val_F1.append(best_F1_value)
      # Make a function that outputs the best model for this round or train/test

    for i, _ in enumerate(best_val_accs):
      print("Partion {:1d} : F1 score {:.2f}% | Exact Match accuracy {:.2f}% | mem_acc {:.2f}% | Hamming Loss {:2.4f} | MCC {:2.4f}".format(
        i, best_F1_value*100, best_val_accs[i]*100, best_val_mem_accs[i]*100, best_val_gorodkins[i], best_val_mccs[i]))
      print("\nAverage is: F1 score {:.2f}% | Exact Match accuracy {:.2f}% | mem_acc {:.2f}% | Hamming Loss {:2.4f} | MCC {:2.4f} \n".format(
      sum(best_val_F1)/len(best_val_F1)*100, sum(best_val_accs)/len(best_val_accs)*100, sum(best_val_mem_accs)/len(best_val_accs)*100, sum(best_val_gorodkins)/len(best_val_accs), sum(best_val_mccs)/len(best_val_accs)))


  def tester(self):
    (X_test, y_test, mask_test, mem_test, unk_test) = self.testdata

    best_models = []
    for i in range(1,5):
      model = self.Model(self.args).to(self.args.device)
      load_model(model, self.args, index=i)
      model.eval()
      best_models.append(model)
    
    val_loss, confusion_test, confusion_mem_test,\
    (alphas, targets, seq_lengths, exact_match, HammingLoss, preds, targets)\
     = self.run_test(best_models, X_test, y_test, mask_test, mem_test, unk_test)

    print("ENSAMBLE TEST RESULTS")
    print(confusion_test)
    print(confusion_mem_test)
    print("test Exact Match accuracy:\t\t{:.2f} %".format(exact_match * 100))
    print("test Hamming Loss:\t\t{:.4f}".format(HammingLoss))
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
