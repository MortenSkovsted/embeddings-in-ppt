import numpy as np
import random
import torch
from collections import OrderedDict
from torch.autograd import Variable
from utils.data_utils import tokenize_sequence

def iterate_minibatches(inputs, targets, masks, targets_mem, unk_mem, batchsize, shuffle=True, sort_len=True, sample_last_batch=True):
  """ Generate minibatches of a specific size 
  Arguments:
    inputs -- numpy array of the encoded protein data. Shape: (n_samples, seq_len, n_features)
    targets -- numpy array of the targets. Shape: (n_samples,)
    masks -- numpy array of the protein masks. Shape: (n_samples, seq_len)
    batchsize -- integer, number of samples in each minibatch.
    shuffle -- boolean, shuffle the samples in the minibatches. (default=False)
    sort_len -- boolean, sort the minibatches by sequence length (faster computation, just for training). (default=True) 
  Outputs:
  list of minibatches for protein sequences, targets and masks.

  """ 
  assert len(inputs) == len(targets)

  # Calculate the sequence length of each sample
  len_seq = np.apply_along_axis(np.bincount, 1, masks.astype(np.int32))[:,-1]

  # Sort the sequences by length
  if sort_len:
    indices = np.argsort(len_seq) #[::-1][:len(inputs)] #sort and reverse to get in decreasing order
  else:
    indices = np.arange(len(inputs))

  # Shuffle the data of 3 batches together to avoid identical batches
  if shuffle:
    bucketsize = batchsize*3
    num_buckets = len(inputs) // bucketsize
    for i in range(num_buckets):
      start = i*bucketsize
      end = (i+1)*bucketsize
      random.shuffle(indices[i*bucketsize:(i+1)*bucketsize])
    random.shuffle(indices[num_buckets*bucketsize:])

  # Generate minibatches list
  f_idx = len(inputs) % batchsize
  idx_list = list(range(0, len(inputs) - batchsize + 1, batchsize))
  last_idx = None
  if f_idx != 0 and sample_last_batch:
    last_idx = idx_list[-1] + batchsize
    idx_list.append(last_idx)

  # Shuffle the minibatches
  if shuffle:
    random.shuffle(idx_list)

  # Split the data in minibatches
  for start_idx in idx_list:
    if start_idx == last_idx:
      rand_samp = batchsize - f_idx
      B = np.random.randint(len(inputs),size=rand_samp)
      excerpt = np.concatenate((indices[start_idx:start_idx + batchsize], B))
    else:
      excerpt = indices[start_idx:start_idx + batchsize]
    max_prot = np.amax(len_seq[excerpt])

    # Crop batch to maximum sequence length
    if sort_len:
      in_seq = inputs[excerpt][:,:max_prot]
      in_mask = masks[excerpt][:,:max_prot]
    else:
      in_seq = inputs[excerpt][:,:max_prot]
      in_mask = masks[excerpt][:,:max_prot]

    in_target = targets[excerpt]
    in_target_mem = targets_mem[excerpt]
    in_unk_mem = unk_mem[excerpt]
    shuf_ind = np.arange(batchsize)

    # Return a minibatch of each array
    yield in_seq[shuf_ind], in_target[shuf_ind], in_mask[shuf_ind], in_target_mem[shuf_ind], in_unk_mem[shuf_ind]

def load_data(train_path, test_path, is_raw):
  # Load data
  print("Loading data...")
  test_data = np.load(test_path)
  train_data = np.load(train_path)

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
  if (is_raw):
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