import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import sys
import time
sys.path.insert(0,'..')
import datautils.data as data
from models.seqpred_model import SeqPred 

num_epochs = 5
clip_norm = 0.2

def train():
    train_err = 0
    train_batches = 0
    accuracy = 0
    model.train()
    for idx, batch in enumerate(data_gen.gen_train()):
        seq_lengths = batch['mask'].sum(1).astype(np.int32)
        #sort to be in decending order for pad packed to work
        perm_idx = np.argsort(-seq_lengths)
        seq_lengths = seq_lengths[perm_idx]
        inputs = batch['X'][perm_idx]
        targets = batch['t'][perm_idx]
        mask = batch['mask'][perm_idx]
        inp = Variable(torch.from_numpy(inputs).type(torch.float))
        seq_lens = Variable(torch.from_numpy(seq_lengths).type(torch.int32))
        mask = Variable(torch.from_numpy(mask).type(torch.float))
        targets = Variable(torch.from_numpy(targets).type(torch.long))

        optimizer.zero_grad()
        output = model(inp=inp, seq_lengths=seq_lens)
    
        # calculate loss
        loss = 0
        loss_preds = output.permute(1,0,2)
        loss_mask = mask.permute(1,0)
        loss_targets = targets.permute(1,0)
        for i in range(loss_preds.size(0)):
            loss += sum(F.nll_loss(loss_preds[i], loss_targets[i], reduction='none') * loss_mask[i])/sum(loss_mask[i])
        loss.backward()

        accuracy += calculate_accuracy(preds=output, targets=targets, mask=mask)

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        train_err += loss.item()
        train_batches += 1

    total_accuracy = accuracy / train_batches
    train_loss = train_err / train_batches
    return train_loss, total_accuracy

def calculate_accuracy(preds, targets, mask):
    preds = preds.argmax(2).type(torch.float)
    correct = preds.eq(targets.type(torch.float)).type(torch.float) * mask
    return torch.sum(correct) / torch.sum(mask)
    

# Network compilation
model = SeqPred()
print("Model: ", model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
data_gen = data.gen_data(num_iterations=5001, batch_size=64)

for epoch in range(num_epochs):
    start_time = time.time()

    train_loss, train_accuracy = train()
    #val_loss, val_accuracy = evaluate()

    print('-' * 22, ' epoch: {:3d} / {:3d} - time: {:5.2f}s '.format(epoch, num_epochs, time.time() - start_time), '-' * 22 )
    print('| Train | loss {:.4f} | acc {:.2f}%' 
            ' |'.format(train_loss, train_accuracy))
    #print('| Valid | loss {:.4f} | acc {:.2f}% ' 
    #        ' |'.format(val_loss, valid_accuracy))
    print('-' * 79)

sys.stdout.flush()




