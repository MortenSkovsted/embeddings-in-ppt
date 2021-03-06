import numpy as np
import random
import dataloaders.secpred as data
subcel = False
if subcel:
    train_data = np.load("data/Deeploc/train.npz")
    y_train = train_data['y_train']
    partition = train_data['partition']   
else:
    data_dict, _ = data.load_data(is_raw=True, train_path="data/SecPred_raw/train_raw.npz", test_path="data/SecPred_raw/test_raw.npz")
    y_tr = data_dict["t_test"].astype(np.int32)
    seq_len = data_dict["length_test"].astype(np.int32)

def getUnigram(targets):
    num_targets = len(targets)
    unigramPreds = []
    for i in range(0, num_targets):
        if subcel:
            x = random.randint(0, num_targets-1)
            unigramPreds.append(targets[x])
            #unigramPreds.append(0)
        else:
            pred = []
            for j in range(seq_len[i]):
                x = random.randint(0, num_targets-1)
                y = random.randint(0, seq_len[x]-1)
                pred.append(targets[x][y])
                #pred.append(5)
            unigramPreds.append(pred)

    return unigramPreds

def computeAccuracy(targets, preds):
    assert len(targets) == len(preds)
    hits = 0
    if subcel:
        for i in range(len(targets)):
            if targets[i] == preds[i]:
                hits += 1
        return hits/len(targets)*100
    else:
        total = 0
        for i in range(len(preds)):
            for j in range(len(preds[i])):
                total += 1
                if targets[i][j] == preds[i][j]:
                    hits += 1
        return hits/total*100

if __name__ == "__main__":
    if subcel:
        print("Unigram Model")
        for i in range(1,5):
            train_index = np.where(partition != i)
            y_tr = y_train[train_index].astype(np.int32)
            print("Partition: ", i)
            print("Distribution: ", np.bincount(y_tr))
            preds = getUnigram(y_tr)
            print("Accuracy: ", computeAccuracy(y_tr, preds))
    else:
        preds = getUnigram(y_tr)
        print("Accuracy: ", computeAccuracy(y_tr, preds))

        
    