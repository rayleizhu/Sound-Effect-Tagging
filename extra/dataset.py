import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class MFCCDataset(Dataset):
    def __init__(self, mfccs, labels, transform=None):
        '''
        mfccs: a length b list of mfcc matrix, each matrix has shape (seq_len, 13),
        labels: a numpy array of labels, shape (b, num_classes)
        '''
        self.mfccs = [ torch.from_numpy(x) for x in mfccs ]
        self.labels = torch.from_numpy(labels)
        self.transform = transform
        
    def __getitem__(self, index):
        
        mfcc = self.mfccs[index]
        label = self.labels[index]
        
        if self.transform is not None:
            mfcc = self.transform(mfcc)
     
        return mfcc, label
    
    def __len__(self):
        return len(self.mfccs)

    
class MyPadCollate(object):
    def __init__(self, batch_first=False):
        self.batch_first = batch_first
        
    def __call__(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        xs, ys = list(zip(*batch))
        xs = pad_sequence(xs,
                         batch_first=self.batch_first)
        ys = torch.stack(ys)
        return xs, ys
  