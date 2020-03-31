import torch
import torch.nn as nn

class MyBCEWihLogitLoss(nn.Module):
    def __init__(self, balanced):
        super(MyBCEWihLogitLoss, self).__init__()
        self.balanced = balanced