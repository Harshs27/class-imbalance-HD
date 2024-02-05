"""
Utils for neural graphical models
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F


class DNN(torch.nn.Module):
    """The DNN architecture to map the input to input.
    """
    def __init__(self, I, H, O, model_type='MLP', USE_CUDA=False):
        """Initializing the MLP for the regression 
        network.

        Args:
            I (int): The input dimension
            H (int): The hidden layer dimension
            O (int): The output layer dimension
            USE_CUDA (bool): Flag to enable GPU
        """
        super(DNN, self).__init__() # init the nn.module
        self.dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
        self.I, self.H, self.O = I, H, O
        if model_type=='MLP': self.MLP = self.getMLP()

    def getMLP(self):
        l1 = nn.Linear(self.I, self.H).type(self.dtype)
        l2 = nn.Linear(self.H, self.H).type(self.dtype)
        # l3 = nn.Linear(self.H, self.H).type(self.dtype)
        # l4 = nn.Linear(self.H, self.H).type(self.dtype)
        # l5 = nn.Linear(self.H, self.H).type(self.dtype)
        # l6 = nn.Linear(self.H, self.H).type(self.dtype)
        # l7 = nn.Linear(self.H, self.H).type(self.dtype)
        # l8 = nn.Linear(self.H, self.H).type(self.dtype)
        l9 = nn.Linear(self.H, self.H).type(self.dtype)
        l10 = nn.Linear(self.H, self.O).type(self.dtype)
        return nn.Sequential(
            l1, nn.ReLU(), #nn.Tanh(), #,
            l2, nn.ReLU(), #nn.Tanh(), #nn.ReLU(), #nn.Tanh(),
            # l3, nn.ReLU(),
            # l4, nn.ReLU(),
            # l5, nn.ReLU(), 
            # l6, nn.ReLU(),
            # l7, nn.ReLU(),
            # l8, nn.ReLU(), 
            l9, nn.ReLU(),
            l10,# nn.ReLU(), 
            nn.Sigmoid()
            ).type(self.dtype)


def get_optimizers(model, lr=0.002, use_optimizer='adam'):
    if use_optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr, 
            betas=(0.9, 0.999),
            eps=1e-08,
            # weight_decay=0
        )
    else:
        print('Optimizer not found!')
    return optimizer
