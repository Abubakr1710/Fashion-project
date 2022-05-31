import data_handler as dh
from model import FashionNetwork
import torch
import torch.nn as nn
from torch.optim import optimizer  
 
train_handler, test_handler=dh.load_batch('~/.pytorch/F_MNIST_data/')


