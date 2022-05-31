import torch
import torch.nn as nn
torch.manual_seed(0)

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, num_hidden1, num_hidden2, num_hidden3, output_size):
        super().__init__()
        self.linear1=nn.Linear(input_dim, num_hidden1)
        self.linear2=nn.Linear(num_hidden1, num_hidden2)
        self.linear3=nn.Linear(num_hidden2, num_hidden3)
        self.logits=nn.Linear(num_hidden3, output_size)
        self.dropout=nn.Dropout(0.2)
        self.batchnorm1=nn.BatchNorm1d(num_hidden1)
        self.batchnorm2=nn.BatchNorm1d(num_hidden2)
        self.batchnorm3=nn.BatchNorm1d(num_hidden3)
        self.relu=nn.ReLU()

    def forward(self,x):
        layer1=self.linear1(x)
        act1=self.dropout(self.batchnorm1(self.relu(layer1)))
        layer2=self.linear2(act1)
        act2=self.dropout(self.batchnorm2(self.relu(layer2)))
        layer3=self.linear3(act2)
        act3=self.dropout(self.batchnorm3(self.relu(layer3)))
        out=self.logits(act3)

        return out



