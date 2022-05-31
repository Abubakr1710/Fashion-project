from cProfile import label

import matplotlib.pyplot as plt
import data_handler as dh
from model import FashionNetwork
import torch
import torch.nn as nn
from torch import optim 


train_handler, test_handler=dh.load_batch('~/.pytorch/F_MNIST_data/')

model=FashionNetwork(784, 400,200,100,10)

def torch_fit(train_handler,num_epochs,lr,model):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr)

    print_every=50
    mean_loss_train=[]
    for epoch in range(num_epochs):
        print(f"Epoch: {epoch+1}/{num_epochs}")

        loss_val_train=[]
        for i, (images, labels) in enumerate(iter(train_handler)):
            images.resize_(images.size()[0], 784)

            optimizer.zero_grad()

            pred=model.forward(images)
            loss=criterion(pred, labels)
            loss_val_train.append(loss.item())

            loss.backward()
            optimizer.step()

        mean_loss_value=sum(loss_val_train)/len(loss_val_train)
        mean_loss_train.append(mean_loss_value)
        print(f'Mean loss value for epoch: {mean_loss_value}')

    plt.plot(mean_loss_train)
    plt.show()

ans=torch_fit(train_handler=train_handler,num_epochs=10,lr=0.001,model=model)



    




