from cProfile import label

import matplotlib.pyplot as plt
import data_handler as dh
from model import FashionNetwork
import torch
import torch.nn as nn
from torch import optim 
import torch.nn.functional as F 
torch.manual_seed(0)

train_handler, test_handler=dh.load_batch('~/.pytorch/F_MNIST_data/')

model=FashionNetwork(784,128,64,32,10)

def torch_fit(train_handler,test_handler,num_epochs,lr,model):
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr)

    norm_score=0.86
    print_every=50
    mean_loss_train=[]
    accuracy_list=[]
    mean_loss_test=[]
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
        print(f'Mean loss value for train: {mean_loss_value}')

    ###for --> test
        model.eval()
        with torch.no_grad():
            equals_list=[]
            loss_test=[]
            accuracy_list_epoch=[]
            for j,(imagest,labelst) in enumerate(iter(test_handler)):
                imagest.resize_(imagest.size()[0],784)
                prob=model(imagest)
                pred=prob.argmax(dim=1)
                test_loss=criterion(prob,labelst)
                acc=(pred == labelst).sum()/len(labelst)
                accuracy_list_epoch.append(acc)
            
                loss_test.append(test_loss)
            mean_loss_test_value=sum(loss_test)/len(loss_test)
            mean_loss_test.append(mean_loss_test_value)
            mean_acc=sum(accuracy_list_epoch)/len(accuracy_list_epoch)
            accuracy_list.append(mean_acc)
            print(f'Mean loss value for test: {mean_loss_test_value}')
            print(f'Accuracy: {mean_acc}')
        
            if norm_score < acc:
                torch.save(model,'model.pth')
                norm_score = acc

        model.train()


    plt.plot(mean_loss_train, labels='Train loss')
    plt.plot(mean_loss_test, labels='Test loss')
    plt.plot(accuracy_list, labels='accuracy')
    plt.legend()
    plt.show()

ans=torch_fit(train_handler=train_handler,test_handler=test_handler,num_epochs=10,lr=0.001,model=model)



    




