from operator import index
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_batch(path_dir: str, batch_size:int =128)-> Tuple:

    """
    Return a tuple of tensors batches which will be downloads FashionMNIST dataset from pytorch
    trainloader and testloader: containing the training images and the labels of the dataset

    Parameters:
    ----------

    path_dir: str, Required
        Is the path where the dataset will be downloaded on your working directory

    batch_size: int, default set as (64)
        This determine the number of dataset in each batch for the training
    """
    # Define a transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
    
    # Download and load the training data
    trainset = datasets.FashionMNIST(path_dir, download=True, train=True, 
                                        transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Download and load the test data
    testset = datasets.FashionMNIST(path_dir, download=True, train=False, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)


    return trainloader, testloader



def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))

    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)

    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')

    return ax


def view_classify(img, ps, version="MNIST"):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 28, 28).numpy().squeeze())
    ax1.axis('off')
    ax2.barh(np.arange(10), ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    if version == "MNIST":
        ax2.set_yticklabels(np.arange(10))
    elif version == "Fashion":
        ax2.set_yticklabels(['T-shirt/top',
                            'Trouser',
                            'Pullover',
                            'Dress',
                            'Coat',
                            'Sandal',
                            'Shirt',
                            'Sneaker',
                            'Bag',
                            'Ankle Boot'], size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)

    plt.tight_layout()


