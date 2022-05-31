import click
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


root_dir = './testdata' 
test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5],
                                                           [0.5, 0.5, 0.5])])

# Pass transforms in here, then run the next cell to see how the transforms look
test = datasets.ImageFolder(root_dir + '/test', transform=test_transforms)
testloader = DataLoader(test, batch_size=32, shuffle=True)

images, labels = next(iter(testloader))


@click.group()
def cli():
    pass

@click.command()
def showrawimage():
    click.echo(f'Your model will be test with {test.classes}')
    
@click.command()
def classify():
    click.echo()
    


cli.add_command(showrawimage)
cli.add_command(classify)






if __name__ == '__main__':
    cli()







