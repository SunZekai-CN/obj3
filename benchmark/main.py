from models import *
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import pyfiglet
import click
from datetime import datetime
import json
import numpy as np
trainset = datasets.QMNIST("", train=True, download=True, transform=(transforms.Compose([transforms.ToTensor()])))

testset = datasets.QMNIST("", train=False, download=True, transform=(transforms.Compose([transforms.ToTensor()])))

test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=12, shuffle=True)

def hogwild(model_class, workers, epochs, arch, distributed, nodes, batches,order,add):

    torch.set_num_threads(nodes)
    
    device = torch.device("cpu")
    
    model = model_class.to(device)

    tag=mp.Manager()
    value_table=tag.dict()
    training_time=tag.dict()

    if distributed=='y':

        processes = []

        for rank in range(workers):
            
            #mp.set_start_method('spawn')

            model.share_memory() 

            train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batches, sampler=DistributedSampler(dataset=trainset,num_replicas=workers,rank=rank))

            p = mp.Process(target=train, args=(epochs, arch, model, device, train_loader,value_table,order,add,training_time))

            p.start()

            processes.append(p)

        for p in processes:
            p.join()
        times = []
        for key,value in training_time.items():
            times.append(value)
        tag=np.array(times)
        click.echo(f'Training: sum = {sum(tag)} , average = {np.mean(tag)} , max = {max(times)} , min = {min(times)} , median = {np.median(tag)}')
        test(model, device, test_loader, arch)

    else:
        
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batches, shuffle=True)

        train(epochs, arch, model, device, train_loader)

        test(model, device, test_loader, arch)


def ff_train(arch, epochs, workers, distributed, nodes, batches,order,add):
    click.echo(f'Training neural {arch}-net with {epochs} epochs using {workers} workers and a batch size of {batches}, update paramter in order = {order}, add = {add}')
    model_class = FeedforwardNet()
    hogwild(model_class, workers, epochs, arch, distributed, nodes, batches,order,add)

def conv_train(arch, epochs, workers, distributed, nodes, batches,order,add):    
    click.echo(f'Training neural {arch}-net with {epochs} epochs using {workers} workers and a batch size of {batches}, update paramter in order = {order}, add = {add}')
    model_class = ConvNet()
    hogwild(model_class, workers, epochs, arch, distributed, nodes, batches,order,add)

@click.command()
@click.option('--epochs', default=1, help='number of epochs to train neural network.')
@click.option('--arch', default='ff', help='neural network architecture to benchmark (conv or ff).')
@click.option('--distributed', default='y', help='whether to distribute data or not (y or n).')
@click.option('--workers', default=1, help='number of workers.')
@click.option('--nodes', default=1, help='number of cores to use.')
@click.option('--batches', default=12, help='minibatch size to use.')
@click.option('--order', default='y', help='wether to update paramters in order or not (y or n)')
@click.option('--add', default=0.001, help='number to add while update paramters fail')
def main(epochs, arch, workers, distributed, nodes, batches,order,add):
    
    #print("start training...")
  
    if arch == 'ff':            
        ff_train(arch, epochs, workers, distributed, nodes, batches,order,add)
    
    elif arch == 'conv':
        conv_train(arch, epochs, workers, distributed, nodes, batches,order,add)

    
    #print("finish training...")
    
if __name__ == "__main__":
    main()