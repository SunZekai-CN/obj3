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

def hogwild(model_class, workers, epochs, arch, distributed, batches,order,timeout,update_rate):

    torch.set_num_threads(1)
    
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

            p = mp.Process(target=train, args=(epochs, arch, model, device, train_loader,value_table,order,timeout,update_rate,training_time))

            p.start()

            processes.append(p)

        for p in processes:
            p.join()
        times = []
        need_update=0
        updated=0
        for key,value in training_time.items():
            times.append(value['training_time'])
            need_update+=value['need_update']
            updated+=value['updated']
        tag=np.array(times)
        click.echo(f'Training: sum = {sum(tag)} , average = {np.mean(tag)} , max = {max(times)} , min = {min(times)} , median = {np.median(tag)}')    
        print('need to update {} cases, successfully updated {} cases, rate is {:.2f}'.format( need_update, updated,100. * updated/need_update))
        test(model, device, test_loader, arch)
        return updated/need_update
    else:
        
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batches, shuffle=True)

        train(epochs, arch, model, device, train_loader)

        test(model, device, test_loader, arch)


def ff_train(arch, epochs, workers, distributed, batches,order,timeout,update_rate):
    click.echo(f'Training neural {arch}-net with {epochs} epochs using {workers} workers and a batch size of {batches}, update paramter in order = {order}, timeout = {timeout}')
    model_class = FeedforwardNet()
    return hogwild(model_class, workers, epochs, arch, distributed, batches,order,timeout,update_rate)

def conv_train(arch, epochs, workers, distributed, batches,order,timeout):    
    click.echo(f'Training neural {arch}-net with {epochs} epochs using {workers} workers and a batch size of {batches}, update paramter in order = {order}, timeout = {timeout}')
    model_class = ConvNet()
    hogwild(model_class, workers, epochs, arch, distributed, batches,order,timeout)

@click.command()
@click.option('--epochs', default=1, help='number of epochs to train neural network.')
@click.option('--arch', default='ff', help='neural network architecture to benchmark (conv or ff).')
@click.option('--distributed', default='y', help='whether to distribute data or not (y or n).')
@click.option('--workers', default=1, help='number of workers.')
@click.option('--batches', default=12, help='minibatch size to use.')
@click.option('--timeout', default=10.0, help='tiemout for each update')
def main(epochs, arch, workers, distributed, batches,timeout):
    
    #print("start training...")

    if arch == 'ff': 
        update_rate=1.0 
        update_rate=ff_train(arch, epochs, workers, distributed, batches,'y',timeout,update_rate)
        ff_train(arch, epochs, workers, distributed, batches,'n',timeout,update_rate)

    
    elif arch == 'conv':
        conv_train(arch, epochs, workers, distributed, batches,order,timeout)

    
    #print("finish training...")
    
if __name__ == "__main__":
    main()