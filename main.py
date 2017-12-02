
import argparse
import torch,math
from torch.optim.lr_scheduler import ReduceLROnPlateau , StepLR
import pickle
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
from torch.utils import data
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import os,sys,random,time,glob

from tqdm import tqdm

from visdom import Visdom
import numpy as np



from model import VQ_VAE





from PIL import Image








# Training settings
parser = argparse.ArgumentParser(description='PyTorch VQ-VAE')

# Model hyper-parameters
parser.add_argument('--image_size', type=int, default=32)
parser.add_argument('--d_dim', type=int, default=256)
parser.add_argument('--k_dim', type=int, default=256)




# Training settings
parser.add_argument('--num_epoch', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--vq_beta', type=float, default=0.25)



# Misc
parser.add_argument('--sample_size', type=int, default=8)
parser.add_argument('--model_save_path', type=str, default='./train_4_outputs')
parser.add_argument('--sample_path', type=str, default='./train_4_outputs')



parser.add_argument('--image_path', type=str, default= './caltech256/_original/256_ObjectCategories/' )
parser.add_argument('--log_step', type=int, default=10)
parser.add_argument('--step_for_sampling', type=int, default=350)


parser.add_argument('-n','--exp_number',  type=int, default=random.randint(10,100), help='experiment case number')






##################################
# Load data and create Data loaders
##################################
def load_data(config, mnist = False , cifar = True):
    
    """Create and return Dataloader."""
    
    print('loading data!')
    transform = transforms.Compose([
                    transforms.Scale((config.image_size, config.image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    kwargs = {'num_workers': 1, 'pin_memory': True}
    
    
    if(mnist):
        
        train_loader = data.DataLoader(
            dset.MNIST('./data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=config.batch_size, shuffle=True, **kwargs)
        
        test_loader = data.DataLoader(
            dset.MNIST('./data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),batch_size= config.batch_size, shuffle=True, **kwargs)
    
        return train_loader

        
    elif (cifar):
        # Data with cifar 10
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        trainset = dset.CIFAR10(root='./data/cifar10/train', train=True, download=True, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size = config.batch_size, shuffle=True, num_workers=1)

        testset = dset.CIFAR10(root='./data/cifar10/test', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size = config.batch_size, shuffle=False, num_workers=1)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


        return trainloader
        

    else:
        dataset = dset.ImageFolder(config.image_path, transform)
        print("Loaded images")
        print(len(dataset))
        img, target = dataset[3] # load 4th sample
        print("Image Size: ", img.size())

        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size= config.batch_size,
                                      shuffle=True,
                                      **kwargs)

        return data_loader
















def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)







##################################
# Setting up the Model 
##################################







def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)
 


def sample(x , epoch_load, config, model):
    
   
    z = to_var(torch.randn(config.batch_size , config.d_dim ,config.sample_size, config.sample_size )) 
   
    save_path = os.path.join ( config.sample_path ,  "real_epoch_" +str(epoch_load)+"_.png")
    save_image(denorm(x.data), save_path)
    #torchvision.utils.save_image( I2, 'test.jpg' )
    
    #vis.images(denorm(x.data))
    
    
    
    model.eval()
    S = torch.load('%s/vqvae_epoch_%d.pth' % (config.model_save_path, epoch_load))
                              
    
    

                              
    model.load_state_dict(S['model'])

    reconst, _, _ = model(x)
    
    save_path = os.path.join ( config.sample_path ,  "reconstructed_epoch_" +str(epoch_load)+"_.png")
    save_image(denorm(reconst.data), save_path)
    #vis.images(denorm(reconst.data))
    
    
    
    fake = model.decode(z)
    save_path = os.path.join ( config.sample_path ,  "fake_epoch_" +str(epoch_load)+"_.png")
    save_image(denorm(fake.data), save_path)
    #vis.images(denorm(fake.data))


    


def train(config,data_loader):
    
    # model and optimizer
 
    
    model = VQ_VAE(config.image_size, d_dim = config.d_dim,   k_dim = config.k_dim , batch_size = config.batch_size )
    
    #model = torch.nn.DataParallel(model)

    print(model)
    

    g_optimizer = torch.optim.Adam(model.parameters(), config.lr, [config.beta1, config.beta2])
    scheduler = StepLR( g_optimizer, step_size=30, gamma=0.1)
    if torch.cuda.is_available():
        model= model.cuda()
    
    
    reconst_loss = nn.L1Loss()
    
    fixed_x = to_var(iter(data_loader).next()[0])
    

    
    for epoch in tqdm(range(config.num_epoch)):
        start_epoch = time.time()
        for i, data in enumerate(tqdm(data_loader, 0)):
            start_iter = time.time()
            
            
            target = Variable( data[0].cuda() , requires_grad = False)
            x = to_var(data[0])
            
            
            out , loss_e1, loss_e2 = model(x)

          
            
            loss_rec = reconst_loss(out, target)
            
            
            loss = loss_rec + loss_e1 + config.vq_beta * loss_e2
            
            loss = torch.sum(loss)
            

            
            
            #sys.exit()
            g_optimizer.zero_grad()

            
            # For decoder
            loss.backward(retain_graph=True )

            # For encoder
            model.bwd()
            g_optimizer.step()
            
            
            
        
        

            end_iter = time.time()
            # Print out log info
            if (i+1) % config.log_step == 0:
                print("[ {} iteration in epoch number : {}/{} ] reconstruction loss: {:.4f}, loss_e1: {:.4f} , loss_e2: {:.4f}  Elapsed {:.2f} s ".format( i+1, epoch, config.num_epoch, loss_rec.data[0], loss_e1.data[0] , loss_e2.data[0] , end_iter - start_iter))
                
                
            if( (i+1) % config.step_for_sampling == 0 and epoch >= 1):
                sample(fixed_x, epoch -1 , config, model)

        
        scheduler.step()
        end_epoch = time.time()
        print("took {:.2f}s to complete last epoch".format(end_epoch - start_epoch))
        
        if ( (epoch) % 1 == 0):
            torch.save({'model':model.state_dict()},  '%s/vqvae_epoch_%d.pth' % (config.model_save_path, epoch) )

            
            
def make_d(directory):    
    if not os.path.exists(directory):
        os.makedirs(directory)

            

            
            
            
            

            
if __name__ == '__main__':

    config = parser.parse_args()
    cuda = torch.cuda.is_available()
    
    
    config.sample_path = os.path.join(config.sample_path , str(config.exp_number) + "_experiment/samples/" )
    config.model_save_path = os.path.join(config.model_save_path , str(config.exp_number) + "_experiment/model/" )
    
                                      
    make_d(config.sample_path)
    make_d(config.model_save_path)
    
    
    
    

    print(config)
    seed = 10
    train_labeled_loader = load_data(config)

    train(config,train_labeled_loader)
    

 


    
    
