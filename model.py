
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import os,sys, random, time




# 3x3 Convolution
def conv2d(in_channels, out_channels ,kernel_size ,stride , padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding= padding, bias=False)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.relu1 = nn.ReLU()
        self.conv1 = conv2d(in_channels, out_channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv2 = conv2d(out_channels, out_channels, kernel_size=1 , stride=1, padding = 0)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        
    def forward(self, x):
        residual = x
        
 
        out = self.relu1(x)
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu2(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        
        
        out += residual

        return out

# ResNet Module
class Encoder_vqvae(nn.Module):
    def __init__(self, block , in_channels , out_channels):
        super(Encoder_vqvae, self).__init__()
        
        self.in_channels = in_channels
        d = out_channels 
        
        self.conv1 = conv2d(self.in_channels, d , kernel_size = 4 , stride = 2)
        self.bn1 = nn.BatchNorm2d(d)
        self.relu1 = nn.ReLU()
        
        self.conv2 = conv2d(d, d , kernel_size = 4 , stride = 2)
        self.bn2 = nn.BatchNorm2d(d)
        self.relu2 = nn.ReLU()
        
        
        
        self.residual_block_1 = block(d , d, stride=1)
        
        self.residual_block_2 = block(d , d, stride=1)
        
        
        
        
        #layers.append(block(d , d, stride=1))
        
        
        #self.combined_residual_blocks = nn.Sequential(*layers)


    
    def forward(self, x):
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu2(out)
        
        out = self.residual_block_1(out)
        out = self.residual_block_2(out)
        
        
        return out

    
# ResNet Module
class Decoder_vqvae(nn.Module):
    def __init__(self, block , in_channels , out_channels):
        super(Decoder_vqvae, self).__init__()
        
        d = in_channels 
        self.out_channels = out_channels
        
        
        
        
        
        layers = []
        layers.append(block(d , d, stride=1))
        layers.append(block(d , d, stride=1))
        
        
        self.combined_residual_blocks = nn.Sequential(*layers)


        
        
        
        
        self.deconv1 = nn.ConvTranspose2d( d , d, kernel_size = 4 , stride = 2 , padding=1 )
        self.bn1 = nn.BatchNorm2d(d)
        self.relu1 = nn.ReLU()
        
        self.deconv2 = nn.ConvTranspose2d(d, out_channels , kernel_size = 4 , stride = 2 , padding=1)
        self.bn2 = nn.BatchNorm2d(d)
        self.sigmoid = nn.Sigmoid()
        
        
    
    def forward(self, x):
        
        
       
        out = self.combined_residual_blocks(x)
        out = self.deconv1(out)
        #out = self.bn1(out)
        out = self.relu1(out)
        out = self.deconv2(out)
        #out = self.bn2(out)    
        out = self.sigmoid(out)
        
     
        return out


    


class VQ_VAE(nn.Module):
    """Vector Quantised Variational Auto-Encoder by Aaron Van der Oord"""
    def __init__(self, image_size=32, d_dim=256, k_dim= 20 , batch_size = 128):
        super(VQ_VAE, self).__init__()

        self.k_dim = k_dim
        self.d_dim = d_dim


        self.embed = nn.Embedding(self.k_dim, self.d_dim)

        #output_channels = 256

        self.encoder = Encoder_vqvae(ResidualBlock , 3, d_dim)
        print(self.encoder)


        self.decoder = Decoder_vqvae(ResidualBlock , d_dim , 3)

        print (self.decoder)

        
        self.init_weights()        
        
    def init_weights(self):
        initrange = 1.0 / self.k_dim
        self.embed.weight.data.uniform_(-initrange, initrange)        
            





    def l2(self, x,y):
#         print(" x shape " + str(x.size()))
#         print(" y shape " + str(y.size()))
        
        return  ((x-y) ** 2)




    """ https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/2 for stopgradient"""    


    def forward(self, x):
        

        z_e = self.encoder(x)                             
        sz =  z_e.size() ### B X C x H x W
        org_z = z_e
        
        
        
        
 
        
              
        z_e = z_e.permute(0, 2, 3, 1).contiguous() ## should be B H W C  64 8 8 256
        new_sz =  z_e.size()
        
        """ refer to the following discussion https://discuss.pytorch.org/t/swap-axes-in-pytorch/970 and https://github.com/fxia22/stn.pytorch/issues/7 """
        
        
        
        new_z = z_e.unsqueeze(-2).expand(sz[0] , sz[2] , sz[3] , self.k_dim , sz[1])  ## B H W K C .   64 8 8 20 256
     
        
    
        w = self.embed.weight        
        
        
        w_reshape = w.view(1,1,1,self.k_dim , self.d_dim)
        
        
        
        mins, min_index = torch.norm( new_z - w_reshape , p = 2 , dim=-1).min(dim=-1)
        

    
    
        ## TO DO remove this ugly methods of slicing from another variable.
        
        
        
        
        z_q = w.index_select(0, min_index.long().view(-1)).view(new_sz)
        
        
     

        z_e_sg = new_z.detach()  ## B x H x W x K x C
        z_q_sg = z_q.detach()   ##  B x  H x W x C
        
            
          
         
        z_e = z_q.permute(0,3,1,2)
     
        def hook(grad):
            nonlocal org_z
            
            self.saved_grad = grad     ## copying gradients from decoder input to encoder output
            self.saved_h = org_z
            return grad

        z_e.register_hook(hook)
        
        z_e = self.decoder(z_e)
        
       
        return z_e, self.l2(new_z.permute(3,0,1,2,4), z_q_sg).sum(-1).mean(), self.l2(z_e_sg.permute(3,0,1,2,4),z_q).sum(-1).mean()
    
    
    
    # back propagation for encoder part where we flow the saved gradients of the decoder through the encoder
    def bwd(self):
        self.saved_h.backward(self.saved_grad)
        
    def decode(self, z):

        return self.decoder(z)


