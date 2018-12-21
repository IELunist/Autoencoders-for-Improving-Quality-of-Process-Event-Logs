
# coding: utf-8

# **Note:**
# - nan_index_matrix: 1: missing index, 0: available index --> test function
# - avai_index_matrix: 1: available index, 0: missing index --> cost evaluation
# - **avai_index_matrix != 1 - nan_test** because of 0 padding
# - For training: Use avai_train and avai_val
# - For predicting: Use avai_test
# - For testing: Use nan_test

# In[2]:


import importlib
import argparse
import os, sys
import argparse
import pandas as pd
import numpy as np
import pickle


# In[3]:


import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms


# In[4]:


sys.path.insert(0, './../utils/')
from utils import *


# In[29]:


#Define parser
#name = 'bpi_2012'
name = 'bpi_2013'
#name = 'helpdesk'  

parser = {
    'data_dir': '../data/',
    'data_file': name + '.csv',
    'input_dir': '../input/{}/'.format(name),  
    'batch_size' : 16,
    'epochs' : 200,
    'no_cuda' : True,
    'seed' : 7,
    'log_interval' : 1000,
    'z_dim': 10,
    'h_dim': 200,
    'output_size': 10,
    'lr': 0.001,
    'betas': (0.9, 0.999),   
    'lr_decay': 0.95,
}

args = argparse.Namespace(**parser)


# In[30]:


args.cuda = not args.no_cuda and torch.cuda.is_available()


# In[31]:


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual(args.seed)


# In[32]:


kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}


# In[33]:


with open(args.input_dir + 'preprocessed_data.pkl', 'rb') as f:
    #nan_index_matrix = pickle.load(f)
    #f_dim_list = pickle.load(f)
    #s_dim_list = pickle.load(f)
    #t_dim_list = pickle.load(f)
    min_array = pickle.load(f)
    max_array = pickle.load(f)
    c_train = pickle.load(f) #normalized
    avai_train = pickle.load(f) #index vector
    true_train = pickle.load(f) #true values
    c_val = pickle.load(f)
    avai_val = pickle.load(f)
    nan_val = pickle.load(f)
    true_val = pickle.load(f)
    c_test = pickle.load(f)
    avai_test = pickle.load(f)
    nan_test = pickle.load(f)
    true_test = pickle.load(f)
    m_test = pickle.load(f)


# In[34]:


c_train.shape, c_test.shape, avai_train.shape


# In[35]:


train_loader = torch.utils.data.DataLoader(c_train, batch_size=args.batch_size, shuffle=False, num_workers=2)
avai_train_loader = torch.utils.data.DataLoader(avai_train, batch_size=args.batch_size, shuffle=False, num_workers=2)


# # Build model

# ## Define VAE

# In[36]:


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(args.z_dim, args.h_dim) 
        self.fc2 = nn.Linear(args.h_dim, args.h_dim)
        self.fc3 = nn.Linear(args.h_dim, args.output_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return self.fc3(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(c_train.shape[1]*c_train.shape[2], args.h_dim)
        self.fc2 = nn.Linear(args.h_dim, args.h_dim)
        self.fc3 = nn.Linear(args.h_dim, args.output_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        return self.sigmoid(self.fc3(x))


# In[37]:


D = Discriminator()
G = Generator()

if args.cuda:
    D.cuda()
    G.cuda()


# ## Define loss

# In[13]:


# Define loss
recon_function = nn.BCELoss()
recon_function.size_average = False #loss sum of each mini-batch

def loss_function(recon_x, x, mu, logvar):
    #x = recon_x*index_nan_matrix
    BCE = recon_function(recon_x, x)  
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)

    return BCE + KLD


# In[39]:


D_optimizer = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)
G_optimizer = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)


# In[40]:


#Adjust learning rate per epoch: http://pytorch.org/docs/master/optim.html?highlight=adam#torch.optim.Adam
lambda1 = lambda epoch: args.lr_decay ** epoch
D_scheduler = torch.optim.lr_scheduler.LambdaLR(D_optimizer, lr_lambda=[lambda1])
G_scheduler = torch.optim.lr_scheduler.LambdaLR(G_optimizer, lr_lambda=[lambda1])


# # Train model

# In[16]:


def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (c_data, avai_index) in enumerate(zip(train_loader, avai_train_loader)):
        m_data = c_data*avai_index
        
        c_data = Variable(c_data.float())
        m_data = Variable(m_data.float())
        #Transform: np --> Tensor/Variable: tensor --> tensor with wrapper
        #Wraps a tensor and records the operations applied to it.
        #Variable is a thin wrapper around a Tensor object, that also holds the gradient
        if args.cuda:
            c_data = c_data.cuda()
            m_data = m_data.cuda()
            
        optimizer.zero_grad()
        
        recon_data, mu, logvar = model(m_data)
        
        loss = loss_function(recon_data, c_data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        
        # Track performance of each batch
        #if batch_idx % args.log_interval == 0:
        #    print('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #        epoch, batch_idx * len(m_data), len(train_loader.dataset),
        #        100. * batch_idx / len(train_loader),
        #        loss.data[0] / len(m_data)))
    
    # Track performance of each epoch
    print('====> Epoch {}: Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


# In[ ]:


def train(epoch):
    # Sample data
    #X.size(0) = batch_size
    D_losses = AverageValueMeter()
    G_losses = AverageValueMeter()
    
    
    
    for X, _ in train_loader:
        # Create ones_label and zeros_label
        ones_label = Variable(torch.ones(X.size(0)))
        zeros_label = Variable(torch.zeros(X.size(0)))
        
        # Input: z - latent variables, x - input
        z = Variable(torch.randn(X.size(0), Z_dim))
        X = Variable(X.view(-1, 784))

        # Dicriminator forward-loss-backward-update
        G_sample = G(z) # X_fake: generate from Generator
        D_real = D(X)
        D_fake = D(G_sample)
        
        # Calculate loss
        D_loss_real = F.binary_cross_entropy(D_real, ones_label) # compare D_real with 1
        D_loss_fake = F.binary_cross_entropy(D_fake, zeros_label) # compare D_fake with 0
        D_loss = D_loss_real + D_loss_fake

        # Housekeeping - reset gradient
        reset_grad()
        
        # Tinh dao ham cua D_loss vs cac Variable require_grad = true
        D_loss.backward()
        
        # update params
        D_solver.step()

        #---------------------------------------------------#
        
        # Generator forward-loss-backward-update
        z = Variable(torch.randn(X.size(0), Z_dim))
        G_sample = G(z)
        D_fake = D(G_sample)

        G_loss = F.binary_cross_entropy(D_fake, ones_label) # Compare D_fake with 1

        # Housekeeping - reset gradient
        reset_grad()
        
        # Back-ward
        G_loss.backward()
        
        # Update
        G_solver.step()
        
        #D_losses.add(D_loss.data[0], X.size(0))
        #G_losses.add(G_loss.data[0], X.size(0))
        
        # Test A. Du's loss
        D_losses.add(D_loss.data[0]*X.size(0), X.size(0))
        G_losses.add(G_loss.data[0]*X.size(0), X.size(0))

    print('Epoch-{}; D_loss: {}; G_loss: {}'.format(epoch, D_losses.value()[0], G_losses.value()[0]))


# In[17]:


for epoch in range(1, args.epochs + 1):
    train(epoch)


# # Predict and get probabilities

# ## Predict

# In[18]:


m_test = c_test*avai_test
m_test = Variable(torch.Tensor(m_test).float())


# In[19]:


recon_test, mu, logvar = model(m_test)


# In[20]:


recon_test.size()


# ## Get probability

# In[21]:


# Reshape predicted values
recon_test = recon_test.view(c_test.shape)


# In[22]:


recon_test.size(), c_test.shape


# ```
# softmax = nn.Softmax()
# def getProbabilities(inp, inp_index, start_index):
#     softmax_input = softmax(input[inp_index, :, start_index:])
#     return softmax_input
# ```

# In[23]:


softmax = nn.Softmax()
for i in range(recon_test.size(0)):
    cont_values = recon_test[i, :, 0].contiguous().view(recon_test.size(1),1) #(35,1)
    softmax_values = softmax(recon_test[i, :, 1:])
    if i == 0:
        recon = torch.cat([cont_values, softmax_values], 1)
        recon = recon.contiguous().view(1,recon_test.size(1), recon_test.size(2)) #(1, 35, 8)
    else:
        current_recon = torch.cat([cont_values, softmax_values], 1)
        current_recon = current_recon.contiguous().view(1,recon_test.size(1), recon_test.size(2)) #(1, 35, 8)
        recon = torch.cat([recon, current_recon], 0)


# In[24]:


recon.size()


# # Evaluate

# - predicted data: recon
# - complete data (normalized): c_test
# - nan matrix: nan_test

# In[25]:


evalTime(recon, true_test, nan_test, min_array, max_array)


# In[26]:


evalAct(recon, true_test, nan_test)


# ## Time

# In[27]:


#transform Variable into numpy array
recon = recon.data.numpy()


# In[28]:


inversed_recon = inverse_minmaxScaler(recon, min_array, max_array, cols=[0])
#inversed_c_test = inverse_minmaxScaler(c_test, min_array, max_array, cols=[0])


# In[29]:


(inversed_recon > 0).all()


# In[30]:


inversed_recon.shape


# In[31]:


299*35


# In[32]:


predict_time = inversed_recon[:, :, 0]*nan_test[:, :, 0]
predict_time = predict_time.reshape(c_test.shape[0]*c_test.shape[1], 1)
predict_time = predict_time[~np.all(predict_time == 0, axis=1)]


# In[33]:


true_time = true_test[:, :, 0]*nan_test[:, :, 0]
true_time = true_time.reshape(c_test.shape[0]*c_test.shape[1], 1)
true_time = true_time[~np.all(true_time == 0, axis=1)]


# In[34]:


true_time.shape, predict_time.shape


# In[35]:


from sklearn.metrics import mean_absolute_error


# In[36]:


mean_absolute_error(true_time, predict_time)


# In[37]:


5816926.9375617802/(24*60*60)


# ## Activity

# In[38]:


pred = inversed_recon[:, :, 1:]*nan_test[:, :, 1:]
pred = pred.reshape(c_test.shape[0]*c_test.shape[1], c_test.shape[2]-1)
missing_pred = pred[~np.all(pred == 0, axis=1)]


# In[39]:


gt = true_test[:, :, 1:]*nan_test[:, :, 1:]
gt = gt.reshape(c_test.shape[0]*c_test.shape[1], c_test.shape[2]-1)
missing_gt = gt[~np.all(gt == 0, axis=1)]


# In[40]:


missing_pred.shape, missing_gt.shape


# In[41]:


from sklearn.metrics import accuracy_score, log_loss


# In[42]:


gt_label = missing_gt.argmax(axis=1)
pred_label = missing_pred.argmax(axis=1)


# In[43]:


gt_label.shape


# In[44]:


accuracy_score(gt_label, pred_label)


# In[45]:


log_loss(missing_gt, missing_pred)

