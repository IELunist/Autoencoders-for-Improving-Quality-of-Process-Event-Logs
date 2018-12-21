
# coding: utf-8

# **Outline:**
# - Build model and loss function
# - Train model
# - Observe valdidate
# - Test

# **To do:**
# - Hyperparameter tuning
#     + lr
#     + layer1, layer2
#     + betas

# **Modification**
# - Weight initialization with xavier uniform
# - Adam optimization
# - LR decay

# In[1]:


import importlib
import argparse
import os, sys
import argparse
import pandas as pd
import numpy as np
import pickle
import time


# In[2]:


import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms


# In[3]:


sys.path.insert(0, './../utils/')
from utils import *
from models import *


# In[4]:


#Define parser
#name = 'bpi_2012'
#name = 'bpi_2013'
name = 'small_log'
#name = 'large_log'

parser = {
    'train': True,
    'test': True,
    'model_class': 'AE',
    'model_name': '',
    'data_dir': '../data/',
    'data_file': name + '.csv',
    'nan_pct': 0.5,
    'input_dir': '../input/{}/'.format(name),
    'batch_size' : 16,
    'epochs' : 200,
    'no_cuda' : False,
    'seed' : 7,
    'layer1': 300,
    'layer2': 100,
    'lr': 0.0005,
    'betas': (0.9, 0.999),   
    'lr_decay': 0.90,
}

args = argparse.Namespace(**parser)
args.output_dir = './output/{0}_{1}_{2}/'.format(name, args.nan_pct, args.model_class)


# In[5]:


if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)


# In[6]:


args.cuda = not args.no_cuda and torch.cuda.is_available()


# In[7]:


torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


# In[8]:


kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}


# In[9]:


preprocessed_data_name = os.path.join(args.input_dir, 'preprocessed_data_full_{}.pkl'.format(args.nan_pct))
with open(preprocessed_data_name, 'rb') as f:
    min_max_storage = pickle.load(f)
    complete_matrix_w_normalized_time = pickle.load(f)
    missing_matrix_w_normalized_time = pickle.load(f)
    avai_matrix = pickle.load(f)
    nan_matrix = pickle.load(f)
    pad_matrix = pickle.load(f)
    cols_w_time = pickle.load(f)
    cols_w_normalized_time = pickle.load(f)


# In[10]:


file_name = os.path.join(args.input_dir, 'parameters_{}.pkl'.format(args.nan_pct))
with open(file_name, 'rb') as f:
    most_frequent_activity = pickle.load(f)
    first_timestamp = pickle.load(f)
    avai_instance = pickle.load(f)
    nan_instance = pickle.load(f)
    train_size = pickle.load(f)
    val_size = pickle.load(f)
    test_size = pickle.load(f)
    train_row_num = pickle.load(f)
    val_row_num = pickle.load(f)
    test_row_num = pickle.load(f)


# # Load data

# In[11]:


complete_matrix_w_normalized_time_trainLoader = torch.utils.data.DataLoader(complete_matrix_w_normalized_time, 
                                                                            batch_size=args.batch_size, shuffle=False, 
                                                                            num_workers=2)
missing_matrix_w_normalized_time_trainLoader = torch.utils.data.DataLoader(missing_matrix_w_normalized_time, 
                                                                           batch_size=args.batch_size, shuffle=False, 
                                                                           num_workers=2)
avai_matrix_trainLoader = torch.utils.data.DataLoader(avai_matrix, 
                                                      batch_size=args.batch_size, shuffle=False, 
                                                      num_workers=2)


# In[12]:


normalized_complete_df_name = os.path.join(args.input_dir, 'normalized_complete_df_{}.csv'.format(args.nan_pct))
normalized_complete_df = pd.read_csv(normalized_complete_df_name)

normalized_missing_df_name = os.path.join(args.input_dir, 'normalized_missing_df_{}.csv'.format(args.nan_pct))
normalized_missing_df = pd.read_csv(normalized_missing_df_name)


# In[13]:


missing_true_test = normalized_missing_df
complete_true_test = normalized_complete_df


# In[14]:


nan_time_index, nan_activity_index = getnanindex(missing_true_test)


# In[15]:


row_num = missing_true_test.shape[0]


# In[16]:


complete_matrix_w_normalized_time.shape


# In[17]:


35*8


# # Build model

# ## Define model

# In[18]:


if args.model_class == 'AE':
    model = AE(complete_matrix_w_normalized_time.shape, args.layer1, args.layer2)
        
if args.cuda:
    model.cuda()


# ## Define loss

# In[19]:


# Define loss

def loss_function(recon_x, x, avai_mask):
    #MSE = F.mse_loss(recon_x*avai_mask, x*avai_mask, size_average=False)
    BCE = F.binary_cross_entropy(recon_x, x, weight=avai_mask, size_average=False) 
    return BCE


# In[20]:


optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)


# In[21]:


#Adjust learning rate per epoch: http://pytorch.org/docs/master/optim.html?highlight=adam#torch.optim.Adam

# Method 1:
lambda1 = lambda epoch: args.lr_decay ** epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

# Method 2:
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)


# ## Utils

# In[22]:


def save_model(model, epoch, score):
    model_file = os.path.join(args.output_dir, 'model_{}_epoch{}_score{:.4f}.pth'.format(args.model_class, epoch, score))
    torch.save(model.state_dict(), model_file)


# In[23]:


def load_model(model, model_name):
    model_file = os.path.join(args.output_dir, model_name)
    assert os.path.isfile(model_file), 'Error: no model found!'
    model_state = torch.load(model_file)
    model.load_state_dict(model_state)


# # Train model

# In[24]:


def train(epoch, model, optimizer):
    model.train()
    train_loss = 0
    for batch_idx, (m_data, c_data, avai_mask) in enumerate(zip(missing_matrix_w_normalized_time_trainLoader, 
                                                                complete_matrix_w_normalized_time_trainLoader,
                                                                avai_matrix_trainLoader)):
        c_data = Variable(c_data.float())
        m_data = Variable(m_data.float())
        avai_mask = Variable(avai_mask.float())

        if args.cuda:
            c_data = c_data.cuda()
            m_data = m_data.cuda()
            avai_mask = avai_mask.cuda()

            
        optimizer.zero_grad()
        
        recon_data = model(m_data)
        
        loss = loss_function(recon_data, c_data, avai_mask)
        
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        
    return train_loss / len(complete_matrix_w_normalized_time_trainLoader.dataset)


# In[25]:


if args.train:
    for epoch in range(1, args.epochs + 1):
        init = time.time()
        
        #method 1 scheduler
        scheduler.step()
        train_loss = train(epoch, model, optimizer)
        
        end = time.time()
        print('====> Epoch {} | End time: {:.4f} ms | Train loss: {:.4f}'.
              format(epoch, (end-init)*1000, train_loss))
else:
    load_model(model, args.model_name)


# # Predict and evaluate

# In[26]:


if args.test:
    m_test = missing_matrix_w_normalized_time
    m_test = Variable(torch.Tensor(m_test).float())
    
    if args.cuda:
        m_test = m_test.cuda()
    
    print('Predicting...')
    recon_test = model(m_test)
    
    print('\n')
    print('Converting to dataframe...')
    recon_df_w_normalized_time = convert2df(recon_test, pad_matrix, cols_w_normalized_time, row_num)
    
    print('Transforming Normalized Time to Time...')
    recon_df_w_time = getDfWithTime(recon_df_w_normalized_time, missing_true_test, min_max_storage)
    
    print('Getting submission...')
    submission_df = getSubmission(recon_df_w_time, missing_true_test, complete_true_test, first_timestamp)
    submission = fixTime(submission_df)
    
    print('Testing...')
    mae_time, rmse_time, acc = evaluation(submission, nan_time_index, nan_activity_index, show=True)
    print('\n')
    
    print('Saving submission...')
    submission_df.to_csv(args.output_dir+'submission.csv', index=False)
    print('Done!')


# In[27]:


submission_df.shape


# In[28]:


submission.head(10)


# In[29]:


missing_true_test.head(10)

