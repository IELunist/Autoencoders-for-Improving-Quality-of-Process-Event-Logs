
# coding: utf-8

# This is the implementation of [**Long Short-Term Memory Encoder-Decoder**](https://arxiv.org/pdf/1406.1078.pdf)

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

# **To do:**
# - Add dropout
# - Implement another loss function

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
n_pct = [0.3,0.35,0.4,0.5]
for k in n_pct:
    print('\n')
    print('Nan_Pct_rate : %s'%(str(k)))

    parser = {
        'train': True,
        'test': True,
        'model_class': 'LSTMAE',
        'model_name': ' ',
        'data_dir': '../data/',
        'data_file': name + '.csv',
        'nan_pct': k,
        'input_dir': '../input/{0}/nan_pct_{1}/'.format(name,k),
        'batch_size' : 16,
        'epochs' : 20,
        'no_cuda' : False,
        'seed' : 7,
        'layer1': 25,
    #    'layer2': 100,
    #    'early_stopping': 30,
        'lr': 0.001,
        'betas': (0.9, 0.99),
        'lr_decay': 0.99,
    }

    args = argparse.Namespace(**parser)
    args.output_dir = './output/{0}_{1}_{2}/'.format(name, args.nan_pct, args.model_class)

    for count in range(10):
        print('\n')
        print('Count : %s'%(count))


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


        preprocessed_data_name = os.path.join(args.input_dir, 'preprocessed_data_{0}_ver{1}.pkl'.format(args.nan_pct,str(count)))
        with open(preprocessed_data_name, 'rb') as f:
            min_max_storage = pickle.load(f)
            complete_matrix_w_normalized_time_train = pickle.load(f)
            missing_matrix_w_normalized_time_train = pickle.load(f)
            avai_matrix_train = pickle.load(f)
            nan_matrix_train = pickle.load(f)
            complete_matrix_w_normalized_time_val = pickle.load(f)
            missing_matrix_w_normalized_time_val = pickle.load(f)
            avai_matrix_val = pickle.load(f)
            nan_matrix_val = pickle.load(f)
            pad_matrix_val = pickle.load(f)
            complete_matrix_w_normalized_time_test = pickle.load(f)
            missing_matrix_w_normalized_time_test = pickle.load(f)
            avai_matrix_test = pickle.load(f)
            nan_matrix_test = pickle.load(f)
            pad_matrix_test = pickle.load(f)
            cols_w_time = pickle.load(f)
            cols_w_normalized_time = pickle.load(f)


        # In[10]:


        file_name = os.path.join(args.input_dir, 'parameters_{0}_ver{1}.pkl'.format(args.nan_pct,str(count)))
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

        # ## Train

        # In[11]:


        complete_matrix_w_normalized_time_trainLoader = torch.utils.data.DataLoader(complete_matrix_w_normalized_time_train,
                                                                                    batch_size=args.batch_size, shuffle=False,
                                                                                    num_workers=2)
        missing_matrix_w_normalized_time_trainLoader = torch.utils.data.DataLoader(missing_matrix_w_normalized_time_train,
                                                                                   batch_size=args.batch_size, shuffle=False,
                                                                                   num_workers=2)
        avai_matrix_trainLoader = torch.utils.data.DataLoader(avai_matrix_train,
                                                              batch_size=args.batch_size, shuffle=False,
                                                              num_workers=2)


        # In[12]:


        complete_matrix_w_normalized_time_train.shape


        # ## Validate and test

        # In[13]:


        normalized_complete_df_name = os.path.join(args.input_dir, 'normalized_complete_df_{0}_ver{1}.csv'.format(args.nan_pct,str(count)))
        normalized_complete_df = pd.read_csv(normalized_complete_df_name)

        normalized_missing_df_name = os.path.join(args.input_dir, 'normalized_missing_df_{0}_ver{1}.csv'.format(args.nan_pct,str(count)))
        normalized_missing_df = pd.read_csv(normalized_missing_df_name)


        # In[14]:


        missing_true_val = normalized_missing_df[train_row_num:-test_row_num].reset_index(drop=True)
        complete_true_val = normalized_complete_df[train_row_num:-test_row_num].reset_index(drop=True)


        # In[15]:


        missing_true_test = normalized_missing_df[-test_row_num:].reset_index(drop=True)
        complete_true_test = normalized_complete_df[-test_row_num:].reset_index(drop=True)


        # In[16]:


        missing_true_val.shape, missing_true_test.shape


        # In[17]:


        nan_time_index_val, nan_activity_index_val = getnanindex(missing_true_val)


        # In[18]:


        nan_time_index_test, nan_activity_index_test = getnanindex(missing_true_test)


        # In[19]:


        pd.isnull(normalized_missing_df).sum()


        # In[20]:


        pd.isnull(missing_true_val).sum()


        # In[21]:


        pd.isnull(missing_true_test).sum()


        # ## Define model

        # In[22]:


        if args.model_class == 'LSTMAE':
            model = LSTMAE(complete_matrix_w_normalized_time_train.shape[2], args.layer1, 1, args.cuda)

        if args.cuda:
            model.cuda()


        # In[23]:


        model.parameters


        # ## Define loss

        # In[24]:


        # Define loss
        def loss_function(recon_x, x, avai_mask):
            BCE = F.binary_cross_entropy(recon_x, x, weight=avai_mask, size_average=False)
            return BCE


        # In[25]:


        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)


        # In[26]:


        #Adjust learning rate per epoch: http://pytorch.org/docs/master/optim.html?highlight=adam#torch.optim.Adam

        # Method 1:
        lambda1 = lambda epoch: args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

        # Method 2:
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)


        # ## Utils

        # In[27]:


        def save_model(model, epoch, score):
            model_file = os.path.join(args.output_dir, 'model_{}_epoch{}_score{:.4f}.pth'.format(args.model_class, epoch, score))
            torch.save(model.state_dict(), model_file)


        # In[28]:


        def load_model(model, model_name):
            model_file = os.path.join(args.output_dir, model_name)
            assert os.path.isfile(model_file), 'Error: no model found!'
            model_state = torch.load(model_file)
            model.load_state_dict(model_state)


        # In[29]:


        def val(model, missing_matrix_w_normalized_time_val, complete_matrix_w_normalized_time_val, avai_matrix_val):
            model.eval()
            m_val = missing_matrix_w_normalized_time_val
            m_val = Variable(torch.Tensor(m_val).float())

            c_val = complete_matrix_w_normalized_time_val
            c_val = Variable(torch.Tensor(c_val).float())

            avai_matrix_val = Variable(torch.Tensor(avai_matrix_val).float())

            if args.cuda:
                m_val = m_val.cuda()
                c_val = c_val.cuda()
                avai_matrix_val = avai_matrix_val.cuda()

            recon_val = model(m_val)
            val_loss = loss_function(recon_val, c_val, avai_matrix_val)

            return val_loss.data[0]/len(c_val.data)


        # In[30]:


        missing_true_val.head()


        # In[31]:


        complete_true_val.head()


        # # Train model

        # In[32]:


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


        # In[33]:


        if args.train:
            for epoch in range(1, args.epochs + 1):
                init = time.time()

                #method 1 scheduler
                scheduler.step()
                train_loss = train(epoch, model, optimizer)
                end_train = time.time()

                val_score = val(model, missing_matrix_w_normalized_time_val,
                                complete_matrix_w_normalized_time_val, avai_matrix_val)

                end = time.time()
                print('====> Epoch {} | Train time: {:.4f} ms| End time: {:.4f} ms | Train loss: {:.4f} | Val score: {:.4f}'.
                      format(epoch, (end_train-init)*1000, (end-init)*1000, train_loss, val_score))
        else:
            load_model(model, args.model_name)


        # # Predict and evaluate

        # In[34]:


        if args.test:
            m_test = missing_matrix_w_normalized_time_test
            m_test = Variable(torch.Tensor(m_test).float())

            if args.cuda:
                m_test = m_test.cuda()

            print('Predicting...')
            recon_test = model(m_test)

            print('\n')
            print('Converting to dataframe...')
            recon_df_w_normalized_time = convert2df(recon_test, pad_matrix_test, cols_w_normalized_time, test_row_num)

            print('Transforming Normalized Time to Time...')
            recon_df_w_time = getDfWithTime(recon_df_w_normalized_time, missing_true_test, min_max_storage)

            print('Getting submission...')
            submission_df = getSubmission(recon_df_w_time, missing_true_test, complete_true_test, first_timestamp)
            submission = fixTime(submission_df)

            print('Testing...')
            mae_time, rmse_time, acc = evaluation(submission, nan_time_index_test, nan_activity_index_test,show=True)
            result = [mae_time,rmse_time,acc]
            resultloc=args.output_dir+'result_ver%s.pkl'%(str(count))
            with open(resultloc, 'wb') as f:
                pickle.dump(result, f, protocol=2)

            print('\n')
            print('Saving submission...')
            submission_loc = args.output_dir+'submission_ver%s.csv'%(str(count))
            submission_df.to_csv(submission_loc, index=False)

            print('Done!')


        # In[35]:


        submission_df.head(20)


        # In[36]:


        missing_true_test.head(20)
