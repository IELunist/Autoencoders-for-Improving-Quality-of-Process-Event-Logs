
# coding: utf-8

# This is the implementation of [**Variational Autoencoder**](https://arxiv.org/pdf/1312.6114.pdf)

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


import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve




# In[3]:


import torch
import torch.utils.data
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler
from torch.autograd import Variable
from torchvision import transforms


# In[4]:


sys.path.insert(0, './../utils/')
from utils import *
from models import *


# In[5]:


#Define parser
names = ['bpi_2012','bpi_2013','small_log','large_log']
start_time = time.time()

for name in names:
    parser = {
        'train': True,
        'test': True,
        'model_class': 'VAE_dropout', #VAE
        'model_name': '',
        'data_dir': '../data/',
        'data_file': name + '.csv',
        'anomaly_pct': 0.1,
        'scaler': 'standardization',
        'input_dir': '../input/ver 8/{}/'.format(name),
        'batch_size' : 16,
        'epochs' : 100,
        'no_cuda' : False,
        'seed' : 7,
        'layer1': 50,
        'layer2': 20,
        'lr': 0.001,
        'betas': (0.9, 0.999),
        'lr_decay': 0.99,
    }
    args = argparse.Namespace(**parser)
    out_dir = './output/ver 8/{0}_{1}/'.format(name,args.model_class)
    for count in range(10):
        count = str(count)
        print('\n')
        print('Version Number : %s'%(count))
        args.output_dir = './'+out_dir+'ver'+count+'/'

        # In[6]:


        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)


        # In[7]:


        args.cuda = not args.no_cuda and torch.cuda.is_available()
        kwargs = {'num_workers': 2, 'pin_memory': True} if args.cuda else {}


        # In[8]:


        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)


        # In[9]:


        preprocessed_data_name = os.path.join(args.input_dir, 'preprocessed_data_%s_%s.pkl'%(args.anomaly_pct,count))
        with open(preprocessed_data_name, 'rb') as f:
            input_train = pickle.load(f)
            input_val = pickle.load(f)
            input_test = pickle.load(f)
            pad_index_train = pickle.load(f)
            pad_index_val = pickle.load(f)
            pad_index_test = pickle.load(f)
            activity_label_test = pickle.load(f)
            time_label_test = pickle.load(f)
            train_case_num = pickle.load(f)
            val_case_num = pickle.load(f)
            test_case_num = pickle.load(f)
            train_row_num = pickle.load(f)
            val_row_num = pickle.load(f)
            test_row_num = pickle.load(f)
            min_value = pickle.load(f)
            max_value = pickle.load(f)
            mean_value = pickle.load(f)
            std_value = pickle.load(f)
            cols = pickle.load(f)
            statistics_storage = pickle.load(f)
            true_time = pickle.load(f)
            true_act = pickle.load(f)
            full_true_time = pickle.load(f)
            full_true_act = pickle.load(f)


        # # Load data

        # In[10]:


        #train
        input_trainLoader = torch.utils.data.DataLoader(input_train, batch_size=args.batch_size, shuffle=False, num_workers=2)
        pad_index_trainLoader = torch.utils.data.DataLoader(pad_index_train, batch_size=args.batch_size, shuffle=False, num_workers=2)


        # In[11]:


        #df
        normal_df_name = os.path.join(args.input_dir, 'normal_df_{0}_{1}.csv'.format(args.anomaly_pct,count))
        normal_df = pd.read_csv(normal_df_name)

        anomalous_df_name = os.path.join(args.input_dir, 'anomolous_df_{0}_{1}.csv'.format(args.anomaly_pct,count))
        anomalous_df = pd.read_csv(anomalous_df_name)

        #test
        caseid_test = normal_df['CaseID'][-test_row_num:]
        normal_df_test = normal_df[-test_row_num:]
        anomalous_df_test = anomalous_df[-test_row_num:]


        # In[12]:

        # # Build model

        # ## Define model

        # In[14]:


        if args.model_class == 'VAE':
            model = VAE(input_train.shape, args.layer1, args.layer2, True)

        if args.model_class == 'VAE_dropout':
            model = VAE_dropout(input_train.shape, args.layer1, args.layer2, True)

        if args.cuda:
            model.cuda()


        # In[15]:


        model


        # ## Define loss

        # In[16]:


        def loss_function(recon_x, x, mu, logvar, avai_mask):
            MSE = F.mse_loss(recon_x*avai_mask, x*avai_mask, reduction='sum')
            KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            KLD = torch.sum(KLD_element).mul_(-0.5)
            loss = MSE+KLD
            return loss


        # ## Define optimizer

        # In[17]:


        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.betas)


        # In[18]:


        #Adjust learning rate per epoch: http://pytorch.org/docs/master/optim.html?highlight=adam#how-to-adjust-learning-rate

        # Method 1:

        #lambda1 = lambda epoch: epoch // args.lr_step
        lambda1 = lambda epoch: args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=[lambda1])

        # Method 2:
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10)


        # # Utils

        # In[19]:


        def save_model(model, epoch, score):
            model_file = os.path.join(args.output_dir, 'model_{}_epoch{}_score{:.4f}.pth'.format(args.model_class, epoch, score))
            torch.save(model.state_dict(), model_file)


        # In[20]:


        def load_model(model, model_name):
            model_file = os.path.join(args.output_dir, model_name)
            assert os.path.isfile(model_file), 'Error: no model found!'
            model_state = torch.load(model_file)
            model.load_state_dict(model_state)


        # In[21]:


        def val(model, input_val, pad_index_val):
            model.eval()
            input_val = Variable(torch.Tensor(input_val).float())
            pad_index_val = Variable(torch.Tensor(pad_index_val).float())

            if args.cuda:
                input_val = input_val.cuda()
                pad_index_val = pad_index_val.cuda()

            recon_val, mu, logvar = model(input_val)
            loss = loss_function(recon_val, input_val, mu, logvar, pad_index_val)
            return loss.data[0]/len(input_test.data)


        # # Train

        # In[22]:


        def train(epoch, model, optimizer):
            model.train()
            train_loss = 0
            for batch_idx, (batch_data, batch_index) in enumerate(zip(input_trainLoader, pad_index_trainLoader)):

                batch_data = Variable(batch_data.float())
                batch_index = Variable(batch_index.float())

                if args.cuda:
                    batch_data = batch_data.cuda()
                    batch_index = batch_index.cuda()

                optimizer.zero_grad()

                recon_data, mu, logvar = model(batch_data)

                loss = loss_function(recon_data, batch_data, mu, logvar, batch_index)

                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            return train_loss / len(input_trainLoader.dataset)


        # In[23]:


        if args.train:
            for epoch in range(1, args.epochs + 1):
                init = time.time()

                #method 1 scheduler
                scheduler.step()

                train_loss = train(epoch, model, optimizer)
                end_train = time.time()
                val_score = val(model, input_val, pad_index_val)

                '''
                # To save model
                if epoch == 1:
                    current_best = val_score
                    save_model(model, epoch, val_score)

                else:
                    if val_score < current_best:
                        current_best = val_score
                        save_model(model, epoch, val_score)
                '''

                end = time.time()
                print('====> Epoch {} | Train time: {:.4f} ms| End time: {:.4f} ms | Train loss: {:.4f} | Val loss: {:.4f}'.
                      format(epoch, (end_train-init)*1000, (end-init)*1000, train_loss, val_score))
        else:
            load_model(model, args.model_name)


        # # Predict

        # In[24]:


        if args.test:
            input_test = Variable(torch.Tensor(input_test).float())
            if args.cuda:
                input_test = input_test.cuda()

            print('Predicting...')
            recon_test, mu, logvar = model(input_test)

            print('Separating prediction ...')
            predicted_time, predicted_activity = getPrediction(recon_test, pad_index_test)

            print('Done!')


        # # Evaluate

        # ## Time

        # In[25]:


        #Fix predicted time
        time_df = pd.DataFrame({'CaseID': caseid_test,
                                'PredictedTime': predicted_time})

        groupByCase = time_df.groupby(['CaseID'])
        fixed_time_df = pd.DataFrame(columns=list(time_df))

        for case, group in groupByCase:
            group.iloc[0, 1] = -mean_value/std_value
            fixed_time_df = fixed_time_df.append(group)


        # In[26]:



        # In[27]:


        fixed_predicted_time = fixed_time_df['PredictedTime']


        # In[28]:


        error = np.abs(true_time - fixed_predicted_time)
        error_time_df = pd.DataFrame({'Error': error,
                                      'TimeLabel': time_label_test})


        # In[29]:
        error_time_df['TimeLabel'] = error_time_df['TimeLabel'].astype('float64')
        error_time_df.to_csv(args.output_dir+'error_time_df.csv',index=False)

        # In[30]:

        time_threshold = np.mean(error_time_df['Error'])
        print('Threshold of Time: {}'.format(time_threshold))


        # In[31]:
        y_true = error_time_df.TimeLabel.astype('uint8')
        y_pred = [1 if e > time_threshold else 0 for e in error_time_df.Error.values]
        confusion = confusion_matrix(y_true, y_pred)
        with open(args.output_dir+'anomaloustime_result'+str(count)+'.pkl','wb') as f:
            pickle.dump(confusion_matrix(y_true, y_pred),f,protocol=2)


        # In[35]:


    #    fpr, tpr, thresholds = roc_curve(error_time_df.TimeLabel, error_time_df.Error, pos_label=1)
    #    roc_auc = auc(fpr, tpr)


        # In[36]:


        #get submission
        if args.scaler == 'standardization':
            inverse_scaled_time = [x*std_value+mean_value for x in fixed_predicted_time]
        else:
            inverse_scaled_time = [x*(max_value-min_value)+min_value for x in fixed_predicted_time]

        predicted_time_label_test = [1 if e > time_threshold else 0 for e in error_time_df.Error.values]

        submission_time = pd.DataFrame({'AnomalousDuration': anomalous_df_test['AnomalousDuration'].copy(),
                                        'Activity': normal_df_test['Activity'].copy(),
                                        'AnomalousCumDuration': anomalous_df_test['AnomalousCumDuration'].copy(),
                                        'PredictedCumDuration': inverse_scaled_time,
                                        'TimeLabel': time_label_test,
                                        'PredictedTimeLabel': predicted_time_label_test})


        groupByActivity = submission_time.groupby(['Activity'])

        # In[39]:


        act_list = [i for i in submission_time['Activity'].unique()]
        false_positive_df = submission_time[(submission_time['PredictedTimeLabel']== 1)&(submission_time['TimeLabel']== 0)]
        false_negative_df = submission_time[(submission_time['PredictedTimeLabel']== 0)&(submission_time['TimeLabel']== 1)]

        # ## Activity

        # ### Threshold

        # In[41]:


        # error = np.mean(np.power(true_act - predicted_activity, 2), axis = 1)
        error = np.mean(np.abs(true_act - predicted_activity), axis = 1)
        error_activity_df = pd.DataFrame({'Error': error,
                                          'ActivityLabel': activity_label_test})
        # In[43]:

        error_activity_df.to_csv(args.output_dir+'error_activity_df.csv',index=False)

        activity_threshold = np.mean(error_activity_df['Error'])
        print('Threshold of Activity: {}'.format(activity_threshold))


        # In[45]:

        y_true = error_activity_df.ActivityLabel.astype('uint8')
        y_pred = [1 if e > activity_threshold else 0 for e in error_activity_df.Error.values]
        print(confusion_matrix(y_true, y_pred))
        with open(args.output_dir+'anomalousactivity_result'+str(count)+'.pkl','wb') as f:
            pickle.dump(confusion_matrix(y_true, y_pred),f,protocol=2)


        # In[49]:

        # ### Argmax

        # In[50]:


        # evaluate based on classification
        predicted_act_df = pd.DataFrame(data=predicted_activity, columns=list(true_act))
        predicted_act_label = predicted_act_df.idxmax(axis=1)
        true_act_label = true_act.idxmax(axis=1)
        predicted_time_label = [0 if a==b else 1 for a, b in zip(true_act_label,predicted_act_label)]


        # In[51]:


        #plot confusion matrix
        LABELS = ['Normal', 'Anomaly']
        matrix = confusion_matrix(error_activity_df.ActivityLabel.astype('uint8'), predicted_time_label)

        # In[52]:


        score = precision_recall_fscore_support(error_activity_df.ActivityLabel.astype('uint8'), predicted_time_label, average='weighted')

        print('-------Evaluation of Activity-------')
        print('\n')
        print('--Weighted Evaluation--')
        print('Evaluation')
        print('Precision: {:.2f}'.format(score[0]))
        print('Recall: {:.2f}'.format(score[1]))
        print('Fscore: {:.2f}'.format(score[2]))
        print('\n')
        score_1 = precision_recall_fscore_support(error_activity_df.ActivityLabel.astype('uint8'), predicted_time_label)
        print('--Evaluation for each class--')
        print('Normal')
        print('Precision: {:.2f}'.format(score_1[0][0]))
        print('Recall: {:.2f}'.format(score_1[1][0]))
        print('Fscore: {:.2f}'.format(score_1[2][0]))
        print('\n')
        print('Anomaly')
        print('Precision: {:.2f}'.format(score_1[0][1]))
        print('Recall: {:.2f}'.format(score_1[1][1]))
        print('Fscore: {:.2f}'.format(score_1[2][1]))


        # In[53]:
        with open(args.output_dir+'anomalousactivity_argmax_result'+str(count)+'.pkl','wb') as f:
            pickle.dump(score_1,f,protocol=2)


        from sklearn.metrics import accuracy_score
        accuracy_score(true_act_label, predicted_act_label)


end_time = time.time()
print(end_time - start_time)
