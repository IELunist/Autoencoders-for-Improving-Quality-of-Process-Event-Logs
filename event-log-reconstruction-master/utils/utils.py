import pandas as pd
import numpy as np
import math
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, log_loss
import torch
import torch.nn as nn
from datetime import timedelta

'''
Note: 2 way to index an element in pandas df
By Index:
- df.Time[row]
- df.loc[row, 'Time']

By location:
- df.iloc[1, 1]: iloc only take interger

'''
def calculateTimeInterval(missing_data):
    df = missing_data.copy()
    df['TimeInterval'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].shift(1))
    df.loc[0, 'TimeInterval'] = 0
    return df

def calculateDuration(missing_data):
    df = missing_data.copy()
    df['Duration'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].iloc[0])
    df['Duration'].iloc[0] = 0
    return df

def calculateCumTimeInterval(missing_data):
    df = missing_data.copy()
    df['CumTimeInterval'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].iloc[0])
    return df

def calculateCaseTimeInterval(missing_data):
    df = missing_data.copy()
    df['CaseTimeInterval'] = np.nan

    current_point = {}
    current_point[df.loc[0, 'CaseID']] = df.loc[0, 'CompleteTimestamp']

    for i in range(1, df.shape[0]):
        if df.loc[i, 'CaseID'] != df.loc[i-1, 'CaseID']:
            df.loc[i, 'CaseTimeInterval'] = (df.loc[i, 'CompleteTimestamp'] - current_point[df.loc[i, 'CaseID']-1]).total_seconds()
            current_point[df.loc[i, 'CaseID']] = df.loc[i, 'CompleteTimestamp']
            
    return df

def convert2seconds(x):
    x = x.total_seconds()
    return x


def minmaxScaler(caseid, df_case, missing_df_case):
    epsilon = 0.1
    missing_case_storage = {}
    missing_case_storage[caseid] = {}
    
    temp = df_case.copy()
    missing_temp = missing_df_case.copy()
    
    temp['NormalizedTime'] = temp['CumTimeInterval'].copy()
    missing_temp['NormalizedTime'] = missing_temp['CumTimeInterval'].copy()
    
    min_val = temp['CumTimeInterval'].min()
    max_val = temp['CumTimeInterval'].max()
    
    missing_min_val = missing_temp['CumTimeInterval'].min()
    missing_max_val = missing_temp['CumTimeInterval'].max()
    missing_case_storage[caseid]['missing_min_val'] = missing_min_val
    missing_case_storage[caseid]['missing_max_val'] = missing_max_val
    
    for row in range(temp.shape[0]):
        #scale complete df
        temp.iloc[row, temp.columns.get_loc('NormalizedTime')] = (temp.iloc[row, temp.columns.get_loc('CumTimeInterval')] - min_val)/(max_val-min_val+epsilon)
        
        #scale missing df
        missing_temp.iloc[row, missing_temp.columns.get_loc('NormalizedTime')] = (missing_temp.iloc[row, missing_temp.columns.get_loc('CumTimeInterval')] - missing_min_val)/(missing_max_val-missing_min_val+epsilon)  
    return temp, missing_temp, missing_case_storage


def OHE(df, categorical_variables):
    for i in categorical_variables:
        enc_df = pd.get_dummies(df, columns=categorical_variables, drop_first=False)
    return enc_df

def findLongestLength(groupByCase):
    '''This function returns the length of longest case'''
    #groupByCase = data.groupby(['CaseID'])
    maxlen = 1
    for case, group in groupByCase:
        temp_len = group.shape[0]
        if temp_len > maxlen:
            maxlen = temp_len
    return maxlen

def padwithzeros(vector, maxlen):
    '''This function returns the (maxlen, num_features) vector padded with zeros'''
    npad = ((maxlen-vector.shape[0], 0), (0, 0))
    padded_vector = np.pad(vector, pad_width=npad, mode='constant', constant_values=0)
    return padded_vector

def getInput(groupByCase, cols, maxlen):
    full_list = []
    for case, data in groupByCase:
        temp = data.as_matrix(columns=cols)
        temp = padwithzeros(temp, maxlen)
        full_list.append(temp)
    inp = np.array(full_list)
    return inp

def getMeanVar(array, idx=0):
    temp_array = [a[idx] for a in array if not np.isnan(a[idx])]
    mean_val = np.mean(temp_array)
    var_val = np.var(temp_array)
    return mean_val, var_val

def getProbability(recon_test):
    '''This function takes 3d tensor as input and return a 3d tensor which has 
    probabilities for classes of categorical variable'''
    softmax = nn.Softmax()
    #recon_test = recon_test.cpu() #moving data from gpu to cpu for full evaluation
    
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
    return recon

def convert2df(predicted_tensor, pad_matrix, cols, test_row_num):
    '''
    This function converts a tensor to a pandas dataframe
    Return: Dataframe with columns (NormalizedTime, PredictedActivity)

    - predicted_tensor: recon
    - df: recon_df_w_normalized_time
    '''
    predicted_tensor = getProbability(predicted_tensor) #get probability for categorical variables
    predicted_array = predicted_tensor.data.cpu().numpy() #convert to numpy array
    
    #Remove 0-padding
    temp_array = predicted_array*pad_matrix
    temp_array = temp_array.reshape(predicted_array.shape[0]*predicted_array.shape[1], predicted_array.shape[2])
    temp_array = temp_array[np.any(temp_array != 0, axis=1)]
    
    #check number of row of df
    if temp_array.shape[0] == test_row_num:
        #print('Converting tensor to dataframe...')
        df = pd.DataFrame(temp_array, columns=cols)
        activity_list = [i for i in cols if i!='NormalizedTime']
        df['PredictedActivity'] = df[activity_list].idxmax(axis=1) #get label
        #df['PredictedActivity'] = df['PredictedActivity'].apply(lambda x: x.split('_')[1]) #remove prefix
        df['PredictedActivity'] = df['PredictedActivity'].apply(lambda x: x[9:]) #remove prefix Activity_
        df = df.drop(activity_list, axis=1)
        #print('Done!!!')
    return df

def inversedMinMaxScaler(caseid, min_max_storage, recon_df_w_normalized_time_case):
    epsilon = 0.1
    
    temp = recon_df_w_normalized_time_case.copy()
    temp['PredictedCumTimeInterval'] = recon_df_w_normalized_time_case['NormalizedTime'].copy()
    
    #should check for nan values here
    #min_val = min_max_storage[caseid]['missing_min_val']
    #max_val = min_max_storage[caseid]['missing_max_val']
    min_val, max_val = findValidMinMax(caseid, min_max_storage)

    for row in range(temp.shape[0]):
        temp.iloc[row, temp.columns.get_loc('PredictedCumTimeInterval')] = min_val + temp.iloc[row, temp.columns.get_loc('NormalizedTime')]*(max_val-min_val+epsilon)
        
    return temp

def findValidMinMax(caseid, min_max_storage):
    min_val_before = 0
    max_val_before= 0
    min_val_after = 0
    max_val_after = 0
    min_val = 0
    max_val = 0
    
    if caseid == len(min_max_storage):
        for i in range(caseid):
            min_val = min_max_storage[caseid-i]['missing_min_val']
            max_val = min_max_storage[caseid-i]['missing_max_val']
            if not np.isnan(min_val) and not np.isnan(max_val):
                break
    else:
        for i in range(caseid):
            min_val_before = min_max_storage[caseid-i]['missing_min_val']
            max_val_before = min_max_storage[caseid-i]['missing_max_val']
            if not np.isnan(min_val_before) and not np.isnan(max_val_before):
                break
    
        for j in range(len(min_max_storage) - caseid+1):
            min_val_after = min_max_storage[caseid+j]['missing_min_val']
            max_val_after = min_max_storage[caseid+j]['missing_max_val']
            if not np.isnan(min_val_after) and not np.isnan(max_val_after):
                break
        min_val = (min_val_before+min_val_after)/2
        max_val = (max_val_before+max_val_after)/2
    return min_val, max_val


def getDfWithTime(recon_df_w_normalized_time, missing_true_test, min_max_storage):
    temp = recon_df_w_normalized_time.copy()
    temp['CaseID'] = missing_true_test['CaseID'].copy()
    recon_groupByCase = temp.groupby(['CaseID'])
    recon_df_w_time = pd.DataFrame(columns=list(temp)+['PredictedCumTimeInterval'])
    
    for caseid, data_case in recon_groupByCase:
        temp_case = inversedMinMaxScaler(caseid, min_max_storage, data_case)
        recon_df_w_time = recon_df_w_time.append(temp_case)
    return recon_df_w_time



def getnanindex(missing_true_df):
    nan_time_index = []
    nan_activity_index = []
    for row in range(missing_true_df.shape[0]):
        if np.isnan(missing_true_df.CumTimeInterval[row]):
            nan_time_index.append(row)

        if not type(missing_true_df.Activity[row]) == str:
            nan_activity_index.append(row)
    return nan_time_index, nan_activity_index

def getSubmission(recon_df_w_time, missing_true_test, complete_true_test, first_timestamp):
    temp = pd.DataFrame(columns=['CaseID', 'TrueActivity', 'PredictedActivity', 'TrueTime', 'PredictedTime'])
    temp['CaseID'] = missing_true_test['CaseID'].copy()
    
    #ground truth
    temp['TrueActivity'] = complete_true_test['Activity'].copy()
    temp['TrueTime'] = complete_true_test['CumTimeInterval'].copy()
    temp['TrueCompleteTimestamp'] = complete_true_test['CompleteTimestamp'].copy()

    #predicted activity
    temp['PredictedActivity'] = missing_true_test['Activity'].copy()
    temp['PredictedTime'] = missing_true_test['CumTimeInterval'].copy()
    temp['PredictedCompleteTimestamp'] = missing_true_test['CompleteTimestamp'].copy()

    for row in range(temp.shape[0]):
        if pd.isnull(temp.loc[row, 'PredictedActivity']):
            temp.loc[row, 'PredictedActivity'] = recon_df_w_time.loc[row, 'PredictedActivity']
        if pd.isnull(temp.loc[row, 'PredictedTime']):
            temp.loc[row, 'PredictedTime'] = recon_df_w_time.loc[row, 'PredictedCumTimeInterval']
            temp.loc[row, 'PredictedCompleteTimestamp'] = first_timestamp+timedelta(seconds=recon_df_w_time.loc[row, 'PredictedCumTimeInterval'])
    return temp

def fixTime(recon_df_w_time):
    groupByCase = recon_df_w_time.groupby(['CaseID'])
    temp = pd.DataFrame(columns=list(recon_df_w_time))

    for caseid, data_case in groupByCase:
        for row in range(1, len(data_case)):
            current = data_case.iloc[row, data_case.columns.get_loc('PredictedTime')]
            previous = data_case.iloc[row-1, data_case.columns.get_loc('PredictedTime')]
            if current < previous:
                data_case.iloc[row, data_case.columns.get_loc('PredictedTime')] = previous
                data_case.iloc[row, data_case.columns.get_loc('PredictedCompleteTimestamp')] = data_case.iloc[row-1, data_case.columns.get_loc('PredictedCompleteTimestamp')]
        temp = temp.append(data_case)
    return temp


def evaluation(submission_df, nan_time_index, nan_activity_index, show=False):
    #eval Time
    true_time = submission_df.loc[nan_time_index, 'TrueTime']
    predicted_time = submission_df.loc[nan_time_index, 'PredictedTime']
    mae_time = mean_absolute_error(true_time, predicted_time)
    rmse_time = sqrt(mean_squared_error(true_time, predicted_time))
    
    #eval Activity
    true_activity = submission_df.loc[nan_activity_index, 'TrueActivity']
    predicted_activity = submission_df.loc[nan_activity_index, 'PredictedActivity']
    acc = accuracy_score(true_activity, predicted_activity)
    
    if show==True: 
        print('Number of missing Time: {}'.format(len(nan_time_index)))
        print('Mean Absolute Error: {:.4f} day(s)'.format(mae_time/86400))
        print('Root Mean Squared Error: {:.4f} day(s)'.format(rmse_time/86400))
        
        print('Number of missing Activity: {}'.format(len(nan_activity_index)))
        print('Accuracy: {:.2f}%'.format(acc*100))
    return mae_time, rmse_time, acc

'''
def val(model, missing_matrix_w_normalized_time_val, complete_true_val, missing_true_val,
       pad_matrix_val, cols_w_normalized_time, val_row_num,
       nan_time_index_val, nan_activity_index_val):
    model.eval()
    m_val = missing_matrix_w_normalized_time_val
    m_val = Variable(torch.Tensor(m_val).float())
    
    if args.cuda:
        m_val = m_val.cuda()
        
    recon_val, mu, logvar = model(m_val)
    
    recon_df_w_normalized_time = convert2df(recon_val, pad_matrix_val, cols_w_normalized_time, val_row_num)
    recon_df_w_time = getDfWithTime(recon_df_w_normalized_time, missing_true_val, min_max_storage)
    submission_df = getSubmission(recon_df_w_time, missing_true_val, complete_true_val, first_timestamp)
    
    #evaluate
    mae_time, rmse_time, acc = evaluation(submission_df, nan_time_index_val, nan_activity_index_val)
    
    
    return mae_time/86400+1/acc
    
#val_score = val(model, missing_matrix_w_normalized_time_val, complete_true_val, missing_true_val,
#                 pad_matrix_val, cols_w_normalized_time, val_row_num,
#                 nan_time_index_val, nan_activity_index_val)
'''

'''
def val(model, missing_matrix_w_normalized_time_val, complete_matrix_w_normalized_time_val, nan_matrix_val):
    model.eval()
    m_val = missing_matrix_w_normalized_time_val
    m_val = Variable(torch.Tensor(m_val).float())
    
    c_val = complete_matrix_w_normalized_time_val
    c_val = Variable(torch.Tensor(c_val).float())
    
    nan_matrix_val = Variable(torch.Tensor(nan_matrix_val).float())
    
    if args.cuda:
        m_val = m_val.cuda()
        c_val = c_val.cuda()
        nan_matrix_val = nan_matrix_val.cuda()
        
    recon_data, mu, logvar = model(m_val)
        
    loss = loss_function(recon_data, c_val, mu, logvar, nan_matrix_val)
    return loss.data[0]/missing_matrix_w_normalized_time_val.shape[0]
    
#val_score = val(model, missing_matrix_w_normalized_time_val, complete_matrix_w_normalized_time_val, avai_matrix_val)
'''