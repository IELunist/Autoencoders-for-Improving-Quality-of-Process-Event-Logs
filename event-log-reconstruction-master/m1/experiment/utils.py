import pandas as pd
import numpy as np
import math
from math import sqrt
from datetime import timedelta

import torch
import torch.nn as nn


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc, precision_recall_curve

def calculateCumDuration(df):
    df['CumDuration'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].iloc[0])
    return df

def calculateAnomalousCumDuration(df):
    df['AnomalousCumDuration'] = (df['AnomalousCompleteTimestamp'] - df['AnomalousCompleteTimestamp'].iloc[0])
    return df

def calculateDuration(df):
    df['Duration'] = (df['CompleteTimestamp'] - df['CompleteTimestamp'].shift(1)).fillna(0)
    return df

def convert2seconds(x):
    x = x.total_seconds()
    return x

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

def getModifiedInput(groupByCase, cols, maxlen):
    full_list = []
    for case, data in groupByCase:
        temp = data.as_matrix(columns=cols)
        #temp = padwithzeros(temp, maxlen)
        full_list.append(temp)
    inp = np.array(full_list)
    return inp

def getProbability(recon_test):
    '''This function takes 3d tensor as input and return a 3d tensor which has probabilities for 
    classes of categorical variable'''
    softmax = nn.Softmax()
    #recon_test = recon_test.view(c_test.shape)
    
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


def getPrediction(predicted_tensor, pad_matrix):
    '''
    This function converts a tensor to a pandas dataframe
    Return: Dataframe with columns (NormalizedTime, PredictedActivity)

    - predicted_tensor: recon
    - df: recon_df_w_normalized_time
    '''
    predicted_tensor = getProbability(predicted_tensor) #get probability for categorical variables
    predicted_array = predicted_tensor.data.cpu().numpy() #convert to numpy array
    
    #Remove 0-padding
    temp_predicted_array = predicted_array*pad_matrix
    temp_predicted_array = temp_predicted_array.reshape(predicted_array.shape[0]*predicted_array.shape[1], predicted_array.shape[2])
    temp_predicted_array = temp_predicted_array[np.any(temp_predicted_array != 0, axis=1)]
    
    predicted_time = temp_predicted_array[:, 0]
    predicted_activity = temp_predicted_array[:, 1:]

    return predicted_time, predicted_activity



def getError(predicted_tensor, true_tensor, pad_matrix):
    '''
    This function converts a tensor to a pandas dataframe
    Return: Dataframe with columns (NormalizedTime, PredictedActivity)

    - predicted_tensor: recon
    - df: recon_df_w_normalized_time
    '''
    predicted_tensor = getProbability(predicted_tensor) #get probability for categorical variables
    predicted_array = predicted_tensor.data.cpu().numpy() #convert to numpy array

    true_array = true_tensor.data.cpu().numpy()
    
    #Remove 0-padding
    temp_predicted_array = predicted_array*pad_matrix
    temp_predicted_array = temp_predicted_array.reshape(predicted_array.shape[0]*predicted_array.shape[1], predicted_array.shape[2])
    temp_predicted_array = temp_predicted_array[np.any(temp_predicted_array != 0, axis=1)]
    
    temp_true_array = true_array*pad_matrix
    temp_true_array = temp_true_array.reshape(true_array.shape[0]*true_array.shape[1], true_array.shape[2])
    temp_true_array = temp_true_array[np.any(temp_true_array != 0, axis=1)]

    predicted_time = temp_predicted_array[:, 0]
    predicted_activity = temp_predicted_array[:, 1:]

    true_time = temp_true_array[:, 0]
    true_activity = temp_true_array[:, 1:]
    return predicted_time, predicted_activity, true_time, true_activity


def plotConfusionMaxtrix(error_df, threshold, variable='Activity', output_dir='./', save=False):
    LABELS = ['Normal', 'Anomaly']
    y_pred = [1 if e > threshold else 0 for e in error_df.Error.values]
    
    if variable == 'Activity':
        matrix = confusion_matrix(error_df.ActivityLabel.astype('uint8'), y_pred)
    else:
        matrix = confusion_matrix(error_df.TimeLabel.astype('uint8'), y_pred)
        
    plt.figure(figsize=(7, 7))
    sns.heatmap(matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
    plt.title('Confusion matrix of {}'.format(variable))
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    if save == True:
        plt.savefig(output_dir+'confusion_matrix_'+variable)
    plt.show()


def plotOverlapReconstructionError(error_df, variable='Activity', output_dir='./', save='False'):
    if variable == 'Activity':
        normal_error_df = error_df[(error_df['ActivityLabel']== 0)]['Error']
        anomaly_error_df = error_df[(error_df['ActivityLabel']== 1)]['Error']
    else:
        normal_error_df = error_df[(error_df['TimeLabel']== 0)]['Error']
        anomaly_error_df = error_df[(error_df['TimeLabel']== 1)]['Error']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hist(normal_error_df, color='blue', label='Normal data ', alpha=0.5)
    ax.hist(anomaly_error_df, color='green', label='Anomalous data ', alpha=0.5)

    #plt.title('Reconstruction error')
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save == True:
        plt.savefig(output_dir+'reconstruction_error_'+variable, bbox_inches='tight')
    plt.show()
	

def plotReconstructionError(error_df, variable='Activity'):
    if variable == 'Activity':
        normal_error_df = error_df[(error_df['ActivityLabel']== 0)]['Error']
        anomaly_error_df = error_df[(error_df['ActivityLabel']== 1)]['Error']
    else:
        normal_error_df = error_df[(error_df['TimeLabel']== 0)]['Error']
        anomaly_error_df = error_df[(error_df['TimeLabel']== 1)]['Error']
    
    plt.figure(figsize=(15, 5))
    
    #normal cases
    plt.subplot(121)
    normal_error_df.hist()
    plt.title('Reconstruction error of Normal cases')
    
    #anomalous cases
    plt.subplot(122)
    anomaly_error_df.hist()
    plt.title('Reconstruction error of Anomaly cases')
    
    plt.show()


def evalScore(error_df, threshold, variable='Activity'):
    y_pred = [1 if e > threshold else 0 for e in error_df.Error.values]
    
    if variable=='Activity':
        y_true = error_df.ActivityLabel.astype('uint8')
    else:
        y_true = error_df.TimeLabel.astype('uint8')
    
    score = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print('-------Evaluation of {}-------'.format(variable))
    print('\n')
    print('--Weighted Evaluation--')
    print('Evaluation of {}'.format(variable))
    print('Precision: {:.2f}'.format(score[0]))
    print('Recall: {:.2f}'.format(score[1]))
    print('Fscore: {:.2f}'.format(score[2]))
    print('\n')
    score_1 = precision_recall_fscore_support(y_true, y_pred)
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
    #print('Support: {:.2f}'.format(score[3]))

def plotDurationofPredictedTimeLabel(activity, df, statistics_storage, output_dir='./', save=False):
    fig, ax = plt.subplots(figsize=(10, 5))
    anomaly = df[df['PredictedTimeLabel']==1]
    normal = df[df['PredictedTimeLabel']==0]
    ax.plot(anomaly.index, anomaly.AnomalousDuration, marker='o', ms=3.5, linestyle='', color='green', label='Anomalous data: '+str(len(anomaly)))
    ax.plot(normal.index, normal.AnomalousDuration, marker='o', ms=3.5, linestyle='', color='blue', label='Normal data: '+str(len(normal)))
    ax.hlines(statistics_storage[activity]['mean']+statistics_storage[activity]['std'], ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Border')
    plt.title('Duration of '+ activity)
    plt.xlabel('Data point index')
    plt.ylabel('Duration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save == True:
        plt.savefig(output_dir + 'duration_'+activity)
    plt.show()
    plt.close()


def plotFalseDuration(false_positive_df, false_negative_df, activity, statistics_storage):
    selected_fp_df = false_positive_df[false_positive_df['Activity']==activity]['AnomalousDuration']
    selected_fn_df = false_negative_df[false_negative_df['Activity']==activity]['AnomalousDuration']
    
    #false positive
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    selected_fp_df.hist()
    plt.axvline(statistics_storage[activity]['mean'], color='w', linestyle='dashed', linewidth=2)
    plt.axvline(statistics_storage[activity]['mean']+statistics_storage[activity]['std'], color='r', linestyle='dashed', linewidth=2)
    plt.title('False Positive: ' + activity)
    
    #false negative
    plt.subplot(122)
    selected_fn_df.hist()
    plt.axvline(statistics_storage[activity]['mean'], color='w', linestyle='dashed', linewidth=2)
    plt.axvline(statistics_storage[activity]['mean']+statistics_storage[activity]['std'], color='r', linestyle='dashed', linewidth=2)
    plt.title('False Negative: ' + activity)
    
    plt.show()

    
'''
#swap 2 activity within a case
groupByCase = duration_df.groupby(['CaseID'])

anomalous_act_index = []
caseid_list = []
temp_df = duration_df.copy()
temp_df['AnomalousActivity'] = temp_df['Activity'].copy()
temp_df['ActivityLabel'] = 0

while len(anomalous_act_index) < anomalous_act_num:
    caseid = np.random.randint(1, len(groupByCase))
    if caseid not in caseid_list:
        group = groupByCase.get_group(caseid)
        row1 = np.random.randint(0, group.shape[0])
        row2 = np.random.randint(0, group.shape[0])
        index1 = group.index.values[row1]
        index2 = group.index.values[row2]
        act1 = duration_df['Activity'].iloc[index1]
        act2 = duration_df['Activity'].iloc[index2]
        if act1 != act2:
            anomalous_act_index.append(index1)
            anomalous_act_index.append(index2)
            temp_df['AnomalousActivity'].iloc[index1] = act2
            temp_df['AnomalousActivity'].iloc[index2] = act1
            temp_df['ActivityLabel'].iloc[index1] = 1
            temp_df['ActivityLabel'].iloc[index2] = 1
            
temp_act = temp_df[['AnomalousActivity', 'ActivityLabel']]
'''