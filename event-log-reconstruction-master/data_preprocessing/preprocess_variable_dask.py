
# coding: utf-8

# **Outline:**
# - Preprocess Activity and CumTimeInterval
# - Get padding mask
# - Get avai/nan mask

# In[1]:


import os, sys
import argparse
import pandas as pd
import numpy as np
import pickle
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from sklearn import preprocessing
# In[2]:


from dateutil.parser import parse
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da


pd.options.mode.chained_assignment = None #to run loop quicker without warnings


# In[3]:
start_time = time.time()

name = 'bpi_2012'
#name = 'bpi_2013'
#name = 'small_log'
#name = 'large_log'
n_pct = [0.35]
for k in n_pct:
    print('Nan_pct : %s'%(k))
    args = {
        'data_dir': '../data/',
        'data_file': name + '.csv',
        'input_dir': '../input/{0}/nan_pct_{1}'.format(name,k),
        'nan_pct': k,
        'train_pct': 0.6,
        'val_pct': 0.2,
    }

    args = argparse.Namespace(**args)


    # In[4]:

    for count in range(9):
        print(count)

        file_name = os.path.join(args.input_dir, 'parameters_{0}_ver{1}.pkl'.format(args.nan_pct,count))
        print(file_name)
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


        # In[5]:


        sys.path.insert(0, './../utils/')
        from utils import *


        # # Load data

        # In[6]:
        pbar = ProgressBar()
        pbar.register()

        # # Preprocess data

        # **To do:**
        # - Normalize CumTimeInterval using minmaxScaler for each case.
        # - One hot encode
        # - Get avai/nan mask
        # - Replace nan with 0
        # - Pad with 0

        # In[7]:

        # ## Normalize

        # In[10]:

        df_name = os.path.join(args.input_dir, 'complete_df_full_{0}.csv'.format(args.nan_pct))
        df = pd.read_csv(df_name)
        missing_df_name = os.path.join(args.input_dir, 'missing_df_full_{0}_ver{1}.csv'.format(args.nan_pct,count))
        missing_df = pd.read_csv(missing_df_name)

        normalized_complete_df = dd.from_pandas(df,npartitions=5)
        normalized_missing_df = dd.from_pandas(missing_df,npartitions=5)
        groupByCase = df.groupby('CaseID')
        misgroupByCase = missing_df.groupby('CaseID')
        caseid = normalized_complete_df['CaseID'].unique().compute()
        complete_normal=pd.DataFrame(columns=list(df.columns.values)+['NormalizedTime'])
        mis_normal = pd.DataFrame(columns=list(df.columns.values)+['NormalizedTime'])
        epsilon =0.1

        toconcat =[]
        tomisconcat=[]
        for k in list(caseid):
            group = groupByCase.get_group(k)
            misgroup = misgroupByCase.get_group(k)

            maxv = group['CumTimeInterval'].max()
            minv = group['CumTimeInterval'].min()
            mismaxv = misgroup['CumTimeInterval'].max()
            misminv = misgroup['CumTimeInterval'].min()

            cnp_scaled = [(x-minv)/(maxv-minv+epsilon) for x in group['CumTimeInterval'].values.reshape(-1,1)]
            group['NormalizedTime']=cnp_scaled
            group['NormalizedTime'] = group['NormalizedTime'].astype('float')
            toconcat.append(group)

            mnp_scaled = [(x-misminv)/(mismaxv-misminv+epsilon) for x in misgroup['CumTimeInterval'].values.reshape(-1,1)]
            misgroup['NormalizedTime']=cnp_scaled
            misgroup['NormalizedTime'] = misgroup['NormalizedTime'].astype('float')
            tomisconcat.append(misgroup)

        complete_normal = pd.concat([complete_normal]+toconcat)
        mis_normal = pd.concat([mis_normal]+tomisconcat)
        #normal =normal.append(toconcat)
        complete_normal.to_csv('daskcomplete.csv',index=False)
        mis_normal.to_csv('daskmis.csv',index=False)


        missing_groupByCase = missing_df.groupby('CaseID')

end_time = time.time()
print(end_time-start_time)


        # In[11]:

        #normalized_complete_df = pd.DataFrame(columns=list(df)+['NormalizedTime'])
        #normalized_missing_df = pd.DataFrame(columns=list(df)+['NormalizedTime'])
        #min_max_storage = {}


'''
        for i, j in zip(groupByCase, missing_groupByCase):
            if i[0] % 1000 ==0:
                print('%s / %s'%(i[0],howlong))
            temp, missing_temp, missing_case_storage = minmaxScaler(i[0], i[1], j[1])
            normalized_complete_df = normalized_complete_df.append(temp)
            normalized_missing_df = normalized_missing_df.append(missing_temp)
            min_max_storage.update(missing_case_storage)


        normalized_complete_df_name = os.path.join(args.input_dir, 'normalized_complete_df_{0}_ver{1}.csv'.format(args.nan_pct,count))
        normalized_complete_df.to_csv(normalized_complete_df_name, index=False)

        normalized_missing_df_name = os.path.join(args.input_dir, 'normalized_missing_df_{0}_ver{1}.csv'.format(args.nan_pct,count))
        normalized_missing_df.to_csv(normalized_missing_df_name, index=False)

        enc_complete_df = OHE(normalized_complete_df, ['Activity'])
        enc_missing_df = OHE(normalized_missing_df, ['Activity'])

        c_df = enc_complete_df.copy()
        m_df = enc_missing_df.copy()
        enc_complete_df_w_normalized_time = c_df.drop(['CompleteTimestamp', 'CumTimeInterval'], axis=1)
        enc_missing_df_w_normalized_time = m_df.drop(['CompleteTimestamp', 'CumTimeInterval'], axis=1)

        c_df = enc_complete_df.copy()
        m_df = enc_missing_df.copy()
        enc_complete_df_w_time = c_df.drop(['CompleteTimestamp', 'NormalizedTime'], axis=1)
        enc_missing_df_w_time = m_df.drop(['CompleteTimestamp', 'NormalizedTime'], axis=1)

        avai_index_df = enc_missing_df_w_time.copy()
        nan_index_df = enc_missing_df_w_time.copy()


        for row in range(enc_missing_df_w_time.shape[0]):
            if np.isnan(enc_missing_df_w_time.loc[row, 'CumTimeInterval']): # if nan Time
                avai_index_df.loc[row, 'CumTimeInterval'] = 0
                nan_index_df.loc[row, 'CumTimeInterval'] = 1
            else:
                avai_index_df.loc[row, 'CumTimeInterval'] = 1
                nan_index_df.loc[row, 'CumTimeInterval'] = 0

        for row in range(enc_missing_df_w_time.shape[0]):
            if np.any(enc_missing_df_w_time.iloc[row,2:]>0): #if avai Time
                avai_index_df.iloc[row, 2:] = 1
                nan_index_df.iloc[row, 2:] = 0
            else:
                avai_index_df.iloc[row, 2:] = 0
                nan_index_df.iloc[row, 2:] = 1


        pad_index_df = enc_complete_df.copy()
        cols = [x for x in list(pad_index_df) if x != 'CaseID']
        pad_index_df.loc[:, cols] = 1
        #padding_df[cols] = 1


        enc_missing_df_w_normalized_time.fillna(0, inplace=True)
        enc_missing_df_w_time.fillna(0, inplace=True)


        enc_complete_w_normalized_time_groupByCase = enc_complete_df_w_normalized_time.groupby(['CaseID'])
        enc_missing_w_normalized_time_groupByCase = enc_missing_df_w_normalized_time.groupby(['CaseID'])

        enc_complete_w_time_groupByCase = enc_complete_df_w_time.groupby(['CaseID'])
        enc_missing_w_time_groupByCase = enc_missing_df_w_time.groupby(['CaseID'])

        avai_index_df_groupByCase = avai_index_df.groupby(['CaseID'])
        nan_index_df_groupByCase = nan_index_df.groupby(['CaseID'])
        pad_index_df_groupByCase = pad_index_df.groupby(['CaseID'])

        maxlen = groupByCase.size().max()

        cols_w_time = [i for i in list(enc_complete_df_w_time) if i != 'CaseID']

        cols_w_normalized_time = [i for i in list(enc_complete_df_w_normalized_time) if i != 'CaseID']

        vectorized_complete_df_w_normalized_time = getInput(enc_complete_w_normalized_time_groupByCase, cols_w_normalized_time, maxlen)
        vectorized_missing_df_w_normalized_time = getInput(enc_missing_w_normalized_time_groupByCase, cols_w_normalized_time, maxlen)

        vectorized_complete_df_w_time = getInput(enc_complete_w_time_groupByCase, cols_w_time, maxlen)
        vectorized_missing_df_w_time = getInput(enc_missing_w_time_groupByCase, cols_w_time, maxlen)

        vectorized_avai_index_df = getInput(avai_index_df_groupByCase, cols_w_time, maxlen)
        vectorized_nan_index_df = getInput(nan_index_df_groupByCase, cols_w_time, maxlen)
        vectorized_pad_index_df = getInput(pad_index_df_groupByCase, cols_w_time, maxlen)

        complete_matrix_w_normalized_time_train = vectorized_complete_df_w_normalized_time[:train_size]
        missing_matrix_w_normalized_time_train = vectorized_missing_df_w_normalized_time[:train_size]

        avai_matrix_train = vectorized_avai_index_df[:train_size]
        nan_matrix_train = vectorized_nan_index_df[:train_size]

        #validate
        complete_matrix_w_normalized_time_val = vectorized_complete_df_w_normalized_time[train_size:train_size+val_size]
        missing_matrix_w_normalized_time_val = vectorized_missing_df_w_normalized_time[train_size:train_size+val_size]

        avai_matrix_val = vectorized_avai_index_df[train_size:train_size+val_size]
        nan_matrix_val = vectorized_nan_index_df[train_size:train_size+val_size]
        pad_matrix_val = vectorized_pad_index_df[train_size:train_size+val_size]

        #test
        complete_matrix_w_normalized_time_test = vectorized_complete_df_w_normalized_time[train_size+val_size:]
        missing_matrix_w_normalized_time_test = vectorized_missing_df_w_normalized_time[train_size+val_size:]

        avai_matrix_test = vectorized_avai_index_df[train_size+val_size:]
        nan_matrix_test = vectorized_nan_index_df[train_size+val_size:]
        pad_matrix_test = vectorized_pad_index_df[train_size+val_size:]

        print('Checking number of available instances in test set:...')
        check_avai = avai_matrix_test.copy()
        check_avai = check_avai.reshape(avai_matrix_test.shape[0]*avai_matrix_test.shape[1], avai_matrix_test.shape[2])
        check_avai = check_avai[np.all(check_avai == 1, axis=1)]
        print('Number of available row: {}'.format(check_avai.shape[0]))
        print(check_avai.shape[0] == avai_instance)


        #check number of nan instances in test set
        print('Checking number of nan instances in test set:...')
        check_nan = nan_matrix_test.copy()
        check_nan = check_nan.reshape(nan_matrix_test.shape[0]*nan_matrix_test.shape[1], nan_matrix_test.shape[2])
        check_nan = check_nan[np.any(check_nan != 0, axis=1)]
        print('Number of nan row: {}'.format(check_nan.shape[0]))
        print(check_nan.shape[0] == nan_instance)


        # In[41]:


        preprocessed_data_name = os.path.join(args.input_dir, 'preprocessed_data_{0}_ver{1}.pkl'.format(args.nan_pct,count))
        with open(preprocessed_data_name, 'wb') as f:
            pickle.dump(min_max_storage, f, protocol=2)
            pickle.dump(complete_matrix_w_normalized_time_train, f, protocol=2)
            pickle.dump(missing_matrix_w_normalized_time_train, f, protocol=2)
            pickle.dump(avai_matrix_train, f, protocol=2)
            pickle.dump(nan_matrix_train, f, protocol=2)
            pickle.dump(complete_matrix_w_normalized_time_val, f, protocol=2)
            pickle.dump(missing_matrix_w_normalized_time_val, f, protocol=2)
            pickle.dump(avai_matrix_val, f, protocol=2)
            pickle.dump(nan_matrix_val, f, protocol=2)
            pickle.dump(pad_matrix_val, f, protocol=2)
            pickle.dump(complete_matrix_w_normalized_time_test, f, protocol=2)
            pickle.dump(missing_matrix_w_normalized_time_test, f, protocol=2)
            pickle.dump(avai_matrix_test, f, protocol=2)
            pickle.dump(nan_matrix_test, f, protocol=2)
            pickle.dump(pad_matrix_test, f, protocol=2)
            pickle.dump(cols_w_time, f, protocol=2)
            pickle.dump(cols_w_normalized_time, f, protocol=2)

end_time = time.time()
print(end_time-start_time)
'''
