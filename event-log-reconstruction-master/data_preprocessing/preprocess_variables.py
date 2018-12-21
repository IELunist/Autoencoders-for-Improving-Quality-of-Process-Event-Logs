
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


# In[2]:


from dateutil.parser import parse
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.options.mode.chained_assignment = None #to run loop quicker without warnings


# In[3]:


#name = 'bpi_2012'
#name = 'bpi_2013'
name = 'small_log'
#name = 'large_log'

args = {
    'data_dir': '../data/',
    'data_file': name + '.csv',
    'input_dir': '../input/{}/'.format(name),  
    'nan_pct': 0.3,
    'train_pct': 0.6,
    'val_pct': 0.2,
}

args = argparse.Namespace(**args)


# In[4]:


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


# In[5]:


sys.path.insert(0, './../utils/')
from utils import *


# # Load data

# In[6]:


df_name = os.path.join(args.input_dir, 'complete_df_full_{}.csv'.format(args.nan_pct))
df = pd.read_csv(df_name)

missing_df_name = os.path.join(args.input_dir, 'missing_df_full_{}.csv'.format(args.nan_pct))
missing_df = pd.read_csv(missing_df_name)


# # Preprocess data

# **To do:**
# - Normalize CumTimeInterval using minmaxScaler for each case.
# - One hot encode
# - Get avai/nan mask
# - Replace nan with 0
# - Pad with 0

# In[7]:


df.head()


# In[8]:


len(df['Activity'].unique())


# In[9]:


missing_df.head()


# ## Normalize

# In[10]:


groupByCase = df.groupby(['CaseID'])
missing_groupByCase = missing_df.groupby(['CaseID'])


# In[11]:


normalized_complete_df = pd.DataFrame(columns=list(df)+['NormalizedTime'])
normalized_missing_df = pd.DataFrame(columns=list(df)+['NormalizedTime'])
min_max_storage = {}

for i, j in zip(groupByCase, missing_groupByCase):
    temp, missing_temp, missing_case_storage = minmaxScaler(i[0], i[1], j[1])
    normalized_complete_df = normalized_complete_df.append(temp)
    normalized_missing_df = normalized_missing_df.append(missing_temp)
    min_max_storage.update(missing_case_storage)


# In[12]:


len(min_max_storage), len(groupByCase), len(missing_groupByCase)


# In[13]:


normalized_complete_df.shape


# In[14]:


normalized_missing_df.shape


# In[15]:


normalized_complete_df_name = os.path.join(args.input_dir, 'normalized_complete_df_{}.csv'.format(args.nan_pct))
normalized_complete_df.to_csv(normalized_complete_df_name, index=False)

normalized_missing_df_name = os.path.join(args.input_dir, 'normalized_missing_df_{}.csv'.format(args.nan_pct))
normalized_missing_df.to_csv(normalized_missing_df_name, index=False)


# ## One hot encode

# - One hot encode for all categorical variables
# - All columns are 0 for nan value

# In[16]:


cat_var = ['Activity']


# In[17]:


# OHE: get k dummies out of k categorical levels (drop_first=False)
enc_complete_df = OHE(normalized_complete_df, cat_var)
enc_missing_df = OHE(normalized_missing_df, cat_var)


# In[18]:


enc_complete_df.head()


# In[19]:


enc_missing_df.head()


# ## Get masks

# **Note:**
# - **nan_index_df**: 1: missing element, 0: available element 
# - **avai_index_df**: 1: available element, 0: missing element
# - **pad_index_df**: 1: all element (prepare for padding later)

# ### Mask for nan and avai

# In[20]:


c_df = enc_complete_df.copy()
m_df = enc_missing_df.copy()
enc_complete_df_w_normalized_time = c_df.drop(['CompleteTimestamp', 'CumTimeInterval'], axis=1)
enc_missing_df_w_normalized_time = m_df.drop(['CompleteTimestamp', 'CumTimeInterval'], axis=1)


# In[21]:


c_df = enc_complete_df.copy()
m_df = enc_missing_df.copy()
enc_complete_df_w_time = c_df.drop(['CompleteTimestamp', 'NormalizedTime'], axis=1)
enc_missing_df_w_time = m_df.drop(['CompleteTimestamp', 'NormalizedTime'], axis=1)


# In[22]:


avai_index_df = enc_missing_df_w_time.copy()
nan_index_df = enc_missing_df_w_time.copy()


# In[23]:


#mask for Time
for row in range(enc_missing_df_w_time.shape[0]):
    if np.isnan(enc_missing_df_w_time.loc[row, 'CumTimeInterval']): # if nan Time
        avai_index_df.loc[row, 'CumTimeInterval'] = 0
        nan_index_df.loc[row, 'CumTimeInterval'] = 1
    else:
        avai_index_df.loc[row, 'CumTimeInterval'] = 1
        nan_index_df.loc[row, 'CumTimeInterval'] = 0


# In[24]:


nan_index_df.head()


# In[25]:


#mask for Activity
for row in range(enc_missing_df_w_time.shape[0]):
    if np.any(enc_missing_df_w_time.iloc[row,2:]>0): #if avai Time
        avai_index_df.iloc[row, 2:] = 1
        nan_index_df.iloc[row, 2:] = 0
    else:
        avai_index_df.iloc[row, 2:] = 0
        nan_index_df.iloc[row, 2:] = 1


# In[26]:


nan_index_df.head()


# ### Mask for 0-padding

# In[27]:


pad_index_df = enc_complete_df.copy()
cols = [x for x in list(pad_index_df) if x != 'CaseID']
pad_index_df.loc[:, cols] = 1
#padding_df[cols] = 1


# ## Replace nan with 0

# In[28]:


enc_missing_df_w_normalized_time.head()


# In[29]:


enc_missing_df_w_normalized_time.fillna(0, inplace=True)
enc_missing_df_w_time.fillna(0, inplace=True)


# In[30]:


enc_missing_df_w_normalized_time.head()


# ## Pad with 0

# At this point, eliminate 'CaseID'

# **To do:**
# - Find longest case
# - Vectorize based on CaseID to get input

# In[31]:


enc_complete_w_normalized_time_groupByCase = enc_complete_df_w_normalized_time.groupby(['CaseID'])
enc_missing_w_normalized_time_groupByCase = enc_missing_df_w_normalized_time.groupby(['CaseID'])

enc_complete_w_time_groupByCase = enc_complete_df_w_time.groupby(['CaseID'])
enc_missing_w_time_groupByCase = enc_missing_df_w_time.groupby(['CaseID'])

avai_index_df_groupByCase = avai_index_df.groupby(['CaseID'])
nan_index_df_groupByCase = nan_index_df.groupby(['CaseID'])
pad_index_df_groupByCase = pad_index_df.groupby(['CaseID'])


# In[32]:


maxlen = findLongestLength(groupByCase)
print('Length of longest case: {}'.format(maxlen))


# In[33]:


cols_w_time = [i for i in list(enc_complete_df_w_time) if i != 'CaseID']
cols_w_time


# In[34]:


cols_w_normalized_time = [i for i in list(enc_complete_df_w_normalized_time) if i != 'CaseID']
cols_w_normalized_time


# In[35]:


vectorized_complete_df_w_normalized_time = getInput(enc_complete_w_normalized_time_groupByCase, cols_w_normalized_time, maxlen)
vectorized_missing_df_w_normalized_time = getInput(enc_missing_w_normalized_time_groupByCase, cols_w_normalized_time, maxlen)

vectorized_complete_df_w_time = getInput(enc_complete_w_time_groupByCase, cols_w_time, maxlen)
vectorized_missing_df_w_time = getInput(enc_missing_w_time_groupByCase, cols_w_time, maxlen)

vectorized_avai_index_df = getInput(avai_index_df_groupByCase, cols_w_time, maxlen)
vectorized_nan_index_df = getInput(nan_index_df_groupByCase, cols_w_time, maxlen)
vectorized_pad_index_df = getInput(pad_index_df_groupByCase, cols_w_time, maxlen)


# In[36]:


vectorized_complete_df_w_normalized_time.shape, vectorized_missing_df_w_normalized_time.shape


# In[37]:


vectorized_complete_df_w_time.shape, vectorized_missing_df_w_time.shape


# In[38]:


vectorized_avai_index_df.shape, vectorized_nan_index_df.shape, vectorized_pad_index_df.shape


# # Split and save data

# In[39]:


# Split: 70% train, 10% validate, 20% test

#train
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


# In[40]:


#check number of avai instances in test set
print('Checking number of available instances in test set:...')
check_avai = avai_matrix_test.copy()
check_avai = check_avai.reshape(avai_matrix_test.shape[0]*avai_matrix_test.shape[1], avai_matrix_test.shape[2])
check_avai = check_avai[np.all(check_avai == 1, axis=1)]
print('Number of available row: {}'.format(check_avai.shape[0]))
print(check_avai.shape[0] == avai_instance)

print('\n')

#check number of nan instances in test set
print('Checking number of nan instances in test set:...')
check_nan = nan_matrix_test.copy()
check_nan = check_nan.reshape(nan_matrix_test.shape[0]*nan_matrix_test.shape[1], nan_matrix_test.shape[2])
check_nan = check_nan[np.any(check_nan != 0, axis=1)]
print('Number of nan row: {}'.format(check_nan.shape[0]))
print(check_nan.shape[0] == nan_instance)


# In[41]:


preprocessed_data_name = os.path.join(args.input_dir, 'preprocessed_data_{}.pkl'.format(args.nan_pct))
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

