
# coding: utf-8

# **Outline:**
# - Introduce missing values
# - Split data into train/val/test

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



pd.options.mode.chained_assignment = None #to run loop quicker without warnings


# In[3]:


#name = 'bpi_2012'
#name = 'bpi_2013'
name = 'small_log'
#name = 'large_log'

'''
nan_pct : 0.3, 0.35, 0.4, 0.5
'''

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


if not os.path.isdir('../input/'):
    os.makedirs('../input/')

if not os.path.isdir(args.input_dir):
    os.makedirs(args.input_dir)


# In[5]:


sys.path.insert(0, './../utils/')
from utils import *


# # Load data

# In[6]:


# Only consider Case, Activity, Timestamp
cols = ['CaseID', 'Activity', 'CompleteTimestamp']

# For Timestamp: Convert to time
if name == 'helpdesk':
    data = pd.read_csv(args.data_dir + args.data_file)
else:
    data = pd.read_csv(args.data_dir + args.data_file, usecols=['Case ID', 'Activity', 'Complete Timestamp'])
    data['Case ID'] = data['Case ID'].apply(lambda x: x.split(' ')[1])


# Format for each column
data.columns = cols
data['CompleteTimestamp'] = pd.to_datetime(data['CompleteTimestamp'], errors='coerce')
data['CaseID'] = data['CaseID'].apply(pd.to_numeric)
data['Activity'] = data['Activity'].apply(str)


# In[7]:


data.head()


# # Explore data

# In[8]:


print('There are: {} cases'.format(len(data['CaseID'].unique())))
print('There are: {} activities'.format(len(data['Activity'].unique())))


# In[9]:


print('-----Frequency of different activities-----')
print(data['Activity'].value_counts())


# # Induce missing data

# **To do:**
# - nan_pct: percentage of nan values
# - Induce missingness: 30% data

# In[10]:


data.shape


# In[11]:


total_NA = int(data.shape[0]*(data.shape[1]-1)*args.nan_pct)
print('Number of missing values: {}'.format(total_NA))


# In[12]:


# introduce missing Activities and Timestamps
missing_data = data.copy()
i = 0
while i < total_NA:
    row = np.random.randint(1, data.shape[0]) #exclude first row
    col = np.random.randint(1, data.shape[1]) #exclude CaseID
    if not pd.isnull(missing_data.iloc[row, col]):
        missing_data.iloc[row, col] = np.nan
        i+=1


# In[13]:


print('-----Frequency of different activities-----')
print(missing_data['Activity'].value_counts())


# In[14]:


most_frequent_activity = missing_data['Activity'].value_counts().index[0]
print('Most frequent activity is: {}'.format(most_frequent_activity))


# In[15]:


first_timestamp = missing_data['CompleteTimestamp'][0]


# # Compute CumTimeInverval

# In[16]:


missing_df = calculateCumTimeInterval(missing_data)
missing_df['CumTimeInterval'] = missing_df['CumTimeInterval'].apply(convert2seconds)


# In[17]:


missing_df.head()


# # Split df to train/val/test

# In[18]:


df = calculateCumTimeInterval(data)
df['CumTimeInterval'] = df['CumTimeInterval'].apply(convert2seconds)


# In[19]:


df.head()


# In[20]:


groupByCase = df.groupby(['CaseID'])
missing_groupByCase = missing_df.groupby(['CaseID'])

# Split: 70% train, 10% validate, 20% test
train_size = int(len(groupByCase)*args.train_pct)
val_size = int(len(groupByCase)*args.val_pct)
test_size = len(groupByCase) - train_size - val_size


# In[21]:


df.shape


# In[22]:


df_train = pd.DataFrame(columns=list(df))
df_val = pd.DataFrame(columns=list(df))
df_test = pd.DataFrame(columns=list(df))

for caseid, data_case in groupByCase:
    if caseid <= train_size:
        df_train = df_train.append(data_case)
    elif train_size < caseid <= (train_size+val_size):
        df_val = df_val.append(data_case)
    else:
        df_test = df_test.append(data_case)


# In[23]:


df.shape[0] == df_train.shape[0] + df_val.shape[0] + df_test.shape[0]


# In[24]:


missing_df_train = pd.DataFrame(columns=list(missing_df))
missing_df_val = pd.DataFrame(columns=list(missing_df))
missing_df_test = pd.DataFrame(columns=list(missing_df))

#Note: case start from 1 not 0
for caseid, data_case in missing_groupByCase:
    if caseid <= train_size:
        missing_df_train = missing_df_train.append(data_case)
    elif train_size < caseid <= train_size+val_size:
        missing_df_val = missing_df_val.append(data_case)
    else:
        missing_df_test = missing_df_test.append(data_case)


# In[25]:


missing_df.shape[0] == missing_df_train.shape[0] + missing_df_val.shape[0] + missing_df_test.shape[0]


# In[26]:


len(df_train.groupby(['CaseID'])), len(df_val.groupby(['CaseID'])), len(df_test.groupby(['CaseID']))


# In[27]:


train_size, val_size, test_size


# In[28]:


len(missing_df_train.groupby(['CaseID'])), len(missing_df_val.groupby(['CaseID'])), len(missing_df_test.groupby(['CaseID']))


# In[29]:


#get number of rows
print(df_train.shape, df_val.shape, df_test.shape)
train_row_num = df_train.shape[0]
val_row_num = df_val.shape[0]
test_row_num = df_test.shape[0]


# In[30]:


missing_df_test.head()


# In[31]:


avai_instance = 0
for row in range(len(missing_df_test)):
    if not pd.isnull(missing_df_test['CumTimeInterval'].iloc[row]) and not pd.isnull(missing_df_test['Activity'].iloc[row]):
        avai_instance+=1

print('Number of available row: {}'.format(avai_instance))


# In[32]:


nan_instance = 0
for row in range(len(missing_df_test)):
    if pd.isnull(missing_df_test['CumTimeInterval'].iloc[row]) or pd.isnull(missing_df_test['Activity'].iloc[row]):
        nan_instance+=1

print('Number of nan row: {}'.format(nan_instance))


# In[33]:


missing_df_test.shape[0] == avai_instance + nan_instance


# # Save df

# In[34]:


df_name = os.path.join(args.input_dir, 'complete_df_full_{}.csv'.format(args.nan_pct))
df.to_csv(df_name, index=False)

missing_df_name = os.path.join(args.input_dir, 'missing_df_full_{}.csv'.format(args.nan_pct))
missing_df.to_csv(missing_df_name, index=False)


# In[35]:


#df_train.to_csv(args.input_dir+'complete_df_train.csv', index=False)
#df_val.to_csv(args.input_dir+'complete_df_val.csv', index=False)
#df_test.to_csv(args.input_dir+'complete_df_test.csv', index=False)


# In[36]:


#missing_df_train.to_csv(args.input_dir+'missing_df_train.csv', index=False)
#missing_df_val.to_csv(args.input_dir+'missing_df_val.csv', index=False)
#missing_df_test.to_csv(args.input_dir+'missing_df_test.csv', index=False)


# In[37]:


pd.isnull(missing_df).sum()


# In[38]:


pd.isnull(missing_df_train).sum()


# In[39]:


pd.isnull(missing_df_val).sum()


# In[40]:


pd.isnull(missing_df_test).sum()


# # Save parameters

# In[41]:


file_name = os.path.join(args.input_dir, 'parameters_{}.pkl'.format(args.nan_pct))
with open(file_name, 'wb') as f:
    pickle.dump(most_frequent_activity, f, protocol=2)
    pickle.dump(first_timestamp, f, protocol=2)
    pickle.dump(avai_instance, f, protocol=2)
    pickle.dump(nan_instance, f, protocol=2)
    pickle.dump(train_size, f, protocol=2)
    pickle.dump(val_size, f, protocol=2)
    pickle.dump(test_size, f, protocol=2)
    pickle.dump(train_row_num, f, protocol=2)
    pickle.dump(val_row_num, f, protocol=2)
    pickle.dump(test_row_num, f, protocol=2)
