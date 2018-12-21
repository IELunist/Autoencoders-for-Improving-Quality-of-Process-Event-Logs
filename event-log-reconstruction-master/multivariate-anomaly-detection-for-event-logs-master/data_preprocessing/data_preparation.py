
# coding: utf-8

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


# In[4]:

start_time = time.time()
names = ['bpi_2013','small_log','large_log','bpi_2012']
for name in names:
    for count in range(10):
        print('\n')
        print('Name : %s Count : %s'%(name,count))
        args = {
            'data_dir': '../data/',
            'data_file': name + '.csv',
            'input_dir': '../input/ver 1/{}/'.format(name),
            'train_pct': 0.6,
            'val_pct': 0.2,
            'anomaly_pct': 0.1,
            'scaler': 'standardization',
        }

        args = argparse.Namespace(**args)



        # In[5]:


        if not os.path.isdir('../input/'):
            os.makedirs('../input/')

        if not os.path.isdir(args.input_dir):
            os.makedirs(args.input_dir)


        # In[6]:


        sys.path.insert(0, './../utils/')
        from utils import *


        # # Load data

        # In[7]:


        # Only consider Case, Activity, Timestamp
        cols = ['CaseID', 'Activity', 'CompleteTimestamp']

        # For Timestamp: Convert to time
        data = pd.read_csv(args.data_dir + args.data_file, usecols=['Case ID', 'Activity', 'Complete Timestamp'])
        data['Case ID'] = data['Case ID'].apply(lambda x: x.split(' ')[1])


        # Format for each column
        data.columns = cols
        data['CompleteTimestamp'] = pd.to_datetime(data['CompleteTimestamp'], errors='coerce')
        data['CaseID'] = data['CaseID'].apply(pd.to_numeric)

        # In[9]:

        #Calculate duration and cumulative duration
        groupByCase = data.groupby(['CaseID'])
        duration_df = pd.DataFrame(pd.DataFrame(columns=list(data)+['Duration', 'CumDuration']))

        groups=[]

        for case, group in groupByCase:
            if case % 1000 ==0:
                print('%s/%s'%(case,len(groupByCase)))
            group = calculateDuration(group)
            group = calculateCumDuration(group)
            group['Duration'] = group['Duration'].apply(convert2seconds)
            group['CumDuration'] = group['CumDuration'].apply(convert2seconds)
            groups.append(group)

        duration_df = pd.concat([duration_df]+groups)

        # In[11]:


        #get statistics storage for activity
        groupByActivity = duration_df.groupby(['Activity'])
        statistics_storage = {}

        for act, act_data in groupByActivity:
            act_storage = {}
            act_storage[act] = {}
            mean_value = act_data['Duration'].mean()
            std_value = act_data['Duration'].std()
            act_storage[act]['mean'] = mean_value
            act_storage[act]['std'] = std_value
            statistics_storage.update(act_storage)

        # In[12]:


        print('Descriptive statistics: \n{}'.format(statistics_storage))


        # In[13]:


        act_list = data['Activity'].unique()


        # # Introduce anomalous data
        # In[16]:


        anomaly_num = int(data.shape[0]*(data.shape[1]-1)*args.anomaly_pct)
        anomalous_act_num = int(anomaly_num/2)
        anomalous_time_num = anomaly_num - anomalous_act_num

        print('Number of anomalous values: {}'.format(anomaly_num))
        print('Number of anomalous activities: {}'.format(anomalous_act_num))
        print('Number of anomalous time: {}'.format(anomalous_time_num))


        # ## Activity

        # **Mutation:**
        # - Replace an activity by another

        # In[17]:


        temp_act_df = pd.DataFrame({'Activity': duration_df['Activity'].copy(),
                                    'AnomalousActivity': duration_df['Activity'].copy(),
                                    'ActivityLabel': 0})


        # In[19]:


        rows = np.random.choice(temp_act_df.shape[0],anomalous_act_num,replace=False)
        anoactlist=[]
        for row in rows:
            act = temp_act_df.loc[row, 'Activity']
            anomalous_act_list = [i for i in act_list if i != act]
            anomalous_act = anomalous_act_list[int(np.random.choice(len(anomalous_act_list),1))]
            anoactlist.append(anomalous_act)
        temp_act_df.loc[rows, 'AnomalousActivity'] = anoactlist
        temp_act_df.loc[rows, 'ActivityLabel'] = 1



        # In[21]:


        temp_act = temp_act_df[['AnomalousActivity', 'ActivityLabel']]


        # In[22]:
        # ## Time

        # **Mutation:**
        # - Extreme duration

        # In[23]:

        temp_time_df = duration_df.copy()
        temp_time_df['AnomalousDuration'] = temp_time_df['Duration'].copy()
        temp_time_df['TimeLabel'] = 0

        # In[25]:

        #get anomalous duration
        anomalous_time_index = []

        timerows = list(np.random.choice(temp_act_df.shape[0],anomalous_time_num,replace=False))
        anomalous_values=[]
        changedrow=[]
        for row in timerows:
            anomalous_time_index.append(row)
            act = temp_time_df.loc[row, 'Activity']
            if act != 'A_SUBMITTED-COMPLETE' and act != 'Activity A':
                anomalous_value = (np.random.random_sample() + 1)*(statistics_storage[act]['mean'] + statistics_storage[act]['std'])
                anomalous_values.append(anomalous_value)
                changedrow.append(row)
        temp_time_df.loc[changedrow, 'AnomalousDuration'] = anomalous_values
        temp_time_df.loc[changedrow, 'TimeLabel'] = 1

        # In[26]:

        # In[27]:


        #get anomalous cumulative duration
        temp_cum_time_df = pd.DataFrame(columns=list(temp_time_df)+['AnomalousCompleteTimestamp'])
        groupByCase = temp_time_df.groupby(['CaseID'])

        for case, group in groupByCase:
            if case % 1000 == 0:
                print('%s/%s'%(case,len(groupByCase)))
            group['AnomalousCompleteTimestamp'] = group['CompleteTimestamp'].copy()
            if group['TimeLabel'].sum() > 0:
                for row in range(group.shape[0]-1):
                    previous_timestamp = group['CompleteTimestamp'].iloc[row]
                    current_duration = group['AnomalousDuration'].iloc[row+1]
                    current_timestamp = previous_timestamp + timedelta(seconds=current_duration)
                    group['AnomalousCompleteTimestamp'].iloc[row+1] = current_timestamp
            temp_cum_time_df = temp_cum_time_df.append(group)

        # In[28]:


        # In[29]:


        groupByCase = temp_cum_time_df.groupby(['CaseID'])
        temp_time = pd.DataFrame(pd.DataFrame(columns=list(temp_cum_time_df)+['AnomalousCumDuration']))
        groups = []
        for case, group in groupByCase:
            group = calculateAnomalousCumDuration(group)
            group['AnomalousCumDuration'] = group['AnomalousCumDuration'].apply(convert2seconds)
            groups.append(group)
        temp_time = pd.concat([temp_time]+groups)

        # ## Get full df

        # In[31]:

        full_df = pd.concat([temp_time, temp_act], axis=1)

        # In[34]:


        normal_df = full_df[['CaseID', 'Activity', 'CompleteTimestamp', 'Duration', 'CumDuration']]
        anomalous_df = full_df[['CaseID', 'AnomalousActivity', 'AnomalousCompleteTimestamp', 'AnomalousDuration',
                                'AnomalousCumDuration', 'ActivityLabel', 'TimeLabel']]


        # In[35]:


        print('Saving dataframes...')
        normal_df_name = os.path.join(args.input_dir, 'normal_df_%s_%s.csv'%(args.anomaly_pct,count))
        normal_df.to_csv(normal_df_name, index=False)

        anomalous_df_name = os.path.join(args.input_dir, 'anomolous_df_%s_%s.csv'%(args.anomaly_pct,count))
        anomalous_df.to_csv(anomalous_df_name, index=False)
        print('Done!')


        # # Preprocess data

        # In[36]:


        groupByCase = anomalous_df.groupby(['CaseID'])

        # Split: 60% train, 20% validate, 20% test
        train_case_num = int(len(groupByCase)*args.train_pct)
        val_case_num = int(len(groupByCase)*args.val_pct)
        test_case_num = len(groupByCase) - train_case_num - val_case_num


        # In[37]:


        anomalous_df_train = pd.DataFrame(columns=list(anomalous_df))
        anomalous_df_val = pd.DataFrame(columns=list(anomalous_df))
        anomalous_df_test = pd.DataFrame(columns=list(anomalous_df))

        for caseid, data_case in groupByCase:
            if caseid <= train_case_num:
                anomalous_df_train = anomalous_df_train.append(data_case)
            elif train_case_num < caseid <= (train_case_num+val_case_num):
                anomalous_df_val = anomalous_df_val.append(data_case)
            else:
                anomalous_df_test = anomalous_df_test.append(data_case)


        # In[38]:


        print('Checking shapes of sub data: ', anomalous_df.shape[0] == anomalous_df_train.shape[0] + anomalous_df_val.shape[0] + anomalous_df_test.shape[0])


        # In[39]:


        train_row_num = anomalous_df_train.shape[0]
        val_row_num = anomalous_df_val.shape[0]
        test_row_num = anomalous_df_test.shape[0]

        print('Number of rows for training: {}'.format(train_row_num))
        print('Number of rows for val: {}'.format(val_row_num))
        print('Number of rows for testing: {}'.format(test_row_num))


        # In[40]:


        print('Number of anomalous values in train set: {}'.format(anomalous_df_train['ActivityLabel'].sum() + anomalous_df_train['TimeLabel'].sum()))
        print('Number of anomalous activities in train set: {}'.format(anomalous_df_train['ActivityLabel'].sum()))
        print('Number of anomalous time in train set: {}'.format(anomalous_df_train['TimeLabel'].sum()))
        print('\n')
        print('Number of anomalous values in validate set: {}'.format(anomalous_df_val['ActivityLabel'].sum() + anomalous_df_val['TimeLabel'].sum()))
        print('Number of anomalous activities in validate set: {}'.format(anomalous_df_val['ActivityLabel'].sum()))
        print('Number of anomalous time in validate set: {}'.format(anomalous_df_val['TimeLabel'].sum()))
        print('\n')
        print('Number of anomalous values in test set: {}'.format(anomalous_df_test['ActivityLabel'].sum() + anomalous_df_test['TimeLabel'].sum()))
        print('Number of anomalous activities in test set: {}'.format(anomalous_df_test['ActivityLabel'].sum()))
        print('Number of anomalous time in test set: {}'.format(anomalous_df_test['TimeLabel'].sum()))


        # # Prepare input



        # ## Labels

        # In[42]:


        activity_label = anomalous_df['ActivityLabel']
        time_label = anomalous_df['TimeLabel']


        # In[43]:


        activity_label_train = activity_label[:train_row_num]
        activity_label_val = activity_label[train_row_num:train_row_num+val_row_num]
        activity_label_test = activity_label[-test_row_num:]

        time_label_train = time_label[:train_row_num]
        time_label_val = time_label[train_row_num:train_row_num+val_row_num]
        time_label_test = time_label[-test_row_num:]


        # In[45]:


        anomaly = anomalous_df[['CaseID', 'AnomalousActivity', 'AnomalousCumDuration']]


        # ## Activity

        # In[46]:


        cat_var = ['AnomalousActivity']


        # In[47]:


        enc_data = OHE(anomaly, cat_var)


        # ## Time

        # In[49]:


        min_value = np.min(enc_data['AnomalousCumDuration'].iloc[:train_row_num])
        max_value = np.max(enc_data['AnomalousCumDuration'].iloc[:train_row_num])


        # In[50]:


        print('Min used for normalization: {}'.format(min_value))
        print('Max used for normalization: {}'.format(max_value))


        # In[51]:


        mean_value = np.mean(enc_data['AnomalousCumDuration'].iloc[:train_row_num])
        std_value = np.std(enc_data['AnomalousCumDuration'].iloc[:train_row_num])


        # In[52]:


        print('Mean used for standardization: {}'.format(mean_value))
        print('STD used for standardization: {}'.format(std_value))


        # In[53]:


        enc_data['NormalizedCumDuration'] = enc_data['AnomalousCumDuration'].apply(lambda x: (x-min_value)/(max_value-min_value))
        enc_data['StandardizedCumDuration'] = enc_data['AnomalousCumDuration'].apply(lambda x: (x-mean_value)/(std_value))


        # In[55]:


        if args.scaler == 'standardization':
            scaled_enc_data = enc_data.drop(['AnomalousCumDuration', 'NormalizedCumDuration'], axis=1)
        if args.scaler == 'normalization':
            scaled_enc_data = enc_data.drop(['AnomalousCumDuration', 'StandardizedCumDuration'], axis=1)



        # ## 0-padding

        # In[57]:


        #re arrange cols
        cols = list(scaled_enc_data)
        cols = ['CaseID', cols[-1]] + cols[1:-1]
        scaled_enc_data = scaled_enc_data[cols]


        # In[58]:


        scaled_enc_data.head()


        # In[59]:


        true_time = scaled_enc_data.iloc[-test_row_num:, 1]
        true_act = scaled_enc_data.iloc[-test_row_num:, 2:]


        # In[60]:


        full_true_time = scaled_enc_data.iloc[:, 1]
        full_true_act = scaled_enc_data.iloc[:, 2:]


        # In[61]:


        cols = [i for i in list(scaled_enc_data) if i != 'CaseID']
        cols


        # In[62]:


        pad_index = scaled_enc_data.copy()
        pad_index[cols] = 1.0


        # In[63]:


        pad_index.head()


        # ## Vectorize

        # In[64]:


        groupByCase = scaled_enc_data.groupby(['CaseID'])

        maxlen = findLongestLength(groupByCase)
        print('Maxlen: ', maxlen)


        # In[65]:


        vectorized_data = getInput(groupByCase, cols, maxlen)

        pad_index_groupByCase = pad_index.groupby(['CaseID'])
        vectorized_pad_index = getInput(pad_index_groupByCase, cols, maxlen)


        # # Split in to train/val/test

        # In[66]:


        print('Shape of vectorized data: {}'.format(vectorized_data.shape))
        print('Shape of vectorized pad index: {}'.format(vectorized_pad_index.shape))
        print('\n')
        print('Number of case for train: {}'.format(train_case_num))
        print('Number of case for validate: {}'.format(val_case_num))
        print('Number of case for test: {}'.format(test_case_num))


        # In[67]:


        input_train = vectorized_data[0:train_case_num]
        input_val = vectorized_data[train_case_num:train_case_num+val_case_num]
        input_test = vectorized_data[-test_case_num:]

        pad_index_train = vectorized_pad_index[0:train_case_num]
        pad_index_val = vectorized_pad_index[train_case_num:train_case_num+val_case_num]
        pad_index_test = vectorized_pad_index[-test_case_num:]


        # In[68]:


        print('Check shape of input for training: {}'.format(input_train.shape[0]==train_case_num))
        print('Check shape of input for validation: {}'.format(input_val.shape[0]==val_case_num))
        print('Check shape of input for testing: {}'.format(input_test.shape[0]==test_case_num))


        # # Save data

        # In[69]:


        preprocessed_data_name = os.path.join(args.input_dir, 'preprocessed_data_%s_%s.pkl'%(args.anomaly_pct,count))
        with open(preprocessed_data_name, 'wb') as f:
            pickle.dump(input_train, f, protocol=2)
            pickle.dump(input_val, f, protocol=2)
            pickle.dump(input_test, f, protocol=2)
            pickle.dump(pad_index_train, f, protocol=2)
            pickle.dump(pad_index_val, f, protocol=2)
            pickle.dump(pad_index_test, f, protocol=2)
            pickle.dump(activity_label_test, f, protocol=2)
            pickle.dump(time_label_test, f, protocol=2)
            pickle.dump(train_case_num, f, protocol=2)
            pickle.dump(val_case_num, f, protocol=2)
            pickle.dump(test_case_num, f, protocol=2)
            pickle.dump(train_row_num, f, protocol=2)
            pickle.dump(val_row_num, f, protocol=2)
            pickle.dump(test_row_num, f, protocol=2)
            pickle.dump(min_value, f, protocol=2)
            pickle.dump(max_value, f, protocol=2)
            pickle.dump(mean_value, f, protocol=2)
            pickle.dump(std_value, f, protocol=2)
            pickle.dump(cols, f, protocol=2)
            pickle.dump(statistics_storage, f, protocol=2)
            pickle.dump(true_time, f, protocol=2)
            pickle.dump(true_act, f, protocol=2)
            pickle.dump(full_true_time, f, protocol=2)
            pickle.dump(full_true_act, f, protocol=2)

        end_time = time.time()

print(end_time - start_time)
