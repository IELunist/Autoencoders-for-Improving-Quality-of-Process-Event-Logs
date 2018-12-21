
# coding: utf-8

# In[2]:


import os, sys
import argparse
import pandas as pd
import numpy as np
import pickle


# In[4]:


from dateutil.parser import parse
from datetime import datetime
import time
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

get_ipython().run_line_magic('matplotlib', 'inline')
pd.options.mode.chained_assignment = None #to run loop quicker without warnings


# In[5]:


#name = 'bpi_2012'
name = 'bpi_2013'
#name = 'small_log'
#name = 'large_log'

args = {
    'data_dir': '../data/',
    'data_file': name + '.csv',
    'input_dir': '../input/{}/'.format(name),  
    'train_pct': 0.6,
    'val_pct': 0.2,
    'anomaly_pct': 0.1,
    'scaler': 'standardization',
}

args = argparse.Namespace(**args)


# In[6]:


if not os.path.isdir('../input/'):
    os.makedirs('../input/')
    
if not os.path.isdir(args.input_dir):
    os.makedirs(args.input_dir)


# In[9]:


sys.path.insert(0, './../utils/')
from utils import *


# In[10]:


preprocessed_data_name = os.path.join(args.input_dir, 'preprocessed_data_{}.pkl'.format(args.anomaly_pct))
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

# In[ ]:


normal_df_name = os.path.join(args.input_dir, 'normal_df_{}.csv'.format(args.anomaly_pct))
normal_df = pd.read_csv(normal_df_name)

anomalous_df_name = os.path.join(args.input_dir, 'anomolous_df_{}.csv'.format(args.anomaly_pct))
anomalous_df = pd.read_csv(anomalous_df_name)


# In[ ]:


normal_df.head()


# In[ ]:


anomalous_df.head()


# # Histogram

# In[ ]:


def histogram_plot(df, activity):
    selected_df = df[df['Activity']==activity]['Duration']
    selected_df.hist()
    plt.axvline(selected_df.mean(), color='w', linestyle='dashed', linewidth=2)
    plt.axvline(selected_df.mean()+selected_df.std(), color='r', linestyle='dashed', linewidth=2)
    plt.title('Histogram of '+ act)
    plt.savefig(args.input_dir + 'histogram_'+act)
    plt.show()
    plt.close()


# In[ ]:


act_list = normal_df['Activity'].unique()
print('Activity: {}'.format(act_list))


# In[ ]:


for act in act_list:
    histogram_plot(normal_df, act)


# # Duration

# ## Full set

# In[ ]:


temp = pd.DataFrame({'Activity': normal_df['Activity'].copy(),
                     'AnomalousDuration': anomalous_df['AnomalousDuration'].copy(),
                     'TimeLabel': anomalous_df['TimeLabel'].copy()})


# In[ ]:


temp.head()


# In[ ]:


groupByActivity = temp.groupby(['Activity'])


# In[ ]:


def plotDuration(activity, df, save=True):
    fig, ax = plt.subplots(figsize=(10, 5))
    anomaly = df[df['TimeLabel']==1]
    normal = df[df['TimeLabel']==0]
    #ax.plot(anomaly.index, anomaly.AnomalousDuration, marker='o', ms=3.5, linestyle='', color='green', label='Anomalous data: '+str(len(anomaly)))
    #ax.plot(normal.index, normal.AnomalousDuration, marker='o', ms=3.5, linestyle='', color='blue', label='Normal data: '+str(len(normal)))
    #ax.hlines(statistics_storage[activity]['mean']+statistics_storage[activity]['std'], ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Border')
    ax.plot(anomaly.index, anomaly.AnomalousDuration, marker='o', ms=3.5, linestyle='', color='green', label=str(len(anomaly)))
    ax.plot(normal.index, normal.AnomalousDuration, marker='o', ms=3.5, linestyle='', color='cornflowerblue', label=str(len(normal)))
    ax.hlines(statistics_storage[activity]['mean']+statistics_storage[activity]['std'], ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100)
    #plt.title('Duration of '+ activity)
    plt.xlabel('Data point index')
    plt.ylabel('Duration')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save == True:
        plt.savefig(args.input_dir + 'duration_'+act, bbox_inches='tight')
    plt.show()
    plt.close()


# In[ ]:


for act, group in groupByActivity:
    plotDuration(act, group, True)


# ## Test set

# In[ ]:


temp_test = temp[-test_row_num:]


# In[ ]:


groupByActivity = temp_test.groupby(['Activity'])

for act, group in groupByActivity:
    plotDuration(act, group, False)

