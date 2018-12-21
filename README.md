# Multivariate Anomaly Detection for Event Logs

### Directory structure:

```
multivariate-anomaly-detection-for-event-logs
│   README.md
|   requirement.txt
|
|--- data: original dataset
│   │   bpi_2013.csv
|   |   bpi_2012.csv
|   |   small_log.csv
|   |   large_log.csv
|
|--- data_preprocessing
|   |   data_preparation.ipynb
|   |   data_exploration.ipynb
|   |   descriptive-statistics.ipynb
|
|--- utils
|   |   utils.py
|   |   models.py
|
|--- input: preprocessed data
|
|--- experiment
|   |   output
|   |   VAE.ipynb
|   |   AE.ipynb
|   |   LSTMAE.ipynb
|   



```
### Reference


1. Install requirement

- Install pytorch: ```conda install pytorch torchvision -c soumith```
- ```pip install -r requirements.txt```

2. Run ```data_preparation.ipynb```
3. Run ```VAE.ipynb``` or ```AE.ipynb``` or ```LSTMAE.ipynb```


# Event log Reconstruction

### Directory structure:

```
event-log-reconstruction
│   README.md
│   requirement.txt
|
|--- data: original dataset
│   │   bpi_2013.csv
|   |   bpi_2012.csv
|   |   small_log.csv
|   |   large_log.csv
|
|--- data_preprocessing
|   |   induce_missing_data.py
|   │   preprocess_variables.py
|   |   real_log_preprocessing.sh
|
|--- utils
|   |   utils.py
|   |   models.py
|
|--- input: preprocessed data
|
|--- experiment
|   |   output
|   |   AE.ipynb
|   |   VAE.ipynb
|   |   LSTMAE.ipynb
|
|-- base_model
|   |   dummy_imputation.ipynb
|   |   statistical_description.ipynb
|


```
### Reference


1. Install requirement

- Install pytorch: ```conda install pytorch torchvision -c soumith```
- Install requirements: ```pip install -r requirements.txt```

2. For preprocessing: 
- ```cd data_preprocessing```
- ```source real_log_preprocessing.sh```

3. For training and evaluating:
- ```cd experiment```
- Run ```AE.ipynb``` or ```VAE.ipynb``` or ```LSTMAE```

4. For baseline models:
- ```cd base_model```
- Run ```dummy_imputation.ipynb```
