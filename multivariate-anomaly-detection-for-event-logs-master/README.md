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

