pydata

python induce_missing_data.py -n small_log --nan_pct 0.3
python induce_missing_data.py -n small_log --nan_pct 0.35
python induce_missing_data.py -n small_log --nan_pct 0.4
python induce_missing_data.py -n small_log --nan_pct 0.5

python induce_missing_data.py -n large_log --nan_pct 0.3
python induce_missing_data.py -n large_log --nan_pct 0.35
python induce_missing_data.py -n large_log --nan_pct 0.4
python induce_missing_data.py -n large_log --nan_pct 0.5



python preprocess_variables.py -n small_log --nan_pct 0.3
python preprocess_variables.py -n small_log --nan_pct 0.35
python preprocess_variables.py -n small_log --nan_pct 0.4
python preprocess_variables.py -n small_log --nan_pct 0.5

python preprocess_variables.py -n large_log --nan_pct 0.3
python preprocess_variables.py -n large_log --nan_pct 0.35
python preprocess_variables.py -n large_log --nan_pct 0.4
python preprocess_variables.py -n large_log --nan_pct 0.5



python preprocess_variables_full.py -n small_log --nan_pct 0.3
python preprocess_variables_full.py -n small_log --nan_pct 0.35
python preprocess_variables_full.py -n small_log --nan_pct 0.4
python preprocess_variables_full.py -n small_log --nan_pct 0.5

python preprocess_variables_full.py -n large_log --nan_pct 0.3
python preprocess_variables_full.py -n large_log --nan_pct 0.35
python preprocess_variables_full.py -n large_log --nan_pct 0.4
python preprocess_variables_full.py -n large_log --nan_pct 0.5

pyexit
