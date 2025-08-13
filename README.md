# SetAD
SetAD is implemented using PyTorch. You may need the packages below to run CAD:

- numpy==1.23.1
- torch==1.13.1
- scikit-learn==1.0.2

Raw data is accessed from [ODDS dataset](https://odds.cs.stonybrook.edu/) and [ADbench](https://github.com/Minqi824/ADBench?tab=readme-ov-file) and processed using [data_preprocess.py](/data_preprocess.py), and processed data can be found in [processed_data](/processed_data). 

To perfrom anomaly detection, run

    python SetAD-main.py --dataset Your-dataset --batch_size batch-size --labeled_ratio ratio-of-labeled-anomalies --contamination_rate Contamination_rate

The program will read the propossed data accoring to the specified dataset name from the _processed_data_ dictionary.
