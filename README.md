# SetAD

SetAD is a semi-supervised anomaly detection framework that scores a sample through
its relationship with a contextual set. Instead of learning only point-wise or
pair-wise anomaly scores, SetAD trains an attention-based set encoder with a
graded regression objective: each sampled set is assigned the number of known
anomalies it contains. At inference time, each test point is evaluated across
multiple randomly sampled contexts and calibrated against reference points from
the same contexts.

## Requirements

SetAD is implemented with PyTorch. The experiments were run with:

- numpy==1.23.1
- torch==1.13.1
- scikit-learn==1.0.2

Raw data is accessed from the [ODDS dataset](https://odds.cs.stonybrook.edu/)
and [ADBench](https://github.com/Minqi824/ADBench?tab=readme-ov-file). Data is
processed with [data_preprocess.py](data_preprocess.py), and processed files are
provided in [processed_data](processed_data).

## Running SetAD

```bash
python SetAD-main.py --dataset Your-dataset --batch_size batch-size --labeled_ratio ratio-of-labeled-anomalies --contamination_rate contamination-rate
```

The program reads the requested dataset from the `processed_data` directory.
The main paper uses contamination rate `0.02` and the labeled-anomaly ratios
listed in the dataset statistics table: `0.05` for Cardiotocography,
Mammography, MNIST, and Fraud, and `0.01` for SpamBase, Shuttle, Celeba, Cover,
Campaign, and Census. The default SetAD set size is `k=8`.

## Main Results with Error Bars

The table below summarizes the main experimental results on the 10 benchmark
datasets. Each value is `mean +/- std` over 10 independent runs, where `std` is
the sample standard deviation (`ddof=1`) across the 10 runs.

### AUC-PR

| Dataset | GANomaly | REPEN | DevNet | DeepSAD | FEAWAD | PReNet | SSIF | SetAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cardiotocography | 0.4457 +/- 0.0527 | 0.7483 +/- 0.0906 | 0.8314 +/- 0.0487 | 0.7505 +/- 0.0623 | 0.7984 +/- 0.0444 | 0.8331 +/- 0.0478 | 0.7377 +/- 0.0390 | 0.8447 +/- 0.0516 |
| Mammography | 0.1946 +/- 0.1274 | 0.4526 +/- 0.0664 | 0.5230 +/- 0.0932 | 0.5115 +/- 0.0921 | 0.4969 +/- 0.1324 | 0.5084 +/- 0.0995 | 0.1427 +/- 0.0418 | 0.5650 +/- 0.0440 |
| SpamBase | 0.7427 +/- 0.0319 | 0.7814 +/- 0.0337 | 0.8314 +/- 0.0453 | 0.6690 +/- 0.0300 | 0.8423 +/- 0.0292 | 0.8295 +/- 0.0405 | 0.7736 +/- 0.0373 | 0.8806 +/- 0.0319 |
| MNIST | 0.2886 +/- 0.0545 | 0.8095 +/- 0.0414 | 0.8331 +/- 0.0394 | 0.7387 +/- 0.0511 | 0.7670 +/- 0.0638 | 0.8038 +/- 0.0499 | 0.5120 +/- 0.0467 | 0.8412 +/- 0.0299 |
| Shuttle | 0.9041 +/- 0.0904 | 0.9655 +/- 0.0143 | 0.9770 +/- 0.0123 | 0.9662 +/- 0.0148 | 0.9696 +/- 0.0170 | 0.9739 +/- 0.0089 | 0.9407 +/- 0.0124 | 0.9745 +/- 0.0063 |
| Celeba | 0.0218 +/- 0.0105 | 0.2201 +/- 0.0131 | 0.2508 +/- 0.0294 | 0.1467 +/- 0.0241 | 0.1906 +/- 0.0362 | 0.2496 +/- 0.0361 | OOM | 0.2653 +/- 0.0354 |
| Cover | 0.0225 +/- 0.0044 | 0.7688 +/- 0.1243 | 0.9547 +/- 0.0157 | 0.8981 +/- 0.0465 | 0.8619 +/- 0.0714 | 0.9482 +/- 0.0231 | OOM | 0.9635 +/- 0.0106 |
| Campaign | 0.2395 +/- 0.0561 | 0.4176 +/- 0.0271 | 0.4264 +/- 0.0218 | 0.3315 +/- 0.0337 | 0.3744 +/- 0.0260 | 0.4158 +/- 0.0258 | 0.4643 +/- 0.0162 | 0.5158 +/- 0.0317 |
| Fraud | 0.1841 +/- 0.1496 | 0.6708 +/- 0.0329 | 0.5852 +/- 0.1395 | 0.6059 +/- 0.0751 | 0.5863 +/- 0.0778 | 0.5301 +/- 0.1682 | OOM | 0.6855 +/- 0.0635 |
| Census | 0.0920 +/- 0.0115 | 0.4025 +/- 0.0231 | 0.4011 +/- 0.0345 | 0.2283 +/- 0.0161 | 0.2103 +/- 0.0700 | 0.3823 +/- 0.0272 | OOM | 0.4719 +/- 0.0138 |

### AUC-ROC

| Dataset | GANomaly | REPEN | DevNet | DeepSAD | FEAWAD | PReNet | SSIF | SetAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cardiotocography | 0.7308 +/- 0.0372 | 0.8930 +/- 0.0522 | 0.9332 +/- 0.0196 | 0.8827 +/- 0.0271 | 0.8975 +/- 0.0201 | 0.9325 +/- 0.0185 | 0.8759 +/- 0.0139 | 0.9433 +/- 0.0164 |
| Mammography | 0.8595 +/- 0.0534 | 0.8971 +/- 0.0197 | 0.9024 +/- 0.0304 | 0.8990 +/- 0.0274 | 0.8656 +/- 0.0514 | 0.8844 +/- 0.0258 | 0.6493 +/- 0.0325 | 0.9032 +/- 0.0144 |
| SpamBase | 0.8261 +/- 0.0201 | 0.8533 +/- 0.0306 | 0.8534 +/- 0.0355 | 0.7575 +/- 0.0177 | 0.8715 +/- 0.0270 | 0.8342 +/- 0.0377 | 0.8204 +/- 0.0195 | 0.9185 +/- 0.0193 |
| MNIST | 0.6995 +/- 0.0561 | 0.9707 +/- 0.0073 | 0.9361 +/- 0.0299 | 0.9427 +/- 0.0133 | 0.9592 +/- 0.0209 | 0.9246 +/- 0.0415 | 0.8760 +/- 0.0181 | 0.9586 +/- 0.0180 |
| Shuttle | 0.9737 +/- 0.0134 | 0.9922 +/- 0.0026 | 0.9886 +/- 0.0059 | 0.9966 +/- 0.0019 | 0.9923 +/- 0.0044 | 0.9856 +/- 0.0035 | 0.9942 +/- 0.0014 | 0.9863 +/- 0.0029 |
| Celeba | 0.3773 +/- 0.1251 | 0.9268 +/- 0.0042 | 0.9372 +/- 0.0089 | 0.8630 +/- 0.0299 | 0.8640 +/- 0.0310 | 0.9327 +/- 0.0129 | OOM | 0.9444 +/- 0.0058 |
| Cover | 0.6257 +/- 0.0388 | 0.9960 +/- 0.0024 | 0.9995 +/- 0.0002 | 0.9929 +/- 0.0066 | 0.9961 +/- 0.0027 | 0.9994 +/- 0.0003 | OOM | 0.9996 +/- 0.0001 |
| Campaign | 0.6623 +/- 0.0536 | 0.8066 +/- 0.0172 | 0.8078 +/- 0.0185 | 0.7624 +/- 0.0272 | 0.7986 +/- 0.0191 | 0.7976 +/- 0.0216 | 0.8706 +/- 0.0059 | 0.8813 +/- 0.0140 |
| Fraud | 0.8309 +/- 0.1160 | 0.9703 +/- 0.0091 | 0.9428 +/- 0.0227 | 0.9450 +/- 0.0162 | 0.9466 +/- 0.0179 | 0.9477 +/- 0.0271 | OOM | 0.9710 +/- 0.0137 |
| Census | 0.6670 +/- 0.0431 | 0.8995 +/- 0.0027 | 0.8125 +/- 0.0171 | 0.7603 +/- 0.0218 | 0.8097 +/- 0.0304 | 0.8118 +/- 0.0232 | OOM | 0.8984 +/- 0.0051 |
