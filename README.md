# Seeing the Future at Formation Stage

**Ultra-Early Prediction of Battery Lifetime and Knee Point from Formation-Stage Data**

This repository provides the source code for the computational workflow used in our study on ultra-early prediction of lithium-ion battery lifetime and degradation knee point using formation-stage data.

The workflow includes physics-informed semantic feature extraction, random train/test splitting, training-set-only normalization, multi-scale convolutional neural network modeling with temporal attention, Monte Carlo Dropout uncertainty estimation, and model evaluation.

---

## Software archive

The source code corresponding to Version 1.0.0 is permanently archived on Zenodo.

**Zenodo DOI:**  
https://doi.org/10.5281/zenodo.21890099

**Version:** `1.0.0`

**GitHub repository:**  
https://github.com/513351/formation-stage-battery-lifetime-prediction

---

## Workflow

The computational workflow is divided into five scripts and should be executed in the following order.

### 1. Semantic encoding

`01_semantic_encoding.py`

Extracts a 244-dimensional physics-informed semantic feature vector from the five formation stages of each battery cell.

The encoded representation includes formation-stage boundary features, differential features, sampled capacity/voltage features, and stage separators.

---

### 2. Random batch split

`02_random_batch_split.py`

Randomly splits the 145 cells used in this study into:

- 117 training cells
- 28 test cells

The train/test split is controlled by:

```python
SPLIT_SEED = 40
```

Changing `SPLIT_SEED` generates a different random train/test split.

---

### 3. Training-set-only normalization

`03_train_only_normalization.py`

Fits all normalization parameters using the training set only.

The same training-derived normalization parameters are then applied to both the training and test sets.

For the prediction targets:

- `Knee_Cycle`
- `EOL_Cycle`

Min-Max normalization parameters are calculated exclusively from the training labels.

The test set is not used to fit any normalization parameter, and normalized test values are not clipped.

---

### 4. Model training

`04_train_model.py`

Trains a multi-scale one-dimensional convolutional neural network with temporal attention for joint prediction of:

- degradation knee point
- battery end-of-life (EOL)

The model contains:

- three parallel one-dimensional convolutional branches
- multiple convolution kernel sizes
- adaptive max pooling
- temporal attention
- a fully connected representation layer
- two regression heads for knee-point and EOL prediction

The total training loss is the sum of the mean squared errors of the two prediction targets.

The trained numerical model parameters are saved locally during execution.

Trained model weights are not redistributed in this repository.

---

### 5. Model testing and inverse normalization

`05_test_and_inverse_normalize.py`

Evaluates the trained model on the 28 test cells in the corresponding batch.

Monte Carlo Dropout is used for predictive uncertainty estimation.

The normalized predictions are converted back to cycle numbers using the target Min-Max parameters obtained exclusively from the training set.

No 3-sigma outlier filtering is applied during evaluation.

---

## Data source

The battery data used in this study are derived from the publicly available dataset reported by:

> Cui, X., Kang, S. D., Wang, S., Rose, J. A., Lian, H., Geslin, A., Torrisi, S. B., Bazant, M. Z., Sun, S., & Chueh, W. C.  
> **Data-driven analysis of battery formation reveals the role of electrode utilization in extending cycle life.**  
> *Joule*, 8(11), 3072–3087, 2024.  
> https://doi.org/10.1016/j.joule.2024.07.024

The original dataset is available from:

https://data.matr.io/8/

The original battery dataset is not redistributed in this repository.

Users should obtain the source battery data directly from the original publication and its associated public data repository.

---

## Target labels

The prediction targets used in this study are:

- `Knee_Cycle`: degradation knee cycle
- `EOL_Cycle`: end-of-life cycle defined at 80% state of health (SOH)

The processed target-label file used in the experiments is not redistributed in this repository.

For local execution, the label table should contain:

```text
Cell_ID,Knee_Cycle,EOL_Cycle
```

and should be placed locally at:

```text
data/Batch3_145_ID_Knee_EOL80.csv
```

The processed label file is excluded from version control.

---

## Data preparation

After obtaining the original battery dataset, the formation-stage data should be prepared locally in the following structure:

```text
data/
└── formation_segments/
    ├── Cell_XXX_...
    │   ├── ...
    │   └── ...
    ├── Cell_XXX_...
    │   ├── ...
    │   └── ...
    └── ...
```

The semantic encoding script reads the five pre-segmented formation stages of each cell and generates the corresponding 244-dimensional model input vector.

The local `data/` directory is not included in the public repository.

---

## Repository structure

```text
formation-stage-battery-lifetime-prediction/
│
├── 01_semantic_encoding.py
├── 02_random_batch_split.py
├── 03_train_only_normalization.py
├── 04_train_model.py
├── 05_test_and_inverse_normalize.py
├── requirements.txt
├── README.md
└── .gitignore
```

Local data and generated outputs are excluded from the public repository.

---

## Main experimental settings

| Parameter | Value |
|---|---:|
| Total cells | 145 |
| Training cells | 117 |
| Test cells | 28 |
| Split seed | 40 |
| Training seed | 40 |
| Input feature dimension | 244 |
| Batch size | 8 |
| Training epochs | 500 |
| Learning rate | 0.002 |
| Dropout rate | 0.3 |
| MC Dropout samples | 300 |
| 3-sigma filtering | No |

---

## Model architecture

The prediction model uses three parallel one-dimensional convolutional branches with kernel sizes:

```text
2
5
7
```

Each branch produces 16 feature channels.

The three branches are concatenated and followed by:

```text
AdaptiveMaxPool1d(32)
Temporal Attention
Dropout(0.3)
Fully Connected Layer
Two Regression Heads
```

The two output heads independently predict:

```text
Knee_Cycle
EOL_Cycle
```

---

## Requirements

The main Python dependencies are listed in `requirements.txt`.

They include:

```text
numpy
pandas
scipy
torch
```

Install the required dependencies using:

```bash
pip install -r requirements.txt
```

Using consistent Python and package versions is recommended when reproducing the computational workflow.

---

## Usage

Run the scripts sequentially from the repository root:

```bash
python 01_semantic_encoding.py
python 02_random_batch_split.py
python 03_train_only_normalization.py
python 04_train_model.py
python 05_test_and_inverse_normalize.py
```

The output generated by each step is used as the input to the subsequent step.

The overall workflow is:

```text
Formation-stage data
        ↓
Physics-informed semantic encoding
        ↓
244-dimensional feature representation
        ↓
145-cell dataset
        ↓
Random 117/28 train-test split
        ↓
Training-set-only normalization
        ↓
Multi-scale CNN + Temporal Attention
        ↓
Joint Knee / EOL prediction
        ↓
MC Dropout uncertainty estimation
        ↓
Inverse normalization
        ↓
Performance evaluation
```

---

## Outputs

The pipeline generates local intermediate and final outputs in:

```text
outputs/
├── 01_semantic_encoding/
├── 02_batch_split/
├── 03_normalization/
├── 04_model/
└── 05_test/
```

The final evaluation includes:

- mean absolute error (MAE)
- root mean square error (RMSE)
- mean absolute percentage error (MAPE)
- predictive uncertainty estimated using Monte Carlo Dropout
- 95% confidence interval half-width

The `outputs/` directory is excluded from the public repository.

---

## Model weights

The trained numerical model parameters are generated locally by:

```text
04_train_model.py
```

and saved as a PyTorch model state file.

Pre-trained numerical model weights are not redistributed in this repository or the associated Zenodo software archive.

The model can be retrained locally using the provided model architecture and training pipeline when the required input data and target labels are available.

---

## Reproducibility

This repository provides the source code for:

- formation-stage semantic feature extraction
- random train/test splitting
- training-set-only normalization
- model architecture
- model training
- Monte Carlo Dropout uncertainty estimation
- inverse normalization
- performance evaluation

The processed target-label file and trained numerical model weights used in the study are not redistributed.

Therefore, reproduction of the exact numerical results reported in the study requires access to the corresponding processed target labels in addition to the publicly available source battery data.

The computational methodology and model implementation are provided in this repository.

---

## Data availability

The original battery dataset is publicly available from the data repository associated with Cui et al.:

https://data.matr.io/8/

The original dataset is not redistributed in this repository.

The processed target-label file used in this study is not publicly redistributed.

Users should refer to the original data provider for the applicable data license and terms of use.

---

## Code availability

The source code supporting this study is publicly available on GitHub:

https://github.com/513351/formation-stage-battery-lifetime-prediction

Version 1.0.0 of the source code is permanently archived on Zenodo:

https://doi.org/10.5281/zenodo.21890099

The processed target-label file and trained numerical model weights are not redistributed.

---

## Software citation

If you use this software, please cite the archived software release:

```text
Wang, Y., Wang, C., Wang, R., Li, J., Yu, Q., & Lu, J. (2026).
Seeing the Future at Formation Stage: Ultra-Early Prediction of Battery
Lifetime and Knee Point (Version 1.0.0) [Software].
Zenodo.
https://doi.org/10.5281/zenodo.21890099
```

Please also cite the corresponding research article once its final publication information becomes available.

---

## Original dataset citation

If you use the original battery dataset, please cite:

```text
Cui, X., Kang, S. D., Wang, S., Rose, J. A., Lian, H., Geslin, A.,
Torrisi, S. B., Bazant, M. Z., Sun, S., & Chueh, W. C. (2024).
Data-driven analysis of battery formation reveals the role of electrode
utilization in extending cycle life.
Joule, 8(11), 3072–3087.
https://doi.org/10.1016/j.joule.2024.07.024
```

---

## License

Copyright (C) 2026 The Authors. All rights reserved.

Permission is granted to use and modify this software solely for non-commercial academic and research purposes, provided that appropriate credit is given to the original authors and, once available, the associated publication is properly cited.

Redistribution of the software, in whole or in part, and any commercial use require prior written permission from the authors.

This software is provided "as is", without warranty of any kind, express or implied. The authors shall not be liable for any claim, damages, or other liability arising from the use of this software.

The original battery dataset remains subject to the license and terms specified by its original data provider.

## License

The source code in this repository is provided for academic and research use.

The original battery dataset remains subject to the license and terms specified by its original data provider.
