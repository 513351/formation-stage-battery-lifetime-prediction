# Seeing the Future at Formation Stage

Code for ultra-early prediction of lithium-ion battery lifetime and degradation knee point using formation-stage data.

This repository contains the implementation of the data-processing, semantic-encoding, model-training, and evaluation pipeline used in our study.

## Workflow

The complete workflow is divided into five scripts and should be executed in the following order:

### 1. Semantic encoding

`01_semantic_encoding.py`

Extracts a 244-dimensional physics-informed semantic feature vector from the five formation stages of each battery cell.

### 2. Random batch split

`02_random_batch_split.py`

Randomly splits the 145 cells used in this study into:

- 117 training cells
- 28 test cells

The train/test split is controlled by `SPLIT_SEED`. Changing the random seed generates a different batch split.

### 3. Training-set normalization

`03_train_only_normalization.py`

Fits the normalization parameters using the training set only and applies the same parameters to both the training and test sets.

For the target variables, the minimum and maximum values of the training set are used for Min-Max normalization.

The test set is not used to fit any normalization parameter.

### 4. Model training

`04_train_model.py`

Trains the multi-scale convolutional neural network with temporal attention for joint prediction of:

- degradation knee point
- end-of-life (EOL)

The trained model parameters are saved locally after training.

Numerical trained model weights are not included in this public repository.

### 5. Model testing

`05_test_and_inverse_normalize.py`

Evaluates the trained model using the 28 test cells in the corresponding batch.

Monte Carlo Dropout is used for predictive uncertainty estimation.

The predicted normalized outputs are converted back to cycle numbers using the target Min-Max parameters obtained exclusively from the training set.

No 3-sigma outlier filtering is applied during evaluation.

---

## Data source

The battery data used in this study are derived from the publicly available dataset reported by:

> Xiao Cui, Stephen Dongmin Kang, Sunny Wang, Justin A. Rose, Huada Lian, Alexis Geslin, Steven B. Torrisi, Martin Z. Bazant, Shijing Sun, and William C. Chueh.  
> **Data-driven analysis of battery formation reveals the role of electrode utilization in extending cycle life.**  
> *Joule*, 8(11), 3072–3087, 2024.  
> DOI: https://doi.org/10.1016/j.joule.2024.07.024

The original dataset can be obtained from:

https://data.matr.io/8/

The original battery dataset is not redistributed in this repository. Users should download the data directly from the original data source.

The target variables used in this study are:

- `Knee_Cycle`: degradation knee cycle
- `EOL_Cycle`: end-of-life cycle defined at 80% state of health (SOH)

The present study uses 145 cells from the source dataset for model development and evaluation.

---

## Data preparation

After downloading the original dataset, prepare the formation-stage data in the following structure:

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

The label table used by the pipeline should have the following format:

```text
Cell_ID,Knee_Cycle,EOL_Cycle
...
```

and should be saved as:

```text
data/Batch3_145_ID_Knee_EOL80.csv
```

---

## Project structure

```text
formation-stage-battery-lifetime-prediction/
│
├── 01_semantic_encoding.py
├── 02_random_batch_split.py
├── 03_train_only_normalization.py
├── 04_train_model.py
├── 05_test_and_inverse_normalize.py
│
├── data/
│   ├── Batch3_145_ID_Knee_EOL80.csv
│   └── formation_segments/
│
├── outputs/
│
├── README.md
└── .gitignore
```

Generated files are written to the `outputs/` directory.

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

Changing `SPLIT_SEED` in `02_random_batch_split.py` produces a different random train/test split.

---

## Model architecture

The prediction model consists of:

- three parallel one-dimensional convolutional branches with different kernel sizes
- adaptive max pooling
- temporal attention
- a fully connected layer
- two regression heads for simultaneous prediction of the knee point and EOL

The model is trained using the sum of the mean squared errors of the two prediction targets.

---

## Requirements

The main Python dependencies are:

```text
numpy
pandas
scipy
torch
```

Install the required packages before running the pipeline.

For reproducibility, using consistent Python and package versions across experiments is recommended.

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

The output of each step is used as the input to the subsequent step.

---

## Reproducibility

The train/test split is controlled by a random seed.

With the same:

- source data
- preprocessing procedure
- random seed
- software environment
- model configuration
- training settings

the experimental pipeline can be reproduced from the beginning.

The trained model weights are intentionally not distributed. Running `04_train_model.py` generates the model parameters locally.

---

## Output

The pipeline generates intermediate and final outputs including:

```text
outputs/
├── 01_semantic_encoding/
├── 02_batch_split/
├── 03_normalization/
├── 04_model/
└── 05_test/
```

The final evaluation includes:

- MAE
- RMSE
- MAPE
- predictive uncertainty estimated using MC Dropout
- 95% confidence interval half-width

---

## Data availability

The original battery dataset is publicly available from the data repository associated with Cui et al.:

https://data.matr.io/8/

Users should refer to the original dataset and publication for the applicable data license and terms of use.

---

## Code availability

The source code required to reproduce the computational workflow is provided in this repository.

Trained numerical model weights are not included. They can be regenerated by executing the training pipeline.

---

## Citation

If you use the original battery dataset, please cite:

```text
Cui, X., Kang, S. D., Wang, S., Rose, J. A., Lian, H., Geslin, A.,
Torrisi, S. B., Bazant, M. Z., Sun, S., & Chueh, W. C. (2024).
Data-driven analysis of battery formation reveals the role of electrode
utilization in extending cycle life.
Joule, 8(11), 3072–3087.
https://doi.org/10.1016/j.joule.2024.07.024
```

If you use this repository, please also cite the corresponding publication associated with this work.

Citation information for our study will be updated upon publication.

---

## License

The code in this repository is provided for academic and research use.

The original battery dataset is subject to the license and terms specified by its original data provider.
