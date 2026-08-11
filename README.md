# Seeing the Future at Formation Stage

Code for ultra-early prediction of lithium-ion battery lifetime and degradation knee point using formation-stage data.

This repository contains the implementation of the semantic encoding, dataset splitting, normalization, model training, and model evaluation pipeline used in this study.

---

## Workflow

The computational workflow is divided into five scripts and should be executed in the following order.

### 1. Semantic encoding

`01_semantic_encoding.py`

Extracts a 244-dimensional physics-informed semantic feature vector from the five formation stages of each battery cell.

Each cell is represented by formation-stage boundary features, differential features, sampled capacity/voltage features, and stage separators.

### 2. Random batch split

`02_random_batch_split.py`

Randomly splits the 145 cells used in this study into:

- 117 training cells
- 28 test cells

The train/test split is controlled by `SPLIT_SEED`.

Changing `SPLIT_SEED` generates a different random batch split.

### 3. Training-set normalization

`03_train_only_normalization.py`

Fits the normalization parameters using the training set only and applies the same parameters to both the training and test sets.

For the prediction targets, the minimum and maximum values of the training set are used for Min-Max normalization.

The test set is not used to fit any normalization parameter and test values are not clipped.

### 4. Model training

`04_train_model.py`

Trains a multi-scale one-dimensional convolutional neural network with temporal attention for joint prediction of:

- degradation knee point
- end-of-life (EOL)

The trained model parameters are generated and saved locally during execution.

Numerical trained model weights are not distributed in this public repository.

### 5. Model testing

`05_test_and_inverse_normalize.py`

Evaluates the trained model using the test cells in the corresponding batch.

Monte Carlo Dropout is used for predictive uncertainty estimation.

The normalized predictions are converted back to cycle numbers using the target Min-Max parameters obtained exclusively from the training set.

No 3-sigma outlier filtering is applied during evaluation.

---

## Data source

The battery data used in this study are derived from the publicly available dataset reported by:

> Xiao Cui, Stephen Dongmin Kang, Sunny Wang, Justin A. Rose, Huada Lian, Alexis Geslin, Steven B. Torrisi, Martin Z. Bazant, Shijing Sun, and William C. Chueh.  
> **Data-driven analysis of battery formation reveals the role of electrode utilization in extending cycle life.**  
> *Joule*, 8(11), 3072–3087, 2024.  
> https://doi.org/10.1016/j.joule.2024.07.024

The original dataset is available from:

https://data.matr.io/8/

The original battery data are not redistributed in this repository. Users should obtain the source data directly from the original publication and its associated data repository.

---

## Target labels

The prediction targets used in this study are:

- `Knee_Cycle`: degradation knee cycle
- `EOL_Cycle`: end-of-life cycle defined at 80% state of health (SOH)

The processed target-label file used in the experiments is not redistributed in this repository.

For local execution, the label table should contain the following columns:

```text
Cell_ID,Knee_Cycle,EOL_Cycle
```

and should be placed locally at:

```text
data/Batch3_145_ID_Knee_EOL80.csv
```

The processed label file is excluded from the public repository.

---

## Data preparation

After obtaining the original battery dataset, the formation-stage data should be prepared locally with the following structure:

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

The semantic encoding script reads the five pre-segmented formation stages for each cell and generates the model input features.

The local `data/` directory is not required to be committed to the public repository.

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

Intermediate outputs generated during execution are written to the local `outputs/` directory and are not included in the repository.

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

The main Python dependencies are listed in `requirements.txt`.

They include:

```text
numpy
pandas
scipy
torch
```

Install the dependencies using:

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

The output of each step is used as the input to the subsequent step.

The overall workflow is:

```text
Formation-stage data
        ↓
Semantic encoding
        ↓
145-cell dataset
        ↓
Random 117/28 train-test split
        ↓
Training-set-only normalization
        ↓
Model training
        ↓
Test-set prediction
        ↓
Inverse normalization
        ↓
Performance evaluation
```

---

## Output

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

---

## Model weights

The trained numerical model parameters are generated locally by:

```text
04_train_model.py
```

and saved as a PyTorch model state file.

The trained model weights are not included in this public repository.

The `.gitignore` file excludes model-weight files from version control.

---

## Reproducibility

This repository provides the source code for the computational workflow, including:

- semantic feature extraction
- random train/test splitting
- normalization
- model architecture
- model training
- uncertainty estimation
- model evaluation

The processed target-label file and trained numerical model weights used in the study are not redistributed.

Reproduction of the exact numerical results reported in the study therefore requires access to the corresponding processed target labels in addition to the publicly available source battery data.

---

## Data availability

The original battery dataset is publicly available from the data repository associated with Cui et al.:

https://data.matr.io/8/

The original dataset is not redistributed here.

The processed target-label file used in this study is not included in this repository.

Users should refer to the original dataset provider for the applicable data license and terms of use.

---

## Code availability

The source code for the computational workflow is provided in this repository.

The processed target-label file and trained model weights are not redistributed.

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

The source code in this repository is provided for academic and research use.

The original battery dataset remains subject to the license and terms specified by its original data provider.
