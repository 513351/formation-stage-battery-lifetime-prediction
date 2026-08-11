# Battery Lifetime Prediction Pipeline

Run the scripts in order:

1. `01_semantic_encoding.py`
2. `02_random_batch_split.py`
3. `03_train_only_normalization.py`
4. `04_train_model.py`
5. `05_test_and_inverse_normalize.py`

Place the final 145-cell label file at:

`./data/Batch3_145_ID_Knee_EOL80.csv`

Place the five pre-segmented formation-stage folders under:

`./data/formation_segments/`

Generated files are written to `./outputs/`.

Model weights (`*.pth`) are excluded from the public repository by `.gitignore`.
