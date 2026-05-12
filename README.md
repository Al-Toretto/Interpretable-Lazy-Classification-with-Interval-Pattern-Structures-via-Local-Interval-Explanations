# Interpretable Lazy Classification with Interval Pattern Structures and Local Interval Explanations

This repository contains the experiment code for comparing interpretable lazy classifiers based on interval pattern structures with common baseline classifiers on numerical datasets.

## Setup

Install the Python dependencies from the project root:

```bash
pip install -r requirements.txt
```

The FCALC implementation is kept in [FCALC](FCALC). Its internal details are described in [FCALC/README.md](FCALC/README.md).

## Current Experiments

The current paper results are produced by two repeated-split experiments:

- [experiments/exp_repeated_macro_f1.py](experiments/exp_repeated_macro_f1.py): computes macro-F1.
- [experiments/exp_repeated_sizes.py](experiments/exp_repeated_sizes.py): computes classifier-size and local-compactness statistics from the selected hyperparameters.

Both experiments use the same repeated protocol:

- datasets are loaded from [datasets](datasets);
- each dataset is evaluated on 10 stratified 80/20 train-test splits;
- split seeds are `1998, 1999, ..., 2007`;
- model selection uses 5-fold stratified cross-validation on each training split;
- raw per-repeat results are written incrementally, so interrupted runs can be resumed.

## Macro-F1 Experiment

Run the full repeated macro-F1 experiment:

```bash
python experiments/exp_repeated_macro_f1.py
```

Main outputs:

- `output/repeated_macro_f1/raw_repeat_results.csv`: one row per dataset, classifier, and repeat, including the selected hyperparameters;
- `output/repeated_macro_f1/summary.csv`: mean macro-F1 and confidence intervals over repeats;
- `output/repeated_macro_f1/latex_table_body.tex`: LaTeX table rows only;
- `output/repeated_macro_f1/summary.txt`: short text report.

The runner is incremental. If `raw_repeat_results.csv` already contains successful rows, rerunning the command skips those rows and continues with missing rows.

Useful options:

```bash
python experiments/exp_repeated_macro_f1.py --datasets wine sonar
python experiments/exp_repeated_macro_f1.py --classifiers ips_knn svm fcalc_rand
python experiments/exp_repeated_macro_f1.py --repeat-indices 1 2 3
python experiments/exp_repeated_macro_f1.py --summary-only
```

The merged final macro-F1 artifact used by the size experiment is:

```text
output/merged/raw_repeat_results.csv
```

It contains the selected hyperparameters for every completed dataset-classifier-repeat row.

## Size Experiment

Run the repeated size experiment after the macro-F1 raw results exist:

```bash
python experiments/exp_repeated_sizes.py
```

By default, the size experiment reads selected hyperparameters from:

```text
output/merged/raw_repeat_results.csv
```

and writes:

- `output/repeated_sizes/raw_repeat_sizes.csv`: one row per dataset, classifier, and repeat;
- `output/repeated_sizes/summary.csv`: mean size statistics over repeats;
- `output/repeated_sizes/compactness.csv`: IPS-KNN compactness statistics;
- `output/repeated_sizes/latex_table_body.tex`: LaTeX table rows only, with the smallest comparable local-model size in each row bolded.

To rebuild only the summary files and LaTeX table from an existing raw size file:

```bash
python experiments/exp_repeated_sizes.py --summary-only
```

Useful options:

```bash
python experiments/exp_repeated_sizes.py --datasets wine sonar
python experiments/exp_repeated_sizes.py --classifiers ips_knn svm decision_tree
python experiments/exp_repeated_sizes.py --repeat-indices 1 2 3
python experiments/exp_repeated_sizes.py --source-results-path output/merged/raw_repeat_results.csv
```

## Size Definitions

The size experiment follows the original size-reporting definitions and adds the missing FCALC definitions:

- `FCALC`: `n_train * n_features`.
- `FCALC(rand.)`: `n_classes * num_iters * n_features`, where `num_iters` is the selected randomized-FCALC hyperparameter.
- `IPS-KNN`: refit with selected `k`, then report average and maximum reduced reason for classification size over the test split.
- `k-NN`: `n_features * n_neighbors`.
- `Naive Bayes`: `4 * n_features + 2`.
- `Logistic Regression`: `n_features + 1`.
- `SVM`: linear kernel uses `n_features + 1`; non-linear kernels are refit and reported as `n_features * n_support_vectors + n_support_vectors + 2`.
- `Decision Tree`: refit with selected hyperparameters, then report average and maximum number of unique features along test-sample decision paths.
- `Random Forest`: report selected `n_estimators` and max depth. If selected `max_depth=None`, refit and report the observed maximum depth across fitted trees.
- `XGBoost`: report selected `n_estimators` and `max_depth`.

For IPS-KNN compactness, `compactness.csv` reports:

- mean average reduced reason size across repeats;
- maximum reduced reason size observed across repeats;
- mean reduced-reason/classification-reason ratio, computed per repeat and then averaged.

## Repository Cleanup

Legacy single-split experiment scripts and their old text outputs were removed. The active experiment entry points are only:

```text
experiments/exp_repeated_macro_f1.py
experiments/exp_repeated_sizes.py
```
