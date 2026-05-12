# FCALC

FCALC now uses the repository-level datasets and preprocessing pipeline instead of
local copies under `FCALC/data`.

Run from the repository root or from this directory:

```shell
conda activate fca
python FCALC/run_experiments.py glass
```

To run every dataset listed in `src.dataset.known_datasets`:

```shell
conda activate fca
python FCALC/run_experiments.py
```

The standalone runner writes generated deterministic results to
`FCALC/all-results/results` and randomized results to
`FCALC/all-results/results-randomized`. These directories are generated artifacts
and are not needed for the current repeated macro-F1 and size experiments.

For FCALC(rand.), the scoring rule is treated as a hyperparameter. The runner
searches the standard rules `standard`, `standard-support`, and `ratio-support`,
and the proximity rules `proximity`, `proximity-non-falsified`, and
`proximity-support`. For each candidate rule, the number of sampled groups is
searched over 10, 20, 30, 40, and 50, and the sampled group size is searched
over feasible values from 1 through 10 using 5-fold stratified cross-validation
on the training split with macro-F1 as the selection criterion.
