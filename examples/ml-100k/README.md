Sample code that builds and evaluates a recommender based on N-fold cross validation.

Usage:

```sh
julia --threads auto -- ml-100k.jl
```

`--threads` allows the script to test validation samples in parallel for efficiency.

Change the configurable parameters through in-line variables as needed e.g., for testing the different number of folds, evaluation metric, and/or recommender.
