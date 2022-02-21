# Preparing Data

## Onehot encoding

```@docs
onehot
```

## Load public datasets

```@docs
load_movielens_100k
load_movielens_latest
load_amazon_review
load_lastfm
```

Test a recommender with `cross_validation`:

```julia
using Recommendation

data = load_movielens_100k()
recall = cross_validation(
                          1,            # N-fold
                          Recall,       # Metric
                          5,            # Top-k
                          MostPopular,  # Recommender
                          data          # Data Accessor
                         )
println(recall)
```

## Generate synthetic data

```@docs
SyntheticFeature
SyntheticRule
generate
```

## Helper functions

```@docs
get_data_home
download_file
unzip
load_libsvm_file
```
