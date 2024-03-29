# Getting Started

## Installation

This package is registered in [JuliaRegistries](https://github.com/JuliaRegistries/General).

```julia
julia> using Pkg; Pkg.add("Recommendation")
```

## Usage

The package contains a unified `DataAccessor` module and several non-personalized/personalized recommenders, as well as evaluation metrics such as `Recall`. All of them can be accessible by loading the package as follows:

```julia
using Recommendation
```

First of all, you need to create a data accessor from a matrix:

```julia
using SparseArrays

data = DataAccessor(sparse([1 0 0; 4 5 0]))
```

or set of events:

```julia
n_users, n_items = 5, 10

events = [Event(1, 2, 1), Event(3, 2, 1), Event(2, 6, 4)]

data = DataAccessor(events, n_users, n_items)
```

where `Event()` is a composite type which represents a user-item interaction:

```julia
type Event
    user::Integer
    item::Integer
    value::Infinite
end
```

Note that `Infinite` is a custom type `Union{AbstractFloat, Integer}` that represents either floating point or integer number.

Next, you can pass the data accessor to an arbitrary recommender as:

```julia
recommender = MostPopular(data)
```

and building a recommendation engine should be easy:

```julia
fit!(recommender)
```

Personalized recommenders sometimes require us to specify the hyperparameters:

```julia
help?> Recommendation.MatrixFactorization
  MatrixFactorization(
      data::DataAccessor,
      n_factors::Integer
  )
```

```julia
recommender = MatrixFactorization(data, 2)
fit!(recommender, learning_rate=15e-4, max_iter=100)
```

Once a recommendation engine has been built successfully, top-`2` recommendation for a user `4` is performed as follows:

```julia
# for user#4, pick top-2 from all items
recommend(recommender, 4, 2, collect(1:n_items))
```
