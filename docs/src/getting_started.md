# Getting Started

## Installation

This package is registered in [METADATA.jl](https://github.com/JuliaLang/METADATA.jl).

```julia
julia> using Pkg; Pkg.add("Recommendation")
```

## Usage

This package contains `DataAccessor` and several fundamental recommendation techniques (e.g., non-personalized `MostPopular` recommender, `CF` and `MF`), and evaluation metrics such as `Recall`. All of them can be accessible by loading the package as follows:

```julia
using Recommendation
```

First of all, you need to create a data accessor from a matrix:

```julia
using SparseArrays

da = DataAccessor(sparse([1 0 0; 4 5 0]))
```

or set of events:

```julia
const n_user = 5
const n_item = 10

events = [Event(1, 2, 1), Event(3, 2, 1), Event(2, 6, 4)]

da = DataAccessor(events, n_user, n_item)
```

where `Event()` is a composite type which represents a user-item interaction:

```julia
type Event
    user::Int
    item::Int
    value::Float64
end
```

Next, you can pass the data accessor to an arbitrary recommender as:

```julia
recommender = MostPopular(da)
```

and building a recommendation engine should be easy:

```julia
build(recommender)
```

Personalized recommenders sometimes require us to specify the hyperparameters:

```julia
help?> Recommendation.MF
  MF(
      da::DataAccessor,
      k::Int
  )
```

```julia
recommender = MF(da, 2)
build(recommender, learning_rate=15e-4, max_iter=100)
```

Once a recommendation engine has been built successfully, top-`k` recommendation for a user `u` with item candidates `candidates` is performed as follows:

```julia
u = 4
k = 2
candidates = [i for i in 1:n_item] # all items

recommend(recommender, u, k, candidates)
```
