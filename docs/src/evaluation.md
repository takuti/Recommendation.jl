# Evaluation

```@index
Pages = ["evaluation.md"]
```

## Cross validation

```@docs
cross_validation
```

## Rating metrics

```@docs
RMSE
MAE
```

## Ranking metrics

Let a target user $u \in \mathcal{U}$, set of all items $\mathcal{I}$, ordered set of top-$N$ recommended items $I_N(u) \subset \mathcal{I}$, and set of truth items $\mathcal{I}^+_u$.

```@docs
Recall
Precision
MAP
AUC
ReciprocalRank
MPR
NDCG
```