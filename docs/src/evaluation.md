# Evaluation

```@index
Pages = ["evaluation.md"]
```

## Cross validation

```@docs
cross_validation
leave_one_out
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

## Aggregated metrics

Return a single aggregated score for an array of multiple top-``k`` recommended items. [Recommender Systems Handbook](https://www.bgu.ac.il/~shanigu/Publications/EvaluationMetrics.17.pdf). gives an overview of such aggregated metrics. In particular, the formulation of Gini index and Shannon Entropy can be found at Eq. (20) and (21) on page 26.

```@docs
AggregatedDiversity
ShannonEntropy
GiniIndex
```

## Intra-list metrics

Given a list of recommended items (for a single user), intra-list metrics quantifies the quality of the recommendation list from a non-accuracy perspective. [A Survey of Serendipity in Recommender Systems](https://dl.acm.org/doi/10.1016/j.knosys.2016.08.014) highlights the foundation of these metrics.

```@docs
Coverage
Novelty
IntraListSimilarity
Serendipity
```
