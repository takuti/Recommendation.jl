# Factorization Machines

Beyond numerous discussions about conventional matrix factorization (MF) based recommenders, [Rendle](https://dl.acm.org/doi/10.1145/2168752.2168771) has proposed factorization machines (FMs) as general predictors based on the similar "factorization" idea. In contrast to MF, FMs are formulated by a equation which is very similar to the polynomial regression, and the model can be applied all of the regression, classification and ranking problems depending on a choice of loss functions.

```@docs
FactorizationMachines
```

First of all, for an input vector $\mathbf{x} \in \mathbb{R}^d$, let us imagine a linear model parameterized by $w_0 \in \mathbb{R}$，$\mathbf{w} \in \mathbb{R}^d$ as follows:

```math
\hat{y}^{\mathrm{LR}}(\mathbf{x}) := w_0 + \mathbf{w}^{\mathrm{T}} \mathbf{x}.
```

Next, by incorporating interactions of the $d$ input variables, we extend the linear model into the following second-order polynomial model.

```math
\hat{y}^{\mathrm{PR}}(\mathbf{x}) := w_0 + \mathbf{w}^{\mathrm{T}} \mathbf{x} + \sum_{i=1}^d \sum_{j=i}^d w_{i,j} x_i x_j,
```

where $w_{i,j}$ is an element in a symmetric matrix $W \in \mathbb{R}^{d \times d}$, and it indicates a weight of $x_i x_j$, an interaction between the $i$-th and $j$-th variable.

FMs assume that $W$ can be approximated by a low-rank matrix; $w_{ij}$ in $\hat{y}^{\mathrm{PR}}$ is approximated by using a rank-$k$ matrix $V \in \mathbb{R}^{d \times k}$, and the weights are replaced with inner products of $k$ dimensional vectors as $w_{i, j} \approx \mathbf{v}_i^{\mathrm{T}} \mathbf{v}_j$ for $\mathbf{v}_1, \cdots, \mathbf{v}_d \in \mathbb{R}^k$. Finally, the model is formulated as follows:

```math
\hat{y}^{\mathrm{FM}}(\mathbf{x}) := \underbrace{w_0}_{\textbf{global bias}} + \underbrace{\mathbf{w}^{\mathrm{T}} \mathbf{x}_{ }}_{\textbf{linear}} + \sum_{i=1}^d \sum_{j=i}^d \underbrace{\mathbf{v}_i^{\mathrm{T}} \mathbf{v}_j}_{\textbf{interaction}} x_i x_j.
```

Several studies prove that a variety of feature representations $\mathbf{x}$ (e.g. concatenation of one-hot vectors for several categorical variables) work well with FMs, and the flexibility in feature representation is one of the most important characteristics of FMs.

Note that the original paper specially referred to the formulation above as *second-order FM* as a specific case that $p=2$ of the following $p$-th order FM:

```math
\hat{y}^{\mathrm{FM}^{(p)}}(\mathbf{x}) := w_0 + \mathbf{w}^{\mathrm{T}} \mathbf{x} + \sum^p_{\ell=2} \sum^d_{j_1 = 1} \cdots \sum^d_{j_p = j_{p-1} + 1} \left( \prod^{\ell}_{i=1} x_{j_i} \right) \sum^{k_{\ell}}_{f=1} \prod^{\ell}_{i=1} v_{j_i,f},
```

with the model parameters:

```math
w_0 \in \mathbb{R}, \ \mathbf{w} \in \mathbb{R}^d, \ V_{\ell} \in \mathbb{R}^{d \times k_{\ell}},
```

where $\ell \in \{2, \cdots, p\}$. The higher-order FMs are actually promising to capture more complex underlying concepts from dynamic data, but accordingly the computational cost should be more expensive.

