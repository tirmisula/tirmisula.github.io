---
author: "X. Wang"
title: "Bayesian Linear Regression"
date: "2023-09-18"
description: "A brief introduction."
tags: ["machine learning"]
categories: ["themes", "syntax"]
# series: ["Themes Guide"]
aliases: ["migrate-from-jekyl"]
math: true
ShowBreadCrumbs: false
ShowToc: true
TocOpen: true
draft: true
---

:                                                         

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" integrity="sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn" crossorigin="anonymous">

<!-- The loading of KaTeX is deferred to speed up page rendering -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js" integrity="sha384-cpW21h6RZv/phavutF+AuVYrr+dA8xD9zs6FwLpaCct6O9ctzYFfFr4dgmgccOTx" crossorigin="anonymous"></script>

<!-- To automatically render math in text elements, include the auto-render extension: -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"
    onload="renderMathInElement(document.body);"></script>
{{ end }}

{{ if .Page.Store.Get "hasMermaid" }}
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true });
  </script>
{{ end }}
{{</ math.inline >}}

<style>
    /* Set the font size of all math elements to 16px */
    .katex {
        font-size: 16px !important;
    }
</style>

<style>
/* Custom CSS styles */
.graph {
    background-color: white;
  /* padding: 10px; */
  /* border-radius: 5px; */
}
.graph pre {
    background-color: white;
  /* font-family: 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.5; */
}
</style>

## Background

### Overview of PGM

<cite>[^1]</cite>

$$
\text{PGM}: \begin{cases}
    \text{Representation} \begin{cases}
        \text{directed graph}\rarr  \text{Bayesian network} \\\
        \text{undirected graph}\rarr \text{Markov network(MRF)} \\\
        \text{continous variable}\rarr \text{\color{red}{Gaussian BN/Gaussian MRF}} \\\
        \text{time$\rarr$} \underset{\text{$x_i$ not i.i.d.}}{\text{ Dynamic model}} \begin{cases}
            \text{discrete state$\rarr$Hidden Markov Model} \\\
            \text{continous state} \begin{cases}
                \text{Linear model$\rarr$Karman Filter} \\\
                \text{Nonlinear model$\rarr$Particle Filter}
            \end{cases}
        \end{cases}
    \end{cases} \\\
    \text{Inference} \begin{cases}
        \text{MAP inference$\rarr \hat{x_A}=\argmax_{x_A}p(x_A|x_B)\propto\argmax p(x_A,x_B)$} \\\
        \text{exact inference} \begin{cases}
          \text{Variable elimination(VE)} \\\
          \text{Belief propagation(BP)$\rarr$sum-product algorithm(Tree)} \\\
          \text{Junction tree algorithm(Normal graph)}
        \end{cases} \\\
        \text{approximate inference} \begin{cases}
            \text{Loop belief propagation(Cyclic graph)} \\\
            \text{Variational inference} \\\
            \text{MCMC: importance sampling}
        \end{cases} 
    \end{cases} \\\
    \text{Learning} \begin{cases}
        \text{parameter learning} \begin{cases}
            \text{complete data: $(x,z)$} \\\
            \text{hidden variable: $z$}
        \end{cases} \\\
        \text{structure learning}
    \end{cases}
\end{cases}
$$


## Gaussian network

For nodes in Gaussian network(GN) we have definition:

$$
\begin{cases}
\text{nodes: } \lbrace x^{(1)},\cdots,x^{(N)} \rbrace \\\
x^{(i)} = (x_1,x_2,\cdots,x_p)^T \\\
x^{(i)} \sim \mathcal{N}(\mu_i,\Sigma_i) \\\
p(x) = \frac{1}{(2\pi)^{\frac{p}{2}}|\Sigma|^{\frac{1}{2}}} \exp(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)) \\\
\Sigma = \left( \sigma_{ij} \right)_{p\times p}
\end{cases}
$$

We have marginal independence:

$$
x_i\perp x_j \iff \sigma_{ij}=0
$$

However we want <b>conditional independence</b> in probalistic graphical model(PGM) for simplification, it can be achieved by precision matrix in Gaussian network, [we will prove it later](https://tirmisula.github.io/posts/gaussian-network/#pairwise-markov-property-in-gmrf):

$$
x_A\perp x_B|x_C  \\\
\dArr \\\
\text{Let }\Lambda = \Sigma^{-1} = \lambda_{ij} \text{, $\quad\Lambda$ is precision matrix} \\\
x_i\perp x_j | x\setminus\lbrace x_i,x_j \rbrace \iff \lambda_{ij}=0
$$

{{< math.inline >}}
<p>
Since \( h(\cdot) \) and \( g(\cdot) \) are not linear, \( \epsilon \) and \( \delta \) are not gaussian noise, we can't find a analytic expression for filtering problem like Kalman filter, sampling method is used on the other hand.
</p>
{{</ math.inline >}}

## Gaussian bayesian network

### Review gaussian linear model
Gaussian bayesian network(GBN) is a directed graph based on local <b>linear gaussian model</b> [which described before](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/#solve-the-joint-pdf):

$$
x \sim \mathcal{N} (\mu, \Lambda^{-1}) \\\
y=Ax+B+\epsilon\\\
\epsilon \sim \mathcal{N}(0, L^{-1}), \epsilon \perp x \\\
\dArr \\\
\begin{cases}
y|x \sim \mathcal{N}(Ax+B, L^{-1}) \\\
y \sim \mathcal{N}(A\mu+B, A \Lambda^{-1} A^T + L^{-1}) \\\
\begin{bmatrix}
    x\\\
    y
\end{bmatrix} \sim \mathcal{N}(\begin{bmatrix}
    \mu \\\
    A\mu + B
\end{bmatrix}, \begin{bmatrix}
    \Lambda^{-1} & \Lambda^{-1} A^T\\\
    (\Lambda^{-1} A^T)^T & A \Lambda^{-1} A^T + L^{-1}
\end{bmatrix}) \\\
x|y \sim \mathcal{N}(\Sigma_{xy}\Sigma_{yy}^{-1} (x_y-\mu_{y}) + \mu_{x}, -\Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}+\Sigma_{xx})
\end{cases}
$$

A typical linear gaussian model is [Kalman filter](https://tirmisula.github.io/posts/kalman-filter/#definition):

$$
\begin{cases}
z_1 \sim \mathcal{N}(\mu_1,\Sigma_1) \\\
\epsilon\sim \mathcal{N}(0,Q) \\\
\delta\sim \mathcal{N}(0,R) \\\
z_t = Az_{t-1}+B+\epsilon \\\
x_t = Cz_{t}+D+\delta \\\
p(z_t|z_{t-1}) = \mathcal{N}(Az_{t-1}+B, Q) \\\
p(x_t|z_{t}) = \mathcal{N}(Cz_{t}+D, R)
\end{cases}
$$

### Modeling p(x) in GBN

Recall that the factorization in [PGM chapter](https://tirmisula.github.io/posts/probabilistic-graphical-model/#definition):

$$
p(x_1,\cdots,x_p)=\prod_{i=1}^p p(x_i|x_{pa(i)})
$$

{{< math.inline >}}
<p>
For each node, it is gaussian distribution in GBN, so conditional distribution \( p(x_i|x_{pa(i)}) \) is still gaussian:
</p>
{{</ math.inline >}}

$$
x_i|x_{pa(i)} \sim \mathcal{N}(w_i^T(x_{pa(i)}-\mu_{pa(i)}) + \mu_i, \sigma_i^2)
$$

We can derive it's linear gaussian model:

$$
\text{Let $x_i$ has k parents $\in x_{pa(i)}$} \\\
\begin{cases}
x_{pa(i)} = (x_{pa(i)1},\cdots,x_{pa(i)k})^T \\\
x_{pa(i)j} \sim \mathcal{N}(\mu_{ij}, \sigma_{ij}^2\epsilon) \\\
\epsilon \sim \mathcal{N}(0,1) \\\
w_i = (w_{i1}, \cdots, w_{ik})^T
\end{cases} \\\
\dArr
$$

$$
\begin{cases}
x_i &= w_i^T(x_{pa(i)}-\mu_{pa(i)}) + \mu_i + \sigma_i\epsilon \\\
x_i-\mu_i &= \sum_{j=1}^k w_{ij}(x_{pa(i)j}-\mu_{ij}) + \sigma_i\epsilon \\\
x_i-\mu_i|x_{pa(i)}-\mu_{pa(i)} &\sim \mathcal{N}(w_i^T(x_{pa(i)}-\mu_{pa(i)}), \sigma_i^2)
\end{cases}
$$

### Vectorize p(x)

Let:

$$
x=\begin{bmatrix}
    x_1 \\\
    \vdots \\\
    x_p
\end{bmatrix},
\mu = \begin{bmatrix}
    \mu_1 \\\ 
    \vdots \\\
    \mu_p 
\end{bmatrix}
$$

Let:

$$
w_{i} = \begin{bmatrix}
    w_{i1} \\\
    \vdots \\\
    w_{ip}
\end{bmatrix},
\begin{cases}
    w_{ij} = 0, \text{if $x_j\notin x_{pa(i)}$} \\\
    w_{ij} \neq 0, \text{if $x_j\in x_{pa(i)}$}
\end{cases}
$$

We have:

$$
\begin{align*}
x_i-\mu_i &= \sum_{j=1}^k w_{ij}(x_{pa(i)j}-\mu_{ij}) + \sigma_i\epsilon \\\
&= \sum_{j} w_{ij}(x_{pa(i)j}-\mu_{ij})+ \sum_{a,x_a\notin x_{pa(i)}}0\cdot(x_a-\mu_a) + \sigma_i\epsilon \\\
&= \begin{bmatrix}
    w_{i1} \\\
    \vdots \\\
    w_{ip}
\end{bmatrix}^T ( 
    \begin{bmatrix}
    x_1 \\\
    \vdots \\\
    x_p
\end{bmatrix} - \begin{bmatrix}
    \mu_1 \\\
    \vdots \\\
    \mu_p
\end{bmatrix}
 ) + \epsilon\sigma_i \\\
 &= w_i^T (x-\mu) + \epsilon\cdot \sigma_i
\end{align*}
$$

Let:

$$
W  =\begin{bmatrix}
    w_1^T \\\
    \vdots \\\
    w_p^T
\end{bmatrix},
S = \begin{bmatrix}
    \sigma_1 &  \\\
    & \ddots \\\
    & & \sigma_p
\end{bmatrix},
\epsilon = \begin{bmatrix}
    \epsilon_1 \\\
    \vdots \\\
    \epsilon_p
\end{bmatrix}, \epsilon_i\sim\mathcal{N}(0,1)
$$

We have:

$$
\begin{align*}
\begin{bmatrix}
    x_1-\mu_1 \\\
    \vdots \\\
    x_p-\mu_p
\end{bmatrix} &= \begin{bmatrix}
    w_1^T (x-\mu) + \epsilon_1\cdot \sigma_1 \\\
    \vdots \\\
    w_p^T (x-\mu) + \epsilon_p\cdot \sigma_p
\end{bmatrix} \\\
&= \begin{bmatrix}
    w_1^T \\\
    \vdots \\\
    w_p^T
\end{bmatrix} (x-\mu) + \begin{bmatrix}
    \sigma_1 &  \\\
    & \ddots \\\
    & & \sigma_p
\end{bmatrix}\begin{bmatrix}
    \epsilon_1 \\\
    \vdots \\\
    \epsilon_p
\end{bmatrix} \\\
&\dArr \\\
x-\mu &= W\cdot(x-\mu)+S\cdot\epsilon \\\
(I-W)(x-\mu) &= S\cdot\epsilon \\\
x-\mu &= (I-W)^{-1}S\cdot\epsilon
\end{align*}
$$

Then the covariance matrix can be derived:

$$
\begin{align*}
Cov(x) &= E[(x-\mu)(x-\mu)^T] \\\
&= E[(x-\mu-E[x-\mu])(x-\mu-E[x-\mu])^T] \\\
&=Cov(x-\mu) \\\
&= E[((I-W)^{-1}S\epsilon)((I-W)^{-1}S\epsilon)^T] \\\
&= (I-W)^{-1}S\cdot E[\epsilon\epsilon^T]\cdot S^T ((I-W)^{-1})^T \\\
&= (I-W)^{-1}S S^T ((I-W)^{-1})^T
\end{align*}
$$

{{< math.inline >}}
<p>
The vectorized form of \( p(x) \) is:
</p>
{{</ math.inline >}}

$$
\begin{bmatrix}
    x_1 \\\
    \vdots \\\
    x_p
\end{bmatrix} \sim \mathcal{N}\left(
    \mu,
    (I-W)^{-1}S S^T ((I-W)^{-1})^T \right) 
$$

<!-- Define weight at time t as:

$$
\begin{align*}
\begin{bmatrix}
x_1 \\\
\vdots \\\
x_p
\end{bmatrix} - 
\begin{bmatrix}
\mu_1 \\\
\vdots \\\
\mu_p
\end{bmatrix} &= 
\begin{bmatrix}
\sum_{j=1}^k w_{ij}(x_{pa(i)j} - \mu_{ij}) + \sigma_1^2 \\\
\vdots \\\
w_p^T(x_{pa(p)}-\mu_{p}) + \sigma_p^2
\end{bmatrix} \\\
&= 
\end{align*}
$$

$$
X = 
\begin{bmatrix}
x_1 \\
\vdots \\
x_p
\end{bmatrix}, 
\mu = 
\begin{bmatrix}
\mu_1 \\
\vdots \\
\mu_p
\end{bmatrix}
$$

We can then express the linear model in vector form:

$$
X - \mu = WX + \epsilon
$$

where \(W\) is a weight matrix and \(\epsilon\) is a vector of noise terms. 

The conditional distribution of \(X\) given its parents is then:

$$
X - \mu | Pa(X) \sim \mathcal{N}(W(Pa(X) - \mu), \Sigma)
$$

where \(\Sigma\) is the covariance matrix.

在高斯贝叶斯网络中，每个节点的条件分布可以通过线性回归模型来表示。具体来说，给定节点 (i) 的父节点 (pa(i))，节点 (i) 的值 (x_i) 可以表示为其父节点的线性组合加上一个高斯噪声项 (\epsilon_i)：

$$ x_i = w_i^Tx_{pa(i)} + \mu_i + \epsilon_i $$

其中，(w_i) 是权重向量，(\mu_i) 是偏置项，(\epsilon_i) 是服从高斯分布的噪声项。因此，给定父节点的条件下，节点 (i) 的条件分布为：

$$ x_i|x_{pa(i)} \sim \mathcal{N}(w_i^Tx_{pa(i)} + \mu_i, \sigma_i^2) $$

这里，(\sigma_i^2) 是噪声项 (\epsilon_i) 的方差。 -->

## Gaussian markov random field
### Potential function and pdf in GMRF
Gaussian markov random field(GMRF) is a undirected graph, we use [MRF factorization theorem introduced before](https://tirmisula.github.io/posts/probabilistic-graphical-model/#formula-for-nodes-and-edges):

$$
\begin{align*}
p(x) &= \frac{1}{z} \prod_{i=1}^p \underset{\text{node potential}}{\psi_i(x_i)} \prod_{i,j\in edge} \underset{\text{edge potential}}{\psi_{i,j}(x_i,x_j)} \\\
&= \frac{1}{z} \exp( -\sum_{i=1}^p E_i(x_{i})-\sum_{i,j\in edge}E_{i,j}(x_i,x_j)) \\\
\end{align*}
$$

On the other hands, it is a multivariate gaussian distribution:

$$
\begin{align*}
p(x) &= \frac{1}{(2\pi)^{\frac{p}{2}}|\Sigma|^{\frac{1}{2}}} \exp(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)) \\\
&\propto \exp(-\frac{1}{2} (x-\mu)^T\Sigma^{-1}(x-\mu)) \\\
&= \exp( -\frac{1}{2} \left( x^T\Lambda-\mu^T\Lambda(x-\mu) \right) ) \\\
&= \exp( -\frac{1}{2} \left( x^T\Lambda x-\mu^T\Lambda x-x^T\Lambda\mu+\mu^T\Lambda\mu \right) ) \\\
&= \exp( -\frac{1}{2} \left( x^T\Lambda x-2\mu^T\Lambda x+\mu^T\Lambda\mu \right) ) \\\
&\propto \exp( -\frac{1}{2} x^T\Lambda x+\mu^T\Lambda x ) \\\
&\because \Lambda^T=\Lambda \\\
&= \exp( -\frac{1}{2} \underset{\text{quadratic form}} {x^T\Lambda x}+\underset{\text{linear form}}{(\Lambda\mu)^T x} ) \\\
&\text{Let $\Lambda\mu = \begin{bmatrix}
h_1 \\\
\vdots \\\
h_p
\end{bmatrix}$ be the potential vector,}  \\\
&= \exp( -\frac{1}{2}\sum_{i=1}^p\sum_{j=1}^px_i\lambda_{ij}x_j+\sum_{i=1}^p h_ix_i )
\end{align*}
$$

{{< math.inline >}}
<p>
Now we can say that GMRF's pdf has relation with GMRF's potential function:
</p>
{{</ math.inline >}}

$$
\exp( -\sum_{i=1}^p E_i(x_{i})-\sum_{i,j\in edge}E_{i,j}(x_i,x_j)) \triangleq \exp( -\frac{1}{2}\sum_{i=1}^p\sum_{j=1}^px_i\lambda_{ij}x_j+\sum_{i=1}^p h_ix_i ) \\\
\dArr
$$

$$
\begin{align*}
E_i(x_i) &\propto -\frac{1}{2} x_i^2\lambda_{ii} + h_ix_i \\\
E_{i,j}(x_i,x_j) &\propto -\frac{1}{2} (x_i\lambda_{ij}x_j+x_j\lambda_{ji}x_i) = -\lambda_{ij}x_ix_j
\end{align*}
$$

### Pairwise markov property in GMRF

The precision matrix can be used for determine conditional independence:

$$
\lambda_{ij}=0 \implies \begin{cases}
\text{$\psi_i$ and $\psi_j$ exist} \\\
\text{$\psi_{i,j}$ not exist}
\end{cases} \implies 
\text{$x_i,x_j$ not connected} \implies 
\text{$x_i\perp x_j|x\setminus \lbrace x_i,x_j\rbrace$}
$$

### Marginal pdf in GMRF

#### Natural parameterization

<b>Natural parameterization</b><cite>[^2]<cite> <cite>[^5]</cite> gives:

$$
\begin{align*}
p(x|\mu, \Sigma) &= \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu)\right) \\\
&\propto \exp\left( -\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu) \right) \\\
&\text{Let $\Lambda=\Sigma^{-1},\eta=\Lambda\mu$} \\\
p(x|\eta, \Lambda) &= \frac{|\Lambda|^{1/2}}{(2\pi)^{d/2}} \exp\left(-\frac{1}{2}(x^T\Lambda x - 2x^T\eta + \eta^T\Lambda^{-1}\eta)\right) \\\
&\propto \exp\left( (\Lambda\mu)^T x-\frac{1}{2}x^T\Lambda x \right)
\end{align*} \\\
\text{$\eta$ and $\Lambda$ are information vector and matrix respectively}
$$

The joint distribution by canonical parameterization and natural parameterization are listed as follows:

$$
\begin{align*}
\begin{bmatrix}
    x_a \\\
    x_b
\end{bmatrix} &\sim \mathcal{N}\left(\begin{bmatrix}
    \mu_a \\\
    \mu_b
\end{bmatrix}, \begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab} \\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix} \right) \\\
\begin{bmatrix}
    x_a \\\
    x_b
\end{bmatrix} &\sim \tilde{\mathcal{N}}\left(\begin{bmatrix}
    \eta_a \\\
    \eta_b
\end{bmatrix}, \begin{bmatrix}
    \Lambda_{aa} & \Lambda_{ab} \\\
    \Lambda_{ba} & \Lambda_{bb}
\end{bmatrix} \right)
\end{align*}
$$

In which we have:

$$
\begin{bmatrix}
    \eta_{a} \\\
    \eta_{b}
\end{bmatrix} = \Lambda \mu = \begin{bmatrix}
    \Lambda_{aa}\mu_a + \Lambda_{ab}\mu_b \\\
    \Lambda_{ba}\mu_a + \Lambda_{bb}\mu_b
\end{bmatrix}
$$

#### The relation between covariance and precision matrix
The <b>matrix's blockwise inversion</b><cite>[^3]</cite> formula shows:

$$
\begin{bmatrix} A & B \\\ C & D \end{bmatrix}^{-1} = \begin{bmatrix} A^{-1} + A^{-1}B(D - CA^{-1}B)^{-1}CA^{-1} & -A^{-1}B(D - CA^{-1}B)^{-1} \\\ -(D - CA^{-1}B)^{-1}CA^{-1} & (D - CA^{-1}B)^{-1} \end{bmatrix}
$$

{{< math.inline >}}
<p>
Since \(\Sigma\) is symmetric, we can get the mapping relations between submatrices of \( \Sigma \) and \( \Lambda \):
</p>
{{</ math.inline >}}

$$
\begin{bmatrix}
\Lambda_{aa} & \Lambda_{ab} \\\
\Lambda_{ab}^T & \Lambda_{bb}
\end{bmatrix} = \begin{bmatrix} \Sigma_{aa}^{-1} + \Sigma_{aa}^{-1}\Sigma_{ab}(\Sigma_{bb} - \Sigma_{ab}^T\Sigma_{aa}^{-1}\Sigma_{ab})^{-1}\Sigma_{ab}^T\Sigma_{aa}^{-1} & -\Sigma_{aa}^{-1}\Sigma_{ab}(\Sigma_{bb} - \Sigma_{ab}^T\Sigma_{aa}^{-1}\Sigma_{ab})^{-1} \\\ -(\Sigma_{bb} - \Sigma_{ab}^T\Sigma_{aa}^{-1}\Sigma_{ab})^{-1}\Sigma_{ab}^T\Sigma_{aa}^{-1} & (\Sigma_{bb} - \Sigma_{ab}^T\Sigma_{aa}^{-1}\Sigma_{ab})^{-1} \end{bmatrix} \\\
% P_{22}-P_{12}^TP_{11}^{-1}P_{12}\implies \\\
% = (\Sigma_{bb}-\Sigma_{ab}^T\Sigma_{aa}^{-1}\Sigma_{ab})^{-1} - (\Sigma_{bb} - \Sigma_{ab}^T\Sigma_{aa}^{-1}\Sigma_{ab})^{-1}\Sigma_{ab}^T\Sigma_{aa}^{-1} (\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ab}^T) \Sigma_{aa}^{-1}\Sigma_{ab}(\Sigma_{bb} - \Sigma_{ab}^T\Sigma_{aa}^{-1}\Sigma_{ab})^{-1}
$$

$$
\begin{bmatrix}
\Sigma_{aa} & \Sigma_{ab} \\\
\Sigma_{ab}^T & \Sigma_{bb}
\end{bmatrix} = \begin{bmatrix} \Lambda_{aa}^{-1} + \Lambda_{aa}^{-1}\Lambda_{ab}(\Lambda_{bb} - \Lambda_{ab}^T\Lambda_{aa}^{-1}\Lambda_{ab})^{-1}\Lambda_{ab}^T\Lambda_{aa}^{-1} & -\Lambda_{aa}^{-1}\Lambda_{ab}(\Lambda_{bb} - \Lambda_{ab}^T\Lambda_{aa}^{-1}\Lambda_{ab})^{-1} \\\ -(\Lambda_{bb} - \Lambda_{ab}^T\Lambda_{aa}^{-1}\Lambda_{ab})^{-1}\Lambda_{ab}^T\Lambda_{aa}^{-1} & (\Lambda_{bb} - \Lambda_{ab}^T\Lambda_{aa}^{-1}\Lambda_{ab})^{-1} \end{bmatrix}
$$

And the <b>Woodbury matrix identity</b><cite>[^4]</cite> gives that:

$$
A^{-1} + A^{-1}B(D - CA^{-1}B)^{-1}CA^{-1} = (A-BD^{-1}C)^{-1}
$$
<!-- $$
(D - CA^{-1}B)^{-1} - (D - CA^{-1}B)^{-1}CA^{-1}(A-BD^{-1}C) (A-BD^{-1}C)^{-1}BD^{-1}
= (D - CA^{-1}B)^{-1} - (D - CA^{-1}B)^{-1}CA^{-1}BD^{-1} \\\
= (D - B^TA^{-1}B)^{-1} - (D - B^TA^{-1}B)^{-1}B^TA^{-1}BD^{-1} \\\
= (D - CA^{-1}B)^{-1} - D^{-1}C(A-BD^{-1}C)^{-1} BD^{-1}
$$ -->

So we have:

$$
\Lambda_{aa} = (\Sigma_{aa}-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ab}^T)^{-1} \\\
\Sigma_{aa} = (\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ab}^T)^{-1}
$$

#### Marginal and conditional pdf from canonical parameterization 

We get the conlusion for [Gaussian marginal pdf and conditional pdf from joint pdf](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/#solve-the-marginal-pdf) before:

$$
\begin{cases}
x = \begin{bmatrix}
    x_a \\\
    x_b
\end{bmatrix} \\\
x \sim \mathcal{N}(\mu,\Sigma) \\\
\mu = \begin{bmatrix}
    \mu_a \\\
    \mu_b
\end{bmatrix} \\\
\Sigma = \begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab} \\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix}
\end{cases} \implies 
\begin{cases}
x_a \sim \mathcal{N}(\mu_a,\Sigma_{aa}) \\\
x_b \sim \mathcal{N}(\mu_b,\Sigma_{bb}) \\\
x_b|x_a \sim \mathcal{N}(\Sigma_{ba}\Sigma_{aa}^{-1} (x_a-\mu_{a}) + \mu_{b}, {-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb} } ) \\\
x_a|x_b \sim \mathcal{N}(\Sigma_{ab}\Sigma_{bb}^{-1} (x_b-\mu_{b}) + \mu_{a}, {-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}+\Sigma_{aa} } )
\end{cases}
$$

#### Marginal and conditional pdf from natural parameterization 

Combined with above conclusions, we can write down natural parameterization results of marginal and conditional pdf:

1. Marginal
    $$
    x_a \sim \mathcal{N}(\mu_a,\Sigma_{aa})
    $$

    For the variance:

    $$
    \begin{align*}
    Var(x_a) &= \Sigma_{aa} \\\
    &= (\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})^{-1}
    \end{align*}
    $$

    For the expectation:

    $$
    \begin{align*}
    Var(x_a)^{-1}E[x_a] &= (\Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})\mu_a \\\
    &= \Lambda_{aa}\mu_a - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}\mu_a \\\
    &= \Lambda_{aa}\mu_a - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba}\mu_a + \Lambda_{ab}\mu_b - \Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{bb}\mu_b\\\
    &= \Lambda_{aa}\mu_a + \Lambda_{ab}\mu_b - \Lambda_{ab}\Lambda_{bb}^{-1}(\Lambda_{ba}\mu_a + \Lambda_{bb}\mu_b) \\\
    &= \eta_a - \Lambda_{ab}\Lambda_{bb}^{-1}\eta_b
    \end{align*}
    $$

    Overall,

    $$
    x_a \sim \tilde{\mathcal{N}}(\eta_a-\Lambda_{ab}\Lambda_{bb}^{-1}\eta_b, \Lambda_{aa}-\Lambda_{ab}\Lambda_{bb}^{-1}\Lambda_{ba})
    $$

2. Conditional

    $$
    x_a|x_b \sim \mathcal{N}(\Sigma_{ab}\Sigma_{bb}^{-1} (x_b-\mu_{b}) + \mu_{a}, {-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}+\Sigma_{aa} } )
    $$

    For the variance we have:

    $$
    \begin{align*}
    Var(x_a|x_b) &= -\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}+\Sigma_{aa} \\\
    &= \Lambda_{aa}^{-1}
    \end{align*}
    $$

    For the expectation, notice that:

    $$
    \begin{align*}
    \Sigma_{ab}\Sigma_{bb}^{-1} &= -\Lambda_{aa}^{-1}\Lambda_{ab}(\Lambda_{bb} - \Lambda_{ab}^T\Lambda_{aa}^{-1}\Lambda_{ab})^{-1} (\Lambda_{bb} - \Lambda_{ab}^T\Lambda_{aa}^{-1}\Lambda_{ab}) \\\
    &= -\Lambda_{aa}^{-1}\Lambda_{ab}
    \end{align*}
    $$

    So we have<cite>[^6]</cite>:

    $$
    \begin{align*}
    E[x_a|x_b] &= \Sigma_{ab}\Sigma_{bb}^{-1}(x_b-\mu_b)+\mu_a \\\
    &= \mu_a-\Lambda_{aa}^{-1}\Lambda_{ab}(x_b-\mu_b) \\\
    Var(x_a|x_b)^{-1}E[x_a|x_b] &= \Sigma_{a|b}^{-1}(\mu_a-\Lambda_{aa}^{-1}\Lambda_{ab}(x_b-\mu_b)) \\\
    &= \Lambda_{aa}\mu_a - \Lambda_{ab}x_b + \Lambda_{ab}\mu_b \\\
    &= \eta_a - \Lambda_{ab}x_b
    \end{align*}
    $$

    Overall,

    $$
    x_a|x_b \sim \tilde{\mathcal{N}}(\eta_a - \Lambda_{ab}x_b, \Lambda_{aa} )
    $$

#### Solving marginal pdf
{{< math.inline >}}
<p>
In pairwise markov propety, we are solving \( p(x_i|x_{\setminus i}) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
x &= \begin{bmatrix}x_i \\\
x_{\setminus i}
\end{bmatrix} \\\
% x &\sim \mathcal{N}(0, \begin{bmatrix}
%     \sigma_{ii} & \Sigma_{i\setminus{i}} \\\
%     \Sigma_{\setminus{i}i} & \Sigma_{\setminus{i}\setminus{i}}
% \end{bmatrix}) \\\
% x_i|x_{\setminus i} &\sim \mathcal{N}(\Sigma)
\end{align*}
$$

Use the marginalization conclusions above, we have:

$$
\begin{align*}
Var(x_i|x_{\setminus{i}}) &= \Lambda_{ii}^{-1}=\sigma_{ii} \\\
E[x_i|x_{\setminus{i}}] &= \mu_{i}-\Lambda_{ii}^{-1}\Lambda_{i\setminus{i}}(x_{\setminus{i}}-\mu_{\setminus{i}}) \\\
\text{assume } &\text{$x$ is centralized} \\\
&= \lambda_{ii}^{-1}\Lambda_{i\setminus{i}}x_{\setminus{i}} \\\
&= \lambda_{ii}^{-1}\begin{bmatrix}
    \lambda_{i1} \\\
    \vdots \\\
    \lambda_{ij} \\\
    \vdots \\\
    \lambda_{iN}
\end{bmatrix}^T \begin{bmatrix}
    x_1 \\\
    \vdots \\\
    x_j \\\
    \vdots \\\
    x_N
\end{bmatrix} \\\
&= \sum_{j,j\neq i} \frac{\lambda_{ij}}{\lambda_{ii}}x_j \\\
&\dArr \\\
x_i|x_{\setminus{i}} &\sim \mathcal{N}(\sum_{j,j\neq i} \frac{\lambda_{ij}}{\lambda_{ii}}x_j, \sigma_{ii})
\end{align*}
$$

{{< math.inline >}}
<p>
Apparently the central of marginal \( p(x_i) \) is represented by a linear combination of connected \( x_j \).
</p>
{{</ math.inline >}}

## Summary

<!-- If you found any mistakes, please contact me via email. -->

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=105).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^3]: From [Tzon-Tzer, Lu; Sheng-Hua, Shiou (2002). "Inverses of 2 × 2 block matrices"](https://doi.org/10.1016%2FS0898-1221%2801%2900278-4).
[^2]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
