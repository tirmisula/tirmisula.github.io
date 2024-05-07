---
author: "X. Wang"
title: "Generative Adversarial Network"
date: "2023-10-01"
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

## Generative Model Intro

<cite>[^1]</cite>

{{< math.inline >}}
<p>
Generative model is a kind of probabilistic model that either model on  \( p(X,Y) \), \( p(X,Z) \) or \( p(X) \).
</p>
{{</ math.inline >}}

## Taxonomy of Machine Learning Models

### Non-probabilistic Model

Non-probabilistic models refer to models that directly learn the mappings from the input space to the output space, without involving the estimation and integration of probability densities, or similar operations. The key process is usually solving a optimization problem.

#### Supervised Model

1. PLA (perceptron learning algorithm)

2. SVM (support vector machine)

3. KNN (k-nearest neighbors)

4. NN (nerual network for hard margin)

5. Tree Model (random forest)

#### Unsupervised Model

1. PCA (principal component analysis)

2. LSA (latent semantic analysis)

3. K-means (k-means clustering)

4. Autoencoder (nerual network without label)

### Probabilistic Model

The input/output variables are represented by assumed probability distributions. Common estimation methods are MLE (Maximum likelihood estimation) or MAP (Maximum A Posteriori).


### Discriminative Model

Modeling on p(Y|X).

1. LR (linear regression)

2. MEMM (max entropy markov model)

3. CRF (conditional random field)

### Generative Model

Modeling on p(X,Y),p(X,Z) or p(X).

1. Naive Bayes

2. Mixture Model

    2.1. GMM (gaussian mixture model)

3. Time Series Model

    3.1. HMM (hidden markov model)

    3.2. Kalman Filter

    3.3. Particle Filter

4. Non-parametric Model

    4.1. Gaussian Process

    4.2. Dirichlet Process

5. Mixed Membership Model

    5.1. LDA (latent dirichlet allocation)

6. Factorial Model

    6.1. FA (factor analysis)

    6.2. P-PCA (probabilistic principal component analysis)

    6.3. ICA (independent component analysis)

7. Energy-based Model

    7.1. Boltzmann Machine

    7.2. Restricted Boltzmann machine
    
    7.3. Deep Belief Network

    7.4. Sigmoid Belief Network

    7.5. Deep Boltzmann machine

8. VAE (variational autoencoder)

9. GAN (generative adversarial network)

10. Autoregressive Model

11. Flow-based Model

## Generative Model Taxonomy

$$
\begin{cases}
    \text{Representation} : \begin{cases}
        \text{r.v.} : \text{discrete}, \text{continuous} \\\
        \text{edges} : \text{directed}, \text{undirected} \\\
        \text{nodes} \begin{cases}
            \text{latent} \\\
            \text{fully observed}
        \end{cases} \\\
        \text{layers} \begin{cases}
            \text{shallow} \\\
            \text{deep}
        \end{cases} \\\
        \text{connections} \begin{cases}
            \text{sparse} \\\
            \text{dense}
        \end{cases} \\\
        \text{pdf} \begin{cases}
            \text{parametric} \\\
            \text{non-parametric}
        \end{cases} \\\
        \text{whether solve $p(x)$} \begin{cases}
            \text{implicit density} \\\
            \text{explicit density}
        \end{cases} \\\
    \end{cases} \\\
    \text{Inference} : \text{tractability} : \text{exact inference}, \text{approximate inference} \\\
    \text{Learning} : \text{func} : \text{likelihood-based model}, \text{likelihood-free model}
\end{cases}
$$

## DBM intro

Consider a DBM with 3 layers, it's graph structure is similar to DBN:

DBM is proposed by Salakhutdinov and Hinton<cite>[^2]</cite>. The joint distribution of DBM (referenced in [RBM](https://tirmisula.github.io/posts/restricted-boltzman-machine/#rbm-model-definition)) is given by:

$$
\begin{align*}
p(v,h^{(1)},h^{(2)}) &= \frac{1}{z}\exp(-E(v,h^{(1)},h^{(2)})) \\\
&= \frac{1}{z}\exp(v^Tw^{(1)}h^{(1)}+h^{T(1)}W^{(2)}h^{(2)})
\end{align*}
$$

And the conditonal distribution (referenced in [RBM](https://tirmisula.github.io/posts/restricted-boltzman-machine/#posterier-inference)) is given by:

$$
\begin{align*}
p(v_i=1|h^{(1)}) &= \sigma(\sum_j w\_{ij}^{(1)}h\_{j}^{(1)}) \\\
p(h^{(2)}\_{k}=1|h^{(1)}) &= \sigma(\sum_j w\_{jk}^{(2)}h\_{j}^{(1)}) \\\
p(h^{(1)}\_{j}=1|v,h^{(2)}) &= \sigma(\sum_i w\_{ij}^{(1)}v_{i} + \sum_k w\_{jk}^{(2)}h\_{k}^{(2)}) \\\
w_{ij} &: \text{weight from bottom node $\langle i \rangle$ to top node $\langle j \rangle$} \\\
w_{jk} &: \text{weight from bottom node $\langle j \rangle$ to top node $\langle k \rangle$} 
\end{align*}
$$

We can use the general BM learning methods for DBM, However a combination of pre-training (find initial weights) and SGD (sthochastic gradient descent) is applicable to accelerate the learning process due to computation efficiency and similarity to DBN.

## Pre-tranining in DBM
### Intuition
Recall that in [Deep Belief Network](https://tirmisula.github.io/posts/deep-belief-network/#maximize-ph) chapter, we show p(h) is determined by:

$$
p(h^{(1)}) = \begin{cases}
    p(h^{(1)}|w^{(1)}) \text{ in RBM} \\\
    p(h^{(1)}|w^{(2)}) \text{ in DBN}
\end{cases}
$$

{{< math.inline >}}
<p>
DBM's definition shows \(p(h^{(1)})\) is determined by both \( w^{(1)} \) and \( w^{(2)} \). An intuition for DBM pre-training is to perform greedy layerwise pre-training introduced in DBN:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(h^{(1)}|w^{(1)},w^{(2)}) \text{ is a composed model of $p(h^{(1)}|w^{(1)})$ and $p(h^{(1)}|w^{(1)})$}
\end{align*}
$$

{{< math.inline >}}
<p>
\( p(h^{(1)}|w^{(1)}) \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(h^{(1)}|w^{(1)}) &= \sum_{v} p(v,h^{(1)}|w^{(1)}) \\\
&= \sum_{v}p(v|w^{(1)})p(h^{(1)}|v,w^{(1)}) \\\
&\approx \frac{1}{N}\sum_{v\in V}p(h^{(1)}|v,w^{(1)}) = \frac{1}{N}\sum_{v\in V}\prod_{j}\sigma(\sum_i w\_{ij}^{(1)}v_{i})
\end{align*} \\\
\frac{1}{N}\sum_{v\in V}p(v,h^{(1)}|w^{(1)}) \text{ is aggregate posterier}, v \sim p(v)=p(v|w^{(1)})
$$

{{< math.inline >}}
<p>
\( p(h^{(1)}|w^{(2)}) \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(h^{(1)}|w^{(2)}) &= \sum_{h^{(2)}} p(h^{(1)},h^{(2)}|w^{(2)}) \\\
&= \sum_{h^{(2)}}p(h^{(2)}|w^{(2)})p(h^{(1)}|h^{(2)},w^{(2)}) \\\
&\approx \frac{1}{N}\sum_{h^{(2)}\in H}p(h^{(1)}|h^{(2)},w^{(2)}) = \frac{1}{N}\sum_{h^{(2)}\in H}\prod_{j}\sigma(\sum_k w^{(2)}\_{jk}h^{(2)}\_{k})
\end{align*}
$$

$$
\begin{align*}
&h^{(2)} \sim p(h^{(2)}|h^{(1)},w^{(2)}), h^{(1)} \sim p(h^{(1)}|v,w^{(1)}), v\sim p(v)
\end{align*}
$$

{{< math.inline >}}
<p>
Notice that \( h^{(2)} \) depends on sampling from observation distribution \( p(v) \). Directly using \( p(h^{(1)}|w^{(1)}) \) and \( p(h^{(1)}|w^{(2)}) \) would amount to double-counting the evidence, which causes represented approximation distribution \( p(h^{(1)}|w^{(1)},w^{(2)}) \) concentrated. The intuition is to average two models \( p(h^{(1)}|\frac{1}{2}w^{(1)},\frac{1}{2}w^{(2)}) \), this is done by duplicate connections between layers except marginal layers (the top and bottom layers).
</p>
{{</ math.inline >}}

### Pre-tranining procedure

{{< math.inline >}}
<p>
Suppose DBM contains \( L+1 \) layers, learn \( w^{(1)},\cdots,w^{(L)} \) in bottom-up manner, pre-train each two layers as RBM (CD-k).
</p>
{{</ math.inline >}}

1. For bottom two layers, the conditonal probability is given by:

$$
\begin{align*}
    p(h^{(1)}\_{j}=1|v) &= \sigma(\sum_i w\_{ij}^{(1)}v_{i}+\sum_i w\_{ij}^{(1)}v_{i}) \\\ 
    p(v_i=1|h^{(1)}) &= \sigma(\sum_j w\_{ij}^{(1)}h^{(1)}\_{j})
\end{align*}
$$

2. For any intermediate layer, the conditonal probability is given by:

$$
\begin{align*}
    p(h^{(l)}\_{j}=1|h^{(l-1)},h^{(l+1)}) &= \sigma(\sum_i w\_{ij}^{(l)}h^{(l-1)}\_{i} + \sum_k w\_{jk}^{(l+1)}h^{(l+1)}\_{k})
\end{align*}
$$

3. For top two layers, the conditonal probability is given by:

$$
\begin{align*}
    p(h^{(L)}\_{k}=1|h^{(L-1)}) &= \sigma(\sum_j w\_{jk}^{(L)}h^{(L)}\_{j}) \\\
    p(h^{(L-1)}\_{j}=1|h^{(L)}) &= \sigma(\sum_k w\_{jk}^{(L)}h^{(L)}\_{k} + \sum_k w\_{jk}^{(L)}h^{(L)}\_{k}) 
\end{align*}
$$

## DBM's ELBO of p(v)

Salakhutdinov and Hinton<cite>[^2]</cite> proved that DBM's ELBO is further improved on the basis of DBN's ELBO.

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=161).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [Deep Boltzmann Machines. Ruslan Salakhutdinov, Geoffrey Hinton](http://www.cs.toronto.edu/~hinton/absps/dbm.pdf).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
