---
author: "X. Wang"
title: "Deep Belief Network"
date: "2023-09-28"
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

## Overview of PGM

<cite>[^1]</cite>

$$
\text{PGM}: \begin{cases}
    \text{Representation} \begin{cases}
        \text{directed graph}\rarr \begin{cases} 
            \text{Naive Bayes:} \begin{cases}
            p(x_i|y) = \prod_{j=1}^p p(x_i^j|y) \\\
            \argmax \prod_{i=1}^N \left(\prod_{j=1}^p p(x_i^j|y_i)\right) p(y_i)
            \end{cases} \\\
            \text{Gaussian Mixture} \\\
            {\text{MEMM: }} \begin{cases}
                p(y_t|y_{t-1}) \\\
                p(y_t|x_{1:T},x_t)
            \end{cases} \\\
            \text{Sigmoid Belief Network} : \text{BM}+\text{directed graph} \\\
            \text{Bayesian network} 
        \end{cases} \\\
        \text{undirected graph}\rarr \begin{cases}
            \text{CRF}\rarr\text{Linear Chain CRF: } \begin{cases}
                p(y_t|y_{t-1}) \\\
                p(y_{t-1}|y_t) \\\
                p(y_t|x_{1:T},x_t)
            \end{cases} \\\
            \text{Boltzman Machine(BM): } \begin{cases}
                \text{visible nodes} \\\
                \text{hidden nodes}
            \end{cases} \\\
            \text{{Restricted Boltzman Machine(RBM)}} : \text{Bipartite BM} \\\
            \text{\color{red}{Boltzman Machine}} : \text{Fully connected graph} \\\
            \text{Markov network(MRF)}
        \end{cases} \\\
        \text{continous variable}\rarr \text{{Gaussian BN/Gaussian MRF}} \\\
        \text{time$\rarr$} \underset{\text{$x_i$ not i.i.d.}}{\text{ Dynamic model}} \begin{cases}
            \text{discrete state$\rarr$Hidden Markov Model} \\\
            \text{continous state} \begin{cases}
                \text{Linear model$\rarr$Karman Filter} \\\
                \text{Nonlinear model$\rarr$Particle Filter}
            \end{cases}
        \end{cases} \\\
        \text{{Deep Belief Network(DBN)}} : \text{RBM}+\text{SBN}
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


## Boltzman Machine Intro

Boltzman machine is a fully connected graph which is proposed to solve local minimum of Hopfield network.

{{< math.inline >}}
<p>
The BM model is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
x &: \text{nodes in BM}, x \sim \text{Bernoulli}
\\\
x &= \begin{bmatrix}
    x_1 \\\
    \vdots \\\
    x_p
\end{bmatrix}=\begin{bmatrix}
    h \\\
    v
\end{bmatrix}, h = \begin{bmatrix}
    h_1 \\\
    \vdots \\\
    h_m
\end{bmatrix}, v = \begin{bmatrix}
    v_1 \\\
    \vdots \\\
    v_n
\end{bmatrix} \\\
W_{n\times m} &: \text{connection weights between $v,h$} \\\
L_{n\times n} &: \text{connection weights inside $v$} \\\
J_{m\times m} &: \text{connection weights inside $h$} \\\
\theta &= (W,L,J)
\end{align*}
$$

$$
W = \begin{bmatrix}
    w_{11} & \cdots & w_{1m} \\\
    \vdots & \ddots & \vdots \\\
    w_{n1} & \cdots & w_{nm}
\end{bmatrix} L = \begin{bmatrix}
    L_{11} & \cdots & L_{1n} \\\
    \vdots & \ddots & \vdots \\\
    L_{n1} & \cdots & L_{nn}
\end{bmatrix} J = \begin{bmatrix}
    J_{11} & \cdots & J_{1m} \\\
    \vdots & \ddots & \vdots \\\
    J_{m1} & \cdots & J_{mm}
\end{bmatrix}
$$

In previous chpater, the joint probabilty of [restricted Boltzaman machine(RBM)](https://tirmisula.github.io/posts/restricted-boltzman-machine/#rbm-model-definition) is given by:

$$
\begin{align*}
p(o,h) &= \frac{1}{z}\exp(h^Two+\alpha^To+\beta^Th) \\\
&= \frac{1}{z} \prod_{i=1}^m\prod_{j=1}^n\exp(h_iw_{ij}o_j) \prod_{i=1}^m\exp(\beta_i h_i) \prod_{j=1}^n\exp(\alpha_j o_j)
\end{align*}
$$

{{< math.inline >}}
<p>
The BM model is a general form of RBM, by replacing \( \alpha,\beta \) with \( L,J \), we have closed form of joint distribution of BM:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(v,h) &= \frac{1}{z}\exp(-E(v,h)) \\\
&= \frac{1}{z}\exp(v^TWh+\frac{1}{2}v^TLv+\frac{1}{2}h^TJh) \\\
&= \frac{1}{z}\exp(\sum_{i=1}^n\sum_{j=1}^mv_iw_{ij}h_j+\frac{1}{2}\sum_{i=1}^n\sum_{j=1}^nv_iL_{ij}v_j+\frac{1}{2}\sum_{i=1}^m\sum_{j=1}^mh_iJ_{ij}h_j) \\\
&= \frac{1}{z} \prod_{i=1}^n\prod_{j=1}^m\exp(v_iw_{ij}h_j) \prod_{i=1}^n\prod_{j=1}^n\exp(v_iL_{ij}v_j) \prod_{i=1}^m\prod_{j=1}^m\exp(h_iJ_{ij}h_j)
\end{align*}
$$

## Gradient of BM's Log-likelihood

{{< math.inline >}}
<p>
Let \(V,H\) be the whole set of visible and hidden nodes from dataset:
</p>
{{</ math.inline >}}

$$
V,H \in P_{\text{data}} \\\
|V|,|H| = N
$$

For any [energy-based model](https://tirmisula.github.io/posts/partition-function/#the-log-likelihood-gradient-of-energy-based-model) we have the conclusion for log-likelihood gradient:

$$
\begin{align*}
\nabla_{\theta}\mathcal{L}(\theta) &= \frac{1}{N}\sum_{i=1}^N\left( \sum_{o,h}p(o,h)\nabla_{\theta}E(o,h)-\sum_{h}p(h^{(i)}|o^{(i)})\nabla_{\theta}E(o^{(i)},h^{(i)}) \right)
\end{align*}
$$

The derivation of log-likelihood gradient of BM is pretty much the same:

$$
\begin{align*}
    \mathcal{L}(\theta) &= \frac{1}{N}\log\prod_{i=1}^N p(v^{(i)}) \\\
    &= \frac{1}{N}\sum_{v\in V}\log p(v) \\\
    &= \frac{1}{N} \sum_{v\in V} \log \sum_h\frac{1}{z}\exp\left(-E(v,h)\right) \\\
    \nabla_{\theta}\mathcal{L} &= \frac{1}{N}\sum_{v\in V} \left( \sum_{v,h}p(v,h)\nabla_{\theta}E(v,h)-\sum_{h}p(h|v)\nabla_{\theta}E(v,h) \right)
\end{align*}
$$

{{< math.inline >}}
<p>
Take the partial derivative of \( W \), we have:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial W}E(v,h) &= -\frac{\partial}{\partial W} v^TWh \\\
&= -vh^T \\\
\\\
\frac{\partial}{\partial W}\mathcal{L} &= \frac{1}{N}\sum_{v\in V} \sum_{h}p(h|v)vh^T - \frac{1}{N}\sum_{v\in V}\sum_{v,h}p(v,h)vh^T \\\
&= \frac{1}{N}\sum_{v\in V} \mathbb{E}_{h\sim P_{\text{model}}(h|v)}[vh^T] - \frac{1}{N}\sum_{v\in V}\mathbb{E}_{v,h\sim P_{\text{model}}(v,h)}[vh^T] \\\
&= \mathbb{E}_{\begin{subarray}{c}
    v \sim P_{\text{data}}(v) \\\
    h|v \sim P_{\text{model}}(h|v)
    \end{subarray}}[vh^T] - \mathbb{E}_{v,h\sim P_{\text{model}}(v,h)}[vh^T] \\\
\end{align*}
$$
 
{{< math.inline >}}
<p>
Similarly, the partial derivative of \( L \) and \( J \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial L}E(v,h) &= -\frac{\partial}{\partial L} v^TLv \\\
&= -vv^T \\\
\frac{\partial}{\partial L}\mathcal{L} &= \mathbb{E}_{\begin{subarray}{c}
    v \sim P_{\text{data}}(v) \\\
    h|v \sim P_{\text{model}}(h|v)
    \end{subarray}}[vv^T] - \mathbb{E}_{v,h\sim P_{\text{model}}(v,h)}[vv^T] \\\
\end{align*}
$$

$$
\begin{align*}
\frac{\partial}{\partial J}E(v,h) &= -\frac{\partial}{\partial J} v^TLv \\\
&= -hh^T \\\
\frac{\partial}{\partial J}\mathcal{L} &= \mathbb{E}_{\begin{subarray}{c}
    v \sim P_{\text{data}}(v) \\\
    h|v \sim P_{\text{model}}(h|v)
    \end{subarray}}[hh^T] - \mathbb{E}_{v,h\sim P_{\text{model}}(v,h)}[hh^T] \\\
\end{align*}
$$

In conslusion the gradient is given by,

$$
\begin{align*}
\frac{\partial}{\partial W}\mathcal{L} &= \mathbb{E}_{\begin{subarray}{c}
    v \sim P_{\text{data}}(v) \\\
    h|v \sim P_{\text{model}}(h|v)
    \end{subarray}}[vh^T] - \mathbb{E}_{v,h\sim P_{\text{model}}(v,h)}[vh^T] \\\
\frac{\partial}{\partial L}\mathcal{L} &= \mathbb{E}_{\begin{subarray}{c}
    v \sim P_{\text{data}}(v) \\\
    h|v \sim P_{\text{model}}(h|v)
    \end{subarray}}[vv^T] - \mathbb{E}_{v,h\sim P_{\text{model}}(v,h)}[vv^T] \\\
\frac{\partial}{\partial J}\mathcal{L} &= \mathbb{E}_{\begin{subarray}{c}
    v \sim P_{\text{data}}(v) \\\
    h|v \sim P_{\text{model}}(h|v)
    \end{subarray}}[hh^T] - \mathbb{E}_{v,h\sim P_{\text{model}}(v,h)}[hh^T]
\end{align*}
$$

In [RBM learning](https://tirmisula.github.io/posts/partition-function/#rbm-learning) we know that:

$$
\text{RBM }\begin{cases}
P_{\text{model}}(h|v) \text{ has closed form} \\\
P_{\text{model}}(v,h) \text{ is intractable, sampled by CD-k}
\end{cases}
$$

However, BM's posterier and joint distribution are both intractable due to graph structure:

$$
\text{BM }\begin{cases}
P_{\text{model}}(h|v) \text{ is intractable} \\\
P_{\text{model}}(v,h) \text{ is intractable}
\end{cases} 
$$

## Gradient Ascent Based on MCMC
### Gibbs sampling for BM

Recall from [RBM learning chapter](https://tirmisula.github.io/posts/partition-function/#rbm-learning) that the stochastic gradient ascent is performed by Gibbs sampling. Suppose we sample M particles to approximate BM's posterier and joint distribution, it  looks like:

$$
\mathbb{E}_{\begin{subarray}{c}
    v \sim P_{\text{data}}(v) \\\
    h|v \sim P_{\text{model}}(h|v)
    \end{subarray}}[vh^T] \approx \frac{1}{M}\sum_{a=1}^M v^{(a)}h^{(a)T} \\\
\begin{array}{c}
    v^{(1)} \\\
    \cdots \\\
    v^{(M)}
\end{array} \sim P_{\text{data}}(v), h_{j}^{(a)} \sim P_{\text{model}}(h_j|h_{\neg j},v)
$$

$$
\mathbb{E}_{\begin{subarray}{c}
    v,h \sim P_{\text{model}}(v,h)
    \end{subarray}}[vh^T] \approx \frac{1}{M}\sum_{a=1}^M v^{(a)}h^{(a)T} \\\
x^{(a)}=(v^{(a)},h^{(a)}), x_i^{(a)} \sim P_{\text{model}}(x_i|x_{\neg i}) \\\
v_{i}^{(a)} \sim P_{\text{model}}(v_i|v_{\neg i},h)\text{ or } h_{j}^{(a)} \sim P_{\text{model}}(h_j|h_{\neg j},v)
$$

The effectiveness of MCMC approximation is limited to graph's size. Next step is to derive the conditional probability:

$$
P_{\text{model}}(x_i|x_{\neg i}) = \begin{cases}
    P_{\text{model}}(v_i|v_{\neg i},h) \\\
    P_{\text{model}}(h_j|h_{\neg j},v)
\end{cases}
$$

### Conditional probability of BM

$$
\begin{align*}
p(v_i=1|v_{\neg i},h) &= \frac{p(v,h)}{p(v_{\neg i},h)} \\\
&= \frac{\frac{1}{z}\exp(v^TWh+\frac{1}{2}v^TLv+\frac{1}{2}h^TJh)}{\sum_{v_i}\frac{1}{z}\exp(v^TWh+\frac{1}{2}v^TLv+\frac{1}{2}h^TJh)} \\\
&= \frac{\exp(v^TWh+\frac{1}{2}v^TLv+\frac{1}{2}h^TJh)}{\exp(\frac{1}{2}h^TJh)\sum_{v_i}\exp(v^TWh+\frac{1}{2}v^TLv)} \\\
&= \frac{\exp(v^TWh+\frac{1}{2}v^TLv)}{\sum_{v_i}\exp(v^TWh+\frac{1}{2}v^TLv)} \\\
&= \frac{\exp(v^TWh+\frac{1}{2}v^TLv)}{\exp(v^TWh+\frac{1}{2}v^TLv)+} \\\
\end{align*}
$$

## Fine-tuning

Fine-tuning is the second stage of DBN tranining which is used to improve model performance.

+ Supervised method: Treat DBN as feed forward nerual network, use existed labeled data to fine tune weights (back propagation algorithm)

+ Unsupervised method: Fine-tunes the weights using a contrastive version of the wake-sleep algorithm<cite>[^2]</cite>

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=150).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^3]: - [Deep belief networks. Geoffrey E. Hinton (2009)](http://scholarpedia.org/article/Deep_belief_networks).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^2]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
