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

The derivation of log-likelihood gradient of BM is pretty much similar to [the derivation in RBM](https://tirmisula.github.io/posts/partition-function/#the-log-likelihood-gradient-of-rbm):

In conlusion, we have the gradient:

$$
\begin{align*}
\nabla_{\theta}\mathcal{L}(\theta) &= \frac{1}{N}\sum_{i=1}^N\nabla\_{\theta}\mathcal{L}\_{i}(\theta) \\\
&= \frac{1}{N}\sum\_{i=1}^N\left( \sum\_{o,h}p(o,h)\nabla\_{\theta}E(o,h)-\sum\_{h}p(h^{(i)}|o^{(i)})\nabla\_{\theta}E(o^{(i)},h^{(i)}) \right)
\end{align*}
$$

$$
\begin{align*}
    \mathcal{L}(\theta) &= \frac{1}{N}\log\prod_{i=1}^N p(v^{(i)}) \\\
    &= \frac{1}{N} \sum_{v\in V} \log \sum_h\frac{1}{z}\exp\left(-E(v,h)\right) \\\
    &= \frac{1}{N}\sum_{v\in V}\log p(v) \\\
    \nabla_{\theta}\mathcal{L} &= \frac{1}{N}\sum_{v\in V}\frac{\partial}{\partial \theta}\log \frac{p(v,h)}{p(h|v)} \\\
    &= 
\end{align*}
$$

{{< math.inline >}}
<p>
By adding a layer \( h^{(2)} \) on top of \( h^{(1)} \), we see that \( p(h^{(1)}) \) is no longer a prior but a learnable object, the learning problem is maximizing likelihood of \( p(h^{(1)}|w) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{w}^{(2)} &= \argmax_{w^{(2)}} p(h^{(1)}|h^{(2)},w^{(2)}) \\\
\hat{p}(h^{(1)}) &= p(h^{(1)}|h^{(2)},\hat{w}^{(2)})=\max p(h^{(1)}|w)
\end{align*}
$$
 
So for the stacked RBM model, we have marginal probability:

$$
\begin{align*}
p(v) &= \sum_{h^{(1)}} p(h^{(1)})p(v|h^{(1)}) \\\
&\text{replace $p(h^{(1)})$ with $\hat{p}(h^{1})$} \\\
&= \sum_{h^{(1)}} \hat{p}(h^{(1)})p(v|h^{(1)}) \\\
&\geq \text{single layer RBM } p(v)
\end{align*}
$$

<!-- {{< math.inline >}}
<p>
To improve \( p(v) \) we can improve the prior \( p(h^{(1)}) \) and leave p(v|h^{(1)}) fixed. This is done by adding a layer \( h^{(2)} \) on top of \( h^{(1)} \), remove top-down connections from \( h^{(1)} \) to \( v \), and learn \(w^{(1)},w^{(2)}\) in bottom-up manner:
</p>
{{</ math.inline >}} -->

{{< math.inline >}}
<p>
A notable thing for above inequality is to fix \( p(v|h^{(1)}) \). This is done by removing bottom-up(\( v \) to \( h^{(1)} \)) connections from first RBM layer:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{improve $p(v)$} &\hArr \text{improve $p(h^{(1)})$, fix $p(v|h^{(1)})$} \\\
\text{$p(v|h^{(1)})$} &\hArr \text{learn $w^{(1)}$ from first layer RBM inference} \\\
\text{$\hat{p}(h^{(1)})$} &\hArr \text{learn $w^{(2)}$ from second layer RBM inference}
\end{align*} \\\
\text{Thus $w=(w^{(1)},w^{(2)}\cdots)$ is learned in bottom-up manner} \\\
$$

From [ELBO](https://tirmisula.github.io/posts/expectation-maximization/#generalized-em-algorithm) perspective, we know that:

$$
\begin{align*}
\log p(v) &= \log \sum_{h^{(1)}} p(v,h^{(1)}) \\\
&\geq \text{ELBO} \\\
\text{ELBO} &= \sum_{h^{(1)}}q(h^{(1)}|v)\log\frac{p(v,h^{(1)})}{q(h^{(1)}|v)} \\\
% &\geq \mathbb{E}\_{h^{(1)}\sim q(h^{(1)}|v)}\left[\log\frac{p(v,h^{(1)})}{q(h^{(1)}|v)}\right] \\\
&= \sum_{h^{(1)}}q(h^{(1)}|v)\left[\log p(h^{(1)})+\log p(v|h^{(1)})-\log q(h^{(1)}|v)\right] \\\
&\because w^{(1)} \text{ is determined during RBM learning} \\\
&\therefore \text{posterier }p(v|h^{(1)}),q(h^{(1)}|v) \text{ is fixed} \\\
&= \sum_{h^{(1)}}q(h^{(1)}|v)\log p(h^{(1)})+C \\\
&\text{Let } p(h^{(1)}) = \hat{p}(h^{(1)}) \\\
&\leq \sum_{h^{(1)}}q(h^{(1)}|v)\log \hat{p}(h^{(1)})+C \\\ 
&\therefore\text{ELBO is improved by $\hat{p}(h^{(1)})$}
\end{align*}
$$

In conslusion,

$$
\text{stacking RBM} \hArr \text{maximize likelihood of $p(h^{(1)})$} \hArr \text{maximize $p(v)$'s ELBO}
$$

## Pre-training

Pre-training is the first stage of DBN tranining which is used to initialize weights. 

+ Perform bottom-up greedy training layer by layer

+ Treat each layer as RBM model, trained with [CD-k algorithm](https://tirmisula.github.io/posts/partition-function/#cd-k-for-rbm)

The tranining process roughly looks like:

$$
\begin{align*}
&\text{$v$ is known, learn $w^{(1)},b^{(0)}$ by CD-k} \\\
&\text{sample $h^{(1)}\sim p(h^{(1)}|v,w^{(1)})$} \\\
&\text{$h^{(1)}$ is known, learn $w^{(2)},b^{(1)}$ by CD-k} \\\
&\text{sample $h^{(2)}\sim p(h^{(2)}|h^{(1)},w^{(2)})$}
\end{align*} \\\
\cdots
$$

{{< math.inline >}}
<p>
However, posterier \( p(h|v) \) in SBN is intractable because of head to head structure. Instead of directly solving \( p(h|v) \), \( q(h|v) \) is computed to approximate \( p(h|v) \), \( q(h|v) \) is assumed to be factorable like posterier in RBM, so we have:
</p>
{{</ math.inline >}}

$$
\begin{cases}
p(v|h^{(1)}) = \prod_{i}\sigma( w_{:,i}^{T(1)}h^{(1)}+b_{i}^{(0)}) \\\
\because\text{complementary} \text{ prior}\text{, inference is reversible with $w^T$} \\\
q(h^{(1)}|v) = \prod_{j}\sigma( w_{j,:}^{(1)}v+b_{j}^{(0)}) \\\
w_{:,i}^{T(1)}, w_{j,:}^{(1)} \in w^{(1)}
\end{cases}
$$

The trainining process becomes:

$$
\begin{align*}
&\text{$v$ is known, learn $w^{(1)},b^{(0)}$ by CD-k} \\\
&\text{sample $h^{(1)}\sim q(h^{(1)}|v,w^{(1)})$} \\\
&\text{$h^{(1)}$ is known, learn $w^{(2)},b^{(1)}$ by CD-k} \\\
&\text{sample $h^{(2)}\sim q(h^{(2)}|h^{(1)},w^{(2)})$}
\end{align*} \\\
\cdots
$$

Drawback:

$$
\because \text{$q(h|v)$ is factorable but $p(h|v)$ is not factorable} \\\
\therefore \text{KL}(q||p) \gg 0 \rArr \text{ELBO of $p(v)$ is relatively loose in DBN}
$$

## Fine-tuning

Fine-tuning is the second stage of DBN tranining which is used to improve model performance.

+ Supervised method: Treat DBN as feed forward nerual network, use existed labeled data to fine tune weights (back propagation algorithm)

+ Unsupervised method: Fine-tunes the weights using a contrastive version of the wake-sleep algorithm<cite>[^2]</cite>

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=146).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^3]: - [Deep belief networks. Geoffrey E. Hinton (2009)](http://scholarpedia.org/article/Deep_belief_networks).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^2]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
