---
author: "X. Wang"
title: "Deep Belief Network"
date: "2023-09-27"
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
        \text{\color{red}{Deep Belief Network(DBN)}} : \text{RBM}+\text{SBN}
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


## DBN Intro

Deep Belief Network(DBN) is a bayesian network and a hybrid model proposed by Hinton<cite>[^3]</cite>. The top two layers of DBN have undirected connections while the lower layers have directed top-down connections. DBN can be considered as a stack of RBMs, and the directed RBM layers is considered to be SBN.


<div style="text-align: center;">

```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': 'white',
      'primaryTextColor': '#000',
      'primaryBorderColor': '#7C0200',
      'lineColor': '#F8B229',
      'secondaryColor': 'red',
      'tertiaryColor': '#fff'
    }
  }
}%%
graph TD
    subgraph "RBM layer 3"
    id1(("$$h^{(3)}$$"))
    id2(("$$h^{(3)}$$"))
    id3(("$$h^{(3)}$$"))
    end
    subgraph "RBM layer 2"
    id4(("$$h^{(2)}$$"))
    id5(("$$h^{(2)}$$"))
    end
    subgraph "SBN layer 1"
    id6(("$$h^{(2)}$$"))
    id7(("$$h^{(2)}$$"))
    id8(("$$h^{(2)}$$"))
    end
    subgraph "visible layer"
    id9(("$$v$$"))
    id10(("$$v$$"))
    end
    id1(("$$h^{(3)}$$")) --- id4(("$$h^{(2)}$$"))
    id1(("$$h^{(3)}$$")) --- id5(("$$h^{(2)}$$"))
    id2(("$$h^{(3)}$$")) --- id4(("$$h^{(2)}$$"))
    id2(("$$h^{(3)}$$")) --- id5(("$$h^{(2)}$$"))
    id3(("$$h^{(3)}$$")) --- id4(("$$h^{(2)}$$"))
    id3(("$$h^{(3)}$$")) --- id5(("$$h^{(2)}$$"))
    id4(("$$h^{(2)}$$")) --> id6(("$$h^{(2)}$$"))
    id4(("$$h^{(2)}$$")) --> id7(("$$h^{(2)}$$"))
    id4(("$$h^{(2)}$$")) --> id8(("$$h^{(2)}$$"))
    id5(("$$h^{(2)}$$")) --> id6(("$$h^{(2)}$$"))
    id5(("$$h^{(2)}$$")) --> id7(("$$h^{(2)}$$"))
    id5(("$$h^{(2)}$$")) --> id8(("$$h^{(2)}$$"))
    id6(("$$h^{(2)}$$")) --> id9(("$$v$$"))
    id6(("$$h^{(2)}$$")) --> id10(("$$v$$"))
    id7(("$$h^{(2)}$$")) --> id9(("$$v$$"))
    id7(("$$h^{(2)}$$")) --> id10(("$$v$$"))
    id8(("$$h^{(2)}$$")) --> id9(("$$v$$"))
    id8(("$$h^{(2)}$$")) --> id10(("$$v$$"))

    classDef shaded fill:#b6b8d6,stroke:#333,stroke-width:2px;
    class id9,id10 shaded
```

</div>

{{< math.inline >}}
<p>
For each layer in DBN, we have:
</p>
{{</ math.inline >}}

$$
k = 1\cdots \infty \\\
\text{hidden layer k} : \begin{cases}
h^{(k)} &: \text{nodes}, h^{(k)} \sim \text{Bernoulli}
\\\
w^{(k)} &: \text{weights of edges} \\\
b^{(k)} &: \text{bias}
\end{cases}
$$

$$
\text{visible layer} : \begin{cases}
v &: \text{nodes}, v \sim \text{Bernoulli} \\\
b^{(0)} &: \text{bias}
\end{cases}
$$

Then we get all the parameters that need to be learned:

$$
\theta = ( w^{(1)},\cdots,w^{(\infty)},b^{(0)},b^{(1)},\cdots,b^{(\infty)} )
$$

Suppose DBN contains L layers, the joint distribution of DBN can be given by [factorization of Bayesian network](https://tirmisula.github.io/posts/probabilistic-graphical-model/#conclusion):

$$
\begin{align*}
p(v,h^{(1)},\cdots,h^{(L)}) &= p(v|h^{(1)},\cdots,h^{(L)})p(h^{(1)},\cdots,h^{(L)}) \\\
&= p(v|h^{(1)})p(h^{(1)}|h^{(2)})\cdots p(h^{(L-2)}|h^{(L-1)})p(h^{(L-1)},h^{(L)}) \\\
&\because \text{tail to tail structure, nodes in same layer are mutually conditional independent} \\\
&= \prod_{i}p(v_i|h^{(1)})\prod_{j_1}p(h_{j}^{(1)}|h^{(2)})\cdots \prod_{j_{\ast}}p(h_{j}^{(L-2)}|h^{(L-1)})p(h^{(L-1)},h^{(L)})
\end{align*}
$$

We have given the [RBM's conditional probability](https://tirmisula.github.io/posts/restricted-boltzman-machine/#posterier-inference):

$$
\begin{align*}
p(o_k=1|h) &= \sigma(\sum_{i=1}^mh_iw_{ik}+\alpha_k) \\\
p(o_k=0|h) &= 1-\sigma(\sum_{i=1}^mh_iw_{ik}+\alpha_k)
\end{align*}
$$

Similarly, in [Sigmoid belief network](https://tirmisula.github.io/posts/sigmoid-belief-network/#sigmoid-belief-network-definition) we have the conditional probability:

$$
\begin{align*}
p(s_i=1 | \lbrace s_j \rbrace\in\text{parents}(x_i)) &= \sigma(\sum_jw_{ji}s_j) \\\
p(s_i=0 | \lbrace s_j \rbrace\in\text{parents}(x_i)) &= \sigma(-\sum_jw_{ji}s_j)
\end{align*}
$$

Thus we can conclude:

$$
\begin{align*}
p(h_{j}^{(k)}=1|h^{(k+1)}) &= \sigma( w_{j}^{T(k+1)}h^{(k+1)}+b_{j}^{(k)}) \\\
p(v_i=1|h^{(1)}) &= \sigma( w_{i}^{T(1)}h^{(1)}+b_{i}^{(0)})
\end{align*}
$$

The joint probabilty of [restricted Boltzaman machine](https://tirmisula.github.io/posts/restricted-boltzman-machine/#rbm-model-definition) is given by:

$$
\begin{align*}
p(o,h) &= \frac{1}{z}\exp(h^Two+\alpha^To+\beta^Th) \\\
&= \frac{1}{z} \prod_{i=1}^m\prod_{j=1}^n\exp(h_iw_{ij}o_j) \prod_{i=1}^m\exp(\beta_i h_i) \prod_{j=1}^n\exp(\alpha_j o_j)
\end{align*}
$$

So we have the joint distribution of top two layers:

$$
p(h^{(L-1)},h^{(L)}) = \frac{1}{z}\exp\left(h^{T(L)}w^{(L)}h^{(L-1)}+h^{T(L-1)}b^{(L-1)}+h^{T(L)}b^{(L)} \right) \\\
$$

## Stacking RBM Improves ELBO of p(v)

Consider an original RBM model, we have marginal probability:

$$
\begin{align*}
p(v) &= \sum_{h^{(1)}} p(v,h^{(1)}) \\\
&= \sum_{h^{(1)}} p(h^{(1)})p(v|h^{(1)}) \\\
&\text{$p(h^{(1)})$ is prior}
\end{align*}
$$

We introduced in [HMM chapter](https://tirmisula.github.io/posts/hidden-markov-model/#learning) that learning problem is equivalent to maximum likelihood estimation:

$$
\hat{\lambda} = \argmax p(O|\lambda)
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
However, posterier \( p(h|v) \) is intractable because head to head structure exists in SBN.
</p>
{{</ math.inline >}}

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