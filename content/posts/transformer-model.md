---
author: "X. Wang"
title: "Transformer"
date: "2023-10-05"
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
### Softmax function

Softmax function is commonly used in multiclass classification. A formal definition:

$$
\begin{align*}
\sigma &: \mathbb{R}^K \mapsto (0,1)^K \\\
z &= \begin{bmatrix}
    z_1 & \cdots & z_K
\end{bmatrix}^T, z\in \mathbb{R}^K \\\
\sigma(z) &= \begin{bmatrix}
    \sigma(z)_1 & \cdots & \sigma(z)_K
\end{bmatrix}^T, \sigma(z) \in (0,1)^K \\\
\sigma(z)_i &= \frac{\exp(z_i)}{\sum_{j=1}^K\exp(z_j)} \\\
\end{align*}
$$

{{< math.inline >}}
<p>
Given a \( K \) dimensional input vector \( z \), the softmax result is a normalized probability distribution consisting of \( K \) probabilities. Each class's probability is proportional to the exponential of it's input.
</p>
{{</ math.inline >}}

### Data representation

{{< math.inline >}}
<p>
The most common application area is NLP, given a dataset \( X \):
</p>
{{</ math.inline >}}

$$
X = [X_{ij}]_{N\times D} \\\
X \text{ is a set of $N$ samples, $D$ features}
$$

{{< math.inline >}}
<p>
Each row vector is a data sample, define it as \( X_i \):
</p>
{{</ math.inline >}}

$$
X_i = \begin{bmatrix}
    X_{i1} & \cdots & X_{iD}
\end{bmatrix}
$$

{{< math.inline >}}
<p>
We can represent \( X \) with row vectors:
</p>
{{</ math.inline >}}

$$
X = \begin{bmatrix}
    X_1 \\\
    \vdots \\\
    X_N
\end{bmatrix}
$$

### Matrix multiplication as linear operator

{{< math.inline >}}
<p>
Given a matrix \( A \), right multiply \( X \) by \( A \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
XA &= \begin{bmatrix}
    X_1 \\\
    \vdots \\\
    X_N
\end{bmatrix}A \\\
&= \begin{bmatrix}
    X_1A \\\
    \vdots \\\
    X_NA
\end{bmatrix} \\\
&= \begin{bmatrix}
    \mathcal{A}(X_1) \\\
    \vdots \\\
    \mathcal{A}(X_N)
\end{bmatrix} = \mathcal{A}(X)
\end{align*} \\\
\mathcal{A}(\cdot) \text{ is linear operator}, \mathcal{A} : \mathbb{R}^D \mapsto \mathbb{R}^{}
$$

### Seq2seq

Seq2seq model which used in NLP area is commonly composed of an encoder and a decoder.

+ Encoder part

    + raw tokens to word embeddings

    $$
    (x_1,\cdots,x_N) \rarr (X_1,\cdots,X_N)
    $$

    + (multi-layers) word embeddings to latent semantic vectors

    $$
    (X_1,\cdots,X_N) \rarr (Y_1,\cdots,Y_N)
    $$

+ Decoder part

    + Decode Y1...YN in autoregressive manner

    $$
    (Y_1,\cdots,Y_N) \rarr (y_1,\cdots,y_N) \\\
    p(y_{1:N};\theta) = \prod_{i=1}^N p(y_i|y_{1:i-1},Y;\theta)
    $$

## Motivation


## Reference

[^1]: - [video](https://www.bilibili.com/video/BV19H4y1G73r).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [Denoising Diffusion Probabilistic Models. Jonathan Ho, Ajay Jain, Pieter Abbee](https://arxiv.org/pdf/2006.11239).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
