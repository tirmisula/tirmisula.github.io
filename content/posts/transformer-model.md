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

## Transformer versus Earlier Models

### RNN

The recurrent neural network (RNN) cell is given by:

$$
\begin{align*}
&Y_{i}=\tanh(X_iW+Y_{i-1}U) \\\
&\tanh : \text{hyperbolic tangent activation function, $\tanh(x)\in[-1,1]$} \\\
&W : \text{weight matrix that transforms input $X$} \\\
&U : \text{weight matrix that transforms previous hidden state $Y_{i-1}$} \\\
\end{align*}
$$

By the definition, it is clear that the problem is:

1. Encoding in RNN is serialized.

    $$
    Y_0 \rarr Y_1 \rarr \cdots Y_N
    $$

2. Each hidden state depends on previous states.

    $$
    Y_i \text{ directly depends on $X_i,Y_{i-1}$} \\\
    Y_i \text{ indirectly depends on $X_{1:i-1}$}
    $$

### CNN

The convolutional neural network (CNN) has encoder like:

$$
\begin{align*}
&Y_{1:N} = f(X_{1:N}\circledast W+b) \\\
&Y_i = (X_{i-1},X_i,X_{i+1})\circledast W \\\
&f : \text{activation function, e.g. ReLU} \\\
&W : \text{conv window, suppose the kernel size is 3} \\\
&b : \text{bias term} \\\
\end{align*}
$$

By the definition:

1. Encoding in CNN is parallelized.

    $$
    Y_1 \larr (\text{pad},X_1,X_2)\circledast W \\\
    Y_2 \larr (X_1,X_2,X_3)\circledast W \\\
    \cdots \\\
    Y_N \larr (X_{N-1},X_{N},\text{pad})\circledast W \\\
    $$

2. Each hidden state depends on states inside the conv window.

    $$
    Y_i \text{ depends on $X_{i-1:i+1}$}
    $$

### Transformer

The motivation of Transformer is to design a encoder so that:

$$
\begin{align*}
&Y_i = f(X_1,\cdots,X_N) \\\
&\text{ $Y_i$ depends on all tokens} \\\
&Y_1 \cdots Y_N \text{ computation can be parallelized}
\end{align*} 
$$

{{< math.inline >}}
<p>
In order to acheive these 2 goals, Transformer is designed to have 2 matrix transformations from \( X \) to \( Y \), denote the matrix transformation by 2 operators:
</p>
{{</ math.inline >}}

$$
Y = (\mathcal{F}\circ\mathcal{A})(X)
$$

1. Multi-head Attention

    $$
    \begin{align*}
    &Y' = \mathcal{A}(X) \\\
    &\mathcal{A}(X_i) = \frac{\sum_{j=1}^N\text{sim}(X_i,X_j)X_j}{\sum_{j=1}^N\text{sim}(X_i,X_j)} \\\
    &\text{sim}(X_i,X_j) : \text{defined with attention mechanism}
    \end{align*}
    $$

    The first operator computes weighted average of similarities of different word embeddings.

2. Position-Wise Feed-Forward

    $$
    \begin{align*}
    &Y = \mathcal{F}(Y') \\\
    &\mathcal{F} : \text{non-linear transformation}
    \end{align*}
    $$

## Single-head Attention
### Attention score versus similarity

Recall the similarity mentioned above, a conventional function like cosine similarity provides no parameter which is not suitable for a complex model:

$$
\text{sim}(X_i,X_j) = \frac{X_i \cdot X_j}{||X_i||||X_j||}
$$

{{< math.inline >}}
<p>
An alternative way is to project \( X \) to other spaces first, then compute similarity in a dot-product manner:
</p>
{{</ math.inline >}}

$$
\begin{align*}
&\text{Let } \begin{array}{l}
    X_i \rarr X_iW_Q \\\
    X_j \rarr X_jW_K \text{ , } X_jW_V \\\
    X_l \rarr W_K
\end{array} \text{ we have } \\\
&\mathcal{A}(X_i) = \frac{\sum_{j=1}^N\text{sim}(X_iW_Q,X_jW_K)X_jW_V}{\sum_{l=1}^N\text{sim}(X_iW_Q,X_lW_K)} \\\
\\\
&\text{Denote } \begin{array}{l}
    Q_i = X_iW_Q \\\
    K_j = X_jW_K \\\
    V_j = X_jW_V
\end{array} \text{ we have } \\\
&\mathcal{A}(X_i) = \frac{\sum_{j=1}^N\text{sim}(Q_i,K_j)V_j}{\sum_{l=1}^N\text{sim}(Q_i,K_l)} \\\
\\\
&\text{Let } \text{sim}(Q_i,K_j) = \frac{Q_iK^T_j}{\sqrt{d}} = \text{score}(Q_i,K_j) \\\
&\text{$\sqrt{d}$ is the scaling factor, $Q_i,K_i\in\mathbb{R}^d$} \\\
\\\
&\text{The final form: } \\\
&\mathcal{A}(X_i) = \frac{\sum_{j=1}^N\text{score}(Q_i,K_j)V_j}{\sum_{l=1}^N\text{score}(Q_i,K_l)} \\\
&\quad Q_i : \text{query related to token $X_i$} \\\
&\quad K_i,V_i : \text{key-value pair of token $X_i$}
\end{align*}
$$

{{< math.inline >}}
<p>
\( \text{score}(Q_i,K_j) \) is the <b>attention score</b> which take the place of similarity measurement.
</p>
{{</ math.inline >}}


### Attention weights

{{< math.inline >}}
<p>
\( \mathcal{A}(X_i) \) can be further written as:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\mathcal{A}(X_i) &= \sum_{j=1}^N\frac{\text{score}(Q_i,K_j)}{\sum_{l=1}^N\text{score}(Q_i,K_l)}V_j \\\
&\text{Let } \alpha_{ij}=\frac{\text{score}(Q_i,K_j)}{\sum_{l=1}^N\text{score}(Q_i,K_l)} \\\
&= \sum_{j=1}^N\alpha_{ij}V_j \\\
&\alpha_{ij} \text{ is the normalized weight}
\end{align*}
$$

{{< math.inline >}}
<p>
From this perspective, \( \mathcal{A}(X_i) \) is the weighted sum of projected values. Since \( V_1 \cdots V_N \) are different semantics which is similar to multiple classes, a better choice for getting weights \( \alpha_{i1}\cdots\alpha_{iN} \) is passing similarities to Softmax function:
</p>
{{</ math.inline >}}

$$
\text{Let } \begin{bmatrix}
    \alpha_{i1} \\\
    \vdots \\\
    \alpha_{iN}
\end{bmatrix} = \text{Softmax}(\begin{bmatrix}
    \text{score}(Q_i,K_1) \\\
    \vdots \\\
    \text{score}(Q_i,K_N)
\end{bmatrix}) \\\
\\\
\begin{align*}
\alpha_{ij} &= \frac{\exp(\text{score}(Q_i,K_j))}{\sum_{l=1}^N\exp(\text{score}(Q_i,K_l))}
\end{align*}
$$

{{< math.inline >}}
<p>
The new \( \alpha_{i1}\cdots\alpha_{iN} \) weights are called <b>attention weights</b>. A summary of attention concepts in these 2 sections:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{score}(Q_i,K_j) &: \text{attention score $\frac{Q_iK^T_j}{\sqrt{d}}$} \\\
\frac{\text{score}(Q_i,K_j)}{\sum_{j=1}^N\text{score}(Q_i,K_j)} &: \text{nomalized attention score} \\\
\frac{\exp(\text{score}(Q_i,K_j))}{\sum_{l=1}^N\exp(\text{score}(Q_i,K_l))} &: 
\text{attention weight} \\\
\mathcal{A}(X_i) = \sum_{j=1}^N\frac{\text{score}(Q_i,K_j)}{\sum_{l=1}^N\text{score}(Q_i,K_l)}V_j &: \text{scaled dot-product attention}
% \begin{array}{l}
%     \text{weighted sum of all value vectors for query $Q_i$}
% \end{array}
\end{align*}
$$

In Transformer, attention mechanism is referred to as **Self-Attention**, because it used the sequence itself for queries, keys, and values.

$$
\text{score}(Q_1 , \begin{array}{l}
    K_1 \\\
    \vdots \\\
    K_N
\end{array}) \xrightarrow{attention\space weights} \begin{array}{l}
    \alpha_{11} \\\
    \vdots \\\
    \alpha_{1N}
\end{array} \xrightarrow{weighted\space sum} \sum_{j=1}^N\alpha_{1j}V_j \\\
\cdots \\\
\text{score}(Q_N , \begin{array}{l}
    K_1 \\\
    \vdots \\\
    K_N
\end{array}) \xrightarrow{attention\space weights} \begin{array}{l}
    \alpha_{N1} \\\
    \vdots \\\
    \alpha_{NN}
\end{array} \xrightarrow{weighted\space sum} \sum_{j=1}^N\alpha_{Nj}V_j \\\
$$

### Single-head Attention formalization

{{< math.inline >}}
<p>
Define the projection matrices \( W_Q,W_K,W_V \) as:
</p>
{{</ math.inline >}}

$$
W_Q \in \mathbb{R}^{D\times M} , W_K \in \mathbb{R}^{D\times M}, W_V \in \mathbb{R}^{D\times M}
$$

{{< math.inline >}}
<p>
We can get the \( Q,K,V \) dimensions:
</p>
{{</ math.inline >}}

$$
Q_i,K_i,V_i \in \mathbb{R}^{1\times M} \\\
Q,K,V \in \mathbb{R}^{N\times M}
$$

{{< math.inline >}}
<p>
For \( Q_i \) and all \( K_j \)'s similarity, calculate attention scores based on dot product:
</p>
{{</ math.inline >}}

$$
\text{score}(Q_i,K_j) = \frac{Q_i \cdot K^T_j}{\sqrt{M}} \\\
\sqrt{M} \text{ is the scaling factor}\\\
$$

$$
\begin{align*}
\text{score}(Q_i , \begin{array}{l}
    K_1 \\\
    \vdots \\\
    K_N
\end{array}) &= \begin{bmatrix}
    \frac{1}{\sqrt{M}}Q_iK^T_1 & \cdots & \frac{1}{\sqrt{M}}Q_iK^T_N
\end{bmatrix} \\\
&= \frac{Q_iK^T}{\sqrt{M}} \\\
\end{align*}
$$

{{< math.inline >}}
<p>
Use Softmax function to obtain \( Q_i \)'s attention weights:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\alpha_{i} &= \text{softmax}(\frac{Q_iK^T}{\sqrt{M}}) \\\
&= \begin{bmatrix}
    \alpha_{i1} & \cdots & \alpha_{iN}
\end{bmatrix} \\\
\alpha_{ij} &= \frac{\exp(\frac{Q_iK^T_j}{\sqrt{M}})}{\sum_{l=1}^N\exp(\frac{Q_iK^T_l}{\sqrt{M}})}
\end{align*}
$$

{{< math.inline >}}
<p>
Compute weighted sum (Attention):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\mathcal{A}(X_i) &= \sum_{j=1}^N\alpha_{ij}V_j \\\
&= \begin{bmatrix}
    \alpha_{i1} & \cdots & \alpha_{iN}
\end{bmatrix} \begin{bmatrix}
    V_1 \\\
    \vdots \\\
    V_N
\end{bmatrix} \\\
&= \alpha_iV \\\
&= \text{softmax}(\frac{Q_iK^T}{\sqrt{M}})V \\\
&= \text{Attention}(Q_i,K,V)
\end{align*}
$$

{{< math.inline >}}
<p>
Convert it to matrix form (Self-Attention):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\mathcal{A}(X) &= \begin{bmatrix}
    \mathcal{A}(X_1) \\\
    \vdots \\\
    \mathcal{A}(X_N)
\end{bmatrix} \\\
&= \begin{bmatrix}
    \text{softmax}(\frac{Q_1K^T}{\sqrt{M}})V \\\
    \vdots \\\
    \text{softmax}(\frac{Q_NK^T}{\sqrt{M}})V
\end{bmatrix} \\\
&= \begin{bmatrix}
    \begin{bmatrix}
    \alpha_{11} & \cdots & \alpha_{1N}
\end{bmatrix} \\\
    \vdots \\\
    \begin{bmatrix}
    \alpha_{N1} & \cdots & \alpha_{NN}
\end{bmatrix}
\end{bmatrix}V \\\
&\because Q_iK^T_j \xmapsto{\text{Softmax}} \alpha_{ij} \\\
&\therefore \begin{bmatrix}
    Q_1 \\\
    \vdots \\\
    Q_N
\end{bmatrix} \begin{bmatrix}
    K^T_1 & \cdots & K^T_N
\end{bmatrix} \xmapsto{\text{Softmax}} \begin{bmatrix}
    \alpha_{11} & \cdots & \alpha_{1N} \\\
    \vdots & \ddots & \vdots \\\
    \alpha_{N1} & \cdots & \alpha_{NN}
\end{bmatrix} \\\
&= \text{Softmax}(\frac{QK^T}{\sqrt{M}})V \\\
&= \text{Attention}(Q,K,V)
\end{align*}
$$

{{< math.inline >}}
<p>
Then the Single-head Attention is given:
</p>
{{</ math.inline >}}

$$
\text{Attention}(Q,K,V) = \text{Softmax}(\frac{QK^T}{\sqrt{M}})V \\\
Q,K,V \in \mathbb{R}^{N\times M}
$$

## Multi-head Attention



## Reference

[^1]: - [video](https://www.bilibili.com/video/BV19H4y1G73r).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [Denoising Diffusion Probabilistic Models. Jonathan Ho, Ajay Jain, Pieter Abbee](https://arxiv.org/pdf/2006.11239).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
