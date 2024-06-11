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
draft: false
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
In order to acheive these 2 goals, Transformer is designed to have 2 transformations from \( X \) to \( Y \), denote the transformations by 2 operators:
</p>
{{</ math.inline >}}

$$
Y = (\mathcal{F}\circ\mathcal{A})(X)
$$

1. Multi-head Self-Attention

    $$
    \begin{align*}
    &Y = \mathcal{A}(X) \\\
    &\mathcal{A}(X) : \text{ matrix transformation which is linear} \\\
    &\mathcal{A}(X_i) = \frac{\sum_{j=1}^N\text{sim}(X_i,X_j)X_j}{\sum_{j=1}^N\text{sim}(X_i,X_j)} \\\
    &\text{sim}(X_i,X_j) : \text{defined by attention mechanism}
    \end{align*}
    $$

    The first operator indicates weighted sum (based on normalized similarity weights) of all word embeddings. In other words, the output is a weighted average representation of all input tokens (depends on all tokens). Matrix transformation indicates getting outputs parallel.

2. Position-Wise Feed-Forward

    $$
    \begin{align*}
    &Y' = \mathcal{F}(Y) \\\
    &\mathcal{F} : \text{non-linear transformation}
    \end{align*}
    $$

    While self-attention captures dependencies and relationships, the FFN processes these representations further by adding non-linearity.

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

{{< math.inline >}}
<p>
For tokens \( X_1 \cdots X_N \), the encoding result is:
</p>
{{</ math.inline >}}

$$
(Y_1, \cdots, Y_N) = \text{Attention}(XW_Q,XW_K,XW_V)
$$

It's corresponding function operator representation is:

$$
Y = \text{Softmax}(\frac{QK^T}{\sqrt{M}})V
$$

## Multi-head Attention

{{< math.inline >}}
<p>
Multi-head Attention (MHA) is composed of multiple Single-head Attention (SHA) encoding results, each SHA has it's own set of weight matrices \( W_Q^{(i)},W_K^{(i)},W_V^{(i)} \).
</p>
{{</ math.inline >}}

{{< math.inline >}}
<p>
Suppose MHA has \( H \) heads, each head \( \text{head}^{(i)} \) has the same dimensionality of weight matrices. The math formulation is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{head}^{(i)} &= \text{Attention}(XW_Q^{(i)},XW_K^{(i)},XW_V^{(i)}) \\\
W_Q^{(i)},W_K^{(i)},W_V^{(i)} &\in \mathbb{R}^{D\times M} \\\
\text{head}^{(i)} &\in \mathbb{R}^{N\times M}
\end{align*}
$$

These heads are concatenated and linear transformed in MHA:

$$
\begin{align*}
\text{MultiHead}(Q,K,V) &= \text{concat}(\text{head}^{(1)},\cdots,\text{head}^{(H)})W_O \\\
W_O &\in \mathbb{R}^{(H M)\times D} \text{ , output projection matrix} \\\
\text{MultiHead} &\in \mathbb{R}^{N\times D}
\end{align*}
$$

The corresponding function operator representation is:

$$
\text{head}^{(i)} = \mathcal{A}^{(i)}(X) \\\
Y = \mathcal{A}(X) = \text{concat}(\mathcal{A}^{(1)}(X),\cdots,\mathcal{A}^{(H)}(X))W_O
$$

{{< math.inline >}}
<p>
The outputs of MHA is designed to have the same dimensionality with input tokens \( X \).
</p>
{{</ math.inline >}}

In summary of MHA:

+ Each single head focuses on (a single aspect of the input sequence)/(a specific part of the input sequence).

+ Compared to Single-head Attention which uses a single set of query, key, value projections, Multi-head Attention uses multiple sets of query, key, value projections. Raw information is projected to different subspaces to get different representations.

+ Allows the model to learn from multiple perspectives and features simultaneously.

## Position-Wise Feed-Forward

In Transformer, the Position-wise Feed-Forward Network (FFN) is applied independently to each position (encoded token) in the sequence. It consists of 2 fully connected layers:

$$
\begin{align*}
\text{FFN}(Y_i) &= f(Y_iW_1+b_1)W_2+b_2 \\\
f(\cdot) &: \text{ReLu activation function} \\\
W_1,b_1 &: \text{first layer weights and biases} \\\
W_2,b_2 &: \text{second layer weights and biases}
\end{align*}
$$

The corresponding function operator representation is:

$$
Y_i' = \mathcal{F}(Y_i) = f(Y_iW_1+b_1)W_2+b_2 \\\
Y' = \mathcal{F}(Y) = \begin{bmatrix} 
    f(Y_1W_1+b_1)W_2+b_2 \\\
    \vdots \\\
    f(Y_NW_1+b_1)W_2+b_2
\end{bmatrix}
$$

## Layer Normalization and Residual Nets

Layer normalization is defined as:

$$
\begin{align*}
\text{LayerNorm}(X_i) &= \frac{X_i-\mu}{\sqrt{\sigma^2}+\epsilon}\gamma + \beta \\\
\mu &: \text{the mean of $X_i$} \\\
\sigma^2 &: \text{the variance of $X_i$} \\\
\epsilon &: \text{small constant for numerical stability} \\\
\gamma,\beta &: \text{learnable parameters}
\end{align*}
$$

In Transformer, layer normalization can be applied either before or after the main sub-layers (MHA and FFN). In this post, layer normalization is assumed to be applied before sub-layers, it's operator is given by:

$$
\mathcal{N}(X) = \begin{bmatrix}
    \mathcal{N}(X_1) \\\
    \vdots \\\
    \mathcal{N}(X_N)
\end{bmatrix} = \begin{bmatrix}
    \text{LayerNorm}(X_1) \\\
    \vdots \\\
    \text{LayerNorm}(X_N)
\end{bmatrix}
$$

And the residual connection is added afterward:

$$
X_{\text{output}} = X_{\text{input}} + \text{SubLayer}(\mathcal{N}(X))
$$

## Transformer Block Definition

Summarize briefly each module mentioned above:

$$
\begin{align*}
\text{Single-head Attention} &: \mathcal{A}^{(h)}(X) = \text{Softmax}(\frac{QK^T}{\sqrt{M}})V \\\
\text{Multi-head Attention} &: \mathcal{A}(X) = \text{concat}(\mathcal{A}^{(1)}(X),\cdots,\mathcal{A}^{(H)}(X))W_O \\\
\text{Position-Wise Feed-Forward} &: \mathcal{F}(Y) = \begin{bmatrix} 
    f(Y_1W_1+b_1)W_2+b_2 \\\
    \vdots \\\
    f(Y_NW_1+b_1)W_2+b_2
\end{bmatrix} \\\
\text{LayerNorm+Residual Connect} &: X_{\text{output}} = X + \text{SubLayer}(\mathcal{N}(X))
\end{align*}
$$

A complete Transformer block has one Multi-head Self-Attention layer and one feed-forward layer, which can be formalized:

$$
\begin{align*}
\text{Let } Y &= \mathcal{T}(X) , \text{ $\mathcal{T}$ is the Transformer block} \\\
X' &= X + \mathcal{A}\circ\mathcal{N}(X) \\\
\mathcal{T}(X) &= X' + \mathcal{F}\circ\mathcal{N}(X') \\\
&\text{$X$ is the input to the Transformer block} \\\
&\text{$X'$ is the output of Self-Attention layer} \\\
&\text{$Y$ is the output of Transformer block}
\end{align*}
$$

## Number of Parameters

### Parameters in Multi-head Attention

{{< math.inline >}}
<p>
There are \( 3H \) projection matrices and one output projection matrix:
</p>
{{</ math.inline >}}

$$
W_Q, W_K, W_V \in \mathbb{R}^{D\times M} \\\
W_O \in \mathbb{R}^{HM\times D}
$$

{{< math.inline >}}
<p>
Usually \( H\times M=D \). The total number of parametes in MHA is:
</p>
{{</ math.inline >}}

$$
3 \times H \times D \times M + H\times M\times D = 4D^2
$$

### Parameters in Feed-Forward Net

In the original paper<cite>[^2]</cite>, the number of parameters of the first and second linear layer are 512 and 2048 respectively. The total number of parametes in MHA is:

$$
D\times 4D + 4D \times D = 8D^2
$$

### Total number of parameters

Consider input word embeddings \( X\in\mathbb{R}^{N\times D} \) are parameters, the total number of parameters is given by:

$$
12D^2 +ND
$$

## Computation Complexity

### Multi-head Attention complexity

{{< math.inline >}}
<p>
The computation complexity is mainly based on \( Q,K,V \) matrix multiplications:
</p>
{{</ math.inline >}}


1. Projection of Queries, Keys, and Values:

$$
\begin{align*}
\text{Each projection has a complexity of } O(N\times M\times D)
\end{align*}
$$

2. Scaled Dot-Product Attention:

$$
\text{$QK^T$ has a complexity of } O(N\times N\times M) \\\
\text{$\text{Softmax}(\cdot)$ has a complexity of } O(N^2) \\\
\text{$\text{Softmax}(\cdot)V$ has a complexity of } O(N\times N\times M)
$$

3. Output projection:

$$
\text{Under the assumption: $D=HM$} \\\
\text{$\text{concat}(\cdots)W_O$ has a complexity of } O(N\times D\times D) \\\
$$

The total complexity for a single head is:

$$
O(3N\times M\times D+N^2(2M+1)) = O(N^2M)
$$

The total complexity for multi-head is:

$$
O(N^2M\times H)+O(ND^2) = O(N^2D+ND^2)
$$

### Feed-Forward Net complexity

1. First layer:

$$
\text{$XW_1$ has a complexity of } O(N\times 4D\times D) \\\
$$

2. Second layer:

$$
\text{$XW_2$ has a complexity of } O(N\times D\times 4D) \\\
$$

The total complexity of FFN is:

$$
O(8ND^2) = O(ND^2)
$$

## Linear Transformer

### Motivation

{{< math.inline >}}
<p>
When token number is large enough (\( N \gg D \)), the complexity from above can be reduced to:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{Multi-head Self-Attention} &: O(N^2D+ND^2) = O(N^2) \\\
\text{Feed-Forward Net} &: O(ND^2) = O(N)
\end{align*}
$$

{{< math.inline >}}
<p>
The main complexity \( O(N^2) \) comes from \( \text{Softmax}(QK^T) \). An alternative way is to compute \( K^TV \) first, then compute \( Q(K^TV) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{Complexity of $K^TV$} &: O(M\times M\times N)=O(N) \\\
\text{Complexity of $Q(K^TV)$} &: O(N\times M\times M) = O(N)
\end{align*}
$$

The overall complexity is reduced from quadratic to linear.

### Linearized Attention

Recall that the weighted sum representation:

$$
\mathcal{A}(X_i) = \frac{\sum_{j=1}^N\text{sim}(Q_i,K_j)V_j}{\sum_{l=1}^N\text{sim}(Q_i,K_l)} \\\
\text{sim}(Q_i,K_j) = \exp(\frac{Q_iK^T_j}{\sqrt{M}})
$$

<cite>[^3]</cite>Linear Transformer uses a kernel function to represent similarity measurement:

$$
\text{Let } \text{sim}(Q_i,K_j) = k(Q_i,K_j) = \langle\phi(Q_i),\phi(K_j)\rangle \\\
\mathcal{A}(X_i) = \frac{\sum_{j=1}^N\phi(Q_i)\phi(K_j)^TV_j}{\sum_{l=1}^N\phi(Q_i)\phi(K_l)^T} \\\
$$

The numerator can be simplified:

$$
\begin{align*}
\sum_{j=1}^N\phi(Q_i)\phi(K_j)^TV_j &= \sum_{j=1}^N\begin{bmatrix}
    \phi(Q_{i1}) & \cdots & \phi(Q_{iM})
\end{bmatrix} \begin{bmatrix}
    \phi(K_{j1}) \\\
    \vdots \\\
    \phi(K_{jM})
\end{bmatrix} \begin{bmatrix}
    V_{j1} & \cdots & V_{jM}
\end{bmatrix} \\\
&= \sum_{j=1}^N(\sum_{z=1}^M\phi(Q_{iz})\phi(K_{jz}))\begin{bmatrix}
    V_{j1} & \cdots & V_{jM}
\end{bmatrix} \\\
&= \sum_{j=1}^N\begin{bmatrix}
    \sum_{z=1}^M\phi(Q_{iz})\phi(K_{jz})V_{j1} & \cdots & \sum_{z=1}^M\phi(Q_{iz})\phi(K_{jz})V_{jM}
\end{bmatrix} \\\
&= \begin{bmatrix}
    \sum_{z=1}^M\sum_{j=1}^N\phi(Q_{iz})\phi(K_{jz})V_{j1} & \cdots & \sum_{z=1}^M\sum_{j=1}^N\phi(Q_{iz})\phi(K_{jz})V_{jM}
\end{bmatrix} \\\
&= \begin{bmatrix}
    \sum_{z=1}^M\phi(Q_{iz})\sum_{j=1}^N\phi(K_{jz})V_{j1} & \cdots & \sum_{z=1}^M\phi(Q_{iz})\sum_{j=1}^N\phi(K_{jz})V_{jM}
\end{bmatrix} \\\
&= \begin{bmatrix}
    \phi(Q_i)\sum_{j=1}^N\phi(K_{j})^TV_{j1} & \cdots & \phi(Q_i)\sum_{j=1}^N\phi(K_{j})^TV_{jM}
\end{bmatrix} \\\
&= \phi(Q_i)\sum_{j=1}^N\begin{bmatrix}
    \phi(K_{j})^TV_{j1} & \cdots & \phi(K_{j})^TV_{jM}
\end{bmatrix} \\\
&= \phi(Q_i)\sum_{j=1}^N\phi(K_{j})^TV_{j} \\\
\text{Furthermore} \\\
\phi(Q_i)\sum_{j=1}^N\phi(K_{j})^TV_{j} &= \phi(Q_i)\sum_{j=1}^N \begin{bmatrix}
    \phi(K_{j1})V_{j1} & \cdots & \phi(K_{j1})V_{jM} \\\
    \vdots & \ddots & \vdots \\\
    \phi(K_{jM})V_{j1} & \cdots & \phi(K_{jM})V_{jM}
\end{bmatrix} \\\
&= \phi(Q_i) \begin{bmatrix}
    \sum_{j=1}^N\phi(K_{j1})V_{j1} & \cdots & \sum_{j=1}^N\phi(K_{j1})V_{jM} \\\
    \vdots & \ddots & \vdots \\\
    \sum_{j=1}^N\phi(K_{jM})V_{j1} & \cdots & \sum_{j=1}^N\phi(K_{jM})V_{jM}
\end{bmatrix} \\\
&= \phi(Q_i) \begin{bmatrix}
    \phi(K_{:,1})^TV_{:,1} & \cdots & \phi(K_{:,1})^TV_{:,M} \\\
    \vdots & \ddots & \vdots \\\
    \phi(K_{:,M})^TV_{:,1} & \cdots & \phi(K_{:,M})^TV_{:,M}
\end{bmatrix} \\\
&= \phi(Q_i) \begin{bmatrix}
    \begin{bmatrix}
        \phi(K_{:,1})^T \\\
        \vdots \\\
        \phi(K_{:,M})^T
    \end{bmatrix}V_{:,1} & \cdots & \begin{bmatrix}
        \phi(K_{:,1})^T \\\
        \vdots \\\
        \phi(K_{:,M})^T
    \end{bmatrix}V_{:,M}
\end{bmatrix} \\\
&= \phi(Q_i) \begin{bmatrix}
    \phi(K)^TV_{:,1} & \cdots & \phi(K)^TV_{:,M}
\end{bmatrix} \\\
&= \phi(Q_i) \phi(K)^TV
% &= \frac{\phi(Q_i)\sum_{j=1}^N\phi(K_j)V^T_j}{\phi(Q_i)\sum_{l=1}^N\phi(K_l)^T} \\\
\end{align*}
$$

The denominator can be simplified:

$$
\begin{align*}
\sum_{l=1}^N\phi(Q_i)\phi(K_l)^T &= \sum_{l=1}^N\sum_{z=1}^M\phi(Q_{iz})\phi(K_{lz}) \\\
&= \sum_{z=1}^M\phi(Q_{iz})\sum_{l=1}^N\phi(K_{lz}) \\\
&= \phi(Q_i)\sum_{l=1}^N\phi(K_l)^T
\end{align*}
$$

{{< math.inline >}}
<p>
So we have \( K^TV \) computed first:
</p>
{{</ math.inline >}}

$$
\mathcal{A}(X_i) = \frac{\phi(Q_i) \phi(K)^TV}{\phi(Q_i)\sum_{l=1}^N\phi(K_l)^T}
$$

$$
\begin{align*}
\mathcal{A}(X) &= \left(\begin{bmatrix}
    \phi(Q_1) \\\
    \vdots \\\
    \phi(Q_N)
\end{bmatrix} \phi(K)^TV\right) \oslash \left(\begin{bmatrix}
    \phi(Q_1) \\\
    \vdots \\\
    \phi(Q_N)
\end{bmatrix} \sum_{l=1}^N\phi(K_l)^T\right)  \\\
&= \left(\phi(Q) \phi(K)^TV\right) \oslash \left(\phi(Q)\sum_{l=1}^N\phi(K_l)^T\right)
\end{align*} \\\
\oslash \text{ is element-wise division}
$$

{{< math.inline >}}
<p>
Linear Transformer compute \( \phi(K)^TV \) and \( \sum_{l=1}^N\phi(K_l)^T \) once for all queries, it has linear complexity:
</p>
{{</ math.inline >}}

$$
\text{The complexity of }
\begin{cases}
\text{$\phi(K)^TV$} &: O(M\times M\times N) = O(N) \\\
\text{$\phi(Q)(\phi(K)^TV)$} &: O(N\times M\times M) = O(N) \\\
\text{$\sum_{l=1}^N\phi(K_l)^T$} &: O(N+N\times M\times D) = O(N) \\\
\text{$\phi(Q)\sum_{l=1}^N\phi(K_l)^T$} &: O(N\times 1\times M) = O(N)
\end{cases} \\\
\\\
\text{The overall complexity of $\mathcal{A}(X)$ is } O(N)
$$

<cite>[^3]</cite>The feature mapping function is suggested to be the following:

$$
\phi(x) = \text{elu}(x)+1 \\\
\text{elu}(\cdot) : \text{exponential linear unit function}
$$

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV19H4y1G73r).
[^2]: From [Attention Is All You Need. Ashish Vaswani, Llion Jones, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Aidan N. Gomez, Łukasz Kaiser](https://arxiv.org/pdf/1706.03762).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^4]: - [Denoising Diffusion Probabilistic Models. Jonathan Ho, Ajay Jain, Pieter Abbee](https://arxiv.org/pdf/2006.11239).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, Franc ̧ois Fleuret](https://arxiv.org/pdf/2006.16236).
