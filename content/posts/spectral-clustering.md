---
author: "X. Wang"
title: "Spectral Clustering"
date: "2023-10-03"
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

## Background of Clustering

<cite>[^1]</cite>In the field of clustering algorithms, we often categorize methods based on their primary characteristics:

$$
\begin{cases}
    \text{compactness: K-means, GMM} \\\
    \text{connectivity: Spectral clustering}
\end{cases}
$$

Compactness-based methods like K-means and GMMs tend to form clusters that are spatially close to each other, while connectivity-based methods like Spectral Clustering group together points that are connected or have a high degree of similarity.

## Model Definition

Spectral clustering is a clustering method based on graph. Given a dataset:

$$
X = \lbrace x_1,\cdots,x_N \rbrace, x_i\in\mathbb{R}^p
$$

Define the corresponding undirected weighted graph:

$$
\begin{align*}
G &= (V,E) \\\
% V &:\text{vertices $\lbrace 1,\cdots,N \rbrace$ correspond to $\lbrace x_1,\cdots,x_N \rbrace$} \\\
V &= \lbrace \text{vertices}|i \rbrace, i=1\cdots N \\\
E &= \lbrace \text{edges}|(i,j) \rbrace, 1 \leq i,j \leq N \\\
W &: \text{similarity matrix, where } w_{ij} = \begin{cases}
    K(x_i,x_j)=\exp(-\frac{\lVert x_i-x_j \rVert^2_2}{2\sigma^2}) & (i,j)\in E \\\
    0 & (i,j)\notin E
\end{cases}
\end{align*}
$$

{{< math.inline >}}
<p>
Define connectivity between two clusters \( A,B \) as \( W(A,B) \):
</p>
{{</ math.inline >}}

$$
A,B \in V, A \cap B = \empty \\\
W(A,B) = \sum_{\begin{subarray}{c}
    i\in A \\\
    j\in B
\end{subarray}}w_{ij}
$$

{{< math.inline >}}
<p>
Suppose we are splitting \( X \) into \( K \) clusters, which corresponds to a graph cut:
</p>
{{</ math.inline >}}

$$
\text{Let }V = \cup_{j=1}^K A_k, \space A_i \cap A_j = \empty, \space i,j\in1\cdots K \\\
\\\
\begin{align*}
\text{cut}(V) &= \text{cut}(A_1,\cdots,A_K) \\\
&= \sum_{k=1}^KW(A_k,\neg A_{k}) \\\
&= \sum_{k=1}^KW(A_k,V)-W(A_k,A_k)
\end{align*}
$$

{{< math.inline >}}
<p>
\( \text{cut}(V) \) is a measurement of similarity between clusters, which can be normalized by out-degree:
</p>
{{</ math.inline >}}

$$
\text{Let } \text{degree}(A_k) \text{ be the summation of out-degree from $A_k$} \\\
\text{degree}(A_k) = \sum_{i\in A_k}\sum_{j=1}^Nw_{ij} \\\
\text{norm-}W(A_k,\neg A_k) = \frac{W(A_k,\neg A_{k})}{\text{degree}(A_k)}
$$

The spectral clustering problem can be deduced to minimize objective function:

$$
\min_{A_1\cdots A_K}\text{norm-cut}(V) = \min_{A_1\cdots A_K}\sum_{k=1}^K\frac{W(A_k,\neg A_{k})}{\sum_{i\in A_k}\sum_{j=1}^Nw_{ij}}
$$

## Objective Function in Vector Form

{{< math.inline >}}
<p>
Define a one-hot encode indicator matrix \( Y \), which represents \( A_1\cdots A_K \):
</p>
{{</ math.inline >}}

$$
Y_{N\times K} = \begin{bmatrix}
    y_1 & \cdots & y_N
\end{bmatrix}^T, y_i = \begin{bmatrix}
    y_{i1} \\\
    \vdots \\\
    y_{iK}
\end{bmatrix} \\\
\sum_{j=1}^Ky_{ij}=1, \space y_{ij}=\lbrace 0,1 \rbrace \\\
y_{ij}=1 \hArr \text{i-th sample partitioned to j-th cluster}
$$

{{< math.inline >}}
<p>
Thus the objective function can be written as:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{Y} = \argmin_{Y}\text{norm-cut}(V)
\end{align*}
$$

{{< math.inline >}}
<p>
On the other hand, \( \sum_{k=1}^K\frac{W(A_k,\neg A_{k})}{\sum_{i\in A_k}\sum_{j=1}^Nw_{ij}} \) is representable by matrix trace:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\sum_{k=1}^K\frac{W(A_k,\neg A_{k})}{\sum_{i\in A_k}\sum_{j=1}^Nw_{ij}} &= Tr\left( \begin{bmatrix}
    \frac{W(A_1,\neg A_{1})}{\sum_{i\in A_1}\sum_{j=1}^Nw_{ij}} & & \\\
    & \ddots & \\\
    & & \frac{W(A_K,\neg A_{K})}{\sum_{i\in A_K}\sum_{j=1}^Nw_{ij}}
\end{bmatrix} \right) \\\
&= Tr\left( \begin{bmatrix}
    W(A_1,\neg A_{1}) & & \\\
    & \ddots & \\\
    & & W(A_K,\neg A_{K})
\end{bmatrix} \begin{bmatrix}
    \sum_{i\in A_1}\sum_{j=1}^Nw_{ij} & & \\\
    & \ddots & \\\
    & & \sum_{i\in A_K}\sum_{j=1}^Nw_{ij}
\end{bmatrix}^{-1} \right) \\\
&\because Y^TIY = \begin{bmatrix}
    y_1 & \cdots & y_N
\end{bmatrix} \begin{bmatrix}
    y^T_1 \\\
    \vdots \\\
    y^T_N
\end{bmatrix} = \sum_{i=1}^Ny_i1y^T_i = \sum_{i=1}^N\text{diag}(y_i) = \begin{bmatrix}
    |A_1| & & \\\
    & \ddots & \\\
    & & |A_K|
\end{bmatrix} = \begin{bmatrix}
    \sum_{i\in A_1}1 & & \\\
    & \ddots & \\\
    & & \sum_{i\in A_K}1
\end{bmatrix} \\\
&\text{Let } d_i =\sum_{j=1}^Nw_{ij} \text{, we have } \sum_{i=1}^Ny_id_iy^T_i = \sum_{i=1}^N\text{diag}(d_iy_i) = \begin{bmatrix}
    \sum_{i\in A_1}d_{i} & & \\\
    & \ddots & \\\
    & & \sum_{i\in A_K}d_{i}
\end{bmatrix} = \begin{bmatrix}
    \sum_{i\in A_1}\sum_{j=1}^Nw_{ij} & & \\\
    & \ddots & \\\
    & & \sum_{i\in A_K}\sum_{j=1}^Nw_{ij}
\end{bmatrix} \\\
&\text{Let }D = \begin{bmatrix}
    d_1 & & \\\
    & \ddots & \\\
    & & d_N
\end{bmatrix} =\text{diag}(\begin{bmatrix}
    W_{1,:} \mathbf{1}\_{N} \\\
    \vdots \\\
    W_{N,:} \mathbf{1}\_{N}
\end{bmatrix}) = \text{diag}(W\mathbf{1}\_{N}) \text{, so we have } \sum_{i=1}^Ny_id_iy^T_i = Y^TDY \\\
&= Tr\left( \begin{bmatrix}
    W(A_1,\neg A_{1}) & & \\\
    & \ddots & \\\
    & & W(A_K,\neg A_{K})
\end{bmatrix} (Y^TDY)^{-1} \right) \\\
&\because W(A_k,\neg A_k) = W(A_k,V)-W(A_k,A_k) = \sum_{i\in A_k}d_i-\sum_{i\in A_k}\sum_{j\in A_k}w_{ij}  \\\
&\text{where } \begin{bmatrix}
    \sum_{i\in A_1}d_{i} & & \\\
    & \ddots & \\\
    & & \sum_{i\in A_K}d_{i}
\end{bmatrix}=Y^TDY \\\
&= Tr\left( \left(Y^TDY-\begin{bmatrix}
    \sum_{i\in A_1}\sum_{j\in A_1}w_{ij} & & \\\
    & \ddots & \\\
    & & & \sum_{i\in A_K}\sum_{j\in A_K}w_{ij}
\end{bmatrix}\right)(Y^TDY)^{-1} \right) \\\
&\because Y^TWY \text{ shares the same diagnal elements with the above diagnal matrix, } \\\
&\text{where } Y^TWY=\begin{bmatrix}
    y_1 \cdots y_N
\end{bmatrix}\begin{bmatrix}
    w_{:,1} \cdots w_{:,N}
\end{bmatrix}\begin{bmatrix}
    y^T_1 \\\
    \vdots \\\
    y^T_N
\end{bmatrix}=\begin{bmatrix}
    \sum_{i=1}^Ny_iw_{i1} \cdots \sum_{i=1}^Ny_iw_{iN}
\end{bmatrix}\begin{bmatrix}
    y^T_1 \\\
    \vdots \\\
    y^T_N
\end{bmatrix}=\sum_{i=1}^N\sum_{j=1}^Ny_iy^T_jw_{ij} \\\
&\text{and } \text{diag}(\sum_{i=1}^N\sum_{j=1}^Ny_iy^T_jw_{ij})=\begin{bmatrix}
    \sum_{i\in A_1}\sum_{j\in A_1}w_{ij} & & \\\
    & \ddots & \\\
    & & & \sum_{i\in A_K}\sum_{j\in A_K}w_{ij}
\end{bmatrix} \\\
&= Tr\left( (Y^TDY-Y^TWY)(Y^TDY)^{-1} \right) \\\
&= Tr\left( Y^T(D-W)Y(Y^TDY)^{-1} \right)
\end{align*}
$$

{{< math.inline >}}
<p>
The objective function becomes
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{Y} &= \argmin_{Y} Tr\left( Y^TLY(Y^TDY)^{-1} \right) \\\
L &: \text{Laplace matrix, where } L=D-W
\end{align*}
$$

## Solve objective function

### Rayleigh quotient

The Rayleigh quotient is defined as:

$$
R(A,y) = \frac{y^TAy}{y^Ty}, \text{where } \\\
\text{$y$ is non-zero vector} \\\
\text{$A$ is Hermitian matrix}
$$

And the generalized Rayleigh quotient is defined as:

$$
R(A,B,y) = \frac{y^TAy}{y^TBy} \\\
$$

We can prove that minimizing Rayleigh quotient is euiqvalent to finding the extreme eigenvalue in generalized eigenvalue problem:

$$

$$

### Eigen decomposition

We can prove the objective function is a form of generalized Rayleigh quotient:

$$
\text{Let }Y = \begin{bmatrix}
    \gamma_1 \cdots \gamma_K
\end{bmatrix}, \gamma_k = \begin{bmatrix}
    y_{1k} \\\
    \vdots \\\
    y_{Nk}
\end{bmatrix} \\\
\text{$\gamma_k$ is column vector, $y_i$ is row vector} \\\
\\\
\begin{align*}
&\because Y^TLY \text{ is a $K\times K$ matrix, where $\gamma^T_iL\gamma_j$ is the $(i,j)$-th element}, \text{the same applies to $Y^TDY$} \\\
&\because \text{$D,W,L$ are clearly positive semi-definite} \\\
&\therefore \gamma^T_iL\gamma_j(\gamma^T_iD\gamma_j)^{-1} = \frac{\gamma^T_iL\gamma_j}{\gamma^T_iD\gamma_j} \rArr Y^TLY(Y^TDY)^{-1}=\begin{bmatrix}
    \frac{\gamma^T_1L\gamma_1}{\gamma^T_1D\gamma_1} & \cdots & \frac{\gamma^T_1L\gamma_K}{\gamma^T_1D\gamma_K} \\\
    \vdots & \ddots & \vdots \\\
    \frac{\gamma^T_KL\gamma_1}{\gamma^T_KL\gamma_1} & \cdots & \frac{\gamma^T_KL\gamma_K}{\gamma^T_KD\gamma_K}
\end{bmatrix} \rArr Tr(Y^TLY(Y^TDY)^{-1}) = \sum_{k}^K\frac{\gamma^T_kL\gamma_k}{\gamma^T_kD\gamma_k}
\end{align*}
$$

$$
\because \gamma^T_k\gamma_k = |A_k|
$$

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=123).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [NIPS 2016 Tutorial: Generative Adversarial Networks. Ian Goodfellow](https://arxiv.org/pdf/1701.00160).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
