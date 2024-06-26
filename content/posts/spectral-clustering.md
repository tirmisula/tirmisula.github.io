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
L &: \text{Laplacian matrix, where } L=D-W
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
\text{$y$ is non-zero vector} \\\
\text{$A$ is Hermitian matrix} \\\
\text{$B$ is positive definite}
$$

<cite>[^2]</cite>We can prove that minimizing Rayleigh quotient is euiqvalent to finding the smallest eigenvalue in corresponding generalized eigenvalue problem:

$$
\begin{align*}
&\because \text{$B$ is semi-positive definite} \\\
&\text{there exist orthogonal matrix $Q$ and diagonal matrix $\Lambda$, so that $B=Q\Lambda Q^T$} \\\
&\text{Let $\Lambda^{\frac{1}{2}}=$diag$(\lambda^{\frac{1}{2}}\_1,\cdots,\lambda^{\frac{1}{2}}\_n)$, we have $\Lambda=\Lambda^{\frac{1}{2}}\Lambda^{\frac{1}{2}}$} \\\
&\therefore B=Q\Lambda Q^T=Q\Lambda^{\frac{1}{2}}Q^TQ\Lambda^{\frac{1}{2}}Q^T, \text{ where $B^{\frac{1}{2}}=Q\Lambda^{\frac{1}{2}}Q^T$ is the \textbf{square root} of $B$} \\\
&\dArr \\\
&\text{Since $B$ has square root, we have: } y^TBy = y^TB^{\frac{1}{2}}B^{\frac{1}{2}}y \\\
&\text{Let $B^{\frac{1}{2}}y=z$, we have: $y^TBy=z^Tz$, generalized Rayleigh quotient is converted back to Rayleigh quotient:} \\\
&\frac{y^TAy}{y^TBy} = \frac{y^TAy}{z^Tz} = \frac{z^TB^{-\frac{1}{2}}AB^{-\frac{1}{2}}z}{z^Tz}=R(B^{-\frac{1}{2}}AB^{-\frac{1}{2}},z) \text{ , we can construct Lagrange multiplier to solve it} \\\
&\dArr \\\
&\because \text{while $\alpha$ is a scalar, }\frac{(\alpha y)^TA(\alpha y)}{(\alpha y)^TB(\alpha y)}=\frac{\alpha^2(y^TAy)}{\alpha^2(y^TBy)}=\frac{y^TAy}{y^TBy} \\\
&\therefore\text{Rayleigh quotient is scaling invariant, we restrict $||z||=1$} \\\
&\text{Let $L(z,\lambda)=z^TB^{-\frac{1}{2}}AB^{-\frac{1}{2}}z - \lambda(||z||^2-1)$} \\\
&\frac{\partial L}{\partial z} = 2B^{-\frac{1}{2}}AB^{-\frac{1}{2}}z - 2\lambda z = 0 \space\rarr\space B^{-\frac{1}{2}}AB^{-\frac{1}{2}}z=\lambda z \space\rarr\space Ay=\lambda B^{\frac{1}{2}}B^{\frac{1}{2}}y \space\rarr\space Ay=\lambda By \\\
&\quad\quad\rarr B^{-\frac{1}{2}}AB^{-\frac{1}{2}}z=\lambda z \\\
&\quad\quad\because z=B^{\frac{1}{2}}y \\\
&\quad\quad\rarr B^{-\frac{1}{2}}AB^{-\frac{1}{2}}B^{\frac{1}{2}}y=\lambda B^{\frac{1}{2}}y \\\
&\quad\quad\rarr B^{-\frac{1}{2}}Ay=\lambda B^{\frac{1}{2}}y \\\
&\quad\quad\rarr Ay=\lambda By \\\
&\frac{\partial L}{\partial \lambda} = ||z||^2 - 1 = 0 \\\
&\dArr \\\
&\text{This implies $(y,\lambda)$ is generalized eigenpair of $(A,B)$, and $||B^{\frac{1}{2}}y||=1$} \\\
&Ay=\lambda By \hArr y^TAy=\lambda y^TBy \hArr \lambda = \frac{y^TAy}{y^TBy} \\\
&\text{This implies $(z,\lambda)$ is eigenpair of $B^{-\frac{1}{2}}AB^{-\frac{1}{2}}$, and $||z||=1$} \\\
&B^{-\frac{1}{2}}AB^{-\frac{1}{2}}z=\lambda z \hArr z^TB^{-\frac{1}{2}}AB^{-\frac{1}{2}}z=\lambda z^Tz \hArr \lambda = \frac{z^TB^{-\frac{1}{2}}AB^{-\frac{1}{2}}z}{z^Tz} \\\
&\text{Finding $\lambda_{\min}$} \hArr \text{minimizing } \frac{y^TAy}{y^TBy} \hArr \text{minimizing } \frac{z^TB^{-\frac{1}{2}}AB^{-\frac{1}{2}}z}{z^Tz}
\end{align*}
$$

The conclusion is:

$$
\begin{align*}
&1.\space R(A,B,y) \hArr R(B^{-\frac{1}{2}}AB^{-\frac{1}{2}},z) \text{ , where $z=B^{\frac{1}{2}}y$} \\\
% \min R(A,B,y) \hArr \min_{||z||=1}\frac{z^TB^{-\frac{1}{2}}AB^{-\frac{1}{2}}z}{z^Tz} \hArr Ay=\lambda_{\min}By \hArr B^{-\frac{1}{2}}AB^{-\frac{1}{2}}z=\lambda_{\min}z \\\
&2.\space\min R(A,B,y) \hArr \min\lambda \quad\text{ s.t. } Ay=\lambda By \\\
&3.\space\min R(B^{-\frac{1}{2}}AB^{-\frac{1}{2}},z) \hArr \min\lambda \quad\text{s.t. }  B^{-\frac{1}{2}}AB^{-\frac{1}{2}}z=\lambda z \\\
&4.\space \text{$B^{-1}A$ and $B^{-\frac{1}{2}}AB^{-\frac{1}{2}}$ share the same eigenvalues}
\end{align*}
$$

### Laplacian matrix

The objective function is a form of generalized Rayleigh quotient:

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
\end{bmatrix} \\\
&\therefore Tr(Y^TLY(Y^TDY)^{-1}) = \sum_{k=1}^K\frac{\gamma^T_kL\gamma_k}{\gamma^T_kD\gamma_k}, \text{ which gives us: }
\end{align*}
$$

$$
\begin{align*}
Tr(Y^TLY(Y^TDY)^{-1}) &= \sum_{k=1}^KR(L,D,\gamma_k) \\\
&= \sum_{k=1}^KR(D^{-\frac{1}{2}}LD^{-\frac{1}{2}},z_k), \text{where $z_k=D^{\frac{1}{2}}\gamma_k$}
\end{align*}
$$

<cite>[^3]</cite>

{{< math.inline >}}
<p>
\( L \) is defined as unnormalized Laplacian. Meanwhile, Laplacian matrix corresponds to \( R(L,D,\gamma_k) \) is defined as random walk normalized Laplacian \( L_{rw} \):
</p>
{{</ math.inline >}}

$$
L_{rw} = D^{-1}L
$$

{{< math.inline >}}
<p>
Laplacian matrix corresponds to \( R(D^{-\frac{1}{2}}LD^{-\frac{1}{2}},z_k) \) is defined as symmetric normalized Laplacian \( L_{rw} \):
</p>
{{</ math.inline >}}

$$
L_{sym} = D^{-\frac{1}{2}}LD^{-\frac{1}{2}}
$$

{{< math.inline >}}
<p>
From previous section, \( L_{rw} \) and \( L_{sym} \) share the same eigenvalues clearly:
</p>
{{</ math.inline >}}

$$
L_{rw}\gamma_k = \lambda\gamma_k \\\
L_{sym}z_k = \lambda z_k
$$

And according to the conclusion of minimizing generalized Rayleigh quotient, the objective function turns to minimizing summation of K eigenvalues, which is equivalent to finding the first K smallest eigenpairs: 

$$
\begin{align*}
\min_{Y} Tr(Y^TLY(Y^TDY)^{-1}) &= \min\sum_{k=1}^K\lambda_k \quad\text{s.t. } L_{rw}\gamma_k = \lambda_k\gamma_k, k=1\cdots K \\\
&= \min\sum_{k=1}^K\lambda_k \quad\text{s.t. }L_{sym}z_k = \lambda_kz_k, k=1\cdots K \\\
&z_k=D^{\frac{1}{2}}\gamma_k ,\quad \lambda_{\min}=\lambda_1\leq\cdots\leq\lambda_K
\end{align*}
$$

{{< math.inline >}}
<p>
In previous definition, \( \gamma_k \) is discrete indicator while \( z_k \) is relaxed to \( \mathbb{R} \), there are constraints on \( z_k \) and \( \gamma_k \):
</p>
{{</ math.inline >}}

<!-- $$
D^{-1}L\gamma_k = \lambda_k\gamma_k \\\
D^{\frac{1}{2}}D^{-1}L\gamma_k = \lambda_kD^{\frac{1}{2}}\gamma_k \\\

D^{-\frac{1}{2}}LD^{-\frac{1}{2}}D^{\frac{1}{2}}\gamma_k = \lambda_kD^{-\frac{1}{2}}DD^{-\frac{1}{2}}D^{\frac{1}{2}}\gamma_k \\\
D^{-\frac{1}{2}}LD^{-\frac{1}{2}}z_k=\lambda_kz_k, \text{where $z_k=D^{\frac{1}{2}}\gamma_k$}
$$ -->


### Constraints of eigenvectors

<cite>[^4]</cite>We can easily find the first eigenvector corresponds to the smallest eigenvalue of 0 in both symmetric Laplacian and random walk Laplacian:

$$
\begin{align*}
\text{Let } z_1 &= D^{\frac{1}{2}}1_N, \gamma_1=1_N \\\
&\dArr \\\
L_{sym}z_1 &= D^{-\frac{1}{2}}LD^{-\frac{1}{2}}z_1 \\\
&= D^{-\frac{1}{2}}(D-W)1_N \\\
&= D^{-\frac{1}{2}}D1_N - D^{-\frac{1}{2}}W1_N \\\
&\because D1_N=\begin{bmatrix}
    d_1 \cdots d_N
\end{bmatrix}^T=\begin{bmatrix}
    W_{1,:} 1_{N} \cdots W_{N,:} 1_{N}
\end{bmatrix}^T \\\
&\because W1_N=\begin{bmatrix}
    W_{1,:} 1_{N} \cdots W_{N,:} 1_{N}
\end{bmatrix}^T \\\
&= 0 \\\
L_{rw}\gamma_1 &= D^{-1}L1_N \\\
&= D^{-1}(D-W)1_N \\\
&= 0
\end{align*} \\\
% \text{$z_1$ is the eigenvector of smallest eigenvalue, $\gamma_1=1_N$}
$$

{{< math.inline >}}
<p>
Since \( D^{-\frac{1}{2}}LD^{-\frac{1}{2}} \) is positive semidefinite. All eigenvectors are perpendicular to each other, so we have \( z \perp z_1 \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
z^Tz_1 = 0 &\hArr (D^{\frac{1}{2}}\gamma_2)^TD^{\frac{1}{2}}1_N=0 \\\
&\hArr \gamma^T_2D1_N=0 \\\
&\cdots \\\
&\hArr \gamma^T_{N}D1_N=0
\end{align*}
$$

{{< math.inline >}}
<p>
On the other hand, \( Y^TY=\text{diag}(|A_1|,\cdots,|A_K|) \) is given in previous definition which can be detailed:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\gamma^T_k\gamma_k &= |A_k| \\\
\gamma^T_kD\gamma_k &= \sum_{i=1}^Nd_iy_{ik}y_{ik} = \sum_{i\in A_k}\sum_{j=1}^Nw_{ij} = \text{degree}(A_k) \\\
Y^TDY &= \begin{bmatrix}
    \text{degree}(A_1) & & \\\
    & \ddots & \\\
    & & \text{degree}(A_K)
\end{bmatrix} \\\
\\\
\text{Let } \gamma_k &= D^{-\frac{1}{2}}z_k \\\
z^T_kD^{-\frac{1}{2}}D^{-\frac{1}{2}}z_k &= z^T_kD^{-1}z_k = |A_k| \\\
z^T_kD^{-\frac{1}{2}}DD^{-\frac{1}{2}}z_k &= z^T_kz_k = \text{degree}(A_k) \\\
Z^TZ &= \begin{bmatrix}
    \text{degree}(A_1) & & \\\
    & \ddots & \\\
    & & \text{degree}(A_K)
\end{bmatrix}
\end{align*}
$$

{{< math.inline >}}
<p>
In conclusion, constraints of \( z \) and \{ \gamma \} are given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
z_1 &= D^{\frac{1}{2}}1_N \\\
z^TD^{\frac{1}{2}}1_N &= 0 \\\
z^T_kz_k &= \text{degree}(A_k) \\\
Z^TZ &= \text{diag}(\text{degree}(A_1),\cdots,\text{degree}(A_K)) \\\
\gamma_1 &= 1_N \\\
\gamma^TD1_N &= 0 \\\
\gamma^T_kD\gamma_k &= \text{degree}(A_k) \\\
Y^TDY &= \text{diag}(\text{degree}(A_1),\cdots,\text{degree}(A_K))
\end{align*}
$$

### Upper bound of eigenvalues
<!-- #### Gershgorin Circle Theorem

{{< math.inline >}}
<p>
Gershgorin Circle Theorem: For a matrix \( A \), each eigenvalue of a matrix lies within at least one Gershgorin disc. The Gershgorin disc centered at \( a_{ii} \) with radius \( R_i \) is given by:
</p>
{{</ math.inline >}}

$$
R_i = \sum_{j\neq i}|a_{ij}| \\\
|\lambda-a_{ii}| \leq R_i
$$

{{< math.inline >}}
<p>
Applying Gershgorin Circle Theorem to \( A_{norm} \), the Gershgorin disc centered at \( (A_{norm})_{ii} \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
R_i &= \sum_{j\neq i}|(A_{norm})_{ij}| \\\
&= \sum_{j\neq i,j=1}^N\frac{W_{ij}}{\sqrt{d_id_j}} \\\
&\because W_{ij}\in\text{Gaussian kernel} \\\
&\therefore W_{ii}=w_{ii}=1 \\\
&= \sum_{j\neq i,j=1}^N\frac{w_{ij}}{\sqrt{\sum_{j=1}^Nw_{ij}}\sqrt{\sum_{k=1}^Nw_{jk}}} \\\
\end{align*}
$$ -->

{{< math.inline >}}
<p>
Let \( A_{norm} \) be the normalized adjacency matrix:
</p>
{{</ math.inline >}}

$$
A_{norm} = D^{-\frac{1}{2}}WD^{-\frac{1}{2}}
$$

<cite>[^5]</cite>

{{< math.inline >}}
<p>
Because \( D^{-\frac{1}{2}}LD^{-\frac{1}{2}} \) is positive semidefinite, we have:
</p>
{{</ math.inline >}}

$$
\begin{align*}
z^TD^{-\frac{1}{2}}LD^{-\frac{1}{2}}z \geq 0 &\rArr z^T(I-A_{norm})z \geq 0 \\\
&\rArr 1 \geq \frac{z^TA_{norm}z}{z^Tz}
\end{align*} \\\
\lambda \text{ correspond to $A_{norm}$ has, $\lambda\leq 1$} 
$$

{{< math.inline >}}
<p>
Next, we can prove that \( I+A_{norm} \) is positive semidefinite:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(I+A_{norm})v &= v + A_{norm}v \\\
&= (1 + \lambda)v
\end{align*} \\\
\text{Since $0\leq\lambda\leq 1$, $1\leq 1+\lambda\leq 2$} \\\
\text{All eigenvalues correspond to $I+A_{norm}$ are positve, $I+A_{norm}$ is PSD}
$$

{{< math.inline >}}
<p>
Since \( I+A_{norm} \) is positive semidefinite, we have:
</p>
{{</ math.inline >}}

$$
\begin{align*}
z^T(I+A_{norm})z \geq 0 &\rArr z^Tz+z^TA_{norm}z\geq0 \\\
&\rArr -1 \leq \frac{z^TA_{norm}z}{z^Tz}
\end{align*} \\\
\lambda \text{ correspond to $A_{norm}$ has, $-1\leq\lambda$} 
$$

Combine together, we have:

$$
-1 \leq \frac{z^TA_{norm}z}{z^Tz} \leq 1
$$

{{< math.inline >}}
<p>
Recall that \( R(D^{-\frac{1}{2}}LD^{-\frac{1}{2}},z_k) \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
R(D^{-\frac{1}{2}}LD^{-\frac{1}{2}},z) &= \frac{z^TD^{-\frac{1}{2}}LD^{-\frac{1}{2}}z}{z^Tz} \\\
&= \frac{z^T(D^{-\frac{1}{2}}DD^{-\frac{1}{2}}-D^{-\frac{1}{2}}WD^{-\frac{1}{2}})z}{z^Tz} \\\
&= \frac{z^T(I-A_{norm})z}{z^Tz} \\\
% &= \frac{z^Tz-z^TD^{-\frac{1}{2}}WD^{-\frac{1}{2}}z}{z^Tz} \\\
&= 1 - \frac{z^TA_{norm}z}{z^Tz}
\end{align*} \\\
\dArr \\\
0 \leq R(D^{-\frac{1}{2}}LD^{-\frac{1}{2}},z) \leq 2
$$

{{< math.inline >}}
<p>
Thus eigenvalues of \( L_{sym} \) and \( L_{rw} \) are bounded by 2.
</p>
{{</ math.inline >}}

<!-- 1. x^TA_{norm}x=y^TWy \geq 0. Since W is the adjacency matrix of a graph with non-negative weights, A_{norm} is positive semidefinite.

2. eigenvalues correspond to A_{norm} lies in [-1,1], A_{norm} is generally not positive semidefinite. -->

<!-- {{< math.inline >}}
<p>
Each element of \( A_{norm} \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(A_{norm})_{ij} &= \sum_k (D^{-\frac{1}{2}})_{ik}\sum_lW_{kl}(D^{-\frac{1}{2}})_{lj} \\\
&\because D^{-\frac{1}{2}}=\text{diag}(\frac{1}{\sqrt{d_1}},\cdots,\frac{1}{\sqrt{d_N}}) \\\
&\therefore (D^{-\frac{1}{2}})_{ii}=\frac{1}{\sqrt{d_i}}, (D^{-\frac{1}{2}})_{ij}=0 \\\
&= \sum_k (D^{-\frac{1}{2}})_{ik}W_{kj}(D^{-\frac{1}{2}})_{jj} \\\
&= (D^{-\frac{1}{2}})_{ii}W_{ij}(D^{-\frac{1}{2}})_{jj} \\\
&= \frac{W_{ij}}{\sqrt{d_id_j}} \\\
\end{align*}
$$ -->

### Formalize optimization object

{{< math.inline >}}
<p>
The minimizing Ncut problem becomes:
</p>
{{</ math.inline >}}

$$
\min_{Y} Tr(Y^TLY(Y^TDY)^{-1}) \hArr \begin{array}{l}
    \min_{\gamma_1\cdots\gamma_K}\sum_{k=1}^K\lambda_k \\\
    \text{s.t. $L\gamma_k=\lambda_kD\gamma_k$} \\\
    \text{s.t. $\gamma^T_1=1_N$} \\\
    \text{s.t. $\gamma^T_kD1_N=0, k=2\cdots K$} \\\
    \text{s.t. $\gamma^T_kD\gamma_k = \text{degree}(A_k)$} \\\
    0=\lambda_1\leq\cdots\leq\lambda_K\leq 2
\end{array}
$$

In matrix form:

$$
\quad\quad\quad\quad\min_{Y} Tr(Y^TLY(Y^TDY)^{-1}) \hArr \begin{array}{l}
    \min_{\gamma_1\cdots\gamma_K}Tr(\Lambda) \\\
    \text{s.t. $LY=\Lambda DY$} \\\
    \text{s.t. $\gamma^T_1=1_N$} \\\
    \text{s.t. $Y_{:,2:K}^TD1_N=0$} \\\
    \text{s.t. $Y^TDY = \text{diag}(\text{degree}(A_1),\cdots,\text{degree}(A_K))$} \\\
    0=\lambda_1\leq\cdots\leq\lambda_K\leq 2
\end{array} \\\
\\\
\text{where } \hat{Y} = \begin{bmatrix}
    \hat{\gamma_1} \cdots \hat{\gamma_K} 
\end{bmatrix}, \space\Lambda = \text{diag}(\lambda_1,\cdots,\lambda_K)
$$

<!-- $$
\begin{align*}
% &\text{For each Rayleigh quotient item, if $\gamma_k$ is relxed to $\mathbb{R}$: } \\\
&\min R(L,D,\gamma_k) \hArr L\gamma_k=\lambda_kD\gamma_k \hArr D^{-\frac{1}{2}}LD^{-\frac{1}{2}}z_k=\lambda_kz_k, \text{where $z_k=D^{\frac{1}{2}}\gamma_k$} \\\
&\text{Using Lagrange multipliers: } \\\
&\text{Let } z_k = D^{\frac{1}{2}}\gamma_k\text{, since we want $||z_k||=1$, redefine $\gamma_k=\frac{\gamma_k}{\sqrt{|A_k|}}$} \\\
&\mathcal{L}(z_k,\lambda_k)=z^T_kD^{-\frac{1}{2}}LD^{-\frac{1}{2}}z_k-\lambda_k(||z_k||^2-1) \\\
&\mathcal{L}(Z,\Lambda) = \sum_{k=1}^K\left( z^T_kD^{-\frac{1}{2}}LD^{-\frac{1}{2}}z_k-\lambda_k(||z_k||^2-1) \right) \\\
&\frac{\partial \mathcal{L}}{\partial z_k} = 0 \hArr L\gamma_k=\lambda_kD\gamma_k \hArr \min\sum_{k=1}^K\lambda_k=\min\sum_{k=1}^K\frac{\gamma^T_kL\gamma_k}{\gamma^T_kD\gamma_k}
\end{align*}
$$ -->

<!-- Thus minimizing objective function is euiqvalent to finding first K smallest eigenpairs in corresponding generalized eigenvalue problem:

$$
\begin{align*}
\argmin_{Y} Tr\left( Y^TLY(Y^TDY)^{-1} \right) &\hArr \argmin_{\hat{\gamma_1} \cdots \hat{\gamma_K}} \sum_{k=1}^K\lambda_k \quad\text{ s.t. } L\hat{Y}=\Lambda D\hat{Y}\quad\text{ s.t. }||\hat{\gamma_k}||^2=1 \\\
\text{where } \\\
\hat{Y} &= \begin{bmatrix}
    \hat{\gamma_1} \cdots \hat{\gamma_K} 
\end{bmatrix} \\\
\Lambda &= \text{diag}(\lambda_1,\cdots,\lambda_K), \lambda_{\min}=\lambda_1\leq\cdots\leq\lambda_K<\lambda_N=\lambda_{\max}
\end{align*}
$$ -->

<!-- \argmin_{\begin{subarray}{c}
\gamma_1\cdots\gamma_K \\\
\lambda_1\leq\cdots\leq\lambda_K
\end{subarray}}L\gamma=\lambda D\gamma -->
<!-- &\hArr \lbrace \text{eigen pairs:}\left(\begin{array}{c}
\hat{\gamma_1} \cdots \hat{\gamma_K} \\\
\lambda_{\min}=\lambda_1\leq\cdots\leq\lambda_K<\lambda_N=\lambda_{\max}
\end{array}\right) | L\gamma=\lambda D\gamma \rbrace \\\ -->

<cite>[^3]</cite>To convert the real valued solution to discrete partition, the common way is to use K-means algorithm:

$$
\begin{align*}
&\text{For each row $y_1\cdots y_N$ in $\hat{Y}$: } \\\
&\quad\text{Assign the points into clusters $C_1\cdots C_K$: } \\\
&\quad k\text{-means}(y_i) \mapsto C_k
\end{align*}
$$

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=123).
[^4]: - [Normalized Cuts and Image Segmentation Jianbo Shi and Jitendra Malik, Member, IEEE](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf).
[^5]: - [Spectral Graph Theory. David P. Williamson, Sam Gutekunst](https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf).
[^2]: - [Math 253: Mathematical Methods for Data Visualization Lecture 4: Rayleigh Quotients. Guangliang Chen](https://arxiv.org/pdf/1701.00160).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A Tutorial on Spectral Clustering. Ulrike von Luxburg](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf).
