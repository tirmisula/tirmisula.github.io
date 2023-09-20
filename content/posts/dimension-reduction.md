---
author: "X. Wang"
title: "Dimensionality Reduction - Principal Component Analysis"
date: "2023-09-05"
description: "A brief introduction."
tags: ["machine learning"]
categories: ["themes", "syntax"]
# series: ["Themes Guide"]
aliases: ["migrate-from-jekyl"]
math: true
ShowBreadCrumbs: false
ShowToc: true
TocOpen: true
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
{{</ math.inline >}}

<style>
    /* Set the font size of all math elements to 16px */
    .katex {
        font-size: 16px !important;
    }
</style>

## Background
### Curse of Dimensionality

Given a p-dimensional unit hypercube and it's incribed hypersphere, we can calculate volume:

$$
V_{\text{hypercube}} = 1 \\\
V_{\text{hypersphere}} = K\cdot0.5^p
$$

If dimension p is large enough, we can get:

$$
\lim_{p\rarr\infty} \frac{V_{\text{hypersphere}}}{V_{\text{hypercube}}} = 0
$$

That means if we randomly sample in this hypercube, most of the sampled data locates at corner rather than in the hypersphere which leads to sparse and uneven data distribution.

### Dimension reduction method

$$
\begin{cases}
\text{ Direct reduction} \rarr \text{feature selection like Lasso} \\\
\text{Linear reduction} \rarr \text{PCA or Multidimensional scaling} \\\
\text{Nonlinear reduction} \rarr \text{Flow model}\begin{cases}
\text{Isomap} \\\
\text{LLE}
\end{cases}
\end{cases}
$$

## Prerequisite
### Express mean and variance in matrix form

Given a dataset:
$$
\begin{align*}
X &= \begin{bmatrix}
    x_{1} & x_2 & \dots & x_N
\end{bmatrix}^T
&= \begin{bmatrix}
    {x_1}^T \\\
    {x_2}^T \\\
    \dots \\\
    {x_N}^T
\end{bmatrix}
&= \begin{bmatrix}
    x_{11} & x_{12} & \dots & x_{1p}\\\
    x_{21} & x_{22} & \dots & x_{2p}\\\
    \dots \\\
    x_{N1} & x_{N2} & \dots & x_{Np}
\end{bmatrix}_{N \times p}
\end{align*}
$$

$$
x_i \in \mathbb{R}^p, i=1,2,\cdots ,N
$$

We can write down samples mean and covariance in matrix form:

$$
\begin{align*}
\mu_x &= \frac{1}{N} \sum_{i=1}^N x_i \\\
&= \frac{1}{N}\begin{bmatrix}
    x_1 & x_2 & \dots & x_N
\end{bmatrix}
\begin{bmatrix}
1 \\\
1 \\\
\vdots \\\
1
\end{bmatrix} \\\
&=\frac{1}{N}X^T 1_N
\end{align*}
$$

$$
\begin{align*}
\Sigma_x &= \frac{1}{N}\sum_{i=1}^N(x_i-\mu_x)(x_i-\mu_x)^T \\\
&= \frac{1}{N}\begin{bmatrix}
    x_1-\mu_x & x_2-\mu_x & \dots & x_N-\mu_x
\end{bmatrix}
\begin{bmatrix}
(x_1-\mu_x)^T \\\
(x_2-\mu_x)^T \\\
\vdots \\\
(x_N-\mu_x)^T
\end{bmatrix} \\\
&= \frac{1}{N}\left(\begin{bmatrix}
    x_1 & x_2 & \dots & x_N
\end{bmatrix} - \mu_x\begin{bmatrix}
    1 & 1 & \dots & 1
\end{bmatrix}\right)
\begin{bmatrix}
(x_1-\mu_x)^T \\\
(x_2-\mu_x)^T \\\
\vdots \\\
(x_N-\mu_x)^T
\end{bmatrix} \\\
&= \frac{1}{N} (X^T-\mu_x1_N^T)(X^T-\mu_x1_N^T)^T \\\
&= \frac{1}{N} (X^T-\frac{1}{N}X^T 1_N 1_N^T)(X^T-\mu_x1_N^T)^T \\\
&= \frac{1}{N}X^T (I_N-\frac{1}{N} 1_N 1_N^T)(X^T-\mu_x1_N^T)^T \\\
\end{align*}
$$

{{< math.inline >}}
<p>
Let \( I_N-\frac{1}{N} 1_N 1_N^T = H \) be the <mark>centering matrix</mark>, it has following attributes:
</p>
{{</ math.inline >}}

$$
\begin{align*}
H^T &= (I_N-\frac{1}{N} 1_N 1_N^T)^T \\\
&= I_N - \frac{1}{N}1_N1_N^T \\\
&= H
\end{align*}
$$

$$
\begin{align*}
H^2 &= HH \\\
&= (I_N-\frac{1}{N} 1_N 1_N^T)(I_N-\frac{1}{N} 1_N 1_N^T) \\\
&= I_N - \frac{2}{N}1_N1_N^T + \frac{1}{N^2}1_N1_N^T1_N1_N^T \\\
&= I_N - \frac{2}{N}1_N1_N^T + \frac{1}{N^2}N1_N1_N^T \\\
&= I_N - \frac{1}{N}1_N1_N^T \\\
&= H \\\
H^3 &= H^2H = H \\\
&\cdots \\\
H^n &= H^{n-1}H = H
\end{align*}
$$

Based on these attributes we can continue simplify covariance:

$$
\begin{align*}
\Sigma_x &= \frac{1}{N}X^THH^TX \\\
&= \frac{1}{N}X^THHX \\\
&= \frac{1}{N}X^THX
\end{align*}
$$

## Understand PCA
### Maximize projection variance

The core idea of Principal component analysis(PCA) is to <mark>maximze projection variance</mark>. When a high dimensional data project to a low dimensional hyperplane, we want the projection position to be as disperse as possible.

First, we need to centerize data sample:

$$
x_i - \mu_x
$$

Then the projection operation([mentioned before](https://tirmisula.github.io/posts/linear-discriminant-analysis/#prerequisite)) is defined by:

$$
(x_i-\mu_x)^Tu_1 \\\
\lVert u_1 \rVert=1, u_1 \text{ is the first principal component}
$$

{{< math.inline >}}
<p>
Then we can define the object function or variance \(J\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
J &= \frac{1}{N} \sum_{i=1}^N((x_i-\mu_x)^Tu_1 - 0)^2 \\\
&= \frac{1}{N} \sum_{i=1}^Nu_1^T(x_i-\mu_x)(x_i-\mu_x)^Tu_1 \\\
&= u_1^T \frac{1}{N} \sum_{i=1}^N(x_i-\mu_x)(x_i-\mu_x)^T u_1 \\\
&= u_1^T\Sigma_xu_1
\end{align*}
$$

{{< math.inline >}}
<p>
The problem is to maximize \(J\):
</p>
{{</ math.inline >}}

$$
\begin{cases}
\hat{u_1} = \argmax_u u_1^T\Sigma_xu_1 \\\
\text{subject to } u_1^Tu_1=1
\end{cases}
$$

$$
\begin{align*}
L(u_1,\lambda) &= u_1^T\Sigma_x u_1+\lambda(1-u_1^Tu_1) \\\
\frac{\partial L}{\partial u_1} &= 2\Sigma_xu_1-2\lambda u_1 = 0 \\\
\Sigma_xu_1 &= \lambda u_1
\end{align*}
$$

{{< math.inline >}}
<p>
We can conclude that finding \(u_1\) equals to finding eigen vector of \(\Sigma_x\) and \(\lambda\) is  the eigen value of \(\Sigma_x\).
</p>
{{</ math.inline >}}

### Minimize reconstruct cost

When we recover a q-dimensional data to original p-dimensional data, we want to <mark>minimize reconstruction cost</mark>. In particular, we want to minimize the distance between original vector and recovered vector.

{{< math.inline >}}
<p>
Given \( \{u_1,u_2,\cdots,u_p\} \) a group of orthonormal basis, \( x_i^Tu_i \) is the coordinate, we can rewrite \(x_i\) like:
</p>
{{</ math.inline >}}

$$
\begin{align*}
x_i &= (x_i^Tu_1)u_1+(x_i^Tu_2)u_2+\cdots+(x_i^Tu_p)u_p \\\
&= \sum_{k=1}^p (x_i^Tu_k)u_k
\end{align*}
$$

{{< math.inline >}}
<p>
Similarly we can write recovered data \(\tilde{x_i}\), since \(q\) dimensions are recovered by first \(q\) principal components and the left \(p-q\) dimensions are zero:
</p>
{{</ math.inline >}}

$$
\tilde{x_i} = \sum_{k=1}^q (x_i^Tu_k)u_k + \sum_{k=q+1}^p 0
$$

Then the object function(recovery loss) is:

$$
\begin{align*}
J &= \frac{1}{N}\sum_{i=1}^N \lVert x_i-\tilde{x_i} \rVert^2 \\\
&= \frac{1}{N}\sum_{i=1}^N \lVert \sum_{k=q+1}^p (x_i^Tu_k)u_k \rVert^2 \\\
&= \frac{1}{N}\sum_{i=1}^N \lVert \overbrace{x_i}^{q+1,q+2,\cdots, p} \rVert^2 \\\
&= \frac{1}{N}\sum_{i=1}^N \sum_{k=q+1}^p (x_i^Tu_k)^2 \\\
\text{replace } x_i \text{ with } &x_i-\mu_x \text{ because of centralization} \\\
&\triangleq \frac{1}{N}\sum_{i=1}^N \sum_{k=q+1}^p ((x_i-\mu_x)^Tu_k)^2 \\\
&= \frac{1}{N}\sum_{i=1}^N \sum_{k=q+1}^p u_k^T(x_i-\mu_x)(x-\mu_x)^Tu_k \\\
&= \sum_{k=q+1}^p u_k^T\left[\frac{1}{N}\sum_{i=1}^N (x_i-\mu_x)(x-\mu_x)^T\right]u_k \\\
&= \sum_{k=q+1}^p u_k^T\Sigma_xu_k \\\
\end{align*}
$$

{{< math.inline >}}
<p>
Similar to previous subsection, the problem is to minimize \(J\):
</p>
{{</ math.inline >}}

$$
\begin{cases}
\hat{u_k} = \argmin_u \sum_{k=q+1}^p u_k^T\Sigma_xu_k \\\
\text{subject to } u_k^Tu_k=1, k=q+1,\cdots,p
\end{cases}
$$

$$
\begin{align*}
L(u_{q+1},\cdots,u_p,\lambda) &= \sum_{k=q+1}^p u_k^T\Sigma_xu_k + \sum_{k=q+1}^p\lambda_k(1-u_k^Tu_k) \\\
\frac{\partial L}{\partial u_k} &= 2\Sigma_xu_k-2\lambda_k u_k = 0 \\\
\Sigma_xu_k &= \lambda_k u_k
\end{align*}
$$

{{< math.inline >}}
<p>
By replacing \( \Sigma_xu_k \) with \( \lambda_ku_k \), we get:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\sum_{k=q+1}^p u_k^T\Sigma_xu_k &= \sum_{k=q+1}^p u_k^T\lambda_ku_k \\\
&= \sum_{k=q+1}^p \lambda_ku_k^Tu_k \\\
&= \sum_{k=q+1}^p \lambda_k
\end{align*}
$$

{{< math.inline >}}
<p>
Then we can conclude that to minimize reconstruction cost, we need to find \(p-q\) smallest eigen values and corresponding eigen vectors, the smallest reconstruction cost is the summation of \(p-q\) smallest eigen values.
</p>
{{</ math.inline >}}

### SVD and Eigendecomposition

{{< math.inline >}}
<p>
In this subsection, we would like to prove that there are 3 matrix factorization methods to achieve PCA. Recall that PCA tries to find top-q principal components from a \(N \times p\) data matrix and it is equivalent to finding top-q eigen vectors from the \(p \times p\) covariance matrix.
</p>
{{</ math.inline >}}

#### method1
So the first method is eigendecomposite covariance matrix:

$$
\Sigma_x = Q\Lambda Q^T \\\
Q^TQ = I, \Lambda = \underset{\lambda_1>\cdots>\lambda_p}{\begin{bmatrix}
\lambda_1 & & \\\
& \ddots & \\\
& & \lambda_p 
\end{bmatrix}}
$$

{{< math.inline >}}
<p>
Each column of \(Q\) represents a eigen vector we want to find. After finding principal components, we can do projection on each component and acquire new coordinates, these includes centralization and projection:
</p>
{{</ math.inline >}}

$$
\text{coordinates} = HXQ, H \text{ is centering matrix}
$$

{{< math.inline >}}
<p>
\(HXQ\) is the result we get.
</p>
{{</ math.inline >}}

#### method2
{{< math.inline >}}
<p>
The second method is perform singular value decomposition(SVD) on centerized data matrix \(HX\):
</p>
{{</ math.inline >}}

$$
HX = U\Sigma V^T \\\
U^TU = I, V^TV=I
$$

$$
\begin{align*}
\because \Sigma_x &= \frac{1}{N}X^THX \\\
&\propto X^THHX \\\
&= X^TH^THX \\\
&= V\Sigma^T U^TU\Sigma V^T \\\
&= V\Sigma^2 V^T \\\
\therefore Q &= V \\\
\Lambda &= \Sigma^2
\end{align*}
$$

{{< math.inline >}}
<p>
Each column of \(V\) represents a eigen vector we want to find. Then find new coordinates:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{coordinates} &= HXV \\\
&= U\Sigma V^TV \\\
&= U\Sigma
\end{align*}
$$

{{< math.inline >}}
<p>
\(U\Sigma\) is the result we get.
</p>
{{</ math.inline >}}

#### method3
{{< math.inline >}}
<p>
The third method is perform eigendecomposition on matrix \(T=HXX^TH^T\)
</p>
{{</ math.inline >}}

$$
\begin{align*}
\because T &= HXX^TH^T \\\
&= U\Sigma V^T V\Sigma U^T \\\
&= U\Sigma^2 U^T
\end{align*}
$$

{{< math.inline >}}
<p>
We do not solve principal component, we solve new coordinates directly by finding eigen vectors of \(T\), this is called principle coordinate analysis(PCoA):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{coordinates} &=U\Sigma \\\
&\dArr \\\
TU\Sigma &= U\Sigma^2 U^TU\Sigma \\\
&= U\Sigma^3 \\\
&= (U\Sigma)\Sigma^2 \\\
&\dArr \\\
U\Sigma\text{ is eigen vector matrix of }&T \text{ just like }U\text{ but with different scale}
\end{align*}
$$

{{< math.inline >}}
<p>
\(U\Sigma\) is the result we get.
</p>
{{</ math.inline >}}

## P-PCA
### PPCA model definition

Probabilistic principal component analysis (PPCA) is a probalistic generation model just like gaussian discriminant analysis, it has following assumptions:

$$
\begin{align*}
\text{observed data } x &\in \mathbb{R}^p \\\
\text{latent variable } z &\in \mathbb{R}^q, q<p \\\
x &= Wz+b+\epsilon \\\
z &\sim \mathcal{N}(0_q, I_q) \\\
\epsilon &\sim \mathcal{N}(0_p, \sigma^2I_p) \\\
z,\epsilon &\text{ are isotropic gaussian variable}
\end{align*}
$$

It represents a linear gaussian model, and the object is to solve:

$$
\text{P-PCA}
\begin{cases}
\text{probability: } p(z|x) \\\
\text{parameters: } W, b, \sigma^2
\end{cases}
$$

### Inference conditional probabilty

{{< math.inline >}}
<p>
\(p(z)\) is known, \(p(x)\) is solvable, so we can compute joint probability \(p(x,z)\) first then conditional probability \(p(z|x)\).
</p>
{{</ math.inline >}}

This part is mentioned in [previous article](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/)

## Conclusion

Given that:

$$
L(\theta) = \prod_{i=1}^N \prod_{j=1}^p p(x_i^j|y_i)p(y_i)
$$

We assume:

$$
y \sim \mathcal{Bernoulli}(\phi) \\\
x_i^j|y=c \sim \mathcal{Cat}(m,p_{j1c},\cdots,p_{jmc})
$$

{{< math.inline >}}
<p>
Then the solved parameters are :
</p>
{{</ math.inline >}}


## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=21).
[^2]: From [source](https://zhuanlan.zhihu.com/p/71960086).