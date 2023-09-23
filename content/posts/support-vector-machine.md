---
author: "X. Wang"
title: "Support Vector Machine"
date: "2023-09-06"
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

Support vector machine(SVM) has 3 parts:

$$
\begin{cases}
\text{Hard-margin SVM} \\\
\text{Soft-margin SVM} \\\
\text{Kernel SVM}
\end{cases}
$$

## Hard-margin SVM
### Definition of hard-margin SVM

Given a dataset:

$$
\mathcal{D} = \lbrace(x_i,y_i)\rbrace_{i=1,\cdots,N} \\\
x_i \in \mathbb{R}^p \\\
y_i = \lbrace1,-1\rbrace
$$

Similar to [perceptron model](https://tirmisula.github.io/posts/perceptron/), hard-margin SVM tries to find a decision hyperplane that 100% correctly classifies each data sample and the <mark>closest</mark> point to decision hyperplane should have it's distance as long as possible:

$$
\max_{w,b}\min_{\forall x_i, i=1,\cdots,N} dist(x_i, w^Tx+b) \\\
\text{subject to } y_i(w^Tx_i+b) > 0
$$

{{< math.inline >}}
<p>
\(w\) is the normal vector of hyperplane \(w^Tx+b\), because given 2 points \(a_1,a_2\) on hyperplane, it has:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(w^Ta_1+b) - (w^Ta_2+b) &= 0 \\\
w^T(a_1-a_2) &= 0 \\\
w &\perp a_1-a_2
\end{align*}
$$

{{< math.inline >}}
<p>
Assuming we have \(x\)'s projection point \(h\), then \(x-h\) must be perpendicular to normal vector \(w\), then we can define distance function:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(x-h) &= kw \\\
h &= x-kw \\\
&\dArr \\\
w^T(x-kw)+b &= 0 \\\
w^Tx - k\lVert w\rVert^2 +b &= 0 \\\
k &= \frac{w^Tx+b}{\lVert w\rVert^2} \\\
&\dArr \\\
dist(x,h) = \lvert x-h \rvert &= \frac{\lvert w^Tx+b \rvert}{\lVert w\rVert^2} \lVert w \rVert \\\
&= \frac{\lvert w^Tx+b \rvert}{\lVert w\rVert}
\end{align*}
$$

The original problem becomes:

$$
\begin{align*}
\max_{w,b}\min_{\forall x_i, i=1,\cdots,N} dist(x_i, w^Tx+b) &= \max_{w,b}\min_{\forall x_i, i=1,\cdots,N} \frac{\lvert w^Tx_i+b \rvert}{\lVert w\rVert} \\\
&= \max_{w,b}\frac{1}{\lVert w\rVert}\min_{\forall x_i, i=1,\cdots,N} \lvert w^Tx_i+b \rvert \\\
&= \max_{w,b}\frac{1}{\lVert w\rVert}\min_{\forall x_i, i=1,\cdots,N} y_i(w^Tx_i+b)
\end{align*}
$$

{{< math.inline >}}
<p>
Since hyperplane \(w^Tx+b \implies 2w^Tx+2b \) is scalable, we normalize \( \min y_i(w^Tx_i+b) \) to a certain value:
</p>
{{</ math.inline >}}

$$
\min_{\forall x_i, i=1,\cdots,N} y_i(w^Tx_i+b) = \gamma
$$

### Minimize reconstruct cost





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
\(U\Sigma\) is the result we get. This method is more effective when \(p>>N\).
</p>
{{</ math.inline >}}

## P-PCA
### PPCA model definition

Probabilistic principal component analysis (PPCA) is a probalistic generation model like gaussian discriminant analysis, it has following assumptions:<cite>[^1]</cite>

$$
\begin{align*}
\text{observed data } x &\in \mathbb{R}^p \\\
\text{latent variable } z &\in \mathbb{R}^q, q<p \\\
x &= Wz+b+\epsilon \\\
z &\sim \mathcal{N}(0_q, I_q) \\\
\epsilon &\sim \mathcal{N}(0_p, \sigma^2I_p) \\\
z,\epsilon &\text{ are isotropic gaussian variables} \\\
z &\perp \epsilon
\end{align*}
$$

It represents a linear gaussian model, and the object is to solve:

$$
\text{PPCA}
\begin{cases}
\text{probability: } p(z|x) \\\
\text{parameters: } W, b, \sigma^2
\end{cases}
$$

### Inference conditional probabilty

{{< math.inline >}}
<p>
\(p(z)\) is known, \(p(x|z)\) is solvable, so we can compute joint probability \(p(x,z)\) first then conditional probability \(p(z|x)\).
</p>
{{</ math.inline >}}

Deduction process of thhis part is mentioned in [previous article](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/):

$$
\begin{align*}
E[x|z] &= E[Wz+b+\epsilon] \\\
&=Wz+b+E[\epsilon] \\\
&= Wz+b
\end{align*}
$$

$$
\begin{align*}
Var[x|z] &= Var[Wz+b+\epsilon] \\\
&= Var[\epsilon] \\\
&= \sigma^2I
\end{align*}
$$

$$
x|z \sim \mathcal{N}(Wz+b,\sigma^2I)
$$

$$
\begin{align*}
E[x] &= E[Wz+b+\epsilon] \\\
&= WE[z]+b+E[\epsilon] \\\
&= b
\end{align*}
$$

$$
\begin{align*}
Var[x] &= Var[Wz+b+\epsilon] \\\
&= Var[Wz] + Var[\epsilon] \\\
&= WIW^T + \sigma^2I \\\
&= WW^T +\sigma^2I
\end{align*}
$$

$$
x \sim \mathcal{N}(b,WW^T+\sigma^2I)
$$

$$
\begin{align*}
E\left[\begin{bmatrix}
x \\\
z
\end{bmatrix}\right] &= \begin{bmatrix}
E[x] \\\
E[z]
\end{bmatrix} \\\
&= \begin{bmatrix}
b \\\
0
\end{bmatrix}
\end{align*}
$$

$$
\begin{align*}
Var\left[\begin{bmatrix}
x \\\
z
\end{bmatrix}\right] &= \begin{bmatrix}
Cov[x,x] & Cov[x,z] \\\
Cov[z,x] & Cov[z,z]
\end{bmatrix} \\\
&= \begin{bmatrix}
WW^T+\sigma^2I & Cov[x,z] \\\
Cov[z,x] & I
\end{bmatrix} \\\
\because Cov[x,z] &= E\left[ (x-b) (z-0)^T \right] \\\
&= E\left[ (Wz+\epsilon) z^T \right] \\\
&= E\left[Wzz^T\right] + E\left[\epsilon z^T\right] \\\
&= WE\left[(z-0)(z-0)^T\right] + E[\epsilon]E[z^T] \\\
&= WI \\\
&= W \\\
\therefore Var\left[\begin{bmatrix}
x \\\
z
\end{bmatrix}\right] &= \begin{bmatrix}
WW^T+\sigma^2I & W \\\
W^T & I
\end{bmatrix}
\end{align*}
$$

$$
\begin{bmatrix}
x \\\
z
\end{bmatrix} \sim \mathcal{N}(\begin{bmatrix}
b \\\
0
\end{bmatrix}, \begin{bmatrix}
WW^T+\sigma^2I & W \\\
W^T & I
\end{bmatrix})
$$

Just use conclusion from [previous article](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/):

$$
\begin{align*}
E[z|x] &= \Sigma_{zx}\Sigma_{xx}^{-1}(x-E[x])+E[z] \\\
&= W^T(WW^T+\sigma^2I)^{-1}(x-b)
\end{align*}
$$

$$
\begin{align*}
Var[z|x] &= -\Sigma_{zx}\Sigma_{xx}^{-1}\Sigma_{xz}+\Sigma_{zz} \\\
&= -W^T(WW^T+\sigma^2I)^{-1}W + I
\end{align*}
$$


## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=27).
[^2]: From [source](https://zhuanlan.zhihu.com/p/71960086).