---
author: "X. Wang"
title: "Solve marginal conditional and joint Gaussian pdf"
date: "2023-08-28"
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

To simplify the calculations, we take the logarithm of the likelihood function:                                                         

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}

<!-- KaTeX -->
<!-- <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script> -->
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

## Lemma 1
If we have the following,
$$
\begin{cases}
x \sim \mathcal{N}(\mu,\Sigma),
x \in \mathbb{R}^p\\\
y = Ax+B,
y \in \mathbb{R}^q
\end{cases}
$$

{{< math.inline >}}
<p>
Then \(y\) is subject to Gaussian distribution,
</p>
{{</ math.inline >}}

$$
y \sim \mathcal{N}(A\mu+B,A\Sigma A^T)
$$
Because,
$$
\begin{align*}
E[y] &= E[Ax+B]\\\
&= AE[x]+B\\\
&= A\mu+B
\end{align*}\\\
\begin{align*}
Var[y] &= Var[Ax+B] \\\
&= AVar[x]A^T \\\
&= A\Sigma A^T
\end{align*}
$$

## Problem clarification <cite>[^1]</cite>
{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{x_1, x_2, ..., x_n\}\), where each data point \(x_i\) is drawn from a p-D Gaussian distribution with unknown mean \(\mu\) and variance \(\Sigma\), we have:
</p>
{{</ math.inline >}}

$$p(x|\mu,\Sigma) = \frac{1}{(2\pi)^\frac{p}{2} |\Sigma|^\frac{1}{2}} \exp\left( -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right)$$

$$
x = \begin{bmatrix}
    x_{1} \\\
    x_{2} \\\
    \vdots \\\
    x_{p}
\end{bmatrix},
\mu = \begin{bmatrix}
    \mu_{1} \\\
    \mu_{2} \\\
    \vdots \\\
    \mu_{p}
\end{bmatrix},
\Sigma =\begin{bmatrix}
    \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\\
    \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\\
    \vdots & \vdots & \ddots & \vdots \\\
    \sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp}
\end{bmatrix}
$$

{{< math.inline >}}
<p>
If x is made up of two parts \(x_a\) and \(x_b\), we can rewrite \(x\), \(\mu\), \(\Sigma\) above with block matrix:
</p>
{{</ math.inline >}}

$$
x=\begin{bmatrix}
    x_1\\\
    x_2\\\
    \vdots\\\
    x_m\\\
    x_{m+1}\\\
    x_{m+2}\\\
    \vdots\\\
    x_{m+n}
\end{bmatrix}
=\begin{bmatrix}
    x_a\\\
    x_b
\end{bmatrix}, m+n=p
$$

$$
\mu = \begin{bmatrix}
    \mu_{a}\\\
    \mu_{b}
\end{bmatrix}
$$

$$
\Sigma = \begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab}\\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix}
$$

{{< math.inline >}}
<p>
We want to obtain the marginal pdf \(p(x_a)\) and \(p(x_b)\), the conditional pdf \(p(x_b|x_a)\) and \(p(x_a|x_b)\)
</p>
{{</ math.inline >}}

## Solve the marginal pdf

{{< math.inline >}}
<p>
Based on **Lemma 1** we can construct \(x_a\) by \(Ax+B\):
</p>
{{</ math.inline >}}

$$
x_a = \underbrace{
    \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}
}_A
\underbrace{\begin{bmatrix}
    x_{a}\\\
    x_{b}
\end{bmatrix}}_x
$$

Then we can get:
$$
\begin{align*}
E[x_a] &= E\left[\begin{bmatrix}
    1_m & 0_n
\end{bmatrix}\begin{bmatrix}
    x_{a}\\\
    x_{b}
\end{bmatrix}\right]\\\
&= \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}E\left[\begin{bmatrix}
    x_{a}\\\
    x_{b}
\end{bmatrix}\right]\\\
&= \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}\begin{bmatrix}
    \mu_{a}\\\
    \mu_{b}
\end{bmatrix}\\\
&= \mu_{a}
\end{align*}
$$

$$
\begin{align*}
Var[x_a] &= \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}
\Sigma
\begin{bmatrix}
    1_m\\\
    0_n
\end{bmatrix}\\\
&= \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}
\begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab}\\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix}
\begin{bmatrix}
    1_m\\\
    0_n
\end{bmatrix}\\\
&= \begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab}
\end{bmatrix}
\begin{bmatrix}
    1_m\\\
    0_n
\end{bmatrix}\\\
&= \Sigma_{aa}
\end{align*}
$$

{{< math.inline >}}
<p>
Deduction of \(x_b\) is similar, so the conclusion is:
</p>
{{</ math.inline >}}

$$
x_a \sim \mathcal{N}(\mu_{a},\Sigma_{aa})
$$
$$
x_b \sim \mathcal{N}(\mu_{b},\Sigma_{bb})
$$

## Solve the conditonal pdf
{{< math.inline >}}
<p>
Before dig into \(p(x_b|x_a)\), let's introduce a new variable \(x_{b \cdot a}\)
</p>
{{</ math.inline >}}

$$
\begin{align*}
\Sigma^{-1} &= \begin{bmatrix}
u_1 & u_2 & \cdots & u_p
\end{bmatrix}
\begin{bmatrix}
    \frac{1}{\lambda_{1}} & 0 & \cdots & 0 \\\
    0 & \frac{1}{\lambda_{2}} & \cdots & 0 \\\
    \vdots & \vdots & \ddots & \vdots \\\
    0 & 0 & \cdots & \frac{1}{\lambda_{p}}
\end{bmatrix}
\begin{bmatrix}
{u_1}^T\\\
{u_2}^T\\\
\vdots\\\
{u_p}^T
\end{bmatrix} \\\
&=\sum_{i=1}^p u_i\frac{1}{\lambda_{i}}{u_i}^T
\end{align*}
$$

{{< math.inline >}}
<p>
Put it back to the formula, we can get:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(x-\mu)^T \Sigma^{-1} (x-\mu) &= (x-\mu)^T \sum_{i=1}^p u_i\frac{1}{\lambda_{i}}{u_i}^T (x-\mu)\\\
&= \sum_{i=1}^p (x-\mu)^T u_i\frac{1}{\lambda_{i}}{u_i}^T (x-\mu)
\end{align*}
$$

{{< math.inline >}}
<p>
Let \(y_i = (x-\mu)^T u_i\), \(y_i\) is a scalar, we can get:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(x-\mu)^T \Sigma^{-1} (x-\mu) &= \sum_{i=1}^p y_i \frac{1}{\lambda_{i}} {y_i}^T\\\
&= \sum_{i=1}^p \frac{{y_i}^2}{\lambda_{i}}
\end{align*}
$$

## Conclusion

{{< math.inline >}}
<p>
Now consider \((x-\mu)^T \Sigma^{-1} (x-\mu)\) equals to a constant,  in other words a fixed Mahalanobis distance:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(x-\mu)^T \Sigma^{-1} (x-\mu) &= \sum_{i=1}^p \frac{{y_i}^2}{\lambda_{i}} = c
\end{align*}
$$

{{< math.inline >}}
<p>
This Equation represented a p-dimensional ellipse with a centroid of \(\{\mu_1, \mu_2, ..., \mu_p\}\), a major axis's length of \(\{\lambda_1, \lambda_2, ..., \lambda_p\}\) \(\lambda_i\) which is the eigen value of \(\Sigma\), a rotation direction of \(\{u_1, u_2, ..., u_p\}\), \(u_i\) which is the eigen vector of \(\Sigma\).
</p>
{{</ math.inline >}}

## Limitations of Gaussian model[^2]</cite>
### Parameter size
$$
\Sigma_{p\times p} \rightarrow \frac{p(p+1)}{2}=O(p^2)
$$
{{< math.inline >}}
<p>
\(\Sigma\) matrix has \(O(p^2)\) number of parameters to be estimated which might leads to a slow computation. It can be optimized by transform \(\Sigma\) to a diagnal matrix.
</p>
{{</ math.inline >}}

### Oversimplified model
{{< math.inline >}}
<p>
The assumption of a Gaussian mixture model for the dataset \(\mathcal{D} = \{x_1, x_2, ..., x_n\}\) can be considered in cases where a single Gaussian model may not adequately capture the underlying distribution. By incorporating multiple Gaussian components, a Gaussian mixture model offers a more flexible and expressive representation, allowing for the modeling of complex and diverse patterns within the data.
</p>
{{</ math.inline >}}

$$
\underbrace{a + b + c}_{\text{Description}}
$$
## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=5).
[^2]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=6).