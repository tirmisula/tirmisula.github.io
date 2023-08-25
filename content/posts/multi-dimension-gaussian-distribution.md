---
author: "X. Wang"
title: " Introduction to Multidimension Gaussian Distribution"
date: "2023-08-21"
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

## Definition <cite>[^1]</cite>

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{x_1, x_2, ..., x_n\}\), where each data point \(x_i\) is drawn from a 1-D Gaussian distribution with unknown mean \(\mu\) and variance \(\sigma^2\).
</p>

{{</ math.inline >}}

$$p(x_i|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{x_1, x_2, ..., x_n\}\), where each data point \(x_i\) is drawn from a N-D Gaussian distribution with unknown mean \(\mu\) and variance \(\Sigma\).
</p>
{{</ math.inline >}}

$$p(x_i|\mu,\Sigma) = \frac{1}{(2\pi)^\frac{p}{2} |\Sigma|^\frac{1}{2}} \exp\left( -\frac{1}{2}(x_i-\mu)^T \Sigma^{-1} (x_i-\mu) \right)$$

{{< math.inline >}}
<p>
If variable \(x\) has \(p\) dimensions, we have:
</p>
{{</ math.inline >}}

$$
\begin{cases}
x = \begin{bmatrix}
    x_{1} \\\
    x_{2} \\\
    \vdots \\\
    x_{p}
\end{bmatrix},
x \in \mathbb{R}^p\\\
\mu = \begin{bmatrix}
    \mu_{1} \\\
    \mu_{2} \\\
    \vdots \\\
    \mu_{p}
\end{bmatrix}\\\
\Sigma =\begin{bmatrix}
    \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\\
    \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\\
    \vdots & \vdots & \ddots & \vdots \\\
    \sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp}
\end{bmatrix}, & \Sigma\text{ is semi-positive definite}
\end{cases}
$$

## Explainable quadratic form

{{< math.inline >}}
<p>
\((x-\mu)^T \Sigma^{-1} (x-\mu)\) can be roughly considered as Mahalanobis distance between \(x\) and \(\mu\), when \(\Sigma=I\) it becomes Euclidean distance, prove it in 2-D situation:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(x-\mu)^T \Sigma^{-1} (x-\mu) &= \begin{bmatrix}
x_1-\mu_1 & x_2-\mu_2
\end{bmatrix} I^{-1} \begin{bmatrix}
x_1-\mu_1\\\
x_2-\mu_2
\end{bmatrix}\\\
&= (x_1-\mu_1)^2 + (x_2-\mu_2)^2
\end{align*}
$$

{{< math.inline >}}
<p>
Since \(\Sigma\) must be symmetric, we can do eigen decomposition on \(\Sigma\) and \(\Sigma^{-1}\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\Sigma &= U\Lambda U^T & \text{, }UU^T=U^TU=I\\\
\Sigma^{-1} &= (U\Lambda U^T)^{-1}\\\
&= (U^T)^{-1}\Lambda^{-1}U^{-1}\\\
&= U\Lambda^{-1}U^T\\\
\end{align*}
$$

{{< math.inline >}}
<p>
Let \(U=\begin{bmatrix}
u_1 & u_2 & \cdots & u_p
\end{bmatrix}_{p*p}\), \(u_i\) is the eigen vector, we get:
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
Equation represented a p-dimensional ellipse with a centroid of \(\{\mu_1, \mu_2, ..., \mu_p\}\), a major axis's length of \(\{\lambda_1, \lambda_2, ..., \lambda_p\}\) \(\lambda_i\) related to eigen value of \Sigma, a rotation direction of \(\{u_1, u_2, ..., u_p\}\), \(u_i\) related to eigen vector of \(\Sigma\).
</p>
{{</ math.inline >}}


## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=5).