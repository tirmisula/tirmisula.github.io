---
author: "X. Wang"
title: "Introduction to Least Squares Method"
date: "2023-08-30"
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

LS method:                                                         

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

## Matrix representation of LS<cite>[^1]</cite>
{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\), where each data point \( (x_i,y_i) \) has attribution \(x_i \in \mathbb{R}^p \), \( y_i \in \mathbb{R} \), we combine all \(x_i\) to matrix form so that each row represents a data point, data point has p dimension:
</p>
{{</ math.inline >}}

$$
\begin{align*}
X &= \begin{bmatrix}
    x_{1} & x_2 & \dots & x_N
\end{bmatrix}^T\\\
&= \begin{bmatrix}
    {x_1}^T \\\
    {x_2}^T \\\
    \dots \\\
    {x_N}^T
\end{bmatrix}\\\
&= \begin{bmatrix}
    x_{11} & x_{12} & \dots & x_{1p}\\\
    x_{21} & x_{22} & \dots & x_{2p}\\\
    \dots \\\
    x_{N1} & x_{N2} & \dots & x_{Np}
\end{bmatrix}_{N \times p}
\end{align*}
$$

$$
Y = \begin{bmatrix}
    y_1 \\\
    y_2 \\\
    \dots \\\
    y_N
\end{bmatrix}_{N \times 1}
$$

{{< math.inline >}}
<p>
The definition of Least Squares method is:
</p>
{{</ math.inline >}}

$$
\begin{align*}
w &= \begin{bmatrix}
    {w_1} \\\
    {w_2} \\\
    \dots \\\
    {w_p}
\end{bmatrix}, w_i \in \mathbb{R} \\\
LS(w) &= \sum_{i=1}^N {\lVert w^Tx_i - y_i \rVert}^2 \\\
&= \sum_{i=1}^N ( w^Tx_i - y_i )^2 \\\
&= \begin{bmatrix}
    w^Tx_1-y_1 & w^Tx_2-y_2 & \dots & w^Tx_N-y_N
\end{bmatrix}
\begin{bmatrix}
    w^Tx_1-y_1 \\\
    w^Tx_2-y_2 \\\
    \dots \\\
    w^Tx_N-y_N
\end{bmatrix}\\\
&= \left(
    \begin{bmatrix}
    w^Tx_1 & w^Tx_2 & \dots & w^Tx_N
\end{bmatrix} - 
\begin{bmatrix}
    y_1 & y_2 & \dots & y_N
\end{bmatrix}
\right)
\left(
    \begin{bmatrix}
    w^Tx_1 \\\
    w^Tx_2 \\\
    \dots \\\
    w^Tx_N
\end{bmatrix} - 
\begin{bmatrix}
    y_1 \\\
    y_2 \\\
    \dots \\\
    y_N
\end{bmatrix}
\right)\\\
&= (w^T X^T - Y^T) (Xw - Y)
\end{align*}
$$

## Solve LS by derivation

{{< math.inline >}}
<p>
We want to find \( \hat{w} \) so that \( L(w) \) is minimized
</p>
{{</ math.inline >}}

The derivative of matrix refers to Matrix Cookbook<cite>[^2]</cite>:

$$
\begin{align*}
\frac{\partial}{\partial w} LS(w) &= 0\\\
\frac{\partial}{\partial w} (w^T X^T - Y^T) (Xw - Y) &= 0\\\
\frac{\partial}{\partial w} w^TX^TXw - \underset{\color{red}{Scalar}}{w^TX^TY} - \underset{\color{red}{Scalar}}{Y^TXw} + Y^TY &= 0\\\
2X^TXw - 2X^TY + 0 &= 0\\\
X^TXw &= X^TY \\\
w &= (X^TX)^{-1}X^TY
\end{align*}
$$

$$
(X^TX)^{-1}X^T \text{ also known as pseudoinverse } X^+\\\
\text{OR left inverse of a non-square matrix } X
$$

{{< math.inline >}}
<p>
Finally we find the analytic expression of \( LS(w) \):
</p>
{{</ math.inline >}}

$$
\hat{w} = \argmin_{w} L(w) = (X^TX)^{-1}X^TY
$$

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=7).
[^2]: From [video](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).