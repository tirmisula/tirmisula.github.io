---
author: "X. Wang"
title: "Linear Regression"
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
## Linear regression

### Matrix representation of LS
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
 <cite>[^1]</cite>The definition of Least Squares method is:
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

### Solve LS

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

### Understand LS by geometric projection

$$
\begin{align*}
X &= \begin{bmatrix}
    x_{11} & x_{12} & \dots & x_{1p}\\\
    x_{21} & x_{22} & \dots & x_{2p}\\\
    \dots & \dots & \dots & \dots\\\
    x_{N1} & x_{N2} & \dots & x_{Np}
\end{bmatrix}
\\\
&= \begin{bmatrix}
    x_{col1} & x_{col2} & \dots & x_{colp}
\end{bmatrix}
\end{align*}
$$

{{< math.inline >}}
<p>
We consider martix \( X \) consists of \( p \) column vectors, \( Xw \) can be written as:
</p>
{{</ math.inline >}}

$$
\begin{align*}
Xw &= \begin{bmatrix}
    x_{col1} & x_{col2} & \dots & x_{col4}
\end{bmatrix}
w
\\\
&= \sum_{i=1}^{p} w_i x_{coli}
\end{align*}
$$

{{< math.inline >}}
<p>
Thus \( Xw \) is a linear combination of column vectors \( x_{coli} \) or in other words \( Xw \) is a vector built up by \( p \) basis. Different \(w_i\) leads to different vectors in this p-D hyperplane.

Since \( Y \) does not belong to this hyperplane which because \( N>>p \) and \( Y \) is highly unlikely be expressed by a linear combination of \( p \) basis, \( Xw-Y \), \( Y \) and \( Xw\) forms a triangular. Minimize \( Xw-Y \) means \( Xw-Y \) is perpendicular to the p-D hyperplane, so \( Xw-Y \) is perpendicular to each basis \( x_{coli} \), then we have:
</p>
{{</ math.inline >}}

$$
\begin{align*}
X^T(Xw-Y) &= 0
\\\
X^TXw &= X^TY\\\
w &= (X^TX)^{-1}X^TY
\end{align*}
$$

So we got the same conclusion.

### Find definition of LS from aspect of probability
{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\), where \(x_i \in \mathbb{R}^p \), \( y_i \in \mathbb{R} \), we assume \(y\) is a linear mapping of \(x\) plus a noise which subjected to Gaussian distribution:<cite>[^3]</cite>
</p>
{{</ math.inline >}}

$$
y = w^Tx + \epsilon\\\
\epsilon \sim \mathcal{N}(0, \sigma^2)
$$

Based on the previous [provement](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/), we know that:

$$
y|w^Tx \sim \mathcal{N}(w^Tx, \sigma^2)\\\
p(y|w^Tx) = \frac{1}{ \sqrt{2\pi} \sigma} \mathrm{e}^{-\frac{(y-w^Tx)^2}{2\sigma^2}}
$$

Then we can do maximum-likelihood estimation:

$$
\begin{align*}
\argmax_w \mathcal{L}(w|y) &= \argmax_w \log\left( \prod_{i=1}^N p(y_i|w^Tx_i) \right)\\\
&= \argmax_w \sum_{i=1}^N \log\left( p(y_i|w^Tx_i) \right)\\\
&= \argmax_w \sum_{i=1}^N \log\left( \frac{1}{ \sqrt{2\pi} \sigma} \mathrm{e}^{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}} \right)\\\
&= \argmax_w \sum_{i=1}^N \log\left( \frac{1}{ \sqrt{2\pi} \sigma} \right) + \sum_{i=1}^N \log\left( \mathrm{e}^{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}} \right)\\\
&= \argmax_w N\log\left( \frac{1}{\sqrt{2\pi}\sigma} \right) + \sum_{i=1}^N -\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\\\
&= \argmax_w \sum_{i=1}^N -\frac{(y_i-w^Tx_i)^2}{2\sigma^2}\\\
&= \argmin_w \sum_{i=1}^N (y_i-w^Tx_i)^2 \iff \argmin_w \sum_{i=1}^N {\lVert w^Tx_i - y_i \rVert}^2
\end{align*}
$$

{{< math.inline >}}
<p>
MLE result is exactly the same as the definition of LS, so we can say that finding the best fitted line by LS indicates \(y\) is close to a linear mapping of \(x\) but with a random Gaussian noise.
</p>
{{</ math.inline >}}

## Linear regression with regularization
{{< math.inline >}}
<p>
If \( N \) < \(p \) it will cause overfitting because:
</p>
{{</ math.inline >}}

$$
Rank(X^TX)=Rank(X)<p\\\
X^TX \text{ is not revertible}
$$

Adding regularization part to LS helps prevent overfitting. A common regularization framework is presented as follows:

$$
\begin{align*}
\argmin_{\theta} \left[ \underset{\text{Loss}}{\mathcal{L}(\theta)} + \lambda \underset{\text{Penalty}}{P(\theta)} \right]
\end{align*}
$$

There are 2 common choices of penalty function, L1 norm and  L2 norm which corresponds to Lasso and Ridge regression respectively.

$$
\argmin_{\theta} \left[ \sum_{i=1}^N {\lVert w^Tx_i-y_i \rVert}^2 + \lambda {P(\theta)} \right] \implies
\begin{cases}
P(\theta) = {\lVert w \rVert}_1 & \text{Lasso regression}\\\
P(\theta) = {\lVert w \rVert}^2_2 = w^Tw & \text{Ridge regression}
\end{cases}
$$

### Ridge regression from aspect of frequency
We use matrix representation form from [Previous Section](#matrix-representation-of-ls) to simplify Ridge regression:

$$
\begin{align*}
J(w) &= (w^TX^T-Y^T)(Xw-Y) + \lambda w^Tw\\\
&= w^TX^TXw - \underset{\color{red}{Scalar}}{w^TX^TY} - \underset{\color{red}{Scalar}}{Y^TXw} + Y^TY + \lambda w^Tw\\\
&= w^TX^TXw + \lambda w^Tw - 2w^TX^TY + Y^TY\\\
&= w^T(X^TX+\lambda I)w - 2w^TX^TY + Y^TY
\end{align*}
$$

$$
\argmin_w J(w) \iff \frac{\partial J(w)}{\partial w}=0
$$

$$
\begin{align*}
\frac{\partial J(w)}{\partial w} &= 0\\\
\frac{\partial}{\partial w} w^T(X^TX+\lambda I)w - 2w^TX^TY + Y^TY &= 0\\\
2(X^TX+\lambda I)w - 2X^TY &= 0\\\
w &= (X^TX+\lambda I)^{-1}X^TY
\end{align*}
$$

{{< math.inline >}}
<p>
The analytic expression of \( J(w) \):
</p>
{{</ math.inline >}}

$$
\hat{w} = \argmin_{w} J(w) = (X^TX+\lambda I )^{-1}X^TY
$$

{{< math.inline >}}
<p>
Compared to \( \mathcal{L}(w) \), the different part is the additional diagnal identity matrix which makes \( X^TX+\lambda I \) invertible:
</p>
{{</ math.inline >}}

$$
\because \text{ Given } \forall{a} \\\
\begin{align*}
a^TX^TXa &= (Xa)^TXa\\\
&={\lVert Xa \rVert}^2 \\\
&\geq 0 
\end{align*}
$$

$$
\begin{align*}
a^T(X^TX+\lambda I)a &= a^TX^TXa + \lambda a^TIa\\\
&={\lVert Xa \rVert}^2+\lambda{\lVert Ia \rVert}^2 , \text{If }\lambda > 0\\\
&> 0
\end{align*}\\\
 \therefore (X^TX+\lambda I)\text{ is positive definite and \color{red}{revertible}}\\\
\therefore \text{Solving a revertible matrix prevents overfitting}
$$

### Ridge regression from aspect of bayesian
Similar to maximum likelihood estimation of LS from [Previous Section](#find-definition-of-ls-from-aspect-of-probability), we have:

$$
y = w^Tx + \epsilon\\\
\epsilon \sim \mathcal{N}(0, \sigma^2)
$$

$$
y|w^Tx \sim \mathcal{N}(w^Tx, \sigma^2)\\\
p(y|w) = \frac{1}{ \sqrt{2\pi} \sigma} \mathrm{e}^{-\frac{(y-w^Tx)^2}{2\sigma^2}}
$$

{{< math.inline >}}
<p>
If we consider \(w\) has a prior distribution, \(y\) is the data already happened, then we can do maximum posterier estimation on \(w\) based on \(y\) we observed:
</p>
{{</ math.inline >}}

$$
w \sim \mathcal{N}(0, \sigma_0^2)\\\
p(w) = \frac{1}{ \sqrt{2\pi} \sigma_0} \mathrm{e}^{-\frac{{\lVert w \rVert}^2}{2\sigma_0^2}}
$$

$$
\begin{align*}
\hat{w} &= \argmax_w p(w|y)\\\
&= \argmax_w \frac{p(y|w)p(w)}{p(y)}\\\
&= \argmax_w \frac{\prod_{i=1}^Np(y_i|w)p(w)}{p(y)}\\\
&= \argmax_w \prod_{i=1}^Np(y_i|w)p(w)\\\
&= \argmax_w \log\left( \prod_{i=1}^Np(y_i|w)p(w) \right)\\\
&= \argmax_w \sum_{i=1}^N \log\left( \frac{1}{ \sqrt{2\pi} \sigma} \mathrm{e}^{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}} \right) + \log\left( \frac{1}{ \sqrt{2\pi} \sigma_0} \mathrm{e}^{-\frac{{\lVert w \rVert}^2}{2\sigma_0^2}} \right)\\\
&= \argmax_w \sum_{i=1}^N -\frac{(y_i-w^Tx_i)^2}{2\sigma^2}  -\frac{{\lVert w \rVert}^2}{2\sigma_0^2}\\\
&= \argmin_w \sum_{i=1}^N (y_i-w^Tx_i)^2 + \frac{\sigma^2}{\sigma_0^2}{\lVert w \rVert}^2 \iff \argmin_{w} \left[ \sum_{i=1}^N {\lVert w^Tx_i-y_i \rVert}^2 + \lambda {w^Tw} \right]
\end{align*}
$$

{{< math.inline >}}
<p>
MAP result is exactly the same as the definition of ridge regression, penalty weight \(\lambda\) equals to the ratio of variances between noise and parameter. Larger variance of noise leads to larger penalty weight.
</p>
{{</ math.inline >}}

## Conclusion
In this chapter we discussed about linear regression, first we give the definition of LS:

$$
\begin{align*}
LS(w) &= \sum_{i=1}^N {\lVert w^Tx_i - y_i \rVert}^2 \\\
&= (w^T X^T - Y^T) (Xw - Y)
\end{align*}
$$

And it's analytic expression:

$$
\hat{w}_{LS} = (X^TX)^{-1}X^TY
$$

And it's geometric explanation:

$$
X^T (X\hat{w}_{LS}-Y) = 0
$$

Then we find that with Gaussian distribution noise, solving MLE equals to solving LS:

$$
\text{MLE} 
\begin{cases}
y = w^Tx + \epsilon\\\
\epsilon \sim \mathcal{N}(0, \sigma^2)
\end{cases}
\iff
\text{LS}
$$

Second part we give the definition of Ridge regression or LS + L2 regularizer and  it's analytic expression:

$$
J(w) = \sum_{i=1}^N {\lVert w^Tx_i-y_i \rVert}^2 + \lambda {w^Tw}
$$

$$
\hat{w}_{Ridge} = (X^TX+\lambda I )^{-1}X^TY
$$

Then we find that with Gaussian distribution noise and Gaussian distribution prior, solving MAP equals to solving Ridge regression:

$$
\text{MAP} \begin{cases}
y = w^Tx + \epsilon\\\
\epsilon \sim \mathcal{N}(0, \sigma^2)\\\
w \sim \mathcal{N}(0, \sigma_0^2)
\end{cases}
\iff
\text{Ridge}
$$

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=9).
[^2]: From [source](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
[^3]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=10).