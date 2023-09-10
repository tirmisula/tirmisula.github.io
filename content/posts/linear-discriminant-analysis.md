---
author: "X. Wang"
title: "Linear Classification - LDA"
date: "2023-08-31"
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

LDA algorithm:                                                         

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

## Prerequisite
### Geometric meaning of vector inner product
Given that,
$$
w \in \mathbb{R}^p, \lvert w \rvert = 1\\\
x \in \mathbb{R}^p
$$

We have the inner product by definition:

$$
\begin{align*}
w^Tx &= \lvert w \rvert \lvert x \rvert \cos(\alpha) \\\
&= \lvert x \rvert \cos(\alpha)
\end{align*}
$$
{{< math.inline >}}
<p>
So \( w^Tx \) means the \(x\)'s projection point on aixs \( w \), it is  a scalar.
</p>
{{</ math.inline >}}


## LDA model

Linear discriminant analysis(LDA) is a linear classification method similar to perceptron model.

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\), it is described as:
</p>
{{</ math.inline >}}

$$
\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\\\
x_i \in \mathbb{R}^p, y_i \in \lbrace 1,-1 \rbrace
$$

We define 2 classes C1 and C2, it has following attributes:

$$
x_{C1} = \{ x_i|y_i=1 \} \\\
x_{C2} = \{ x_i|y_i=-1 \} \\\
|x_{C1}| = N_1 \\\
|x_{C2}| = N_2 \\\
N = N_1+N_2
$$

{{< math.inline >}}
<p>
The core idea of LDA method is reducing dimensions. By projecting \(x_i\) to a 1-dimensional axis, LDA tries to minimize variance of samples inside a class and maximize distance between 2 classes simutaneously.
</p>
{{</ math.inline >}}

From [Prerequisite](#prerequisite) we can define projection first:

$$
\begin{align*}
z_i = w^Tx_i
\end{align*}
$$

{{< math.inline >}}
<p>
\(z_i\) is the projection point scalar of \(x_i\) on \(w\) axis. We can still split 2 classes for \(z_i\):
</p>
{{</ math.inline >}}

$$
z_{C1} = \{ z_i|x_i \in C1 \}\\\
z_{C2} = \{ z_i|x_i \in C2 \}
$$

Then for each class we can define centroid and sample variance:

$$
\begin{align*}
\bar{z_{C1}} &= \frac{1}{N_1} \sum_{z_i \in z_{C1}} z_i \\\
&= \frac{1}{N_1} \sum_{x_i \in x_{C1}} w^Tx_i \\\
&= w^T \bar{x_{C1}}
\end{align*}
$$

$$
\begin{align*}
\bar{z_{C2}} &= \frac{1}{N_2} \sum_{z_i \in z_{C2}} z_i \\\
&= \frac{1}{N_2} \sum_{x_i \in x_{C2}} w^Tx_i \\\
&= w^T \bar{x_{C2}}
\end{align*}
$$

$$
\begin{align*}
\sigma_{z_{C1}}^2 &= Cov(z_{C1},z_{C1})\\\
&= E\left[ (w^Tx_i - \bar{z_{C1}})(w^Tx_i - \bar{z_{C1}})^T \right]\\\
&= \frac{1}{N_1} \sum_{x_i \in x_{C1}} (w^Tx_i - \bar{z_{C1}})(w^Tx_i - \bar{z_{C1}})^T \\\
&= \frac{1}{N_1} \sum_{x_i \in x_{C1}} (w^Tx_i - w^T \bar{x_{C1}})(w^Tx_i - w^T \bar{x_{C1}})^T \\\
&= \frac{1}{N_1} \sum_{x_i \in x_{C1}} w^T(x_i - \bar{x_{C1}})(x_i - \bar{x_{C1}})^Tw \\\
&= w^T \left[\frac{1}{N_1} \sum_{x_i \in x_{C1}} (x_i - \bar{x_{C1}})(x_i - \bar{x_{C1}})^T\right] w \\\
&= w^T \sigma_{x_{C1}}^2 w
\end{align*}
$$

$$
\begin{align*}
\sigma_{z_{C2}}^2 &= Cov(z_{C2},z_{C2})\\\
&= E\left[ (w^Tx_i - \bar{z_{C2}})(w^Tx_i - \bar{z_{C2}})^T \right]\\\
&= \frac{1}{N_2} \sum_{x_i \in x_{C2}} (w^Tx_i - \bar{z_{C2}})(w^Tx_i - \bar{z_{C2}})^T \\\
&= \frac{1}{N_2} \sum_{x_i \in x_{C2}} (w^Tx_i - w^T \bar{x_{C2}})(w^Tx_i - w^T \bar{x_{C2}})^T \\\
&= \frac{1}{N_2} \sum_{x_i \in x_{C2}} w^T(x_i - \bar{x_{C2}})(x_i - \bar{x_{C2}})^Tw \\\
&= w^T \left[\frac{1}{N_1} \sum_{x_i \in x_{C2}} (x_i - \bar{x_{C2}})(x_i - \bar{x_{C1}})^T\right] w \\\
&= w^T \sigma_{x_{C2}}^2 w
\end{align*}
$$

Then we can design the object function:

$$
\begin{align*}
J(w) &= \frac{(\bar{z_{C1}}-\bar{z_{C2}})^2}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 }\\\
&= \frac{(w^T \bar{x_{C1}}-w^T \bar{x_{C2}})^2}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 } \\\
&= \frac{(w^T (\bar{x_{C1}}-\bar{x_{C2}}))^2}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 } \\\
&= \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}})(w^T (\bar{x_{C1}}-\bar{x_{C2}}))^T}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 } \\\
&= \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^T w}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 } \\\
&= \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^T w}{ w^T \sigma_{x_{C1}}^2 w + w^T \sigma_{x_{C2}}^2 w } \\\
&= \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^T w}{ w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w }
\end{align*}
$$

$$
\hat{w} = \argmax_w J(w)
$$

 <cite>[^1]</cite>

## Solve LDA object function

$$
\begin{align*}
J(w) &= \frac{(\bar{z_{C1}}-\bar{z_{C2}})^2}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 }\\\

&= 
\end{align*}
$$

So falsely predict has the opposite form:

$$
y_iw^Tx_i < 0
$$

Then we can design loss function like:

$$
\begin{align*}
L(w) &= \sum_{i=1}^N I\lbrace y_iw^Tx_i < 0 \rbrace
\end{align*}
$$

{{< math.inline >}}
<p>
However indicator function is not <mark>derivable</mark>, because tiny change \( \Delta w \) may results 0 \( \rarr \) 1 or 1 \( \rarr \) 0. The way is to remove indicator operation and multiply -1 to change convergence from \( -\infty \rarr 0\) to \( \infty \rarr 0 \).
</p>
{{</ math.inline >}}

The final loss function is designed as follow:

$$
\begin{align*}
L(w) &= \sum_{x_i \in D_{err}}  -y_iw^Tx_i
\end{align*}\\\
D_{err}: \{x_i | (x_i,y_i) \in \mathcal{D},y_iw^Tx_i < 0\}
$$

## Solve preceptron weights

{{< math.inline >}}
<p>
Since \( L(w) \) is derivable, the most common way is to use stochastic gradient decent(SGD) method to find \( \hat{w} \):
</p>
{{</ math.inline >}}

$$
\hat{w} = \argmin_w L(w)
$$

$$
\text{SGD Procedure:}\\\
\text{Initialize } w \larr w_0, w_0 \in \text{any real num vector}\\\
\begin{rcases}
\text{Update } D_{err} \text{ by } w^t \\\
w^{t+1} \larr w^{t} - \lambda \nabla_w L(w^t)
\end{rcases} \text{iterate t imes}\\\
\hat{w} = w^{t+1}
$$

$$
\because \nabla_{w^t} L(w^t) = \sum_{x_i \in D_{err}} -y_ix_i \\\
\therefore w^{t+1} \larr w^{t} + \lambda \sum_{x_i \in D_{err}} y_ix_i
$$

## Further readings
1. Convergence of perceptron
2. Pocket algorithm: dataset can not be linearly classified

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=14).
[^2]: From [source](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).