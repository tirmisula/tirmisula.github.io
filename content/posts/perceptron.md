---
author: "X. Wang"
title: "Linear Classification - Perceptron"
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

Perceptron algorithm:                                                         

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
## Perceptron model

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\), where each data point \( (x_i,y_i) \) has attribution \(x_i \in \mathbb{R}^p \), \( y_i \in \{ -1,1 \} \), consider put the linear regression result of \(x_i\) into a sign function, then we get perceptron model:
</p>
{{</ math.inline >}}

$$
\begin{align*}
f(x) &= \operatorname{sign}(w^Tx)\\\
x &\in \mathbb{R}^p\\\
w &\in \mathbb{R}^p
\end{align*}
$$

$$
\operatorname{sign}(a) = 
\begin{cases}
1, a \geq 0\\\
-1, a < 0
\end{cases}
$$

 <cite>[^1]</cite>
 {{< math.inline >}}
<p>
Perceptron model outputs \( f(x_i) \) as classification result of \(x_i\)
</p>
{{</ math.inline >}}

## Design perpceptron loss function

{{< math.inline >}}
<p>
The naive thought is to find prediction accuracy of \(f(x_i)\) on \( y_i \), thus we can design a indicator function to count the number of false predictions.
</p>
{{</ math.inline >}}

A correctly predict contains 2 conditions, and can be summarized in 1 equation:

$$
w^Tx_i \geq 0, y_i=1\\\
w^Tx_i < 0, y_i=-1\\\
\dArr\\\
y_iw^Tx_i \geq 0
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