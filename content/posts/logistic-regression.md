---
author: "X. Wang"
title: "Linear Classification - Logistic Regression"
date: "2023-09-02"
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
## Definition

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\), where each data point \( (x_i,y_i) \) has attribution \(x_i \in \mathbb{R}^p \), \( y_i \in \{ 0,1 \} \). Logistic regression is a probability discriminant model, it assumes \( p(y_i|x_i) \) subjects to a certain distribution. Similar to <a href="https://tirmisula.github.io/posts/perceptron/">perceptron model</a> logistic model choose a activate function for further process, the different thing is that logistic model choose sigmoid function, because logistic model solves soft classification problem, and sigmoid function outputs probability ranges in \( [0,1] \) which satisfies the requirement.
<br/>
<br/>
sigmoid function is defined as follows:<cite>[^1]</cite>
</p>
{{</ math.inline >}}

$$
\begin{align*}
\sigma(z) &= \frac{1}{1+\mathrm{e}^{-z}}
\end{align*}
$$

It has following attributes:

$$
\sigma(z)=
\begin{cases}
0, z \rarr -\infty\\\
\frac{1}{2}, z = 0\\\
1, z \rarr \infty
\end{cases}
$$

 
 {{< math.inline >}}
<p>
Then we can define probability of \(y\) by this function:
</p>
{{</ math.inline >}}

$$
p_1=p(y=1|x)=\sigma(w^Tx)=\frac{1}{1+\mathrm{e}^{-w^Tx}}\\\
p_0=p(y=0|x)=1-p_1=\frac{\mathrm{e}^{-w^Tx}}{1+\mathrm{e}^{-w^Tx}}
$$

{{< math.inline >}}
<p>
Combine \(p_1\), \(p_2\) together, we get:
</p>
{{</ math.inline >}}

$$
p(y|x) = p_1^y p_0^{1-y} \\\
\because
\begin{cases}y=1, p_1^y p_0^{1-y}=p_1\\\
y=0, p_1^y p_0^{1-y}=p_0
\end{cases}
$$

## MLE of logistic regression

MLE:

$$
\begin{align*}
\hat{w} &= \argmax_w \prod_{i=1}^N p(y_i|x_i) \\\
&= \argmax_w \log\left( \prod_{i=1}^N p(y_i|x_i) \right) \\\
&= \argmax_w \sum_{i=1}^N \log\left( p(y_i|x_i) \right) \\\
&= \argmax_w \sum_{i=1}^N\left[ \log(p_1^{y_i})+\log(p_0^{1-{y_i}}) \right] \\\
&= \argmax_w \sum_{i=1}^N\left[ y_i\log(p_1)+(1-y_i)\log(p_0) \right] \\\
&= \argmax_w \underset{\color{red}{-\text{ cross entropy loss}}}{\sum_{i=1}^N\left[ y_i\log(\sigma(w^Tx_i))+(1-y_i)\log(1-\sigma(w^Tx_i)) \right]}
\end{align*}
$$

$$
\begin{align*}
\frac{\partial}{\partial w} \sum_{i=1}^N\left[ y_i\log(\sigma(w^Tx_i))+(1-y_i)\log(1-\sigma(w^Tx_i)) \right] &= 0 \\\
\sum_{i=1}^N \left[ y_i(1+e^{-w^Tx})(-1)(1+e^{-w^Tx})^{-2}e^{-w^Tx}(-x)+ (1-y_i)\log(1-\sigma(w^Tx_i)) \right] &= 0 \\\
\sum_{i=1}^N \left[ y_i\frac{x_ie^{-w^Tx_i}}{(1+e^{-w^Tx_i})} + (1-y_i)\frac{1+e^{-w^Tx}}{e^{-w^Tx}}(1+e^{-w^Tx})^{-2}e^{-w^Tx}(-x) \right] &= 0 \\\
\sum_{i=1}^N \left[ y_i\frac{x_ie^{-w^Tx_i}}{(1+e^{-w^Tx_i})} + (y_i-1)\frac{x_i}{(1+e^{-w^Tx_i})} \right] &= 0 \implies
\sum_{x_i \in C1}x_i(1-\sigma(w^Tx_i)) &= \sum_{x_i \in C0} x_i\sigma(w^Tx_i) \\\
\sum_{i=1}^N \left[ y_ix_i -\frac{1}{(1+e^{-w^Tx_i})}x_i \right]  &= 0 \\\
\sum_{i=1}^N \left[ \left(y_i-\sigma(w^Tx_i)\right)x_i \right] &= 0
\end{align*}
$$

Unlike linear regression with normally distributed residuals, it is not possible to find a closed-form expression for the coefficient values that maximize the likelihood function, so that an iterative process must be used instead.<cite>[^2]</cite>

We use stochastic gradient decent here:

$$
\begin{array}{l}
\text{SGD Procedure:} \\\
\text{\textcircled 1 }\text{Initialize } w \larr w_0, w_0 \in \text{any real num vector} \\\
\text{\textcircled 2 }\text{Randomly shuffle }(x_i,y_i) \text{ in }\mathcal{D} \\\
\text{\textcircled 3 }\text{For } i=1,\cdots,N \text{ do: } \\\
\quad w^{t+1} \larr w^{t} - \lambda \nabla_w L \\\
\text{Repeat \text{\textcircled 2}\text{\textcircled 3} until } w \text{ converge OR terminate after T loops} \\\
\text{\textcircled 4 }\text{Output result as }\hat{w}
\end{array}
$$

$$
\because \nabla_{w} L = \left(y_i-\sigma(w^Tx_i)\right)x_i \\\
\therefore w^{t+1} \larr w^{t} + \lambda \left(y_i-\frac{1}{1+\mathrm{e}^{-w^Tx}}\right)x_i
$$

## Conclusion

The pdf function of 2-classes logistic regression is:

$$
p(y|x) = {\frac{1}{1+\mathrm{e}^{-w^Tx}}}^y \cdot {\frac{\mathrm{e}^{-w^Tx}}{1+\mathrm{e}^{-w^Tx}}}^{1-y}
$$

Weight is updated by SGD:

$$
w^{t+1} \larr w^{t} + \lambda \left(y_i-\frac{1}{1+\mathrm{e}^{-w^Tx}}\right)x_i
$$

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=14).
[^2]: From [source](https://en.wikipedia.org/wiki/Logistic_regression).