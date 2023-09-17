---
author: "X. Wang"
title: "Linear Classification - Naive Bayes Classifier"
date: "2023-09-04"
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

## Definition

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\), where each data point \( (x_i,y_i) \) has attribution \(x_i \in \mathbb{R}^p \), \( y_i \in \{ 0,1 \} \). We define 2 classes C1 and C2:
</p>
{{</ math.inline >}}

$$
x_{C1} = \{ x_i|y_i=1 \} \\\
x_{C2} = \{ x_i|y_i=0 \} \\\
|x_{C1}| = N_1 \\\
|x_{C2}| = N_2 \\\
N = N_1+N_2
$$

{{< math.inline >}}
<p>
For each \(x_i\):
</p>
{{</ math.inline >}}

$$
x_i = \begin{bmatrix}
x_i^1 \\\
x_i^2 \\\
\vdots \\\
x_i^p
\end{bmatrix}
, x_i^j \implies \text{j-th element of } x_i
$$

{{< math.inline >}}
<p>
Naive bayes is the simplest graph model. By given a data \(x_i\), it assumes the conditional probability of \(x_i^j\) is independent from each other:
</p>
{{</ math.inline >}}

$$
x_i^{j_1} \perp x_i^{j_2} | y 
\implies
p(x_i|y) = \prod_{j=1}^p p(x_i^j|y)
$$

Motivation for naive bayes thought: <mark>simplify computation</mark>.

{{< math.inline >}}
<p>
Similar to <a href="https://tirmisula.github.io/posts/gaussian-discriminant-analysis/#definition">Gaussian discriminant analysis</a>, the object is to maximize joint probability \( p(x,y) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{y} &= \argmax_y p(y|x) \\\
&= \argmax p(x|y)p(y)
\end{align*}
$$

{{< math.inline >}}
<p>
We assume \(p(y)\) subject to Bernoulli distribution if it is binary class, subject to Categorical distribution if it has k classes:
</p>
{{</ math.inline >}}

$$
\text{ 2 classes} \implies
\begin{cases}
y \sim \mathcal{Bernoulli}(\phi) \\\
p(y) = \phi^y (1-\phi)^{1-y}
\end{cases}
$$

$$
\text{ k classes} \implies
\begin{cases}
y \in \lbrace 1,2,\cdots,k \rbrace \\\
y \sim \mathcal{Cat}(k,\phi_1,\cdots,\phi_k) \\\
p(y) = \prod_{i=1}^k \phi_i^{[y=i]}, \text{ where } [y=i]=1 \text{ if }y=i \text{ otherwise } 0 \\\
\sum_{i=1}^k \phi_i = 1
\end{cases}
$$

{{< math.inline >}}
<p>
We assume \(p(x_i^j|y)\) subject to Categorical distribution if it \(x\) is discrete with \(m\) categories, subject to Gaussian distribution if \(x\) is continous:
</p>
{{</ math.inline >}}

$$
x \text{ is discrete} \implies
\begin{cases}
x_i^j \in \lbrace 1,2,\cdots,m \rbrace \\\
x_i^j|y=c \sim \mathcal{Cat}(m,p_{j1c},\cdots,p_{jmc}) \\\
p(x_i^j|y=c) = \prod_{ii=1}^m p_{jiic}^{[x_i^j=ii]}, \text{ where } [x_i^j=ii]=1 \text{ if }x_i^j=ii \text{ otherwise } 0 \\\
\sum_{i=1}^m p_{jiic} = 1
\end{cases}
$$

$$
x \text{ is continous} \implies
\begin{cases}
x_i^j|y=c \sim \mathcal{N}(\mu_{jc}, \sigma_{jc}) \\\
p(x_i^j|y=c) = \frac{1}{\sqrt{2\pi} \sigma_{jc}} \mathrm{e}^{-\frac{(x_i^j - \mu_{jc})^2}{2\sigma_{jc}^2}}
\end{cases}
$$

{{< math.inline >}}
<p>
For simplification,  we consider prior is Bernoulli model and likelihood is Categorical model, and the log-likelihood function is :
</p>
{{</ math.inline >}}

$$
\theta=(\phi, \underset{m\times p}{\lbrace p_{jii0, \cdots} \rbrace}, \underset{m\times p}{\lbrace p_{jii1, \cdots} \rbrace})
$$

$$
\begin{align*}
L(\theta) &= \log\left(\prod_{i=1}^N p(x_i|y_i)p(y_i)\right) \\\
&= \log\left(\prod_{i=1}^N \left(\prod_{j=1}^p p(x_i^j|y_i)\right) p(y_i)\right) \\\
&= \sum_{i=1}^N \sum_{j=1}^p \log(p(x_i^j|y_i)) + \sum_{i=1}^N \log(p(y_i)) \\\
&= \sum_{i=1}^N \sum_{j=1}^p \log\left(  (\prod_{ii=1}^m p_{jiiy_i}^{[x_i^j=ii]})^{y_i} (\prod_{ii=1}^m p_{jiiy_i}^{[x_i^j=ii]})^{1-y_i} \right) + \sum_{i=1}^N \log\left(\phi^{y_i} (1-\phi)^{1-y_i}\right) \\\
&= \sum_{i=1}^N \sum_{j=1}^p \left[y_i\log\left( \prod_{ii=1}^m p_{jiiy_i}^{[x_i^j=ii]} \right) + (1-y_i)\log\left( \prod_{ii=1}^m p_{jiiy_i}^{[x_i^j=ii]} \right)\right]
+
\sum_{i=1}^N \left[ y_i\log\phi + (1-y_i)\log(1-\phi) \right]\\\
&= \sum_{i=1}^N \sum_{j=1}^p \left[y_i\sum_{ii=1}^m[x_i^j=ii]\log(p_{jiiy_i}) + (1-y_i)\sum_{ii=1}^m[x_i^j=ii]\log(p_{jiiy_i})\right]
+
\sum_{i=1}^N \left[ y_i\log\phi + (1-y_i)\log(1-\phi) \right]\\\
&= \sum_{i=1}^N \sum_{j=1}^p \left[y_i\log(p_{jx_i^jy_i}) + (1-y_i)\log(p_{jx_i^jy_i})\right]
+
\sum_{i=1}^N \left[ y_i\log\phi + (1-y_i)\log(1-\phi) \right]\\\
\end{align*}
$$

{{< math.inline >}}
<p>
By adding Lagrange multiplier, we get:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{\theta} &= \argmax_{\theta} \left[L(\theta) + \sum_{j=1}^p\alpha_{1j}(1-\sum_{ii=1}^m p_{jii0}) + \sum_{j=1}^p\alpha_{2j}(1-\sum_{ii=1}^m p_{jii1})\right]
\end{align*}
$$

## Solve parameters

{{< math.inline >}}
<p>
We solve \(\phi\) first:<cite>[^1]</cite>
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial \phi}\left[L(\theta) + \sum_{j=1}^p\alpha_{1j}(1-\sum_{ii=1}^m p_{jii0}) + \sum_{j=1}^p\alpha_{2j}(1-\sum_{ii=1}^m p_{jii1})\right] &= 0 \\\
\frac{\partial}{\partial \phi} \sum_{i=1}^Ny_i\log\phi + (1-y_i)\log(1-\phi) &= 0 \\\
\sum_{i=1}^N \left[ y_i\frac{1}{\phi} - (1-y_i)\frac{1}{1-\phi} \right] &= 0 \\\
\sum_{i=1}^N \left[ y_i(1-\phi) - (1-y_i)\phi \right] &= 0 \\\
\sum_{i=1}^N \left[ y_i-\phi \right] &= 0 \\\
\sum_{i=1}^N y_i - N\phi &= 0 \\\
\phi &= \frac{1}{N}\sum_{i=1}^N y_i \\\
\phi &= \frac{N_1}{N}
\end{align*}
$$

{{< math.inline >}}
<p>
Then we solve \(\p_{jii1}\):<cite>[^2]</cite>
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial p_{jii1}} \left[L(\theta) + \sum_{j=1}^p\alpha_{1j}(1-\sum_{ii=1}^m p_{jii1}) + \sum_{j=1}^p\alpha_{2j}(1-\sum_{ii=1}^m p_{jii0})\right] &= 0 \\\
\frac{\partial}{\partial p_{jii1}} \left[\sum_{i=1}^N \sum_{j=1}^p \left[y_i\log(p_{jx_i^j1}) \right] + \sum_{j=1}^p\alpha_{1j}(1-\sum_{ii=1}^m p_{jii1}) \right] &= 0 \\\
\frac{\partial}{\partial p_{jii1}} \left[\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} \left[y_i\log(p_{jii1}) \right] + \alpha_{1j}(1- p_{jii1}) \right] &= 0 \\\
\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} y_i\frac{1}{p_{jii1}} - \alpha_{1j} &= 0 \\\
p_{jii1} &= \frac{\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} y_i}{\alpha_{1j}}
\end{align*}
$$

$$
\begin{align*}
\because \sum_{ii=1}^mp_{jii1} &= 1 \\\
\sum_{ii=1}^m\frac{\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} y_i}{\alpha_{1j}} &= 1 \\\
\frac{\sum_{i=1}^N y_i}{\alpha_{1j}} &= 1 \\\
\alpha_{1j} &= \sum_{i=1}^N y_i \\\
\therefore p_{jii1} &= \frac{\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} y_i}{\sum_{i=1}^N y_i}
\end{align*}
$$

{{< math.inline >}}
<p>
The same operation for \(p_{jii0}\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial p_{jii0}} \left[L(\theta) + \sum_{j=1}^p\alpha_{1j}(1-\sum_{ii=1}^m p_{jii1}) + \sum_{j=1}^p\alpha_{2j}(1-\sum_{ii=1}^m p_{jii0})\right] &= 0 \\\
\frac{\partial}{\partial p_{jii0}} \left[\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} \left[(1-y_i)\log(p_{jii0}) \right] + \alpha_{2j}(1- p_{jii0}) \right] &= 0 \\\
\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} \frac{1}{p_{jii0}}-y_i\frac{1}{p_{jii0}} - \alpha_{2j} &= 0 \\\
p_{jii0} &= \frac{\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} (1-y_i)}{\alpha_{2j}}
\end{align*}
$$

$$
\begin{align*}
\because \sum_{ii=1}^mp_{jii0} &= 1 \\\
\sum_{ii=1}^m\frac{\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} (1-y_i)}{\alpha_{2j}} &= 1 \\\
\frac{\sum_{i=1}^N (1-y_i)}{\alpha_{2j}} &= 1 \\\
\alpha_{2j} &= \sum_{i=1}^N (1-y_i) \\\
\therefore p_{jii0} &= \frac{\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} (1-y_i)}{\sum_{i=1}^N (1-y_i)}
\end{align*}
$$

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

$$
\begin{align*}
\hat{\phi} &= \frac{N_1}{N} \\\
p_{jii1} &= \frac{\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} y_i}{\sum_{i=1}^N y_i} \\\
p_{jii0} &= \frac{\sum_{\forall i \in \lbrace x_i^j = ii \rbrace} (1-y_i)}{\sum_{i=1}^N (1-y_i)}
\end{align*}
$$

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=21).
[^2]: From [source](https://zhuanlan.zhihu.com/p/71960086).