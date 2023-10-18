---
author: "X. Wang"
title: "Varational Inference"
date: "2023-09-10"
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

### Bayesian Inference

<cite>[^1]</cite>

$$
\begin{align*}
p(\theta|x) &= \frac{p(x|\theta)p(\theta)}{p(x)} \\\
&= \frac{p(x|\theta)p(\theta)}{\int_{\theta}p(x|\theta)p(\theta)d\theta}
\end{align*}
$$

{{< math.inline >}}
<p>
Bayesian inference is the procedure to solve \(p(\theta|x)\) when posterior cannot be solved by Bayes formula from above.
</p>
{{</ math.inline >}}

### Bayesian Decision

$$
\begin{align*}
x &: \text{observed data} \\\
\tilde{x} &: \text{new coming data} \\\
\text{Bayesian Decision} &: \text{find } p(\tilde{x}|x) \text{ by inference } x\rarr \theta\rarr \tilde{x} \\\
&\dArr \\\
p(\tilde{x}|x) &= \int_{\theta}p(\tilde{x},\theta|x)d\theta \\\
&= \int_{\theta}p(\tilde{x}|\theta,x)p(\theta|x)\space d\theta \\\
\space&\text{$p(\theta|x)$ }\text{found by bayesian inference} \\\ 
&= \int_{\theta}p(\tilde{x}|\theta)p(\theta|x)\space d\theta,\implies\text{$\tilde{x}\perp x$ when $\theta$ fixed}\\\
&= E_{\theta\sim p(\theta|x)}[p(\tilde{x}|\theta)]
\end{align*}
$$


## VI Derivation

Given preset:

$$
\begin{align*}
x &: \text{observed data} \\\
z&: \text{latent variable $+$ parameters} \\\
(x,z) &: \text{complete data}
\end{align*}
$$

We mentioned in [EM algorithm](https://tirmisula.github.io/posts/expectation-maximization/#derive-em-from-elbokl-divergence) before, Baed on the fact that log-likelihood can be written as ELBO+KL-Divergence:

$$
\begin{align*}
\log p(x) &= \log p(x,z) -\log p(z|x) \\\
&= \log p(x,z)-\log q(z) - \log p(z|x)+\log q(z) \\\
&= \log\frac{p(x,z)}{q(z)} - \log\frac{p(z|x)}{q(z)} \\\
&\dArr\text{integrate on both sides} \\\
\int_{z}q(z)\log p(x)\space dz &= \int_{z}q(z)\log\frac{p(x,z)}{q(z)}\space dz - \int_{z}q(z)\log\frac{p(z|x)}{q(z)}\space dz \\\
\log p(x)\int_{z}q(z)\space dz &= \int_{z}q(z)\log\frac{p(x,z)}{q(z)}\space dz + KL(q(z)||p(z|x)) \\\
\log p(x) &= \text{ELBO} + KL(q(z)||p(z|x)) \\\
&\text{Let } L(q) = \text{ELBO} = \int_{z}q(z)\log\frac{p(x,z)}{q(z)}\space dz \\\
\log p(x) &= L(q) + KL(q(z)||p(z|x)) \\\
\end{align*}
$$

{{< math.inline >}}
<p>
We want \(q(z)\) approximate to \(p(z|x)\). The problem transformation is:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{Let } \log p(x) \text{ be fixed}: \\\
\text{finding } q(z)\approx p(z|x) &\iff KL(q(z)||p(z|x))\approx 0 \\\
&\iff \argmax_{q(z)} L(q)
\end{align*}
$$

{{< math.inline >}}
<p>
For \(q(z)\), we introduced <b>mean field theory</b> to represents it:
</p>
{{</ math.inline >}}

$$
q(z) = \prod_{i=1}^N q_i(z_i) \\\
\text{$q_i,z_i$ are independent individuals} \\\
\dArr \\\
\text{Each time solve $q_j$ while fix $q_1,\cdots,q_{j-1},q_{j+1},\cdots,q_{N}$}
$$

{{< math.inline >}}
<p>
Then we can rewrite \(L(q)\) as:
</p>
{{</ math.inline >}}

$$
\begin{align*}
L(q) &= \underset{\text{Part1}}{\int_{z}q(z)\log p(x,z)\space dz} - \underset{\text{Part2}}{\int_{z}q(z)\log q(z)\space dz} \\\
&\dArr \\\
\text{Part1} &= \int_{z1}\cdots\int_{z_N}\prod_{i=1}^Nq_i(z_i)\log p(x,z)\space dz_1\cdots dz_n \\\
&= \int_{z_j}q_j(z_j) \int_{z_1,\cdots z_{j-1},z_{j+1},\cdots,z_N}\prod_{i=1,i\neq j}^Nq_i(z_i)\log p(x,z) dz_1\cdots d_{j-1}d_{j+1}\cdots d_N \space dz_j \\\
&= \int_{z_j}q_j(z_j) E_{z_{i\neq j}\sim q_{i\neq j}(z_{i\neq j})}\left[\log p(x,z)\right] dz_j \\\
&\text{Define a new pdf function: } \tilde{p}(x,z_j) \\\
&\text{Let } \log \tilde{p}(x,z_j)=E_{z_{i\neq j}\sim q_{i\neq j}(z_{i\neq j})}\left[\log p(x,z)\right]+K, \text{other $z_i$ are eliminated in intergration} \\\
&\text{note that $K$ is a normailize constant controls integration of $\tilde{p}(x,z_j)$ equals to 1} \\\
&= \int_{z_j}q_j(z_j)\left(\log\tilde{p}(x,z_j)-K\right)\space dz_j \\\
&= \int_{z_j}q_j(z_j)\log\tilde{p}(x,z_j)\space dz_j - \int_{z_j}q_j(z_j)Kdz_j \\\
&= \int_{z_j}q_j(z_j)\log\tilde{p}(x,z_j)\space dz_j + K^{'} \\\
\text{Part2} &= \int_{z1}\cdots\int_{z_N}\prod_{i=1}^Nq_i(z_i) \sum_{i=1}^N\log q_i(z_i) \space dz_1\cdots dz_n \\\
&= \sum_{i=1}^N \left[\int_{z1}\cdots\int_{z_N}\prod_{n=1}^Nq_n(z_n) \log q_i(z_i) \space dz_1\cdots dz_n\right] \\\
&= \sum_{i=1}^N \left(\int_{z_i}q_i(z_i)\log q_i(z_i)dz_i\right)\prod_{n=1,n\neq i}^N\int_{z_n}q_n(z_n)dz_n \\\
&= \sum_{i=1}^N\int_{z_i}q_i(z_i)\log q_i(z_i)dz_i \\\
&= \int_{z_j}q_j(z_j)\log q_j(z_j) \space dz_j + \sum_{i=1,i\neq j}^N\int_{z_i}q_i(z_i)\log q_i(z_i) \space dz_i \\\
&= \int_{z_j}q_j(z_j)\log q_j(z_j) \space dz_j + C \\\
&\dArr \\\
L(q) &= \text{Part1}-\text{Part2} \\\
&= \int_{z_j}q_j(z_j)\log\frac{\tilde{p}(x,z_j)}{q_j(z_j)}\space dz_j + K^{'}-C \\\
&= -KL(q_j(z_j)||\tilde{p}(x,z_j)) + const
\end{align*}
$$

So the maximize problem becomes:<cite>[^2]</cite>

$$
\begin{align*}
\argmax_{q(z)} L(q) &= \argmax_{q(z)} -KL(q_j(z_j)||\tilde{p}(x,z_j)) + const \\\
&= \argmax_{q(z)} -KL(q_j(z_j)||\tilde{p}(x,z_j)) \\\
&\implies q_j(z_j) = \tilde{p}(x,z_j), \forall\space j=1,\cdots,N \\\
&\implies \log q_j(z_j) =E_{z_{i\neq j}\sim q_{i\neq j}(z_{i\neq j})}\left[\log p(x,z)\right]-K \\\
&\implies q_j(z_j) = \exp(E_{z_{i\neq j}\sim q_{i\neq j}(z_{i\neq j})}\left[\log p(x,z)\right])\exp(-K), \forall\space j=1,\cdots,N \\\
&\implies q_j(z_j) = \frac{\exp(E_{z_{i\neq j}\sim q_{i\neq j}(z_{i\neq j})}\left[\log p(x,z_{i\neq j},z_j)\right])}{\int\exp(E_{z_{i\neq j}\sim q_{i\neq j}(z_{i\neq j})}\left[\log p(x,z_{i\neq j},z_j)\right])dz_j}, \forall\space j=1,\cdots,N
\end{align*}
$$

{{< math.inline >}}
<p>
Finally we can conclude that solving posterior can be done by finding approximate function \(q(z)\) using coordinate-ascent-like method and iterates a number of times:
</p>
{{</ math.inline >}}

$$
\text{Parameter's update order is:} \\\
q_1^{(1)} \rarr q_2^{(1)} \rarr \cdots \rarr q_N^{(1)} \rarr q_1^{(2)} \rarr \cdots q_N^{(2)} \rarr \cdots \rarr q_N^{(t)} \cdots \rarr q_N^{(t)} \\\
\text{It satisfies: } \\\
|L(q^{(t)})-L(q^{(t-1)})| < \epsilon \\\
\text{The approximate posterior is :} \\\
p(z|x) = \prod_{i=1}^N q_i^{(t)}(z_i)
$$

## Conclusion



## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=69).
[^3]: From [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
[^2]: From [Mean field variational inference](https://mbernste.github.io/files/notes/MeanFieldVariationalInference.pdf).