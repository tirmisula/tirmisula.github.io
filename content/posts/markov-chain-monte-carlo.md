---
author: "X. Wang"
title: "Markov chain and Monte Carlo"
date: "2023-09-11"
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

### Monte Carlo in brief

<cite>[^1]</cite>

$$
\begin{align*}
p(z|x) \rarr E_{z|x\sim p(z|x)}\left[ f(z) \right] &= \int_{z}f(z)p(z|x)dz \\\
&\approx \frac{1}{N} \sum_{i=1}^N f(z_i) \\\
z_1,\cdots,z_N&\text{ are randomly sampled from } p(z|x)
\end{align*}
$$

There are serveral ways to do sampling.

### Sampling from cdf

{{< math.inline >}}
<p>
Suppose we have cumulative density function of \(p(z|x)\), CDF:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{CDF}(t) &= \int_{-\infty}^{t}p(z|x)dz \\\
&t\in [-\infty,\infty] \\\
&\text{CDF}(t)\in [0,1]
\end{align*}
$$

{{< math.inline >}}
<p>
And we have \( u_1,\cdots,u_N \) sampled from uniform distribution \( \mathcal{U}(0,1) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
u_1,\cdots,u_N\sim \mathcal{U}(0,1)
\end{align*}
$$

{{< math.inline >}}
<p>
\(z_i\) is given by the inverse of cdf:
</p>
{{</ math.inline >}}

$$
\begin{align*}
&\forall 1,\cdots,N \\\
&z_i = \text{CDF}^{-1}(u_i) \\\
\end{align*}
$$

{{< math.inline >}}
<p>
Apparently CDF may not be solvable for most of complex \(p(z|x)\).
</p>
{{</ math.inline >}}

### Rejection sampling

{{< math.inline >}}
<p>
We purpose a pdf \(q(z)\) to approximate \(p(z|x)\), define acceptance rate \(\alpha\):
</p>
{{</ math.inline >}}

$$
\alpha = \frac{p(z|x)}{Mq(z)}
$$

Then the sampling procedures are:

$$
\begin{align*}
&\forall i=1,\cdots,N \\\
&z_i \sim q(z) \\\
&u_i \sim \mathcal{U}(0,1) \\\
&\begin{cases}
\text{If } u\leq \alpha, \text{accept $z_i$} \\\
\text{Else }, \text{reject $z_i$} 
\end{cases}
\end{align*}
$$

{{< math.inline >}}
<p>
The core idea is the larger difference between \(p(z_i|x)\) and \(q(z_i)\) the more likely \(z_i\) will be rejected. The problem is finding such \(q(z)\) is difficult.
</p>
{{</ math.inline >}}

### Importance sampling

$$
\begin{align*}
E_{z|x\sim p(z|x)}\left[ f(z) \right] &= \int_{z}f(z)p(z|x)dz \\\
&= \int_{z}f(z)\frac{p(z|x)}{q(z)}q(z)dz \\\
&\approx \frac{1}{N} \sum_{i=1}^N f(z_i)\frac{p(z_i|x)}{q(z_i)}, \frac{p(z_i|x)}{q(z_i)}\text{ is weight} \\\
z_1,\cdots,z_N&\text{ are randomly sampled from } q(z)
\end{align*}
$$

{{< math.inline >}}
<p>
It works when \( p(z|x) \) and \( q(z) \) are close, if the weight is not close to 1 we need to do resampling based on the importance sampling result by weights which is called Importance-Sampling-Resampling.
</p>
{{</ math.inline >}}

## Mean Field VI Derivation

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
Finally we can conclude that solving posterior can be done by finding approximate function \(q(z)\) using coordinate-ascend-like method and iterates a number of times:
</p>
{{</ math.inline >}}

$$
\text{Parameter's update order is:} \\\
q_1^{(1)} \rarr q_2^{(1)} \rarr \cdots \rarr q_N^{(1)} \rarr q_1^{(2)} \rarr \cdots q_N^{(2)} \rarr \cdots \rarr q_N^{(t)} \cdots \rarr q_N^{(t)} \\\
\text{It satisfies: } \\\
|L(q^{(t)})-L(q^{(t-1)})| < \epsilon \\\
\text{The approximated posterior is :} \\\
p(z|x) = \prod_{i=1}^N q_i^{(t)}(z_i)
$$

The drawbacks are:

1. Intergration part w.r.t all of q is still intractable.
2. Meanfield assumption is not suitable for complicated z, e.g. z is a nerual network.


## Stochastic gradient VI

{{< math.inline >}}
<p>
Given that \(q(z)\) is a function of \(\phi\) and \(p(x,z)\) is a function of \(\theta\), we have:
</p>
{{</ math.inline >}}

$$
q(z) = f(\phi) \\\
p(x,z) = g(\theta), \phi\notin\theta \\\
\dArr \\\
z \sim q_{\phi}(z) \\\
(x,z) \sim p_{\theta}(x,z) \\\
\dArr \\\
\argmax_{q(z)} L(q) = \argmax_{\phi} L(\phi) \\\
L(\phi) = \int_{z}q_{\phi}(z)\log\frac{p_{\theta}(x,z)}{q_{\phi}(z)}\space dz \\\
$$

{{< math.inline >}}
<p>
Next, we compute the gradient of \(L(\phi)\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\nabla_{\phi}L(\phi) &= \nabla_{\phi}\int_{z}q_{\phi}(z)\log\frac{p_{\theta}(x,z)}{q_{\phi}(z)}\space dz \\\
&= \int_{z}\nabla_{\phi}\space \left(q_{\phi}(z)\left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right]\right)dz \\\
&= \int_{z}\nabla_{\phi}q_{\phi}(z)\left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right]+q_{\phi}(z)\nabla_{\phi}\left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right] dz \\\
&= \underset{Part1}{\int_{z}\nabla_{\phi}q_{\phi}(z)\left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right]dz}+\underset{Part2}{\int_{z}q_{\phi}(z)\nabla_{\phi}\left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right] dz} \\\
&\dArr \\\
\text{Part1} &= \int_{z}\nabla_{\phi}q_{\phi}(z)\cdot\frac{1}{q_{\phi}(z)}q_{\phi}(z) \left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right]dz \\\
&= \int_{z}\left(\nabla_{\phi}\log q_{\phi}(z)\right)\cdot q_{\phi}(z)\left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right]dz \\\
&= E_{z\sim q_{\phi}(z)} \left[ \left(\nabla_{\phi}\log q_{\phi}(z)\right) \left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right] \right] \\\
&\dArr \\\
\text{Part2} &= -\int_{z}q_{\phi}(z)\nabla_{\phi} \log q_{\phi}(z) dz \\\
&= -\int_{z}q_{\phi}(z)\frac{1}{ q_{\phi}(z)}\nabla q_{\phi}(z) dz \\\
&= -\nabla_{\phi}\int_{z}q_{\phi}(z)dz \\\
&= 0 \\\
&\dArr \\\
\nabla_{\phi}L(\phi) &= E_{z\sim q_{\phi}(z)} \left[ \left(\nabla_{\phi}\log q_{\phi}(z)\right) \left[ \log p_{\theta}(x,z) - \log q_{\phi}(z) \right] \right]
\end{align*}
$$

{{< math.inline >}}
<p>
\( \nabla_{\phi}L(\phi) \) has no analytic expression, thus we can use MC sampling to approximate the expectation:
</p>
{{</ math.inline >}}

$$
\text{for } i=1,2,\cdots,L \\\
\nabla_{\phi}L(\phi) \approx \frac{1}{L}\sum_{i=1}^L \left(\nabla_{\phi}\log q_{\phi}(z_i)\right) \left[ \log p_{\theta}(x_i,z_i) - \log q_{\phi}(z_i) \right]
$$

{{< math.inline >}}
<p>
When sampling \(q(z)\) from \( [0,1] \), \( \log q(z) \) varies greatly from \( [-\infty,0] \), it would cause significant error in approximating gradient and cause even larger error in SGD. That is the drawback of SGVI we need to overcome.
</p>
{{</ math.inline >}}

## Reparameterization trick for SGVI

{{< math.inline >}}
<p>
Assuming that \(z\) is a function of random variable \(\epsilon\), we have:
</p>
{{</ math.inline >}}

$$
z = g(\epsilon) \\\
\epsilon \sim p(\epsilon) \\\
z \sim q_{\phi}(z) \\\
\dArr \\\
\int q_{\phi}(z)dz=\int p(\epsilon)d\epsilon=1 \\\
$$

According to <mark>Law of the unconscious statistician</mark>(LOTUS)<cite>[^4]</cite>, we have:

$$
\begin{cases}
E[g(\epsilon)] = \int g(\epsilon)p_{\epsilon}(\epsilon)d\epsilon \\\
E[z] = \int zp_z(z)dz
\end{cases} \overset{\forall\space f(\cdot)}{\implies}
\begin{cases}
E[f(g(\epsilon))] = \int f(g(\epsilon))p_{\epsilon}(\epsilon)d\epsilon \\\
E[f(z)] = \int f(z)p_z(z)dz
\end{cases} \\\
\dArr \\\
\int f(g(\epsilon))p_{\epsilon}(\epsilon)d\epsilon = \int f(z)p_z(z)dz \\\
\dArr \\\
\begin{align*}
\nabla_{\phi}L(\phi) &= \nabla_{\phi}\int_{z}q_{\phi}(z)\underset{\color{red}{f(z)}}{\log\frac{p_{\theta}(x,z)}{q_{\phi}(z)}}\space dz \\\
&= \nabla_{\phi}\int_{\epsilon}p(\epsilon) \left[\log p_{\theta}(x,z)-\log q_{\phi}(z)\right] \space d\epsilon \\\
&= \int_{\epsilon}\nabla_{\phi}\left(p(\epsilon) \left[\log p_{\theta}(x,z)-\log q_{\phi}(z)\right]\right) \space d\epsilon \\\
&= \int_{\epsilon}p(\epsilon) \nabla_{\phi}\left( \left[\log p_{\theta}(x,z)-\log q_{\phi}(z)\right]\right) \space d\epsilon, \space\because p(\epsilon)\perp \phi \\\
&= \int_{\epsilon}p(\epsilon) \nabla_{g(\epsilon)}\left( \left[\log p_{\theta}(x,g(\epsilon))-\log q_{\phi}(g(\epsilon))\right]\right)\cdot \nabla_{\phi}g(\epsilon) \space d\epsilon \\\
&= E_{\epsilon\sim p(\epsilon)} \left[ \nabla_{g(\epsilon)}\left( \left[\log p_{\theta}(x,g(\epsilon))-\log q_{\phi}(g(\epsilon))\right]\right)\cdot \nabla_{\phi}g(\epsilon) \right]
\end{align*}
$$

{{< math.inline >}}
<p>
Then we can conclude the whole method:
</p>
{{</ math.inline >}}

MC sampling: 

$$
\text{for } i=1,2,\cdots,L \\\
\nabla_{\phi}L(\phi) \approx \frac{1}{L}\sum_{i=1}^L \nabla_{g(\epsilon_i)}\left( \left[\log p_{\theta}(x,g(\epsilon_i))-\log q_{\phi}(g(\epsilon_i))\right]\right)\cdot \nabla_{\phi}g(\epsilon_i)
$$

SGVI:

$$
\phi^{(t+1)} \larr \phi^{(t)} + \lambda^{(t)}\nabla_{\phi}L(\phi^{(t)})
$$

## Supplementation

Some common transformation in reparamerterization trick:

![homeinfo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmoAAAHLCAIAAAD3GCP+AAAAA3NCSVQICAjb4U/gAAAAEHRFWHRTb2Z0d2FyZQBTaHV0dGVyY4LQCQAAIABJREFUeNrsnX9cE1fW/48a681jWiYrXSdbuibVron6LIl1C0H7NcZ1hbitgI9VQltDUwUaLdLYAoUqWlSwKGJB0IKi+4CiS8B2QawLxrXyY1v58diSuCqJqzbjis1QcTMu0Xz/4Ff4KShaxPt+8YdmJjfJPeeez7ln7twZ4XA4AIPBYDAYzEAYibsAg8FgMBgsnxgMBoPBYPnEYDAYDAbLJwaDwWAwWD4xGAwGg8HyicFgMBgMBssnBoPBYDBYPjEYDAaDwfKJwWAwGAyWTwwGg8FgsHxiMBgMBoPpGVYfxyiKwh2EwWAwmCcWkiR7OzQCbxmPwWAwGMxAwcVbDAaDwWCwfGIwGAwGg+UTg8FgMBgsnxgMBoPBYPnEYDAYDAbLJwaDwWAwmIcln8zhQPaI7rAX/Y+STQCAQCAQinwQCgUAgEPkkEAgEAmECMlyNMcuy5AURCAQC4YGFpofcxWq4zFsCgUAgEAiDQoK3BAKBQCAQ+SQQCAQCgcgngUAgEAhEPgkEAoFAIPJJIBAIBAKRTwKBQCAQCEQ+CQQCgUAg8kkgEAgEApFPAoFAIBCIfBIIBAKB8KDw/wFJuim74GF+VgAAAABJRU5ErkJggg==
)

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=69).
[^3]: From [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
[^2]: From [Mean field variational inference](https://mbernste.github.io/files/notes/MeanFieldVariationalInference.pdf).
[^4]: From [Ross, Sheldon M. (2019). Introduction to probability models](https://doi.org/10.1016%2FC2017-0-01324-1).
