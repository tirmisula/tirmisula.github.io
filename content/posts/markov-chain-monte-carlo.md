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

## Markov chain
### Description
Given preset:

$$
\begin{align*}
&\lbrace x_t \rbrace : \text{discrete time series} \\\
&S_i : \text{state i} \\\
&\pi_t(x) : \text{probability distribution of state at time $t$} \\\
&\sum_j \pi_t(x=S_j) = 1
\end{align*}
$$

The first order Markov chain is described as:

$$
\begin{align*}
p(x_{t+1}|x_1,x_2,\cdots,x_t) = p(x_{t+1}|x_t)
\end{align*}
$$

{{< math.inline >}}
<p>
The state of \(x_{t+1}\) is only relevant to the state of last time \(x_t\)
</p>
{{</ math.inline >}}

### State transition matrix

{{< math.inline >}}
<p>
The state transition probability can be described as a transition matrix \(P_{ij}\) and has following attribute:
</p>
{{</ math.inline >}}

$$
\begin{align*}
P_{ij} &= p(x_{t+1}=S_j|x_t=S_i) \\\
\sum_j P_{ij} &= 1
\end{align*}
$$

{{< math.inline >}}
<p>
Then we can easily derive the distribution \(\pi_{t+1}\) from \(\pi_t\) by transition matrix:
</p>
{{</ math.inline >}}

$$
\pi_{t+1}(x=S_{j}) = \sum_i \pi_t(x=S_i)P_{ij} \\\
\text{for any state $\ast$} \\\
\pi_{t+1}(x=S_{\ast}) = \sum_i \pi_t(x=S_i)P_{i\ast}
$$

### Stationary distribution and detailed balance

{{< math.inline >}}
<p>
We call \( \pi(x) \) a <mark>stationary distribution</mark> if \( \pi_t(x) \) and \( \pi_{t+1}(x) \) are indentically equal after transition:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\pi_{t+1}(x=S_{\ast}) = \pi_{t}(x=S_{\ast}) &= \sum_i \pi_t(x=S_i)P_{i\ast} \\\
&\dArr \\\
\pi(x=S_{\ast}) &= \sum_i \pi(x=S_i)P_{i\ast}
\end{align*}
$$

{{< math.inline >}}
<p>
If the chain is reversible w.r.t. \(\pi(x)\), we call it <mark>detailed balance</mark> equation:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\pi(x=S_i)P_{ij} = \pi(x=S_j)P_{ji}, \forall i,j
\end{align*}
$$

{{< math.inline >}}
<p>
Next we can prove that detailed balance is sufficient uneccessary condition for stationary distribution:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{detailed balance} &\implies \text{stationary distribution} \\\
&\dArr\\\
\text{Let } j&=\ast, \text{we sum all over $j$} \\\
\sum_j \pi(x=S_i)P_{ij} &= \sum_j \pi(x=S_{j})P_{ji} \\\
\pi(x=S_i)\sum_j P_{ij} &= \sum_j \pi(x=S_{j})P_{ji} \\\
\pi(x=S_i) &= \sum_j \pi(x=S_{j})P_{ji} \\\
\text{Let } i&=\ast \\\
\pi(x=S_{\ast}) &= \sum_{j}\pi(x=S_j)P_{j\ast} \\\
\text{Q.E.D.}
\end{align*}
$$

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
