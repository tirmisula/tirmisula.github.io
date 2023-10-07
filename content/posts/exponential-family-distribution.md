---
author: "X. Wang"
title: "Exponential Family Distribution"
date: "2023-09-07"
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
### Examples of exponential family distributions
Some of the well-known exponential family distributions are:

$$
\text{Exponential family: }
\begin{cases}
\text{Gaussian} \\\
\text{Bernoulli} \\\
\text{Binomial} \\\
\text{Poisson} \\\
\text{Beta} \\\
\text{Dirichlet} \\\
\text{Gamma}
\end{cases}
$$

### Application areas of exponential famlily distributions

$$
\text{Application areas: }
\begin{cases}
\text{Generalize linear model}\begin{cases}
\text{linear combination: } w^Tx \\\
\text{link funciton: } (\text{ReLU})^{-1} \\\
\text{linear regression: } y|x\sim \mathcal{N}(\mu,\Sigma) \\\
\text{calssification: } y|x\sim \mathcal{Bernoulli(\phi)}
\end{cases}\\\
\text{Graph model: undirected graph} \rarr \text{restricted boltzman machine(RBM)} \\\
\text{Variational inference} \rarr \text{simplification}
\end{cases}
$$

## Exponentail family distribution introduction
### Definition of exponential family distribution

Exponential family distribution has the following form:

$$
p(x|\theta) = h(x)\exp(\theta^T\phi(x)-A(\theta)) \\\
x\in\mathbb{R}^p \\\
\theta: \text{parameters} \\\
A(\theta): \text{log-partition funciton} \\\
\phi(x): \text{sufficient statistic} \\\
h(x)=1, \text{ if not important}
$$

The source of log partition function comes from:

$$
\begin{align*}
p(x|\theta) &= h(x)\exp(\theta^T\phi(x)-A(\theta)) \\\
&=\frac{1}{\exp(A(\theta))}h(x)\exp(\theta^T\phi(x))
\end{align*}
$$

$$
\begin{align*}
\because \int p(x|\theta) \space dx &= \int \frac{1}{\exp(A(\theta))}h(x)\exp(\theta^T\phi(x)) \space dx = 1 \\\
\therefore \int \frac{1}{\exp(A(\theta))}h(x)\exp(\theta^T\phi(x)) \space dx &= 1 \\\
\frac{1}{\exp(A(\theta))} \int h(x)\exp(\theta^T\phi(x)) \space dx &= 1 \\\
\exp(A(\theta)) &= \int h(x)\exp(\theta^T\phi(x)) \space dx
\end{align*}
$$

{{< math.inline >}}
<p>
\( \exp(A(\theta)) \) is called partition function because it normalize integration to 1, thus \( A(\theta) \) is called log partition function.
</p>
{{</ math.inline >}}

### Feature of exponential family distribution

$$
\text{Features: }
\begin{cases}
\phi(x) \text{ compress data} \implies \text{online learning} \\\
\text{exponential likelihood} \implies \text{conjugate prior and posterier} \\\
\text{maximize entropy} \implies \text{zero knowledge of prior}
\end{cases}
$$

#### Conjugate distribution

In bayesian statistics, it is difficult to find posterier distribution because the integration part is difficult to solve:

$$
\begin{align*}
p(\theta|x) &= \frac{p(x|\theta)p(\theta)}{\int p(x|\theta)p(\theta) \space d\theta} \\\
p(\theta|x) &\propto p(x|\theta)p(\theta)
\end{align*}
$$

The advantage of exponential family distribution is that exponential-likelihood's <mark>conjugate prior</mark> is often also in exponential family which brings posterier an easy solution, e.g. 

$$
\underset{\text{Beta}}{p(\theta|x)} \propto \underset{\text{Binomial}}{p(x|\theta)}\underset{\text{Beta}}{p(\theta)}
$$

In this case, likelihood is Binomial distribution and prior is Beta distribution so that we can directly conclude posterier is Beta distribution, posterier and prior are conjugate distributions.


## Exponential form for Gaussian dsitribution

The common form of Gaussian distribution is:

$$
p(x|\mu,\sigma) = \frac{1}{\sqrt{2\pi}\sigma} \mathrm{e}^{-\frac{(x-\mu)^2}{2\sigma^2}}
$$

We can transform it into a exponential family distribution form:

$$
\begin{align*}
p(x|\mu,\sigma) &= \frac{1}{\sqrt{2\pi\sigma^2}} \mathrm{e}^{-\frac{(x-\mu)^2}{2\sigma^2}} \\\
&= \frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{1}{2\sigma^2}(x^2-2x\mu+\mu^2)) \\\
&= \frac{1}{\sqrt{2\pi\sigma^2}} \exp(-\frac{1}{2\sigma^2}(x^2-2x\mu)-\frac{1}{2\sigma^2}\mu^2) \\\
&= \exp(\log\left((2\pi\sigma^2)^{-\frac{1}{2}}\right)) \exp(-\frac{1}{2\sigma^2}
    \begin{bmatrix}-2\mu&1\end{bmatrix}
    \begin{bmatrix}
        x \\\
        x^2
    \end{bmatrix}
    -\frac{1}{2\sigma^2}\mu^2) \\\
&= \exp\left( -\frac{1}{2\sigma^2}
    \begin{bmatrix}-2\mu&1\end{bmatrix}
    \begin{bmatrix}
        x \\\
        x^2
    \end{bmatrix}
    -\frac{1}{2\sigma^2}\mu^2+\log\left((2\pi\sigma^2)^{-\frac{1}{2}}\right) \right) \\\
&= \exp\left( 
    \underset{\color{red}{\theta^T}}{\begin{bmatrix}\frac{\mu}{\sigma^2}&-\frac{1}{2\sigma^2}\end{bmatrix}}
    \underset{\color{red}{\phi(x)}}{
    \begin{bmatrix}
        x \\\
        x^2
    \end{bmatrix}}
    -\underset{\color{red}{A(\theta)}}{\left(\frac{\mu^2}{2\sigma^2}+\frac{1}{2}\log\left(2\pi\sigma^2\right)\right)}
    \right)
\end{align*}
$$

$$
\text{Let } \theta=\begin{bmatrix}
    \theta_1 \\\
    \theta_2
\end{bmatrix}, \text{We have }
\begin{cases}
    \theta_1=\frac{\mu}{\sigma^2} \\\
    \theta_2=-\frac{1}{2\sigma^2}
\end{cases} \implies 
\begin{cases}
    \mu=-\frac{\theta_1}{2\theta_2}  \\\
    \sigma^2=-\frac{1}{2\theta_2}
\end{cases} \\\
\begin{align*}
\text{Thus } A(\theta) &= \frac{(\frac{\theta_1}{2\theta_2})^2}{-\frac{1}{\theta_2}}+\frac{1}{2}\log(-\frac{\pi}{\theta_2}) \\\
&= -\frac{(\theta_1)^2}{4\theta_2}+\frac{1}{2}\log(-\frac{\pi}{\theta_2})
\end{align*}
$$

## Log-partition function and sufficient statistic

Given a exponential family distribution, we have:

$$
\begin{align*}
p(x|\theta) &= h(x)\exp(\theta^T\phi(x)-A(\theta)) \\\
&= \frac{1}{\exp(A(\theta))}h(x)\exp(\theta^T\phi(x)) \\\
&\dArr \\\
\int p(x|\theta) \space dx &= \int \frac{1}{\exp(A(\theta))}h(x)\exp(\theta^T\phi(x)) \space dx \\\
1 &= \int \frac{1}{\exp(A(\theta))}h(x)\exp(\theta^T\phi(x)) \space dx \\\
\exp(A(\theta)) &= \int h(x)\exp(\theta^T\phi(x)) \space dx \\\
&\dArr \\\
\frac{\partial}{\partial \theta}\exp(A(\theta)) &= \frac{\partial}{\partial \theta}\int h(x)\exp(\theta^T\phi(x)) \space dx \\\
\exp(A(\theta))A^{'}(\theta) &= \int \frac{\partial}{\partial \theta}h(x)\exp(\theta^T\phi(x)) \space dx \\\
&\dArr \\\
A^{'}(\theta) &= \frac{1}{\exp(A(\theta))} \int h(x)\exp(\theta^T\phi(x))\phi(x) \space dx \\\
&= \int h(x)\exp(\theta^T\phi(x)-A(\theta))\phi(x) \space dx \\\
&= \int p(x|\theta)\phi(x) \space dx \\\
&= E_{x\sim p(x|\theta)} \left[\phi(x)\right]
\end{align*}
$$

{{< math.inline >}}
<p>
The first derivative of log-partition function \(A(\theta)\) equals to the expectation of sufficient statistic, for the second derivative of \(A(\theta)\), we have:
</p>
{{</ math.inline >}}

$$
\begin{align*}
A^{''}(\theta) &= \frac{\partial}{\partial \theta}\int h(x)\exp(\theta^T\phi(x)-A(\theta))\phi(x) \space dx \\\
&= \int h(x)\frac{\partial}{\partial \theta}\exp(\theta^T\phi(x)-A(\theta))\phi(x) \space dx \\\
&= \int h(x) \exp(\theta^T\phi(x)-A(\theta)) (\phi(x)-A^{'}(\theta)) \phi(x) \space dx \\\
&= \int p(x|\theta) \left(\phi(x)-E[\phi(x)]\right) \phi(x) \space dx \\\
&= \int p(x|\theta) \phi(x)^2 \space dx - E[\phi(x)]\int p(x|\theta)\phi(x)\space dx \\\
&= E\left[\phi(x)^2\right] - E^2\left[ \phi(x) \right] \\\
&= Var\left[ \phi(x) \right]
\end{align*}
$$

{{< math.inline >}}
<p>
The second derivative of log-partition function is the variance of sufficient staticitic and is greater than zero, which means log-partition function \( A(\theta) \) is <mark>convex</mark>.
</p>
{{</ math.inline >}}

We can verify this conclusion on Gaussian distribution:

$$
\text{For Gaussian distribution } \mathcal{N}(\mu,\sigma^2)\\\
\begin{align*}
\frac{\partial}{\partial \theta_1}A(\theta) &= \frac{\partial}{\partial \theta_1} -\frac{(\theta_1)^2}{4\theta_2}+\frac{1}{2}\log(-\frac{\pi}{\theta_2}) \\\
&= -\frac{\theta_1}{2\theta_2} \\\
&= - \frac{\frac{\mu}{\sigma^2}}{2\frac{-1}{2\sigma^2}} \\\
&= \mu
\end{align*}
$$

$$
\begin{align*}
\frac{\partial}{\partial \theta_2}A(\theta) &= \frac{\partial}{\partial \theta_2} -\frac{(\theta_1)^2}{4\theta_2}+\frac{1}{2}\log(-\frac{\pi}{\theta_2}) \\\
&= \frac{\theta_1^2}{4}\theta_2^{-2} - \frac{1}{2}\frac{\theta_2}{\pi}\pi\theta_2^{-2} \\\
&= \frac{\theta_1^2}{4}\theta_2^{-2} - \frac{1}{2\theta_2} \\\
&= \frac{\frac{\mu^2}{\sigma^4}}{4}4\sigma^4 + \sigma^2 \\\
&= \mu^2+\sigma^2
\end{align*}
$$

$$
E\left[\phi(x)\right] = \begin{bmatrix}
E[x] \\\
E[x^2]
\end{bmatrix} = \begin{bmatrix}
\mu \\\
\sigma^2+\mu^2
\end{bmatrix} = \begin{bmatrix}
\frac{\partial}{\partial \theta_1}A(\theta) \\\
\frac{\partial}{\partial \theta_2}A(\theta)
\end{bmatrix}
$$

## MLE and sufficient statistic

{{< math.inline >}}
<p>
Given a dataset \( \mathcal{D}=\lbrace x_1,x_2,\cdots,x_N \rbrace \), maximum likelihood estimation on \(\mathcal{D}\) is:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\theta_{MLE} &= \argmax \log \prod_{i=1}^N p(x_i|\theta) \\\
&= \argmax \sum_{i=1}^N \log \left(h(x_i)\exp(\theta^T\phi(x_i)-A(\theta)) \right) \\\
&= \argmax \sum_{i=1}^N \log \left(h(x_i)\right) + \theta^T\phi(x_i)-A(\theta)
\end{align*} \\\
\dArr
$$

$$
\begin{align*}
\frac{\partial}{\partial \theta} \sum_{i=1}^N \log \left(h(x_i)\right) + \theta^T\phi(x_i)-A(\theta) &= 0 \\\
\sum_{i=1}^N  \phi(x_i)-A^{'}(\theta) &= 0 \\\
\frac{1}{N}\sum_{i=1}^N \phi(x_i) &= A^{'}(\theta_{MLE}) \\\
\text{Define } A^{'(-1)}(\cdot) \text{ the inverse function of } A^{'}(\cdot) \\\
\theta_{MLE} &= A^{'(-1)}\left(\frac{1}{N}\sum_{i=1}^N \phi(x_i)\right)
\end{align*}
$$

{{< math.inline >}}
<p>
The conclusion is when calculating MLE, we only need to compute <mark>sufficient statistic</mark> \(\phi(x_I)\) ignoring the original data samples.
</p>
{{</ math.inline >}}

## Max entropy

Quantities of information is defined as:

$$
-\log p \\\
p\text{ is the probability}
$$

Information entropy is defined as the expectation of information quantities:

$$
H[p] = E_{p(x)}\left[ -\log p \right] = \begin{cases}
\int -p(x)\log p(x)\space dx,\text{continous} \\\
\sum_{x} -p(x)\log p(x), \text{discrete}
\end{cases}
$$

Assuming x is discrete and we have the probability table:

| x   | 1     | 2   | ...   | N   |
| --------- | -------- | ------ | ------ | ------ |
| p | $$p_1$$ | $$p_2$$ | ... | $$p_N$$ |

Then the max entropy problem is defined as follows:

$$
\max \sum_{i=1}^N -p_i\log p_i \\\
\text{subject to } \sum_{i=1}^N p_i = 1 \\\
\dArr \\\
\min \sum_{i=1}^N p_i\log p_i \\\
\text{subject to } \sum_{i=1}^N p_1 = 1
$$

This can be solved by lagrange multiplier:

$$
L(p_1,\cdots,p_N,\lambda) = \sum_{i=1}^N p_i\log p_i +\lambda(1-\sum_{i=1}^N p_i) \\\
\begin{align*}
\frac{\partial}{\partial p_i} \sum_{i=1}^N p_i\log p_i +\lambda(1-\sum_{i=1}^N p_i) &= 0 \\\
\log p_i+p_i\frac{1}{p_i} -\lambda &= 0 \\\
\hat{p_i} &= \mathrm{e}^{\lambda -1}
\end{align*} \\\
\therefore \hat{p_1}=\hat{p_2}=\cdots=\hat{p_N}=\frac{1}{N}
$$

So we can conclude that:

$$
\text{max entropy} \iff \text{same probability for all events} \iff \text{zero prior knowledge of events}
$$

{{< math.inline >}}
<p>
Given a dataset \( \mathcal{D}=\lbrace x_1,x_2,\cdots,x_N \rbrace \), we want to find \(p(x)\) from empirical distribution \(\hat{p(x)}\). Empirical distribution can be directly acquired by counting occurrences of each \(x_i\), and \(p(x)\) can be acquired by max entropy problem:
</p>
{{</ math.inline >}}

$$
\hat{p}(x) = \frac{count(x)}{N}
$$

We want the expectation of any function of x to be the same regardless of whether x is subject to the empirical distribution or the actual distribution, so the max entropy problem becomes:

$$
\begin{align*}
&\min \sum_{i=1}^N p(x_i)\log p(x_i) \\\
&\text{subject to } \sum_{i=1}^N p_i =1 \\\
&\text{subject to } E_{p}[f(x)] = E_{\hat{p}}[f(x)], f(x) = \begin{bmatrix}
f_1(x) \\\
\vdots \\\
f_k(x)
\end{bmatrix}
\end{align*}
$$

Solve it by lagrange multiplier:

$$
\text{Let } E_{\hat{p}}[f(x)]=\Delta=\begin{bmatrix}
\Delta_1 \\\
\vdots \\\
\Delta_k
\end{bmatrix}\\\
\text{Let } p(x_i) = p_i \\\
L(p1,\cdots,p_N,\lambda_0,\lambda) = \sum_{i=1}^N p(x_i)\log p(x_i) + \lambda_0(1-\sum_{i=1}^Np_i)+\lambda^T(\Delta-E_{p}[f(x)]) \\\
\begin{align*}
\frac{\partial}{\partial p_i} \sum_{i=1}^N p(x_i)\log p(x_i) + \lambda_0(1-\sum_{i=1}^Np_i)+\lambda^T(\Delta-E_{p}[f(x)]) &= 0 \\\
\log p_i +1-\lambda_0-\lambda^T\frac{\partial}{\partial p_i}\sum_{i=1}^Np(x_i)f(x_i) &= 0 \\\
\log p_i +1-\lambda_0-\lambda^Tf(x_i) &= 0 \\\
\log p_i &= \lambda^Tf(x_i)+\lambda_0-1 \\\
p_i &= \exp(\lambda^Tf(x_i)+\lambda_0-1)
\end{align*} \\\
\therefore p(x=x_i) = \exp(\underset{\color{red}{\theta^T}}{\lambda^T}\underset{\color{red}{\phi(x)}}{f(x_i)}-\underset{\color{red}{A(\theta)}}{(\lambda_0+1)})
$$

By max entropy problem, we can conclude that for unknown distribution it's assumption distribution is exponential family distribution.

## Conclusion

In this chapter, we find some useful features for exponential family distribution, which makes it an important distribution in machine learning, these features are:

1. Sufficient statistic compress data and can replace original data samples when calculating MLE.
2. When we don't know anything about dataset, max entropy thought tells us to assume data samples subject to exponential family distribution.
3. If likelihood function is exponential family distribution, it is very likely prior and posterier are conjugate distributions.

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=27).
[^2]: From [Weak duality](https://en.wikipedia.org/wiki/Weak_duality).
[^3]: From [Strong duality](https://en.wikipedia.org/wiki/Strong_duality).
[^4]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=35).