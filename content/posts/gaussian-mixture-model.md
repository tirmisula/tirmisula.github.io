---
author: "X. Wang"
title: "Gaussian Mixture Model"
date: "2023-09-09"
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

## Definition of GMM

GMM(Gaussian Mixture Model) in brief is the summation of several different weighted gaussian distributions:

$$
\begin{align*}
&x:\text{observed data} \\\
&z:\text{latent variable} \\\
&(x,z):\text{complete data} \\\
&\theta=(p_k,\mu_k,\Sigma_k,...):\text{parameters}
\end{align*}
$$

| z   | c1     | c2   | ...   | cK   |
| --------- | -------- | ------ | ------ | ------ |  
| prob | $$p_1$$ | $$p_2$$ | ... | $$p_K$$ |

$$
\sum_{k=1}^K p_k = 1
$$

$$
x|z=c_K \sim \mathcal{N}(\mu_k,\Sigma_k)
$$

$$
\begin{align*}
p(x) &= \sum_{K}p(x,z) \\\
&= \sum_{K}p(x|z)p(z) \\\
&= \sum_{k=1}^K \mathcal{N}(x|\mu_k,\Sigma_k)p_k
\end{align*}
$$

## MLE of GMM

{{< math.inline >}}
<p>
Given dataset \(\mathcal{D}=\lbracex_1,\cdots,x_N\rbrace\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\theta_{MLE} &= \argmax_{\theta} \prod_{i=1}^N p(x_i) \\\
&= \argmax_{\theta} \sum_{i=1}^N\log p(x_i) \\\
&= \argmax_{\theta} \sum_{i=1}^N\log \left( \sum_{k=1}^K \mathcal{N}(x_i|\mu_k,\Sigma_k)p_k \right)
\end{align*}
$$

{{< math.inline >}}
<p>
Because of summation form inside the \(\log()\) function, when taking derivative, we cannot find analytic expression of \(\theta_{MLE}\).
</p>
{{</ math.inline >}}

## EM algorithm on GMM

We introduced [EM algorithm](https://tirmisula.github.io/posts/expectation-maximization/#generalized-em-algorithm) previously, it has the following form:

$$
\theta^{(t+1)} = \argmax_{\theta}\int_z \log p(x,z|\theta)p(z|x,\theta^{(t)})\space dz
$$

### E-step

For E-step we have:

$$
\begin{align*}
Q(\theta,\theta^{(t)}) &= \int_z \log p(x,z|\theta)p(z|x,\theta^{(t)})\space dz \\\
&\dArr \text{ discrete form} \\\
&= \sum_{k=1}^K \left(\log \prod_{i=1}^Np(x_i,z_{ik}|\theta)\right)\left(\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})\right) \\\
&= \sum_{k=1}^K \left(\sum_{i=1}^N\log p(x_i,z_{ik}|\theta)\right)\left(\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})\right) \\\
&= \sum_{k=1}^K \left[ \log p(x_1,z_{1k}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})+\cdots+\log p(x_N,z_{Nk}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)}) \right] \\\
&= \sum_{k=1}^K \left[ \log p(x_1,z_{1k}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})\right]+\cdots+\sum_{k=1}^K \left[ \log p(x_N,z_{Nk}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})\right]
\end{align*}
$$

$$
\begin{align*}
\sum_{k=1}^K \left[ \log p(x_1,z_{1k}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})\right] &= \sum_{k=1}^K \left[ \log p(x_1,z_{1k}|\theta)p(z_{1k}|x_1,\theta^{(t)})\prod_{i=2}^Np(z_{ik}|x_i,\theta^{(t)})\right] \\\
&= 
\end{align*}
$$

### M-step

## Derive EM from ELBO+Jenson inequlity
### Jenson inequality in brief

$$
\begin{align*}
\text{Given } &f(x):\text{concave funciton} \\\
&a,b \in \mathbb{R} \\\
&t\in [0,1] \\\
&c = ta+(1-t)b \\\
\text{We have } &f(c) \text{ is above line } ab: \\\
f(ta+(1-t)b) &\geq tf(a)+(1-t)f(b) \\\
f(\frac{a}{2}+\frac{b}{2}) &\geq tf(\frac{a}{2})+(1-t)f(\frac{b}{2}), t=\frac{1}{2} \\\
&\dArr \\\
f(E[x]) &\geq E[f(x)]
\end{align*}
$$

### Derivation

{{< math.inline >}}
<p>
Log-likelihood can be written as integration of joint probability \(p(x,z|\theta)\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\log p(x|\theta) &= \log\int_{z}p(x,z|\theta)\space dz
\end{align*}
$$

{{< math.inline >}}
<p>
Then we introduce unknown distribution \(q(z)\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\log p(x|\theta) &= \log\int_{z}\frac{p(x,z|\theta)}{q(z)}q(z)\space dz \\\
&= \log E_{z\sim q(z)}\left[ \frac{p(x,z|\theta)}{q(z)} \right] \\\
\because (\log x)^{''} &= -\frac{1}{x^2} \leq 0 \\\
\therefore \log p(x|\theta) &\geq E_{z\sim q(z)}\left[ \log\frac{p(x,z|\theta)}{q(z)} \right] \\\
&\geq \int_{z} \log\frac{p(x,z|\theta)}{q(z)}q(z)\space dz \\\
&\geq \text{ELBO}
\end{align*}
$$

{{< math.inline >}}
<p>
We want log-likelihood equals to ELBO(evidence lower bound) so:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{p(x,z|\theta)}{q(z)} &= C \\\
p(x,z|\theta) &= Cq(z) \\\
\int_{z}p(x,z|\theta)\space dz &= C\int_{z}q(z)\space dz \\\
p(x|\theta) &= C \\\
&\dArr \\\
q(z) &= \frac{p(x,z|\theta)}{C} \\\
&= \frac{p(x,z|\theta)}{p(x|\theta)} \\\
&= p(z|x,\theta)
\end{align*}
$$

{{< math.inline >}}
<p>
\(q(z)\) must be posterier, then maxmizing log-likelihood becomes:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{\theta} &=  \argmax_{\theta} \log p(x|\theta) \\\
&= \argmax_{\theta} \text{ELBO} \\\
&\text{Let } q(z) = p(z|x,\theta^{(t)}), \text{let posterier be the last iteration result}\\\
&= \argmax_{\theta} \int_{z}p(z|x,\theta^{(t)})\log\frac{p(x,z|\theta)}{p(z|x,\theta^{(t)})}\space dz \\\
&= \argmax_{\theta} \int_{z}p(z|x,\theta^{(t)})\left[\log p(x,z|\theta) - \log p(z|x,\theta^{(t)}\right]\space dz \\\
&=\argmax_{\theta} \int_{z}p(z|x,\theta^{(t)})\log p(x,z|\theta)\space dz
\end{align*}
$$

{{< math.inline >}}
<p>
We derived the same form of EM algorithm when Jenson inequality satisfies indentically equal.
</p>
{{</ math.inline >}}

## Generalized EM algorithm

{{< math.inline >}}
<p>
From previous section we know that log-likelihood can be written as ELBO+KL-divergence:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\log p(x|\theta) &= \int_{z}q(z)\log\frac{p(x,z|\theta)}{q(z)}\space dz - \int_{z}q(z)\log\frac{p(z|x,\theta)}{q(z)}\space dz \\\
&= \text{ELBO} + KL(q||p) \\\
&\text{Let ELBO}=L(q,\theta) \\\
&= L(q,\theta) + KL(q||p)
\end{align*}
$$

$$
\begin{cases}
\text{ELBO} &= \int_{z}q(z)\log\frac{p(x,z|\theta)}{q(z)}\space dz \\\
KL(q||p) &= \int_{z}q(z)\log\frac{q(z)}{p(z|x,\theta)}\space dz
\end{cases}
$$

{{< math.inline >}}
<p>
The original EM assumes KL divergence equals to zero that is to say posterier \(p(z|x,\theta)\) is <mark>tractable</mark>:
</p>
{{</ math.inline >}}

$$
KL(q||p) = 0 \iff q(z) = p(z|x,\theta)
$$

{{< math.inline >}}
<p>
Generalized EM tries to minimize KL divergence because posterier \( p(z|x,\theta) \) is <mark>intractable</mark>:
</p>
{{</ math.inline >}}


$$
\argmin_{q} KL(q||p)
$$

The whole algorithm is listed as follows:

$$
\begin{align*}
\text{E-step} &: \text{Fix } \theta, q^{(t+1)}=\argmin_{q}KL(q||p)=\argmax_{q}L(q,\theta^{(t)}) \\\
\text{M-step} &: \text{Fix } q, \theta^{(t+1)}=\argmax_{\theta} L(q^{(t+1)},\theta)
\end{align*}
$$

Paramerter's update order indicates the <mark>coordinate ascent</mark> thought:

$$
\theta^{(1)}\rArr q^{(2)}\rArr \theta^{(2)}\rArr q^{(3)}\rArr\cdots\rArr \theta^{(t)} \\\
\dArr \\\
L(q,\theta) \text{ is solved by SMO(Sequential Minimal Optimization)}
$$

For ELBO:

$$
\begin{align*}
L(q,\theta) &= \int_{z}q(z)\log\frac{p(x,z|\theta)}{q(z)}\space dz \\\
&= E_{z\sim q(z)}[\log p(x,z|\theta)] - E_{z\sim q(z)}[\log q(z)] \\\
&= E_{z\sim q(z)}[\log p(x,z|\theta)] + \underset{\color{red}{entropy}}{H_{q(z)}[q(z)]}
\end{align*}
$$

## Extensions of EM

<cite>[^1]</cite>:

$$
\begin{cases}
\text{VEM(Variational EM)} \implies \text{use variational inference to find }p(z|x,\theta) \\\
\text{MCEM(Monte Carlo EM)} \implies \text{use monte carlo method to find }p(z|x,\theta)
\end{cases}
$$

## Conclusion

1. For traditional EM, log-likelihood is expressed as:
$$
\log p(x|\theta) = \begin{cases}
\text{ELBO}+KL(q||p) \implies \text{We want }KL(q||p)=0 \text{ or } q=p \iff q(z) \text{ is the posterier} \\\
\text{ELBO}+\Delta \implies \text{We want } \Delta=0 \iff q(z) \text{ is the posterier}
\end{cases}
$$
2. For generalized EM, q(z) is intractable so coordinate ascending method is used.


## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=65).