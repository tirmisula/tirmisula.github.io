---
author: "X. Wang"
title: "EM Algorithm"
date: "2023-09-08"
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

## Definition of EM algorithm

Expectation maximization(EM) algorithm is used to estimate parameters of model with latent variables. It is different from MLE which can directly find analytic expressions of parameters, EM algorithm is an undirect iterate algorithm which is defined as follows:

$$
\begin{align*}
&x:\text{observed data} \\\
&z:\text{latent variable} \\\
&\theta:\text{parameters} \\\
\theta^{(t+1)} &= \argmax_{\theta}\int_z \log p(x,z|\theta)p(z|x,\theta^{(t)})\space dz \\\
\text{E-step} &: E_{z|x,\theta^{(t)} \sim p(z|x,\theta^{(t)})}\left[ \log p(x,z|\theta) \right] = \int_{z}\log p(x,z|\theta)p(z|x,\theta^{(t)})\space dz \\\
\text{M-step} &: \argmax_{\theta} E_{z|x,\theta^{(t)} \sim p(z|x,\theta^{(t)})}\left[ \log p(x,z|\theta) \right]
\end{align*}
$$

## Proof of convergence of EM algorithm

{{< math.inline >}}
<p>
For each iterate step, we want likelihood larger:
</p>
{{</ math.inline >}}

$$
\theta^{(t)} \rarr \theta^{(t+1)} \\\
\log p(x|\theta^{(t)}) \leq \log p(x|\theta^{(t+1)})
$$

{{< math.inline >}}
<p>
To prove algorithm's convergence, we rewrite \(p(x|\theta)\) with latent variable first:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(x,z|\theta) &= p(x|\theta)p(z|x,\theta) \\\
p(x|\theta) &= \frac{p(x,z|\theta)}{p(z|x,\theta)} \\\
\log p(x|\theta) &= \log \frac{p(x,z|\theta)}{p(z|x,\theta)} \\\
\log p(x|\theta) &= \log p(x,z|\theta) -\log p(z|x,\theta) \\\
E_{z \sim p(z|x,\theta^{(t)})}\left[ \log p(x|\theta) \right] &= E_{z \sim p(z|x,\theta^{(t)})}\left[ \log p(x,z|\theta) -\log p(z|x,\theta) \right] \\\
\int_{z}p(z|x,\theta^{(t)})\log p(x|\theta)\space dz &= \int_{z}p(z|x,\theta^{(t)})\left(\log p(x,z|\theta) -\log p(z|x,\theta)\right)\space dz \\\
\log p(x|\theta)\int_{z}p(z|x,\theta^{(t)})\space dz &= \int_{z}p(z|x,\theta^{(t)})\left(\log p(x,z|\theta) -\log p(z|x,\theta)\right)\space dz \\\
\log p(x|\theta) &= \int_{z}p(z|x,\theta^{(t)})\left(\log p(x,z|\theta) -\log p(z|x,\theta)\right)\space dz \\\
\log p(x|\theta) &= \int_{z}p(z|x,\theta^{(t)})\log p(x,z|\theta)\space dz - \int_{z}p(z|x,\theta^{(t)})\log p(z|x,\theta)\space dz
\end{align*}
$$

{{< math.inline >}}
<p>
Next we set auxilary function \(Q\) and \(H\) to simplify the righter part:
</p>
{{</ math.inline >}}

$$
Q(\theta,\theta^{(t)}) = \int_{z}p(z|x,\theta^{(t)})\log p(x,z|\theta)\space dz \\\
H(\theta,\theta^{(t)}) = \int_{z}p(z|x,\theta^{(t)})\log p(z|x,\theta)\space dz \\\
\log p(x|\theta) = Q(\theta,\theta^{(t)}) - H(\theta,\theta^{(t)})
$$

{{< math.inline >}}
<p>
\(Q\) and \(H\) have the following features:
</p>
{{</ math.inline >}}

$$
\text{By definition} \\\
\theta^{(t+1)} = \argmax_{\theta} Q(\theta,\theta^{(t)}) \\\
Q(\theta^{(t+1)},\theta^{(t)}) \geq Q(\theta,\theta^{(t)}) \\\
Q(\theta^{(t+1)},\theta^{(t)}) \geq Q(\theta^{(t)},\theta^{(t)})
$$

$$
\begin{align*}
H(\theta^{(t+1)},\theta^{(t)}) - H(\theta^{(t)},\theta^{(t)}) &= \int_{z}p(z|x,\theta^{(t)})\left[\log p(z|x,\theta^{(t+1)})-\log p(z|x,\theta^{(t)})\right]\space dz \\\
&= \int_{z}p(z|x,\theta^{(t)})\frac{\log p(z|x,\theta^{(t+1)})}{\log p(z|x,\theta^{(t)})}\space dz \\\
&= \underset{Kullback-Leibler Divergence}{-KL(p(z|x,\theta^{(t)}) || p(z|x,\theta^{(t+1)}))} \\\
&\leq 0 \\\
\text{alternatively, by Jenson inequality: } &E[f(x)] \geq f(E(x)) \\\
\int_{z}p(z|x,\theta^{(t)})\frac{\log p(z|x,\theta^{(t+1)})}{\log p(z|x,\theta^{(t)})}\space dz &\leq \log\int_{z}p(z|x,\theta^{(t)})\frac{p(z|x,\theta^{(t+1)})}{p(z|x,\theta^{(t)})}\space dz \\\
\int_{z}p(z|x,\theta^{(t)})\frac{\log p(z|x,\theta^{(t+1)})}{\log p(z|x,\theta^{(t)})}\space dz &\leq \log 1 \\\
\int_{z}p(z|x,\theta^{(t)})\frac{\log p(z|x,\theta^{(t+1)})}{\log p(z|x,\theta^{(t)})}\space dz &\leq 0 \iff H(\theta^{(t+1)},\theta^{(t)}) \leq H(\theta^{(t)},\theta^{(t)}) \\\
\end{align*}
$$

We put these 2 features back to original equation, we proved the convergence:

$$
\log p(x|\theta^{(t+1)}) = Q(\theta^{(t+1)},\theta^{(t)}) - H(\theta^{(t+1)},\theta^{(t)}) \\\
\log p(x|\theta^{(t)}) = Q(\theta^{(t)},\theta^{(t)}) - H(\theta^{(t)},\theta^{(t)}) \\\
\begin{align*}
\therefore \log p(x|\theta^{(t+1)})-\log p(x|\theta^{(t)}) &= \underset{\color{red}{\geq 0}}{Q(\theta^{(t+1)},\theta^{(t)})-Q(\theta^{(t)},\theta^{(t)})} + \underset{\color{red}{\geq 0}}{H(\theta^{(t)},\theta^{(t)})-H(\theta^{(t+1)},\theta^{(t)})} \\\
&\geq 0 \\\
\log p(x|\theta^{(t+1)}) &\geq \log p(x|\theta^{(t)})
\end{align*}
$$

## Derive EM from ELBO+KL Divergence

From bayesian equation we know the likelihood function is:

$$
\log p(x|\theta) = \log p(x,z|\theta) -\log p(z|x,\theta)
$$

{{< math.inline >}}
<p>
We introduce an unknown posterier \( q(z) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\log p(x|\theta) &= \log p(x,z|\theta) -\log p(z|x,\theta) \\\
&= \log p(x,z|\theta)-\log q(z) - \log p(z|x,\theta)+\log q(z) \\\
&= \log\frac{p(x,z|\theta)}{q(z)} - \log\frac{p(z|x,\theta)}{q(z)} \\\
&\dArr \\\
E_{z\sim q(z)}\left[ \log p(x|\theta) \right] &= E_{z\sim q(z)}\left[ \log\frac{p(x,z|\theta)}{q(z)} - \log\frac{p(z|x,\theta)}{q(z)} \right] \\\
\int_{z}q(z)\log p(x|\theta)\space dz &= \int_{z}q(z)\log\frac{p(x,z|\theta)}{q(z)}\space dz - \int_{z}q(z)\log\frac{p(z|x,\theta)}{q(z)}\space dz \\\
\log p(x|\theta) &= \int_{z}q(z)\log\frac{p(x,z|\theta)}{q(z)}\space dz + KL(q||p) \\\
\text{Let } \int_{z}q(z)\log\frac{p(x,z|\theta)}{q(z)}\space dz &= \text{ELBO (evidence lower bound)} \\\
\therefore \log p(x|\theta) &= \text{ELBO} + KL(q(z)||p(z|x,\theta)) \\\
&\geq \text{ELBO}
\end{align*}
$$

{{< math.inline >}}
<p>
Log-likelihood's lower bound is ELBO(evidence lower bound). KL divergence equals to zero when posterier \(p\) is the same as the real distribution \(q\). Assuming KL divergence is zero and we want to maximize likelihood, it becomes:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{\theta} &= \argmax_{\theta} \log p(x|\theta) \\\
&= \argmax_{\theta} \text{ELBO}, (q(z)=p(z|x,\theta)) \\\
&= \argmax_{\theta} \int_{z}q(z)\log\frac{p(x,z|\theta)}{q(z)}\space dz \\\
&\text{Let } q(z) = p(z|x,\theta^{(t)}), \text{let posterier be the last iteration result}\\\
&= \argmax_{\theta} \int_{z}p(z|x,\theta^{(t)})\log\frac{p(x,z|\theta)}{p(z|x,\theta^{(t)})}\space dz \\\
&= \argmax_{\theta} \int_{z}p(z|x,\theta^{(t)})\left[\log p(x,z|\theta) - \log p(z|x,\theta^{(t)}\right]\space dz \\\
&=\argmax_{\theta} \int_{z}p(z|x,\theta^{(t)})\log p(x,z|\theta)\space dz
\end{align*}
$$

{{< math.inline >}}
<p>
In this subsection we derived the same form of EM algorithm by maximizing log-likelihood function or maximizing ELBO when KL divergence of \(p(z|x,\theta^{*})\) is zero.
</p>
{{</ math.inline >}}

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


{{< math.inline >}}
<p>
The original EM assumes KL divergence equals to zero:
</p>
{{</ math.inline >}}

$$
KL(q||p) = 0 \iff q(z) = p(z|x,\theta)
$$

Generalized EM tries to minimize KL divergence:

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

Paramerter's update order is:

$$
\theta^{(1)}\rArr q^{(2)}\rArr \theta^{(2)}\rArr q^{(3)}\rArr\cdots\rArr \theta^{(t)}
$$

Details for ELBO:

$$
\begin{align*}
L(q,\theta) &= \int_{z}q(z)\log\frac{p(x,z|\theta)}{q(z)}\space dz \\\
&= E_{z\sim q(z)}[\log p(x,z)] - E_{z\sim q(z)}[q(z)] \\\
&= E_{z\sim q(z)}[\log p(x,z)] + \underset{\color{red}{entropy}}{H_{q(z)}[q(z)]}
\end{align*}
$$

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

In this chapter, we find some useful features for exponential family distribution, which makes it an important distribution in machine learning, these features are<cite>[^1]</cite>:

1. Sufficient statistic compress data and can replace original data samples when calculating MLE.
2. When we don't know anything about dataset, max entropy thought tells us to assume data samples subject to exponential family distribution.
3. If likelihood function is exponential family distribution, it is very likely prior and posterier are conjugate distributions.

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=45).