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