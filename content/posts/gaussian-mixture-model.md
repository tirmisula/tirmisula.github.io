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

<center>

| z   | c1     | c2   | ...   | cK   |
| --------- | -------- | ------ | ------ | ------ |  
| prob | $$p_1$$ | $$p_2$$ | ... | $$p_K$$ |

</center>

$$
\sum_{k=1}^K p_k = 1
$$

$$
x|z=c_K \sim \mathcal{N}(\mu_k,\Sigma_k)
$$

$$
\begin{align*}
p(x) &= \sum_{z}p(x,z) \\\
&= \sum_{z}p(x|z)p(z) \\\
&= \sum_{k=1}^K \mathcal{N}(x|\mu_k,\Sigma_k)p_k
\end{align*}
$$

$$
\begin{align*}
p(z_i|x) &= \frac{p(x,z_i)}{p(x)} \\\
&= \frac{\mathcal{N}(x|\mu_i,\Sigma_i)p_i}{\sum_{k=1}^K \mathcal{N}(x|\mu_k,\Sigma_k)p_k}
\end{align*}
$$

## MLE of GMM

{{< math.inline >}}
<p>
Given dataset \(\mathcal{D}=\lbrace x_1,\cdots,x_N \rbrace\):
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
&= \int_z \left(\log \prod_{i=1}^Np(x_i,z_{i}|\theta)\right)\left(\prod_{i=1}^Np(z_{i}|x_i,\theta^{(t)})\right) \space dz \\\
&= \int_{z}\log p(x_1,z_1,\cdots,x_N,z_N)p(z1,\cdots,z_N|x1,\cdots,x_N,\theta^{(t)})\space dz \\\
&= \int_{z_1}\cdots\int_{z_N}\log p(x_1,z_1,\cdots,x_N,z_N)p(z1,\cdots,z_N|x1,\cdots,x_N,\theta^{(t)})dz_1\cdots dz_N \\\
&\dArr \text{ discrete form} \\\
&= \sum_{z_1}\cdots\sum_{z_N} \left(\log \prod_{i=1}^Np(x_i,z_{i}|\theta)\right)\left(\prod_{i=1}^Np(z_{i}|x_i,\theta^{(t)})\right) \\\
&= \sum_{z_1}\cdots\sum_{z_N} \left(\sum_{i=1}^N\log p(x_i,z_{i}|\theta)\right)\left(\prod_{i=1}^Np(z_{i}|x_i,\theta^{(t)})\right) \\\
% &\text{Let } \phi_k=\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)}), \gamma_{ik}=\log p(x_i,z_{ik}|\theta)\\\
% &= \sum_{k=1}^K\phi_k\sum_{i=1}^N\gamma_{ik} \\\
% &= \sum_{k=1}^K\sum_{i=1}^N\phi_k\gamma_{ik} \\\
% &= \sum_{k=1}^K\sum_{i=1}^N\phi_k\gamma_{1k}+\sum_{k=1}^K\sum_{i=1}^N\phi_k\gamma_{2k}+\cdots \\\
% &= \sum_{k=1}^K\sum_{i=1}^N\phi_k\gamma_{1k}+\sum_{k=1}^K\sum_{i=1}^N\phi_k\gamma_{2k}+\cdots \\\
% &= \sum_{k=1}^K\left[\sum_{i=1}^1\phi_k\gamma_{ik}+\sum_{i=2}^N\phi_k\gamma_{ik}\right] \\\
% &= \sum_{k=1}^K\left[ \log p(x_1,z_{1k}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})+\sum_{i=2}^N\phi_k\gamma_{ik}\right] \\\
% &= \sum_{k=1}^K\left[ \log p(x_1,z_{1k}|\theta)p(z_{1k}|x_1,\theta^{(t)})\prod_{i=2}^Np(z_{ik}|x_i,\theta^{(t)})+\sum_{i=2}^N\phi_k\gamma_{ik}\right] \\\
% &= \sum_{k=1}^K \left[ \log p(x_1,z_{1k}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})+\cdots+\log p(x_N,z_{Nk}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)}) \right] \\\
% &= \sum_{k=1}^K \left[ \log p(x_1,z_{1k}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})\right]+\cdots+\sum_{k=1}^K \left[ \log p(x_N,z_{Nk}|\theta)\prod_{i=1}^Np(z_{ik}|x_i,\theta^{(t)})\right]
&= \sum_{i=1}^N\left(\sum_{z_1}\cdots\sum_{z_N} \log p(x_i,z_{i}|\theta)\left(\prod_{i=1}^Np(z_{i}|x_i,\theta^{(t)})\right)\right)
\end{align*}
$$

$$
\text{For each item: } \\\
\begin{align*}
\sum_{z_1}\cdots\sum_{z_N} \log p(x_1,z_{1}|\theta)\left(\prod_{i=1}^Np(z_{i}|x_i,\theta^{(t)})\right) &= \sum_{z_1}\cdots\sum_{z_N} \log p(x_1,z_{1}|\theta)p(z_1|x_1,\theta^{(t)})\left(\prod_{i=2}^Np(z_{i}|x_i,\theta^{(t)})\right) \\\
&= \sum_{z_1}\log p(x_1,z_1|\theta)p(z_1|x_1,\theta^{(t)}) + \sum_{z2}\cdots\sum_{z_N}\prod_{i=2}^Np(z_i|x_i,\theta^{(t)}) \\\
&= \sum_{z_1}\log p(x_1,z_1|\theta)p(z_1|x_1,\theta^{(t)}) + \sum_{z2}p(z_2|x_2,\theta^{(t)})+\cdots+\sum_{z_N}p(z_N|x_N,\theta^{(t)}) \\\
&= \sum_{z_1}\log p(x_1,z_1|\theta)p(z_1|x_1,\theta^{(t)})
\end{align*}
$$

$$
\text{So original equation becomes: } \\\
\begin{align*}
Q(\theta,\theta^{(t)}) &= \sum_{i=1}^N\sum_{z_i}\log p(x_i,z_i|\theta)p(z_i|x_i,\theta^{(t)}) \\\
&= \sum_{i=1}^N\sum_{z_i}\log p(x_i,z_i|\theta)\frac{p(x_i,z_i|\theta^{(t)})}{\sum_{z}p(x_i,z|\theta^{(t)})} \\\
&= \sum_{i=1}^N\sum_{z_i}\log p(x_i,z_i|\theta)\frac{p(x_i|z_i,\theta^{(t)})p(z_i|\theta^{(t)})}{\sum_{z}p(x_i,z|\theta^{(t)})} \\\
&= \sum_{i=1}^N\sum_{k=1}^K\log p(x_i,z_i=c_k|\theta)\frac{p(x_i|z_i=c_k,\theta^{(t)})p(z_i=c_k|\theta^{(t)})}{\sum_{z}p(x_i,z|\theta^{(t)})} \\\
&\because \forall\space z_i=c_k \implies z_i\mapsto(p_k,\mu_k,\Sigma_k)\mapsto(p_{z_i},\mu_{z_i},\Sigma_{z_i}) \\\
&= \sum_{i=1}^N\sum_{k=1}^K\log \left(\mathcal{N}(x_i|\mu_{k},\Sigma_{k})p_{k}\right) \frac{\mathcal{N}(x_i|\mu_{k}^{(t)},\Sigma_{k}^{(t)})p_{k}^{(t)}}{\sum_{j=1}^K \mathcal{N}(x_i|\mu_j^{(t)},\Sigma_j^{(t)})p_j^{(t)}}
\end{align*}
$$

{{< math.inline >}}
<p>
Note that \( \mu_j^{(t)},\Sigma_j^{(t)},p_j^{(t)} \) and  \( \mu_k^{(t)},\Sigma_k^{(t)},p_k^{(t)} \) are known constants from last iteration.
</p>
{{</ math.inline >}}

### M-step

For simplification, For M-step we have:

$$
\begin{align*}
\theta^{(t+1)} &= \argmax_{\theta} Q(\theta,\theta^{(t)}) \\\
&= \argmax_{\theta} \sum_{i=1}^N\sum_{k=1}^K\log \left(\mathcal{N}(x_i|\mu_{k},\Sigma_{k})p_{k}\right) \frac{\mathcal{N}(x_i|\mu_{k}^{(t)},\Sigma_{k}^{(t)})p_{k}^{(t)}}{\sum_{j=1}^K \mathcal{N}(x_i|\mu_j^{(t)},\Sigma_j^{(t)})p_j^{(t)}}
\end{align*}
$$

It becomes a optimization problem to solve:

$$
\argmax_{\theta} Q(\theta,\theta^{(t)}) \\\
\text{subject to } \sum_{k=1}^K p_k=1
$$

{{< math.inline >}}
<p>
We take the derivative of \(p_k\) first:
</p>
{{</ math.inline >}}

$$
\text{Let } L(p_1,\cdots,p_K,\mu_1,\cdots,\mu_K,\Sigma_1,\cdots,\Sigma_K,\lambda) = Q(\theta,\theta^{(t)})+\lambda(\sum_{k=1}^Kp_k-1)
$$

$$
\frac{\partial L}{\partial p_k} = 0 \\\
\dArr \\\
\begin{align*}
\frac{\partial}{\partial p_k}\left[\sum_{i=1}^N\sum_{k=1}^K\log \left(\mathcal{N}(x_i|\mu_{k},\Sigma_{k})p_{k}\right) \frac{\mathcal{N}(x_i|\mu_{k}^{(t)},\Sigma_{k}^{(t)})p_{k}^{(t)}}{\sum_{j=1}^K \mathcal{N}(x_i|\mu_j^{(t)},\Sigma_j^{(t)})p_j^{(t)}}\right]+\lambda(\sum_{k=1}^Kp_k-1) &= 0 \\\
\frac{\partial}{\partial p_k}\left[\sum_{i=1}^N\sum_{k=1}^K\log \left(p_{k}\right) \frac{\mathcal{N}(x_i|\mu_{k}^{(t)},\Sigma_{k}^{(t)})p_{k}^{(t)}}{\sum_{j=1}^K \mathcal{N}(x|\mu_j^{(t)},\Sigma_j^{(t)})p_j^{(t)}}\right]+\lambda(\sum_{k=1}^Kp_k-1) &= 0 \\\
\frac{\partial}{\partial p_k}\left[\sum_{i=1}^N\log \left(p_{k}\right) \frac{\mathcal{N}(x_i|\mu_{k}^{(t)},\Sigma_{k}^{(t)})p_{k}^{(t)}}{\sum_{j=1}^K \mathcal{N}(x|\mu_j^{(t)},\Sigma_j^{(t)})p_j^{(t)}}\right]+\lambda(\sum_{k=1}^Kp_k-1) &= 0 \\\
\sum_{i=1}^N \frac{1}{p_k} p(z_i=c_k|x_i,\theta^{(t)}) + \lambda &= 0 \\\
\frac{1}{p_k}\sum_{i=1}^N  p(z_i=c_k|x_i,\theta^{(t)}) + \lambda &= 0 \\\
\sum_{i=1}^N  p(z_i=c_k|x_i,\theta^{(t)}) + p_k\lambda &= 0 \\\
\end{align*}
$$

$$
\dArr \\\
\begin{align*}
\sum_{k=1}^K\frac{\partial L}{\partial p_k} &= 0 \\\
\sum_{k=1}^K \left(\sum_{i=1}^N  p(z_i=c_k|x_i,\theta^{(t)}) + p_k\lambda\right) &= 0 \\\
\sum_{k=1}^K \sum_{i=1}^N  p(z_i=c_k|x_i,\theta^{(t)}) &= -\sum_{k=1}^Kp_k\lambda \\\
\sum_{i=1}^N\sum_{k=1}^K  p(z_i=c_k|x_i,\theta^{(t)}) &= -\lambda \\\
N &= -\lambda
\end{align*}
$$

$$
\dArr \\\
\begin{align*}
\therefore p_k^{(t+1)} &= \frac{\sum_{i=1}^Np(z_i=c_k|x_i,\theta^{(t)})}{N}\\\
&=\frac{\text{sum of sample space's posteriers from last turn}}{\text{samples num}} \\\
&= \text{average posterier $p(z)$ of sample space from last turn}
\end{align*}
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