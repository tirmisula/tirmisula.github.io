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

<cite>[^1]</cite>For E-step we have:

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
We take the partial derivative w.r.t. \(p_k\) first:
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

{{< math.inline >}}
<p>
Then we take the partial derivative w.r.t. \(\mu_k\):
</p>
{{</ math.inline >}}

<cite>[^2]</cite>

$$
\begin{align*}
\frac{\partial}{\partial \mu_k}\left[\sum_{i=1}^N\sum_{k=1}^K\log \left(\mathcal{N}(x_i|\mu_{k},\Sigma_{k})p_{k}\right) p(z_i|x_i,\theta^{(t)})\right]+\lambda(\sum_{k=1}^Kp_k-1) &= 0 \\\
\frac{\partial}{\partial \mu_k}\left[\sum_{i=1}^N\sum_{k=1}^K\log \left(\mathcal{N}(x_i|\mu_k,\Sigma_k)\right) p(z_i|x_i,\theta^{(t)})\right] &= 0 \\\
\frac{\partial}{\partial \mu_k}\left[\sum_{i=1}^N\log \left(\frac{1}{(2\pi)^{\frac{p}{2}}|\Sigma_k|^{\frac{1}{2}}} \exp\left( -\frac{1}{2}(x_i-\mu_k)^T \Sigma_k^{-1} (x_i-\mu_k) \right)\right) p(z_i|x_i,\theta^{(t)})\right] &= 0 \\\
\frac{\partial}{\partial \mu_k}\left[\sum_{i=1}^N \left( -\frac{1}{2}(x_i-\mu_k)^T \Sigma_k^{-1} (x_i-\mu_k) \right) p(z_i|x_i,\theta^{(t)}) \right] &= 0 \\\
\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \left( \frac{1}{2}2\Sigma_k^{-1}(x_i-\mu_k) \right) &= 0 \\\
\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \left( \Sigma_k^{-1}(x_i-\mu_k) \right) &= 0 \\\
\Sigma_k^{-1}\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \left( x_i-\mu_k \right) &= 0 \\\
\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) x_i &= \sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \mu_k
\end{align*}
$$

$$
\dArr \\\
\begin{align*}
\therefore \mu_k^{(t+1)} &= \frac{\sum_{i=1}^N p(z_i=c_k|x_i,\theta^{(t)})x_i}{\sum_{i=1}^N p(z_i=c_k|x_i,\theta^{(t)})}
\end{align*}
$$

{{< math.inline >}}
<p>
Then we take the partial derivative w.r.t. \(\Sigma_k\):
</p>
{{</ math.inline >}}

<cite>[^2]</cite>

$$
\begin{align*}
\frac{\partial}{\partial \Sigma_k}\left[\sum_{i=1}^N\sum_{k=1}^K\log \left(\mathcal{N}(x_i|\mu_{k},\Sigma_{k})p_{k}\right) p(z_i|x_i,\theta^{(t)})\right]+\lambda(\sum_{k=1}^Kp_k-1) &= 0 \\\
\frac{\partial}{\partial \Sigma_k}\left[\sum_{i=1}^N\log \left(\frac{1}{(2\pi)^{\frac{p}{2}}|\Sigma_k|^{\frac{1}{2}}} \exp\left( -\frac{1}{2}(x_i-\mu_k)^T \Sigma_k^{-1} (x_i-\mu_k) \right)\right) p(z_i|x_i,\theta^{(t)})\right] &= 0 \\\
\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \left((2\pi)^{\frac{p}{2}}|\Sigma_k|^{\frac{1}{2}}(-\frac{1}{(2\pi)^p|\Sigma_k|})(2\pi)^{\frac{p}{2}}\frac{1}{2}|\Sigma_k|^{-\frac{1}{2}}|\Sigma_k|\Sigma_k^{-T} + \frac{1}{2}\Sigma_k^{-T}(x_i-\mu_k)(x_i-\mu_k)^T\Sigma_k^{-T} \right) &= 0 \\\
\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \left(-\frac{1}{2}\Sigma_k^{-T} + \frac{1}{2}\Sigma_k^{-T}(x_i-\mu_k)(x_i-\mu_k)^T\Sigma_k^{-T} \right) &= 0 \\\
\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \left(-\Sigma_k^{-1} + \Sigma_k^{-1}(x_i-\mu_k)(x_i-\mu_k)^T\Sigma_k^{-1} \right) &= 0 \\\
\Sigma_k\left[\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \left(-\Sigma_k^{-1} + \Sigma_k^{-1}(x_i-\mu_k)(x_i-\mu_k)^T\Sigma_k^{-1} \right)\right]\Sigma_k &= 0 \\\
\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \left(-\Sigma_k + (x_i-\mu_k)(x_i-\mu_k)^T \right) &= 0 \\\
\sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) (x_i-\mu_k)(x_i-\mu_k)^T &= \sum_{i=1}^N p(z_i|x_i,\theta^{(t)}) \Sigma_k \\\
\end{align*}
$$

$$
\dArr \\\
\begin{align*}
\therefore \Sigma_k^{(t+1)} &= \frac{\sum_{i=1}^N p(z_i=c_k|x_i,\theta^{(t)})(x_i-\mu_k)(x_i-\mu_k)^T}{\sum_{i=1}^N p(z_i=c_k|x_i,\theta^{(t)})}
\end{align*}
$$

## Conclusion

For each iteration step of EM of Gaussian mixture model we have:

$$
\begin{align*}
p_k^{(t+1)} &= \frac{\sum_{i=1}^Np(z_i=c_k|x_i,\theta^{(t)})}{N} \\\
\mu_k^{(t+1)} &= \frac{\sum_{i=1}^N p(z_i=c_k|x_i,\theta^{(t)})x_i}{\sum_{i=1}^N p(z_i=c_k|x_i,\theta^{(t)})} \\\
\Sigma_k^{(t+1)} &= \frac{\sum_{i=1}^N p(z_i=c_k|x_i,\theta^{(t)})(x_i-\mu_k)(x_i-\mu_k)^T}{\sum_{i=1}^N p(z_i=c_k|x_i,\theta^{(t)})}
\end{align*}
$$

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=69).
[^2]: From [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).