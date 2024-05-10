---
author: "X. Wang"
title: "Variational AutoEncoder"
date: "2023-10-02"
description: "A brief introduction."
tags: ["machine learning"]
categories: ["themes", "syntax"]
# series: ["Themes Guide"]
aliases: ["migrate-from-jekyl"]
math: true
ShowBreadCrumbs: false
ShowToc: true
TocOpen: true
draft: true
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

{{ if .Page.Store.Get "hasMermaid" }}
  <script type="module">
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.esm.min.mjs';
    mermaid.initialize({ startOnLoad: true });
  </script>
{{ end }}
{{</ math.inline >}}

<style>
    /* Set the font size of all math elements to 16px */
    .katex {
        font-size: 16px !important;
    }
</style>

<style>
/* Custom CSS styles */
.graph {
    background-color: white;
  /* padding: 10px; */
  /* border-radius: 5px; */
}
.graph pre {
    background-color: white;
  /* font-family: 'Courier New', monospace;
  font-size: 14px;
  line-height: 1.5; */
}
</style>

## VAE Model Representation
### VAE vs GMM
A brief overview of VAEs is: they are similar to [GMM](https://tirmisula.github.io/posts/gaussian-mixture-model/#definition-of-gmm) (Gaussian Mixture Model), but with several differences:

$$
\begin{align*}
\text{GMM} &: \text{finite Gaussians mixture} \\\
\text{VAE} &: \text{infinite Gaussians mixture} \\\
\\\
\text{GMM} &: \text{$z$ is discrete, 1 dimension} \\\
\text{VAE} &: \text{$z$ is continous, multi-dimensions} \\\
\\\
\text{GMM} &: z \sim \text{Categorical}, x|z_k \sim \mathcal{N}(\mu_k,\Sigma_k) \\\
\text{VAE} &: z \sim \mathcal{N}(0,I),z\text{ is prior}, x|z \sim \mathcal{N}(\mu(z;\theta),\Sigma(z;\theta))
\end{align*}
$$

### Objective funciton

Since VAE is a likelihood-based generative model, according to [variational inference chapter](), the log-likelihood of any latent variable model like VAE is given by:

$$
\begin{align*}
\log p(x) &= \int_{z}q(z|x)\log\frac{p(x,z)}{q(z|x)}\space dz - \int_{z}q(z|x)\log\frac{p(z|x)}{q(z|x)}\space dz \\\ \\\
&= \text{ELBO} + \text{KL}(q(z|x,\phi)||p(z|x,\theta))
\end{align*}
$$



## Inference and Learning
### Log-likelihood of p(x) in VAE

{{< math.inline >}}
<p>
Posterier inference in VAE is hard because \( p(z|x) \) is intractable:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(z|x,\theta) &= \frac{p(z)p(x|z,\theta)}{p(x)}, \text{where } \\\ 
p(x) &= \int p(z)p(x|z,\theta)dz \\\
&\because\text{find integral on high-dimensional $z$ is hard} \\\
&\therefore\text{$p(x)$ is intractable, $p(z|x,\theta)$ is intractable}
\end{align*}
$$



GMM shares the similar learning object as VAE, so it will be reviewed first.

### Review learning in GMM

Recall that [EM algorithm](https://tirmisula.github.io/posts/gaussian-mixture-model/#em-algorithm-on-gmm) is used for parameter learning in GMM, the E-step and M-step are given by:

$$
\begin{align*}
\text{E-step} : Q(\theta,\theta^{(t)}) &= E_{z|x,\theta^{(t)} \sim p(z|x,\theta^{(t)})}\left[ \log p(x,z|\theta) \right] \\\
&= \int_z \log p(x,z|\theta)p(z|x,\theta^{(t)})\space dz \\\
&= \sum_{i=1}^N\sum_{k=1}^K\log \left(\mathcal{N}(x_i|\mu_{k},\Sigma_{k})p_{k}\right) \frac{\mathcal{N}(x_i|\mu_{k}^{(t)},\Sigma_{k}^{(t)})p_{k}^{(t)}}{\sum_{j=1}^K \mathcal{N}(x_i|\mu_j^{(t)},\Sigma_j^{(t)})p_j^{(t)}} \\\
\\\
\text{M-step} : \theta^{(t+1)} &= \argmax_{\theta} Q(\theta,\theta^{(t)}), \space\text{s.t.} \sum_{k=1}^K p_k=1 \\\
&= \argmax_{\theta} \sum_{i=1}^N\sum_{k=1}^K\log \left(\mathcal{N}(x_i|\mu_{k},\Sigma_{k})p_{k}\right) \frac{\mathcal{N}(x_i|\mu_{k}^{(t)},\Sigma_{k}^{(t)})p_{k}^{(t)}}{\sum_{j=1}^K \mathcal{N}(x_i|\mu_j^{(t)},\Sigma_j^{(t)})p_j^{(t)}}
\end{align*}
$$

### SGVI for VAE learning

As VAE's name implies, [SGVI]() (stochastic gradient variational inference) is used for posterier inference.

{{< math.inline >}}
<p>
Similar to <a href="https://tirmisula.github.io/posts/generative-adversarial-network/#stochastic-back-propagation-reparametrization-trick">GANs</a>, VAE uses nerual network to approximate \( \mu(z),\Sigma(z) \). However inference is hard because posterier \( p(z|x) \) is intractable:
</p>
{{</ math.inline >}}


$$
\begin{align*}
p(z|x) &= \frac{p(z)p(x|z,\theta)}{p(x)}, \text{where } \\\ 
p(x) &= \int p(z)p(x|z,\theta)dz \\\
&\because\text{integrate on high-dimensional $z$ is hard} \\\
&\therefore\text{$p(x)$ is intractable, $p(z|x)$ is intractable}
\end{align*}
$$

The optimization object is:

$$
\begin{align*}
(\hat{\theta},\hat{\phi}) &= \argmin_{\theta,\phi} \text{KL}(q(z|x,\phi)||p(z|x,\theta)) \\\
&= \argmax_{\theta,\phi} \text{ELBO} \\\
&= \argmax_{\theta,\phi} \mathbb{E}\_{z\sim q(z|x,\phi)}[\log p(x,z|\theta)] + H[q_{\phi}(z)] \\\
&= \argmax_{\theta,\phi} \mathbb{E}_{z\sim q(z|x,\phi)}[\log p(x|z,\theta)+\log p(z|\theta)] - \mathbb{E}_{z\sim q(z|x,\phi)}[\log q(z|x,\phi)] \\\
&= \argmax_{\theta,\phi} \mathbb{E}_{z\sim q(z|x,\phi)}[\log p(x|z,\theta)] - \text{KL}(q(z|x,\phi)||p(z|\theta)) \\\
\end{align*}
$$

The optimization object becomes:

$$
\min_{G}\max_{D}V(D,G)
$$

For any minimax problem, we can solve inner problem first then solve outer problem:



The global optimality is:

$$
(\hat{D},\hat{G}) = \arg\min_{G}\max_{D} V(D,G) \\\
\begin{align*}
\hat{G}(z) &\sim P_{\text{data}} \\\
\hat{D}(x) &= \frac{1}{2} 
\end{align*}
$$

{{< math.inline >}}
<p>
In conslusion, generator's optimality reaches indentically the same distribution as \( P_{\text{data}} \) which satisfies the object: imitating real samples. At the same time, discriminator always outputs \( \frac{1}{2} \) which provides no valuable information for generator's further training.
</p>
{{</ math.inline >}}

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=167).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [NIPS 2016 Tutorial: Generative Adversarial Networks. Ian Goodfellow](https://arxiv.org/pdf/1701.00160).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
