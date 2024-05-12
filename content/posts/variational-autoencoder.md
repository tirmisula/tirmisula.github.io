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

### Object function

Since VAE is a likelihood-based generative model, according to [variational inference chapter](), the log-likelihood of any latent variable models like VAE is given by:

$$
\begin{align*}
\log p(x) &= \int_{z}q(z|x)\log\frac{p(x,z)}{q(z|x)}\space dz - \int_{z}q(z|x)\log\frac{p(z|x)}{q(z|x)}\space dz \\\ \\\
&= \text{ELBO} + \text{KL}(q(z|x,\phi)||p(z|x,\theta))
\end{align*}
$$

{{< math.inline >}}
<p>
Maximizing the ELBO is equivalent to maximizing the log-likelihood while simultaneously minimizing the KL divergence \( \text{KL}(q(z|x,\phi)||p(z|x,\theta))=0 \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
(\hat{\theta},\hat{\phi}) &= \argmin_{\theta,\phi} \text{KL}(q(z|x,\phi)||p(z|x,\theta)) \\\
&= \argmax_{\theta,\phi} \text{ELBO} \\\
&= \argmax_{\theta,\phi} \mathbb{E}\_{z\sim q(z|x,\phi)}[\log p(x,z|\theta)] + H[q_{\phi}(z)] \\\
&= \argmax_{\theta,\phi} \mathbb{E}\_{z\sim q(z|x,\phi)}[\log p(x|z,\theta)+\log p(z|\theta)] - \mathbb{E}\_{z\sim q(z|x,\phi)}[\log q(z|x,\phi)] \\\
&= \argmax_{\theta,\phi} \mathbb{E}\_{z\sim q(z|x,\phi)}[\log p(x|z,\theta)] - \text{KL}(q(z|x,\phi)||p(z|\theta)) \\\
\end{align*}
$$

### Model selection

{{< math.inline >}}
<p>
Given the training object above, \( q(z|x,\phi), p(x|z,\theta), p(z|\theta) \) are unknown and need to be defined. For sampling convenience, prior \( p(z|\theta) \) is assumed to be standard normal distribution:
</p>
{{</ math.inline >}}

$$
z \sim \mathcal{N}(0,I)
$$

{{< math.inline >}}
<p>
As for sampling from \( q(z|x,\phi) \), because posterier \( p(z|x) \) is intractable :
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

{{< math.inline >}}
<p>
We assume \( q(z|x,\phi) \) is approximated by a neural network (Encoder) which follows conditional Gaussian distribution, reparameterization trick is used:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\epsilon &\sim \mathcal{N}(0,I) \\\
z &= \mu(x;\phi) + \Sigma(x;\phi)^{\frac{1}{2}}\cdot\epsilon \\\
z|x,\phi &\sim \mathcal{N}(\mu(x;\phi),\Sigma(x;\phi))
\end{align*} \\\
x,\epsilon \rarr \text{Encoder}(\phi) \rarr z
$$

{{< math.inline >}}
<p>
As for sampling from generative model \( p(x|z,\theta) \),  the simplest distribution for gradient computation is Gaussian. We assume \( p(x|z,\theta) \) is approximated by a neural network (Decoder) which follows conditional Gaussian distribution:
</p>
{{</ math.inline >}}

$$
\begin{align*}
x|z,\theta &\sim \mathcal{N}(\mu(z;\theta),\Sigma(z;\theta))
\end{align*} \\\
z \rarr \text{Decoder}(\theta) \rarr \tilde{x}
$$

Combining encoder and decoder together we have:

$$
\text{For each obsevation $x^{(i)}, i=1\cdots N$} \\\
x^{(i)},\epsilon \rarr \text{Encoder} \rarr z^{(i)} \rarr \text{Decoder} \rarr \tilde{x}^{(i)}
$$

### Understanding object function

Understanding the object function from encoder/decoder model perspective:

$$
\begin{align*}
\because\log p(x|z,\theta) &= \frac{1}{2}\lVert \frac{x-\mu(z;\theta)}{\sigma(z)} \rVert^2+\frac{D}{2}\log 2\pi+\frac{1}{2}\sum_{i=1}^D\log\sigma^2\_{i}(z) \\\
&\text{Let $\sigma^2(z)$ be constant} \\\
&\propto \frac{1}{2\sigma^2}\lVert x-\mu(z;\theta) \rVert^2 \\\
&\propto \frac{1}{2\sigma^2}\lVert x-\tilde{x} \rVert^2
\end{align*} \\\
\dArr
$$

$$
\begin{align*}
\argmax_{\theta,\phi} \mathbb{E}\_{z\sim q(z|x,\phi)}[\log p(x|z,\theta)] &: \argmin \lVert x-\tilde{x} \rVert^2_2 \\\
\argmax_{\theta,\phi} -\text{KL}(q(z|x,\phi)||p(z|\theta)) &: \text{Regularize $q(z|x,\phi)$ by prior $p(z)$}
\end{align*}
$$

In conlusion, VAE tries to minimize reconstruction loss while at the same time preventing overfitting.

## Inference and Learning
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

### SGVI for VAE

As VAE's name implies, [SGVI](https://tirmisula.github.io/posts/variational-inference/#stochastic-gradient-vi) (stochastic gradient variational inference) is used for posterier inference. The ELBO is:

$$
\text{ELBO}(\theta,\phi) = \mathbb{E}\_{z\sim q(z|x,\phi)}[\log p(x|z,\theta)] - \text{KL}(q(z|x,\phi)||p(z|\theta))
$$

#### Gradient with respect to theta

$$
\begin{align*}
\nabla_{\theta}\text{ELBO} &= \nabla_{\theta}\mathbb{E}\_{z\sim q(z|x,\phi)}[\log p(x|z,\theta)] \\\
&= \mathbb{E}\_{z\sim q(z|x,\phi)}[\nabla_{\theta}\log p(x|z,\theta)] \\\
&\text{By MCMC} \\\
&\approx \frac{1}{L}\sum_{i=1}^L\nabla_{\theta}\log p(x|z^{(i)},\theta)
\end{align*}
$$

Referencing [GDA chapter](https://tirmisula.github.io/posts/gaussian-discriminant-analysis/#solve-parameters) (Gaussian Discriminant Analysis), partial derivative w.r.t. mu and sigma are given by:

$$
\begin{align*}
\nabla_{\theta}\log p(x|z^{(i)},\theta) &= \frac{\partial\log p(x|z^{(i)},\theta)}{\partial \mu(z^{(i)};\theta)}\cdot\nabla_{\theta}\mu(z^{(i)};\theta)  + \frac{\partial\log p(x|z^{(i)},\theta)}{\partial \Sigma(z^{(i)};\theta)}\cdot\nabla_{\theta}\Sigma(z^{(i)};\theta)  \\\
\text{where} \\\
\frac{\partial\log p(x|z^{(i)},\theta)}{\partial \mu(z^{(i)};\theta)} &= \Sigma^{(-1)}(z^{(i)};\theta)(x-\mu(z^{(i)};\theta)) \\\
 \frac{\partial\log p(x|z^{(i)},\theta)}{\partial \Sigma(z^{(i)};\theta)} &= -\frac{1}{2}\left( \Sigma^{-1}(z^{(i)};\theta)-Var(x)\Sigma^{-2}(z^{(i)};\theta) \right)
\end{align*}
$$

#### Gradient with respect to phi

{{< math.inline >}}
<p>
Gradient with respect to \( \\phi \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\nabla_{\phi}\text{ELBO} &= \nabla_{\phi}\mathbb{E}\_{z\sim q(z|x,\phi)}[\log p(x|z,\theta)] - \nabla_{\phi}\mathbb{E}\_{z\sim q(z|x,\phi)}[\log q(z|x,\phi)-\log p(z)] \\\
&= \mathbb{E}\_{z\sim q(z|x,\phi)}[\nabla_{\phi}\log p(x|z,\theta)] - \mathbb{E}\_{z\sim q(z|x,\phi)}[\nabla_{\phi}\log q(z|x,\phi)-\nabla_{\phi}\log p(z)] \\\
&\text{By reparameterization trick, } z^{(i)}=g(\epsilon,x,\phi)=\mu(x^{(i)};\phi)+\Sigma(x^{(i)};\phi)^{\frac{1}{2}}\cdot\epsilon \\\
&= \mathbb{E}\_{\epsilon\sim\mathcal{N}(0,I)}[\nabla_{\phi}\log p(x|g(\epsilon,x,\phi),\theta)] - \mathbb{E}\_{\epsilon\sim\mathcal{N}(0,I)}[\nabla_{\phi}\log q(g(\epsilon,x,\phi)|x,\phi)-\nabla_{\phi}\log p(g(\epsilon,x,\phi))] \\\
&\text{By MCMC} \\\
&\approx \frac{1}{L}\sum_{i=1}^L\nabla_{\phi}\log p(x|z^{(i)},\theta) - \frac{1}{L}\sum_{i=1}^L\nabla_{\phi}\log q(z^{(i)}|x,\phi)
\end{align*}
$$

Similarly, we have:

$$
\begin{align*}
\nabla_{\phi}\log q(z^{(i)}|x,\phi) &= \frac{\partial\log q(z^{(i)}|x,\phi)}{\partial \mu(x^{(i)};\phi)}\cdot\nabla_{\phi}\mu(x^{(i)};\phi)  + \frac{\partial\log q(z^{(i)}|x,\phi)}{\partial \Sigma(x^{(i)};\phi)}\cdot\nabla_{\phi}\Sigma(x^{(i)};\phi)  \\\
\text{where} \\\
\frac{\partial\log q(z^{(i)}|x,\phi)}{\partial \mu(x^{(i)};\phi)} &= \Sigma^{(-1)}(x^{(i)};\phi)(z-\mu(x^{(i)};\phi)) \\\
 \frac{\partial\log q(z^{(i)}|x,\phi)}{\partial \Sigma(x^{(i)};\phi)} &= -\frac{1}{2}\left( \Sigma^{-1}(x^{(i)};\phi)-Var(z)\Sigma^{-2}(x^{(i)};\phi) \right)
\end{align*}
$$

$$
\begin{align*}
\nabla_{\phi}\log p(x|z^{(i)},\theta) &= \frac{\partial \log p(x|z^{(i)},\theta)}{\partial z}\cdot\frac{\partial z}{\partial \phi} \\\
&= \left[ \frac{\partial\log p(x|z^{(i)},\theta)}{\partial \mu(z^{(i)};\theta)}\cdot\frac{\partial\mu(z^{(i)};\theta)}{\partial z}  + \frac{\partial\log p(x|z^{(i)},\theta)}{\partial \Sigma(z^{(i)};\theta)}\cdot\frac{\partial\Sigma(z^{(i)};\theta)}{\partial z} \right]\cdot\frac{\partial z}{\partial \phi} \\\
\text{where} \\\
\frac{\partial\log p(x|z^{(i)},\theta)}{\partial \mu(z^{(i)};\theta)} &= \Sigma^{(-1)}(z^{(i)};\theta)(x-\mu(z^{(i)};\theta)) \\\
 \frac{\partial\log p(x|z^{(i)},\theta)}{\partial \Sigma(z^{(i)};\theta)} &= -\frac{1}{2}\left( \Sigma^{-1}(z^{(i)};\theta)-Var(x)\Sigma^{-2}(z^{(i)};\theta) \right) \\\
 \frac{\partial z}{\partial \phi} &= \frac{\partial \mu(x^{(i)};\phi)}{\partial \phi}+\epsilon\frac{\partial\Sigma(x^{(i)};\phi)}{\partial \phi}
\end{align*}
$$

#### Stochastic gradient update

$$
\text{For iteration $t=1\cdots T$} \\\
\begin{align*}
\theta^{(t+1)} &\larr \theta^{(t)} + \eta^{(t)}\left( \frac{1}{L}\sum_{i=1}^L\nabla_{\theta}\log p(x|z^{(i)},\theta) \right) \\\
\phi^{(t+1)} &\larr \phi^{(t)} + \eta^{(t)}\left( \frac{1}{L}\sum_{i=1}^L\nabla_{\phi}\log p(x|z^{(i)},\theta) - \frac{1}{L}\sum_{i=1}^L\nabla_{\phi}\log q(z^{(i)}|x,\phi) \right)
\end{align*}
$$

## Data Generation

Since encoder is used for approximating MLE, it can be dropped after training. We can generate data we want by:

$$
z\sim p(z)=\mathcal{N}(0,I) \rarr \text{Decoder} \rarr \text{new data $x$}
$$

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=167).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [NIPS 2016 Tutorial: Generative Adversarial Networks. Ian Goodfellow](https://arxiv.org/pdf/1701.00160).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
