---
author: "X. Wang"
title: "Diffusion Model"
date: "2023-10-04"
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

## Background

Denoising diffusion probabilistic model (DDPM) is a type of generative model, a type of probabilistic graphical model (PGM) and a type of latent variable model. It has Encoder and Decoder stages which is similar to VAE.

+ Forward diffusion process (encoder)

    Gradually adding Gaussian noise to data sample until it becomes Gaussian noise

+ Reverse diffusion process (decoder)

    Sampling from Gaussian distribution to gradually remove noise until it becomes clear data

$$
x_0 \xtofrom[decoder]{encoder} \cdots \xtofrom[decoder]{encoder}x_{t-1}\xtofrom[decoder]{encoder}x_{t}\xtofrom[decoder]{encoder} \cdots \xtofrom[decoder]{encoder} x_T
$$

The transition process is modeled by Markov chain, by using Markov chain one distribution can be converted to another distribution gradually.

## Forward Diffusion Process

The noise adding operation is given by:

$$
\begin{align*}
x_t &= \sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1} \\\
&\text{where } \\\
\epsilon_t &\sim \mathcal{N}(0,I), \space t=1\cdots T \\\
\beta_t &\text{ is hyperparameter}
\end{align*}
$$

{{< math.inline >}}
<p>
By <a href="https://tirmisula.github.io/posts/generative-adversarial-network/#stochastic-back-propagation-reparametrization-trick">reparameterization trick</a>, we know \( x_t \) follows a Gaussian distribution as well:
</p>
{{</ math.inline >}}

$$
x_t \sim \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t)
$$

{{< math.inline >}}
<p>
By homogeneous markov assumption, assuming the transition function is determined by \( q(x_t|x_{t-1}) \):
</p>
{{</ math.inline >}}

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t|\sqrt{1-\beta_t}x_{t-1}, \beta_t) \\\
x_0 \sim P_{\text{data}}
$$

{{< math.inline >}}
<p>
As introduced in <a href="https://tirmisula.github.io/posts/hidden-markov-model/#forward-algorithm">HMM chapter</a>, recurrence derivation exists in between current state \( h_t \) and it's former state \( h_{t-1} \) (Forward/Backward Algorithm). We can derive \( x_t \) from \( x_0 \) since adding noise is a recurrence formula:
</p>
{{</ math.inline >}}

$$
\begin{align*}
x_t &= \sqrt{1-\beta_t}x_{t-1}+\sqrt{\beta_t}\epsilon_{t-1} \\\
&= \sqrt{1-\beta_t}(\sqrt{1-\beta_{t-1}}x_{t-2}+\sqrt{\beta_{t-1}}\epsilon_{t-2})+\sqrt{\beta_t}\epsilon_{t-1} \\\
&= \sqrt{1-\beta_t}\sqrt{1-\beta_{t-1}}x_{t-2}+\sqrt{1-\beta_{t}}\sqrt{\beta_{t-1}}\epsilon_{t-2}+\sqrt{\beta_t}\epsilon_{t-1} \\\
&\text{Let } \alpha_t = 1-\beta_t \\\
&= \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + \sqrt{\alpha_{t}(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{(1-\alpha_{t})}\epsilon_{t-1} \\\
&\because \mathcal{N}(\epsilon_1|\mu_1,\Sigma_1)+\mathcal{N}(\epsilon_2|\mu_2,\Sigma_2)=\mathcal{N}(\mu_1+\mu_2,\Sigma_1+\Sigma_2) \text{ if $\epsilon_1 \perp \epsilon_2$} \\\
&\therefore \sqrt{\alpha_{t}(1-\alpha_{t-1})}\epsilon_{t-2} + \sqrt{(1-\alpha_{t})}\epsilon_{t-1} \sim \mathcal{N}(0,(\alpha_{t}-\alpha_{t}\alpha_{t-1}+1-\alpha_{t})I) \\\
&= \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t}\alpha_{t-1}}\epsilon \\\
&\dArr \\\
x_t &= \sqrt{\alpha_{t}}x_{t-1} + \sqrt{1-\alpha_{t}}\epsilon \\\
&= \sqrt{\alpha_{t}\alpha_{t-1}}x_{t-2} + \sqrt{1-\alpha_{t}\alpha_{t-1}}\epsilon \\\
&= \sqrt{\alpha_{t}\alpha_{t-1}\alpha_{t-2}}x_{t-3} + \sqrt{1-\alpha_{t}\alpha_{t-1}\alpha_{t-2}}\epsilon \\\
&\cdots \\\
&= \sqrt{\prod_{i=1}^t\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon
\end{align*}
$$

{{< math.inline >}}
<p>
Thus the ending Gaussian noise has closed form from the inital data, \( q(x_T|x_0) \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
q(x_T|x_0) &= \mathcal{N}(\sqrt{\prod_{i=1}^t\alpha_i}x_0, (1-\prod_{i=1}^t\alpha_i)I) \\\
x_T &= \sqrt{\prod_{i=1}^T\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^T\alpha_i}\cdot\epsilon
\end{align*}
$$

## Reverse Diffusion Process

{{< math.inline >}}
<p>
Define a one-hot encode indicator matrix \( Y \), which represents \( A_1\cdots A_K \):
</p>
{{</ math.inline >}}



{{< math.inline >}}
<p>
The objective function becomes
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{Y} &= \argmin_{Y} Tr\left( Y^TLY(Y^TDY)^{-1} \right) \\\
L &: \text{Laplace matrix, where } L=D-W
\end{align*}
$$

## Solve objective function



## Reference

[^1]: - [video](https://www.bilibili.com/video/BV19H4y1G73r).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [NIPS 2016 Tutorial: Generative Adversarial Networks. Ian Goodfellow](https://arxiv.org/pdf/1701.00160).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
