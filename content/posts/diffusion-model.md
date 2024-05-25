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
### Deterministic noise adding
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
x_t \sim \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_tI)
$$

{{< math.inline >}}
<p>
By homogeneous markov assumption, assuming the transition function is determined by \( q(x_t|x_{t-1}) \):
</p>
{{</ math.inline >}}

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t|\sqrt{1-\beta_t}x_{t-1}, \beta_tI) \\\
x_0 \sim P_{\text{data}}
$$

### Ending state is derivable from initial data

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
q(x_T|x_0) &= \mathcal{N}(\sqrt{\prod_{i=1}^T\alpha_i}x_0, (1-\prod_{i=1}^T\alpha_i)I) \\\
x_T &= \sqrt{\prod_{i=1}^T\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^T\alpha_i}\cdot\epsilon
\end{align*}
$$

## Reverse Diffusion Process
### Deterministic reversed data generating
{{< math.inline >}}
<p>
Given \( x_0 \), the reversed inference \( q(x_{t-1})|x_t \) has a closed form:
</p>
{{</ math.inline >}}

$$
\begin{align*}
&\text{By Bayes' theorem: } q(x_{t-1}|x_t) = \frac{q(x_t|x_{t-1})q(x_{t-1})}{q(x_t)} \\\
&\text{Let $x_0$ be in the conditional probability, we have: } \\\
&q(x_{t-1}|x_t,x_0) = \frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)} \\\
&\text{$q(x_t|x_{t-1},x_0),q(x_{t-1}|x_0),q(x_t|x_0)$ are well defined in forward diffision process, which are: } \\\
&\quad \ast q(x_t|x_{t-1},x_0) = q(x_t|x_{t-1}) = \mathcal{N}(x_t|\sqrt{1-\beta_t}x_{t-1}, \beta_tI)=\mathcal{N}(x_t|\sqrt{\alpha_t}x_{t-1}, \beta_tI) \\\
&\quad \ast q(x_t|x_0) = \mathcal{N}(\sqrt{\prod_{i=1}^t\alpha_i}x_0, (1-\prod_{i=1}^t\alpha_i)I) \\\
&\quad \ast q(x_{t-1}|x_0) = \mathcal{N}(\sqrt{\prod_{i=1}^{t-1}\alpha_i}x_0, (1-\prod_{i=1}^{t-1}\alpha_i)I) \\\
&\text{Recall that multivariate Gaussian with diagonal matrix has the following form: } \\\
&|\Sigma| = \prod_{i=1}^p\sigma^2_i, \quad (x-\mu)^T\Sigma^{-1}(x-\mu) = \sum_{i=1}^p\frac{(x_i-\mu_i)^2}{\sigma^2_i} \\\
&\mathcal{N}(x_1\cdots x_p|\mu,\Sigma) = \frac{1}{(2\pi)^\frac{p}{2}\prod_{i=1}^p\sigma_i}\exp(-\frac{1}{2}\sum_{i=1}^p\frac{(x_i-\mu_i)^2}{\sigma^2_i})=\prod_{i=1}^p\mathcal{N}(x_i|\mu_i,\sigma_i) \\\
&\text{For $q(x_t|x_{t-1},x_0),q(x_{t-1}|x_0),q(x_t|x_0)$, we have: } \\\
&\quad q(x_t|x_{t-1}) = \frac{1}{(2\pi)^\frac{p}{2}\beta_t^{\frac{p}{2}}}\exp(-\frac{1}{2\beta_t}(x_t-\sqrt{\alpha_t}x_{t-1})^T(x_t-\sqrt{\alpha_t}x_{t-1}))=\left(\frac{1}{(2\pi)^{\frac{1}{2}}\beta_t^{\frac{1}{2}}}\right)^p\exp(-\frac{1}{2}\frac{(x_t-\sqrt{\alpha_t}x_{t-1})^T(x_t-\sqrt{\alpha_t}x_{t-1})}{\beta_t}) \\\
&\quad q(x_t|x_0) = \left(\frac{1}{(2\pi)^{\frac{1}{2}}(1-\prod_{i=1}^t\alpha_i)^{\frac{1}{2}}}\right)^p\exp(-\frac{1}{2}\frac{(x_t-\sqrt{\prod_{i=1}^t\alpha_i}x_0)^T(x_t-\sqrt{\prod_{i=1}^t\alpha_i}x_0)}{1-\prod_{i=1}^t\alpha_i}) \\\
&\quad q(x_{t-1}|x_0) = \left(\frac{1}{(2\pi)^{\frac{1}{2}}(1-\prod_{i=1}^{t-1}\alpha_i)^{\frac{1}{2}}}\right)^p\exp(-\frac{1}{2}\frac{(x_{t-1}-\sqrt{\prod_{i=1}^{t-1}\alpha_i}x_0)^T(x_{t-1}-\sqrt{\prod_{i=1}^{t-1}\alpha_i}x_0)}{1-\prod_{i=1}^{t-1}\alpha_i}) \\\
&\text{For simplification let $p=1$, we have: } \\\
&\frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)} = \frac{\sqrt{2\pi(1-\prod_{i=1}^t\alpha_i)}}{2\pi\sqrt{\beta_t(1-\prod_{i=1}^{t-1}\alpha_i)}}\exp\left( -\frac{1}{2}\left[ \frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{\beta_t}+\frac{(x_{t-1}-\sqrt{\prod_{i=1}^{t-1}\alpha_i}x_0)^2}{1-\prod_{i=1}^{t-1}\alpha_i}-\frac{(x_{t}-\sqrt{\prod_{i=1}^{t}\alpha_i}x_0)^2}{1-\prod_{i=1}^{t}\alpha_i} \right] \right) \\\
&\quad\quad\quad\propto \exp\left( -\frac{1}{2}\left[ \frac{(x_t-\sqrt{\alpha_t}x_{t-1})^2}{\beta_t}+\frac{(x_{t-1}-\sqrt{\prod_{i=1}^{t-1}\alpha_i}x_0)^2}{1-\prod_{i=1}^{t-1}\alpha_i}-\frac{(x_{t}-\sqrt{\prod_{i=1}^{t}\alpha_i}x_0)^2}{1-\prod_{i=1}^{t}\alpha_i} \right] \right) \\\
&\text{Recall that $q(x_{t-1})$ is assumed to be in a Gaussian form $\mathcal{N}(\mu,\sigma^2)$ : } \\\
&q(x_{t-1}|x_t,x_0) \propto \exp\left( -\frac{1}{2}\left[ \frac{(x_{t-1}-\mu)^2}{\sigma^2} \right] \right) = \exp\left( -\frac{1}{2}\left[ (\sigma^{-2})x_{t-1}^2-(\sigma^{-2}2\mu)x_{t-1}+\sigma^{-2}\mu^2 \right] \right) \\\
&\text{In order to match the Gaussian case, Rearrange the above Bayes equation: } \\\
&\quad \text{quadratic part: }(\sigma^{-2})x_{t-1}^2 \hArr (\beta_t^{-1}\alpha_t+(1-\prod_{i=1}^{t-1}\alpha_i)^{-1})x_{t-1}^2 \\\
&\quad\quad \sigma^2 = \frac{1}{\frac{\alpha_t}{\beta_t}+\frac{1}{1-\prod_{i=1}^{t-1}\alpha_i}} = \frac{\beta_t(1-\prod_{i=1}^{t-1}\alpha_i)}{\alpha_t-\prod_{i=1}^t\alpha_i+\beta_t} = \beta_t\frac{(1-\prod_{i=1}^{t-1}\alpha_i)}{1-\prod_{i=1}^t\alpha_i} \\\
&\quad \text{linear part: }(\sigma^{-2}2\mu)x_{t-1} \hArr \left(2\beta_t^{-1}x_t\sqrt{\alpha_t}+2(1-\prod_{i=1}^{t-1}\alpha_i)^{-1}\sqrt{\prod_{i=1}^{t-1}\alpha_i}x_0\right)x_{t-1} \\\
&\quad\quad \mu = (\frac{x_t\sqrt{\alpha_t}}{\beta_t}+\frac{\sqrt{\prod_{i=1}^{t-1}\alpha_i}x_0}{1-\prod_{i=1}^{t-1}\alpha_i})\beta_t\frac{1-\prod_{i=1}^{t-1}\alpha_i}{1-\prod_{i=1}^t\alpha_i} = \frac{\sqrt{\alpha_t}(1-\prod_{i=1}^{t-1}\alpha_i)}{1-\prod_{i=1}^t\alpha_i}x_t+\frac{\beta_t\sqrt{\prod_{i=1}^{t-1}\alpha_i}}{1-\prod_{i=1}^t\alpha_i}x_0 \\\
&\quad\quad\quad\because x_t=\sqrt{\prod_{i=1}^t\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon, \space\therefore x_0=\frac{x_t-\sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon}{\sqrt{\prod_{i=1}^t\alpha_i}} \\\
&\quad\quad \mu = \frac{\sqrt{\alpha_t}(1-\prod_{i=1}^{t-1}\alpha_i)}{1-\prod_{i=1}^t\alpha_i}x_t+\frac{\beta_t\sqrt{\prod_{i=1}^{t-1}\alpha_i}}{1-\prod_{i=1}^t\alpha_i}\frac{x_t-\sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon}{\sqrt{\prod_{i=1}^t\alpha_i}} \\\
&\quad\quad\quad = \frac{\sqrt{\alpha_t}(1-\prod_{i=1}^{t-1}\alpha_i)}{1-\prod_{i=1}^t\alpha_i}x_t+\frac{\beta_t}{\sqrt{\alpha_t}(1-\prod_{i=1}^t\alpha_i)}x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\prod_{i=1}^t\alpha_i}}\cdot\epsilon \\\
&\quad\quad\quad = \frac{\alpha_t(1-\prod_{i=1}^{t-1}\alpha_i)+1-\alpha_t}{\sqrt{\alpha_t}(1-\prod_{i=1}^t\alpha_i)}x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\prod_{i=1}^t\alpha_i}}\cdot\epsilon \\\
&\quad\quad\quad = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\prod_{i=1}^t\alpha_i}}\epsilon) \\\
&\quad \text{constant part: } \sigma^{-2}\mu^2 \hArr \beta_t^{-1}x_t^2 + (1-\prod_{i=1}^{t-1}\alpha_i)^{-1}\prod_{i=1}^{t-1}\alpha_ix_0^2 - (1-\prod_{i=1}^{t}\alpha_i)^{-1}(x_{t}-\sqrt{\prod_{i=1}^{t}\alpha_i}x_0)^2
\end{align*}
$$

In conclusion, we have:

$$
q(x_{t-1}|x_t,x_0) = \mathcal{N}(\frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\prod_{i=1}^t\alpha_i}}\epsilon), \beta_t\frac{1-\prod_{i=1}^{t-1}\alpha_i}{1-\prod_{i=1}^t\alpha_i}I)
$$

### Noise removing

The noise removing operation is given by:

$$
\begin{align*}
x_{t-1} &= \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\prod_{i=1}^t\alpha_i}}\hat{\epsilon_t}) + \sqrt{\beta_t\frac{1-\prod_{i=1}^{t-1}\alpha_i}{1-\prod_{i=1}^t\alpha_i}}\epsilon \\\
&\text{where } \\\
\epsilon_t &\sim \mathcal{N}(0,I), \space t=1\cdots T \\\
\beta_t &\text{ is hyperparameter}
\end{align*}
$$

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV19H4y1G73r).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [NIPS 2016 Tutorial: Generative Adversarial Networks. Ian Goodfellow](https://arxiv.org/pdf/1701.00160).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
