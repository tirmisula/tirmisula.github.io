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
draft: false
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

    Sampling from Gaussian distribution to gradually remove noise until it recovers data

$$
x_0 \xtofrom[]{} \cdots \xtofrom[]{}x_{t-1}\xtofrom[p(x_{t-1}|x_{t},\theta)]{q(x_t|x_{t-1})}x_{t}\xtofrom[]{} \cdots \xtofrom[]{} x_T
$$

<cite>[^1]</cite>The transition process is modeled by Markov chain, by using Markov chain one distribution can be converted to another distribution gradually.

## Forward Diffusion Process
### Deterministic forward process
The noise adding operation (forward diffusion process) has closed form which is given by:

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

### Derive ending state from initial data

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
Let \( \epsilon=\epsilon_t \), we can directly sample \( x_t \) with the input of \( x_0,\alpha_{1:t},\epsilon_t \). So the forward process can be redefined as:
</p>
{{</ math.inline >}}

$$
\begin{align*}
x_t &= \sqrt{\prod_{i=1}^t\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon_t \\\
q(x_t|x_0) &= \mathcal{N}(\sqrt{\prod_{i=1}^t\alpha_i}x_0, (1-\prod_{i=1}^t\alpha_i)I) \\\
&\text{where } \\\
\epsilon_t &\sim \mathcal{N}(0,I), \space t=1\cdots T \\\
\alpha_{1:t} &\text{ are hyperparameters}
\end{align*}
$$

<!-- {{< math.inline >}}
<p>
Thus the ending Gaussian noise has closed form from the inital data, \( q(x_T|x_0) \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
q(x_T|x_0) &= \mathcal{N}(\sqrt{\prod_{i=1}^T\alpha_i}x_0, (1-\prod_{i=1}^T\alpha_i)I) \\\
x_T &= \sqrt{\prod_{i=1}^T\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^T\alpha_i}\cdot\epsilon
\end{align*}
$$ -->

### Summary

Forward diffusion process has 2 forms:

$$
x_t = \begin{cases}
\sqrt{1-\beta_t}x_{t-1} + \sqrt{\beta_t}\epsilon_{t-1} \\\
\sqrt{\prod_{i=1}^t\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon_t
\end{cases} \quad
\begin{array}{l}
\epsilon_t \sim \mathcal{N}(0,I), \space t=1\cdots T \\\
\beta_{1:t},\alpha_{1:t} \text{ are hyperparameters}
\end{array}
$$

In practice, the second form is used for noise adding.

## Reverse Process
### Deterministic forward process posterier conditioned on x0
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
&\quad\quad\quad\because x_t=\sqrt{\prod_{i=1}^t\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon_t, \space\therefore x_0=\frac{x_t-\sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon_t}{\sqrt{\prod_{i=1}^t\alpha_i}} \\\
&\quad\quad \mu = \frac{\sqrt{\alpha_t}(1-\prod_{i=1}^{t-1}\alpha_i)}{1-\prod_{i=1}^t\alpha_i}x_t+\frac{\beta_t\sqrt{\prod_{i=1}^{t-1}\alpha_i}}{1-\prod_{i=1}^t\alpha_i}\frac{x_t-\sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon_t}{\sqrt{\prod_{i=1}^t\alpha_i}} \\\
&\quad\quad\quad = \frac{\sqrt{\alpha_t}(1-\prod_{i=1}^{t-1}\alpha_i)}{1-\prod_{i=1}^t\alpha_i}x_t+\frac{\beta_t}{\sqrt{\alpha_t}(1-\prod_{i=1}^t\alpha_i)}x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\prod_{i=1}^t\alpha_i}}\cdot\epsilon_t \\\
&\quad\quad\quad = \frac{\alpha_t(1-\prod_{i=1}^{t-1}\alpha_i)+1-\alpha_t}{\sqrt{\alpha_t}(1-\prod_{i=1}^t\alpha_i)}x_t - \frac{\beta_t}{\sqrt{\alpha_t}\sqrt{1-\prod_{i=1}^t\alpha_i}}\cdot\epsilon_t \\\
&\quad\quad\quad = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\prod_{i=1}^t\alpha_i}}\epsilon_t) \\\
&\quad \text{constant part: } \sigma^{-2}\mu^2 \hArr \beta_t^{-1}x_t^2 + (1-\prod_{i=1}^{t-1}\alpha_i)^{-1}\prod_{i=1}^{t-1}\alpha_ix_0^2 - (1-\prod_{i=1}^{t}\alpha_i)^{-1}(x_{t}-\sqrt{\prod_{i=1}^{t}\alpha_i}x_0)^2
\end{align*}
$$

{{< math.inline >}}
<p>
In conclusion, the closed form of forward process posterier conditioned on \( x_0 \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
q(x_{t-1}|x_t,x_0) &= \mathcal{N}(\tilde{\mu}(x_t,x_0), \tilde{\Sigma}(t) ) \\\
&\text{where } \\\
\tilde{\mu}(x_t,x_0) &= \frac{\sqrt{\alpha_t}(1-\prod_{i=1}^{t-1}\alpha_i)}{1-\prod_{i=1}^t\alpha_i}x_t+\frac{\beta_t\sqrt{\prod_{i=1}^{t-1}\alpha_i}}{1-\prod_{i=1}^t\alpha_i}x_0 = \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\prod_{i=1}^t\alpha_i}}\epsilon_t) \\\
\tilde{\Sigma}(t)  &= \beta_t\frac{1-\prod_{i=1}^{t-1}\alpha_i}{1-\prod_{i=1}^t\alpha_i}I
\end{align*}
$$

$$
\begin{align*}
x_{t-1} &= \frac{1}{\sqrt{\alpha_t}}(x_t - \frac{1-\alpha_t}{\sqrt{1-\prod_{i=1}^t\alpha_i}}\epsilon_t) + \sqrt{\beta_t\frac{1-\prod_{i=1}^{t-1}\alpha_i}{1-\prod_{i=1}^t\alpha_i}}\hat{\epsilon_t} \\\
&\text{where } \\\
\hat{\epsilon_t} &\sim \mathcal{N}(0,I), \space t=1\cdots T \\\
\epsilon_t &\text{ is sampled noise in forward diffusion process } (x_0,\epsilon_t)\mapsto x_t \\\
\end{align*}
$$

### Reverse process definition

{{< math.inline >}}
<p>
Like VAE, we finally care about the generative model (the decoder part). Suppose the joint distribution of observations \( x_0 \) and latent variables \( x_{1:T} \) is defined as \( p(x_{0:T}|\theta) \), which is called the reverse process:
</p>
{{</ math.inline >}}

$$
\begin{align*}
&p(x_0|\theta) = \int_{x_1\cdots x_T} p(x_{0:T}|\theta) dx_{1:T} \\\
&p(x_{0:T}|\theta) = p(x_T|\theta)p(x_{T-1}|x_T,\theta)\cdots p(x_0|x_1,\theta)
\end{align*}
$$

<cite>[^2]</cite>
{{< math.inline >}}
<p>
The starting state of reverse process \( p(x_T|\theta) \) is a prior of standard norm because the ending state of forward diffusion process is standard norm:
</p>
{{</ math.inline >}}

$$
p(x_T|\theta) = \mathcal{N}(0,I) \quad\because q(x_T|x_0)\rarr \mathcal{N}(0,I)
$$

{{< math.inline >}}
<p>
In order to better being approximated by \( q(x_{t-1}|x_t) \), \( p(x_{t-1}|x_t,\theta) \) is defined to be Gaussian:
</p>
{{</ math.inline >}}

$$
p(x_{t-1}|x_t,\theta) = \mathcal{N}(\mu(x_t,t;\theta),\Sigma(x_t,t;\theta)) \\\
\text{$p(x_{t-1}|x_t,\theta)$ is defined as reverse sampling}
$$

### Derive loss function

Recall that MLE for latent variable model is equivalent to maximizing it's ELBO via variational inference:

$$
\begin{align*}
(\hat{\theta},\hat{\phi}) &= \argmax_{\theta,\phi} \log p(x|\theta) \\\
&= \argmax_{\theta,\phi} \text{ELBO} \\\
&= \argmax_{\theta,\phi} \mathbb{E}\_{z\sim q(z|x,\phi)}[\log \frac{p(x,z|\theta)}{q(z|x,\phi)} ]
\end{align*}
$$

{{< math.inline >}}
<p>
In DDPM context, forward diffusion process \( q(x_{1:T}|x_0) \) stands for \( q(z|x,\phi) \) because it is easy to sample. There are two options for denoising: 1. tractable posterier \( q(x_{t-1}|x_t,x_0) \) 2.  generative model \( p(x_{0:T}|\theta) \). We want results from generative model close to results from tractable posterier, by replacing \( x=x_0, z=x_{1:T} \), we have the objective function:
</p>
{{</ math.inline >}}

$$
\text{ELBO} =  \mathbb{E}\_{x_{1:T}\sim q(x_{1:T}|x_0,\phi)}[\log \frac{p(x_{0:T}|\theta)}{q(x_{1:T}|x_0,\phi)} ] \\\
\\\
\begin{align*}
(\hat{\theta},\hat{\phi}) &= \argmax_{\theta,\phi}\mathbb{E}\_{x_{1:T}\sim q(x_{1:T}|x_0,\phi)}[\log \frac{p(x_{0:T}|\theta)}{q(x_{1:T}|x_0,\phi)} ] \\\
&= \argmin_{\theta,\phi}\mathbb{E}\_{x_{1:T}\sim q(x_{1:T}|x_0,\phi)}[\log \frac{q(x_{1:T}|x_0,\phi)}{p(x_{0:T}|\theta)} ] \\\
&\because \text{loss is computed on whole dataset, take average on $x_0$} \\\
&= \argmin_{\theta,\phi}\mathbb{E}\_{x_{0}\sim q(x_0|\phi)}[\mathbb{E}\_{x_{1:T}\sim q(x_{1:T}|x_0,\phi)}[\log \frac{q(x_{1:T}|x_0,\phi)}{p(x_{0:T}|\theta)} ]] \\\
&= \argmin_{\theta,\phi}\mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\log \frac{q(x_{1:T}|x_0,\phi)}{p(x_{0:T}|\theta)} ] \\\
&\text{Let } L = \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\log \frac{q(x_{1:T}|x_0,\phi)}{p(x_{0:T}|\theta)} ] \\\
&= \argmin_{\theta,\phi} L
\end{align*}
$$

{{< math.inline >}}
<p>
Let the approximation distribution be related with the Forward diffusion process  \( q(x_{1:T}|x_0,\phi)=\prod_{t=1}^Tq(x_t|x_{t-1}) \). We can expand the negative evidence lower bound:
</p>
{{</ math.inline >}}


$$
\begin{align*}
L &= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\log \frac{q(x_{1:T}|x_0,\phi)}{p(x_{0:T}|\theta)} ] \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\log \frac{\prod_{t=1}^Tq(x_t|x_{t-1})}{p(x_T|\theta)\prod_{t=1}^Tp(x_{t-1}|x_t,\theta)} ] \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\log \prod_{t=1}^T\frac{q(x_t|x_{t-1})}{p(x_{t-1}|x_t,\theta)}-\log p(x_T|\theta) ] \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=1}^T\log\frac{q(x_t|x_{t-1})}{p(x_{t-1}|x_t,\theta)}-\log p(x_T|\theta) ] \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\log\frac{q(x_t|x_{t-1},x_0)}{p(x_{t-1}|x_t,\theta)}+\log\frac{q(x_1|x_{0})}{p(x_{0}|x_1,\theta)}-\log p(x_T|\theta) ] \\\
&\because q(x_{t-1}|x_t,x_0) = \frac{q(x_t|x_{t-1},x_0)q(x_{t-1}|x_0)}{q(x_t|x_0)} \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\log\frac{q(x_{t-1}|x_t,x_0)q(x_t|x_0)\frac{1}{q(x_{t-1}|x_0)}}{p(x_{t-1}|x_t,\theta)}+\log\frac{q(x_1|x_{0})}{p(x_{0}|x_1,\theta)}-\log p(x_T|\theta) ] \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\log\frac{q(x_{t-1}|x_t,x_0)}{p(x_{t-1}|x_t,\theta)}+\sum_{t=2}^T\log\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)}+\log\frac{q(x_1|x_{0})}{p(x_{0}|x_1,\theta)}-\log p(x_T|\theta) ] \\\
&\because \sum_{t=2}^T\log\frac{q(x_t|x_0)}{q(x_{t-1}|x_0)} = \log(\frac{q(x_2|x_0)}{q(x_1|x_0)}\frac{q(x_3|x_0)}{q(x_2|x_0)}\cdots\frac{q(x_T|x_0)}{q(x_{T-1}|x_0)}) = \log\frac{q(x_T|x_0)}{q(x_1|x_0)} \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\log\frac{q(x_{t-1}|x_t,x_0)}{p(x_{t-1}|x_t,\theta)}+\log\frac{q(x_T|x_0)}{q(x_1|x_0)}+\log\frac{q(x_1|x_{0})}{p(x_{0}|x_1,\theta)}-\log p(x_T|\theta) ] \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\log\frac{q(x_{t-1}|x_t,x_0)}{p(x_{t-1}|x_t,\theta)}+\log\frac{q(x_T|x_0)q(x_1|x_{0})}{q(x_1|x_0)p(x_{0}|x_1,\theta)p(x_T|\theta)} ] \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\log\frac{q(x_{t-1}|x_t,x_0)}{p(x_{t-1}|x_t,\theta)}+\log\frac{q(x_T|x_0)}{p(x_T|\theta)}-\log p(x_0|x_1,\theta) ] \\\
\\\
&\text{Let }A=\mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\log\frac{q(x_{t-1}|x_t,x_0)}{p(x_{t-1}|x_t,\theta)} ] \\\
A &= \mathbb{E}\_{q(x_0,x_{t-1},x_t|\phi)}[\sum_{t=2}^T\log\frac{q(x_{t-1}|x_t,x_0)}{p(x_{t-1}|x_t,\theta)} ] \quad (\text{integrate on irrelavent $x_i$}) \\\
&\because \mathbb{E}\_{p(a)}\mathbb{E}\_{p(b|a)}[f(a,b)] = \int \left(\int p(b|a)p(a)f(a,b)\space db\right)da = \mathbb{E}\_{p(a,b)}[f(a,b)] \\\
&= \mathbb{E}\_{q(x_0,x_t|\phi)}[\mathbb{E}\_{q(x_{t-1}|x_0,x_t,\phi)}[\sum_{t=2}^T\log\frac{q(x_{t-1}|x_t,x_0)}{p(x_{t-1}|x_t,\theta)} ]] \\\
&= \mathbb{E}\_{q(x_0,x_t|\phi)}\sum_{t=2}^T\mathbb{E}\_{q(x_{t-1}|x_0,x_t,\phi)}[\log\frac{q(x_{t-1}|x_t,x_0)}{p(x_{t-1}|x_t,\theta)} ] \\\
&= \mathbb{E}\_{q(x_0,x_t|\phi)}\sum_{t=2}^T\text{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_t,\theta)) \\\
&= \mathbb{E}\_{q(x_{0:T}|\phi)}\sum_{t=2}^T\text{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_t,\theta)) \quad (\text{add integral of irrelavent $x_i$}) \\\
&\text{Let }B=\mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[ \log\frac{q(x_T|x_0)}{p(x_T|\theta)} ] \\\
B &= \mathbb{E}\_{q(x_0,x_T|\phi)}[ \log\frac{q(x_T|x_0)}{p(x_T|\theta)} ] \quad (\text{integrate on irrelavent $x_i$}) \\\
&= \mathbb{E}\_{q(x_0|\phi)}\mathbb{E}\_{q(x_T|x_0|\phi)}[ \log\frac{q(x_T|x_0)}{p(x_T|\theta)} ] \\\
&= \mathbb{E}\_{q(x_0|\phi)}\text{KL}(q(x_T|x_0 || p(x_T|\theta))) \\\
&= \mathbb{E}\_{q(x_{0:T}|\phi)}\text{KL}(q(x_T|x_0 || p(x_T|\theta))) \\\
\\\
&\text{Let } \begin{array}{l}
L_T = \text{KL}(q(x_T|x_0 || p(x_T|\theta))) \\\
L_{t-1} = \text{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_t,\theta)) \\\
L_0 = -\log p(x_0|x_1,\theta)
\end{array}, \text{ we have} \\\
L &= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\text{KL}(q(x_{t-1}|x_t,x_0)||p(x_{t-1}|x_t,\theta))+\text{KL}(q(x_T|x_0 || p(x_T|\theta)))-\log p(x_0|x_1,\theta) ] \\\
&= \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[ L_T+L_{T-1}+\cdots+L_{1}+L_0 ] \\\
\end{align*}
$$

{{< math.inline >}}
<p>
Understanding each part of loss \( L_T,L_t,L_0 \) in negative evidence lower bound \( L \):
</p>
{{</ math.inline >}}



{{< math.inline >}}
<p>
<ul>
  <li>
    \( L_T \)
    <p>Distance between termination of forward process and start of reverse process</p>
    <p>\min L_T \hArr \sqrt{\prod_{i=1}^T\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^T\alpha_i}\cdot\epsilon_t \approx \epsilon_t</p>
    <p>\text{if } T\rarr\infty, \prod_{i=1}^T\alpha_i\rarr0, \text{ equation satisfies naturally}</p>
  </li>
  <li>
    \( -L_0 \)
    <p>Probability of last stage of reverse process</p>
    <p>\min L_0 \hArr \max\mathcal{N}(x_0|\mu(x_1;\theta),\Sigma(x_1;\theta))</p>
    <p>\text{where } x_0 = \mu(x_1;\theta), \text{$x_0$ is close to $x_1$}, meaningless item</p>
  </li>
  <li>
    \( L_t \)
    <p>Distance between intermediate stage of forward process posterier conditioned on \( x_0 \) and intermediate stage of reverse process</p>
    <p>Distance between mean of \( q(x_t|x_{t+1}) \) and mean of \( p(x_t|\theta) \)</p>
  </li>
</ul>
</p>
{{</ math.inline >}}

$$
\begin{align*}
\min L_t &\rArr \min \text{KL}(q(x_{t}|x_{t+1},x_0)||p(x_{t}|x_{t+1},\theta)) \\\
&= \min\int q(x_{t}|x_{t+1},x_0)\log\frac{q(x_{t}|x_{t+1},x_0)}{p(x_{t}|x_{t+1},\theta)} dx_t \\\
&\because q(x_{t}|x_{t+1},x_0) \text{ has closed form which is proved previously} \\\
&\text{For simplification, let } q(x_{t}|x_{t+1},x_0)=\mathcal{N}(\tilde{\mu}(x_{t+1},x_0), \tilde{\Sigma}(t+1)) \text{ , let dim$(x)=1$} \\\
&= \min\int q(x_{t}|x_{t+1},x_0)\log\frac{\frac{1}{(2\pi)^\frac{1}{2}\tilde{\sigma}^2(t+1)}\exp(-\frac{1}{2}\frac{(x_{t}-\tilde{\mu}(x_{t+1},x_0))^2}{\tilde{\sigma}^2(t+1)})}{\frac{1}{(2\pi)^\frac{1}{2}\sigma^2(x_{t+1};\theta)}\exp(-\frac{1}{2}\frac{(x_t-\mu(x_{t+1};\theta))^2}{\sigma^2(x_{t+1};\theta)})} dx_t \\\
&= \min\int q(x_{t}|x_{t+1},x_0)\left[\log\left( \frac{\sigma^2(x_{t+1};\theta)}{\tilde{\sigma}^2(t+1)} \right) - \frac{(x_{t}-\tilde{\mu}(x_{t+1},x_0))^2}{2\tilde{\sigma}^2(t+1)} + \frac{(x_{t}-\mu(x_{t+1};\theta))^2}{2\sigma^2(x_{t+1};\theta)} \right] dx_t \\\
&= \min\log\left( \frac{\sigma^2(x_{t+1};\theta)}{\tilde{\sigma}^2(t+1)} \right) - \int q(x_{t}|x_{t+1},x_0)\frac{(x_{t}-\tilde{\mu}(x_{t+1},x_0))^2}{2\tilde{\sigma}^2(t+1)} dx_t + \int q(x_{t}|x_{t+1},x_0) \frac{(x_{t}-\mu(x_{t+1};\theta))^2}{2\sigma^2(x_{t+1};\theta)} dx_t \\\
&\because \sigma = \int (x-\mu)^2f(x)dx \\\
&= \min\log\frac{\sigma^2(x_{t+1};\theta)}{\tilde{\sigma}^2(t+1)} - \frac{\tilde{\sigma}^2(t+1)}{2\tilde{\sigma}^2(t+1)} + \frac{1}{2\sigma^2(x_{t+1};\theta)}\int q(x_{t}|x_{t+1},x_0) (x_{t}-\mu(x_{t+1};\theta))^2 dx_t \\\
&= \min\log\frac{\sigma^2(x_{t+1};\theta)}{\tilde{\sigma}^2(t+1)} - \frac{1}{2} + \frac{1}{2\sigma^2(x_{t+1};\theta)}\int q(x_{t}|x_{t+1},x_0) \left(x_{t}-\mu(x_{t+1};\theta)+\tilde{\mu}(x_{t+1},x_0)-\tilde{\mu}(x_{t+1},x_0)\right)^2 dx_t \\\
&= \min\log\frac{\sigma^2(x_{t+1};\theta)}{\tilde{\sigma}^2(t+1)} - \frac{1}{2} + \frac{1}{2\sigma^2(x_{t+1};\theta)}\int q(x_{t}|x_{t+1},x_0) \left[(x_{t}-\tilde{\mu}(x_{t+1},x_0))^2+(\tilde{\mu}(x_{t+1},x_0)-\mu(x_{t+1};\theta))^2-2(x_t\tilde{\mu}(x_{t+1},x_0)-x_t\mu(x_{t+1};\theta)-\tilde{\mu}(x_{t+1},x_0)^2+\tilde{\mu}(x_{t+1},x_0)\mu(x_{t+1};\theta))\right] dx_t \\\
&x_t\tilde{\mu}(x_{t+1},x_0)-x_t\mu(x_{t+1};\theta)-\tilde{\mu}(x_{t+1},x_0)^2+\tilde{\mu}(x_{t+1},x_0)\mu(x_{t+1};\theta) \\\
&\because \mu = \int xf(x)dx \\\
&= \min\log\frac{\sigma^2(x_{t+1};\theta)}{\tilde{\sigma}^2(t+1)} - \frac{1}{2} + \frac{1}{2\sigma^2(x_{t+1};\theta)}\left[\tilde{\sigma}^2(t+1)+(\tilde{\mu}(x_{t+1},x_0)-\mu(x_{t+1};\theta))^2-2(\tilde{\mu}(x_{t+1},x_0)^2-\tilde{\mu}(x_{t+1},x_0)\mu(x_{t+1};\theta)-\tilde{\mu}(x_{t+1},x_0)^2+\tilde{\mu}(x_{t+1},x_0)\mu(x_{t+1};\theta)) \right] \\\
&= \min\log\frac{\sigma^2(x_{t+1};\theta)}{\tilde{\sigma}^2(t+1)} - \frac{1}{2} + \frac{1}{2\sigma^2(x_{t+1};\theta)}\left[\tilde{\sigma}^2(t+1)+(\tilde{\mu}(x_{t+1},x_0)-\mu(x_{t+1};\theta))^2 \right] \\\
&= \min\log\frac{\sigma^2(x_{t+1};\theta)}{\tilde{\sigma}^2(t+1)} - \frac{1}{2} + \frac{\tilde{\sigma}^2(t+1)}{2\sigma^2(x_{t+1};\theta)}+\frac{(\tilde{\mu}(x_{t+1},x_0)-\mu(x_{t+1};\theta))^2}{2\sigma^2(x_{t+1};\theta)} \\\
&\text{Notice that variances are the same in forward and reverse process in assumption} \\\
&\therefore \tilde{\sigma}^2(t+1) = \sigma^2(x_{t+1};\theta) \\\
&= \min\frac{(\tilde{\mu}(x_{t+1},x_0)-\mu(x_{t+1};\theta))^2}{2\sigma^2(x_{t+1};\theta)} \\\
&= \min\space(\tilde{\mu}(x_{t+1},x_0)-\mu(x_{t+1};\theta))^2 \\\
&= \min||\tilde{\mu}(x_{t+1},x_0)-\mu(x_{t+1};\theta)||^2 \\\
&= \min\lVert\frac{1}{\sqrt{\alpha_{t+1}}}(x_{t+1} - \frac{1-\alpha_{t+1}}{\sqrt{1-\prod_{i=1}^{t+1}\alpha_i}}\epsilon_{t+1})-\mu(x_{t+1};\theta)\rVert^2 \\\
&\text{Since we are actually predicting forward process noise $\epsilon_{t+1}$, while $\alpha$ fixed} \\\
&\text{Design } \mu(x_{t+1};\theta) = \frac{1}{\sqrt{\alpha_{t+1}}}(x_{t+1} - \frac{1-\alpha_{t+1}}{\sqrt{1-\prod_{i=1}^{t+1}\alpha_i}}\epsilon(x_{t+1};\theta)) \\\
&\rArr \min\lVert \epsilon(x_{t+1};\theta)-\epsilon_{t+1} \rVert^2
\end{align*}
$$

<!-- $$
\min \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[ L_T+L_{T-1}+\cdots+L_{1}+L_0 ] \\\
L_T : \text{distance between termination of forward process and start of reverse process} \\\
\min L_T \hArr \sqrt{\prod_{i=1}^T\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^T\alpha_i}\cdot\epsilon_t \approx \epsilon_t \\\
\text{if } T\rarr\infty, \prod_{i=1}^T\alpha_i\rarr0, \text{ equation natrually satisfies} \\\
-L_0 : \text{termination probability of reverse process} \\\
\min L_0 \hArr \max\mathcal{N}(x_0|\mu(x_1;\theta),\Sigma(x_1;\theta)) \\\
\text{where } x_0 = \mu(x_1;\theta), \text{$x_0$ is close to $x_1$} \\\
L_t : \text{distance between denoising process and reverse process}
$$ -->



The objective loss function can be reduced to:

$$
\begin{align*}
\min_{\theta} L &= \min_{\theta} \mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^TL_{t-1}] \\\
&= \min_{\theta}\mathbb{E}\_{x_{0:T}\sim q(x_{0:T}|\phi)}[\sum_{t=2}^T\lVert \epsilon(x_{t};\theta)-\epsilon_{t} \rVert^2] \\\
&= \min_{\theta}\left\lVert \epsilon\left(\sqrt{\prod_{i=1}^t\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon_t;\theta\right)-\epsilon_{t} \right\rVert^2 \space, t=2\cdots T,\space x_0\in P_{\text{data}} \\\
\end{align*}
% \text{$\epsilon_t$ is used as both input of NN for $p(x_{t-1}|x_t,\theta)$ inference and partial closed form for $q(x_{t-1}|x_t)$ inference}
$$

{{< math.inline >}}
<p>
For the loss function, we are actually finding a forward process noise predictor with parameter \( \theta \).
</p>
{{</ math.inline >}}


{{< math.inline >}}
<p>
noise predictor \( \epsilon(x_{t};\theta) \) is designed to be learned with a nerual network (autoencoder).
</p>
{{</ math.inline >}}

{{< math.inline >}}
<p>
\( \epsilon_t \) is used as both input of NN for \( p(x_{t-1}|x_t,\theta) \) inference and partial closed form for \( q(x_{t-1}|x_t) \) inference.
</p>
{{</ math.inline >}}

## Optimize objective function
### Training

{{< math.inline >}}
<p>
In practical training, timestamp \( t \) is encoded with time embedding as a part of input of nerual network. With the new NN input \( \epsilon(x_t,t;\theta) \), a gradient descent method is used:
</p>
{{</ math.inline >}}

$$
\begin{align*}
&\text{For } l=1\cdots\infty \\\
&\quad x_0 \sim q(x_0)=P_{\text{data}} \\\
&\quad t \sim \mathcal{U}(1,T) \\\
&\quad \epsilon_t \sim \mathcal{N}(0,I) \\\
&\quad \theta^{(l+1)} = \theta^{(l)} + \nabla_{\theta}\left\lVert \epsilon\left(\sqrt{\prod_{i=1}^t\alpha_i}\cdot x_0 + \sqrt{1-\prod_{i=1}^t\alpha_i}\cdot\epsilon_t,t;\theta\right)-\epsilon_{t} \right\rVert^2 \\\
&\text{Loop until converge} \\\
\end{align*}
$$

### Inference

{{< math.inline >}}
<p>
While \( \theta \) is learned for reverse process \( p(x_{0:T}|\theta) \). We can perform reverse sampling step by step from \( p(x_T|\theta) \). Each step \( p(x_{t-1}|x_t,\theta) \) is given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(x_{t-1}|x_t,\theta) &= \mathcal{N}(\mu(x_t,t;\theta),\Sigma(x_t,t;\theta)) \\\
&= \mathcal{N}(\frac{1}{\sqrt{\alpha_{t}}}(x_{t} - \frac{1-\alpha_{t}}{\sqrt{1-\prod_{i=1}^{t}\alpha_i}}\epsilon(x_{t},t;\theta)), \beta_t\frac{1-\prod_{i=1}^{t-1}\alpha_i}{1-\prod_{i=1}^t\alpha_i}I)
\end{align*}
$$

{{< math.inline >}}
<p>
Notice that for the marginal condition \( L_0 \), we have: 
</p>
{{</ math.inline >}}

$$
\begin{align*}
\min L_0 &\hArr \max\mathcal{N}(x_0|\mu(x_1;\theta),\Sigma(x_1;\theta)) \\\
&\hArr p(x_0|x_1,\theta) = 1, \text{replace $\Sigma(x_1;\theta)$ with zero} \\\
&\hArr x_0 = \mu(x_1;\theta)
\end{align*}
$$

The sampling algorithm is given by:

$$
\begin{align*}
&x_T \sim \mathcal{N}(0,I) \\\
&\text{For } t=T\cdots 1 \\\
&\quad z\sim\mathcal{N}(0,I) \text{ if } t>1 \text{ else } z=0 \\\
&\quad x_{t-1} = \frac{1}{\sqrt{\alpha_{t}}}(x_{t} - \frac{1-\alpha_{t}}{\sqrt{1-\prod_{i=1}^{t}\alpha_i}}\epsilon(x_{t},t;\theta)) + \sqrt{\beta_t\frac{1-\prod_{i=1}^{t-1}\alpha_i}{1-\prod_{i=1}^t\alpha_i}}\cdot z \\\
&\text{End} \\\
&\text{Return $x_0$}
\end{align*}
$$

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV19H4y1G73r).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^2]: - [Denoising Diffusion Probabilistic Models. Jonathan Ho, Ajay Jain, Pieter Abbee](https://arxiv.org/pdf/2006.11239).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [A fast learning algorithm for deep belief nets. Geoffrey E. Hinton, Simon Osindero, Yee-Whye Teh](https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf).
