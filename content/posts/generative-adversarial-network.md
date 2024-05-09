---
author: "X. Wang"
title: "Generative Adversarial Network"
date: "2023-10-01"
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
### Stochastic Back Propagation (Reparametrization Trick)

<cite>[^1]</cite>Since NNs (Nerual Networks) are function approximators (universal approximators), an intuition is to use NN to approximate probability density just like what PGM represents, this technique is called stochastic back propagation or reparametrization trick.

Case 1: Approximating the Marginal Distribution

{{< math.inline >}}
<p>
Let's consider a scenario where \(y\) follows a gaussian distribution, we use an easy to sample variable \( z \) to represents \( y \), \( z \) follows standard normal distribution:
</p>
{{</ math.inline >}}

$$
\text{Let } y \sim \mathcal{N}(\mu,\sigma^2), z \sim \mathcal{N}(0,1) \\\
y = \mu+\sigma z = f(z,\mu,\sigma)
$$

{{< math.inline >}}
<p>
Suppose we approximate \( f(z,\mu,\sigma) \) with a neural network:
</p>
{{</ math.inline >}}

$$
z \rarr \text{NN}(\theta) \rarr y, \text{ where } \theta=(\mu,\sigma)
$$

{{< math.inline >}}
<p>
We define the loss function \(J(y)\) as the mean squared error between the predicted and actual values of \(y\):
</p>
{{</ math.inline >}}

$$
J(y)=\frac{1}{N}\sum_{i=1}^N\Vert y-y^{(i)} \rVert^2
$$

{{< math.inline >}}
<p>
We perform gradient descent to minimize this loss, where the gradient of the loss with respect to the parameters \( \theta \) is given by the chain rule:
</p>
{{</ math.inline >}}

$$
\nabla_{\theta}J(y) = \frac{\nabla J(y)}{\nabla y}\cdot\frac{\nabla y}{\nabla \theta}
$$

Case 2: Approximating the Conditional Distribution

{{< math.inline >}}
<p>
Consider \( y|x \) follows a gaussian distribution, we use an easy to sample variable \( z \) to represents \( y|x \), \( z \) follows standard normal distribution:
</p>
{{</ math.inline >}}

$$
\text{Let } y|x \sim \mathcal{N}(x;\mu,\sigma^2), z \sim \mathcal{N}(0,1) \\\
y = \mu(x)+\sigma(x) z = f(z,\mu_x,\sigma_x) \\\
\mu_x = \mu(x), \sigma_x = \sigma(x)
$$

{{< math.inline >}}
<p>
Suppose we approximate \( \mu(x),\sigma(x) \) with a neural network:
</p>
{{</ math.inline >}}

$$
x \rarr \text{NN}(\theta) \rarr \begin{bmatrix}
    \mu_x \\\
    \sigma_x
\end{bmatrix} \rarr f(z,\mu,\sigma) \rarr y
$$

{{< math.inline >}}
<p>
Define the same loss function \(J(y)\):
</p>
{{</ math.inline >}}

$$
J(y)=\frac{1}{N}\sum_{i=1}^N\Vert y-y^{(i)} \rVert^2
$$

{{< math.inline >}}
<p>
Perform gradient descent to minimize this loss:
</p>
{{</ math.inline >}}

$$
\nabla_{\theta}J(y) = \frac{\nabla J(y)}{\nabla y}\frac{\nabla y}{\nabla \mu(x)}\frac{\nabla \mu(x)}{\nabla \theta} + \frac{\nabla J(y)}{\nabla y}\frac{\nabla y}{\nabla \sigma(x)}\frac{\nabla \sigma(x)}{\nabla \theta}
$$

### Deep generative models

<cite>[^2]</cite>

$$
\text{Maximum likelihood} \begin{cases}
    \text{likelihood-based} \begin{cases}
        \text{explicit density} \begin{cases}
            \text{exact inference} 
            \begin{cases}
                \text{fully-observed} : \text{Autoregressive} \\\
                \text{change of variable} : \text{Flow-based},\text{Nonlinear ICA}
            \end{cases}\\\
            \text{approximate inference} \begin{cases}
                \text{variational inference} : \text{VAE} \\\
                \text{MCMC} : \text{Energy-based}
            \end{cases}
        \end{cases}
    \end{cases} \\\
    \text{likelihood-free} \begin{cases}
        \text{implicit density} \begin{cases}
            \text{direct sample} : \text{GAN} \\\
            \text{MCMC} : \text{GSN}
        \end{cases}
    \end{cases}
\end{cases}
$$

GANs are likelihood-free models, which do not model the probability density.

## GAN Model Definition
### Generator and discriminator

{{< math.inline >}}
<p>
There are two roles in GAN: generator and discriminator. Given real data samples:
</p>
{{</ math.inline >}}

$$
P_{\text{data}} : \lbrace x_i \rbrace_{i=1}^N
$$


As a adversarial game, the generator tries to generate fake data samples to imitate the real ones so that discriminator cannot figure it out. The generation starts from a simple distribution ([Reparameterization trick](#stochastic-back-propagation-reparametrization-trick)).

$$
z \rarr \text{NN}(\theta_g) \rarr x \\\
G(z;\theta_g) = \text{NN}(\theta_g) \\\
z \sim p(z|\theta_z), x=G(z;\theta_g) \sim p(x|\theta_g)  \\\
$$

{{< math.inline >}}
<p>
\( G(z;\theta_g) \) is the generator function approximated by a nerual network. As mentioned before, \( x \) comes from easy to sample \( z \).
</p>
{{</ math.inline >}}

{{< math.inline >}}
<p>
The discriminator tries to dsicriminate every real and fake sample accurately. It uses a nerual network to approximate the discriminator function: \( D(x;\theta_d) \).
</p>
{{</ math.inline >}}

$$
x\in \lbrace P_{\text{data}}, G(z) \rbrace \rarr \text{NN}(\theta_d) \rarr Pr(x\text{ is real}) \\\
D(x;\theta_d) = \text{NN}(\theta_d)
$$

### Objective function

For the generator, the object is clear, we want discriminator has the lowest accuracy recognizing fake samples:

$$
\text{If $x\in G(z)$}, D(x) \rarr 1 \\\
\min_{G} \mathbb{E}\_{z\sim p(z|\theta_z)}\left[\log\left(1-D(G(z;\theta_g);\theta_d)\right)\right]
$$

For the discriminator, we want discriminator has the highest accuracy recognizing fake or real samples:

$$
\begin{cases}
\text{If $x\in G(z)$}, D(x) \rarr 0 \\\
\text{If $x\in P_{\text{data}}$}, D(x) \rarr 1
\end{cases} \\\
\max_{D} \mathbb{E}\_{z\sim p(z|\theta_z)}\left[\log\left(1-D(G(z;\theta_g);\theta_d)\right)\right] + \mathbb{E}\_{x\sim P_{\text{data}}}\left[ \log D(x;\theta_d) \right]
$$

Combining together, the total objective function is given by:

$$
\min_{G}\max_{D} \mathbb{E}\_{z\sim p(z|\theta_z)}\left[\log\left(1-D(G(z;\theta_g);\theta_d)\right)\right] + \mathbb{E}\_{x\sim P_{\text{data}}}\left[ \log D(x;\theta_d) \right]
$$

## Global Optimal Solution

Let:

$$
V(D,G) = \mathbb{E}\_{z\sim p(z|\theta_z)}\left[\log\left(1-D(G(z;\theta_g);\theta_d)\right)\right] + \mathbb{E}\_{x\sim P_{\text{data}}}\left[ \log D(x;\theta_d) \right]
$$

The optimization object becomes:

$$
\min_{G}\max_{D}V(D,G)
$$

For any minimax problem, we can solve inner problem first then solve outer problem:

$$
\begin{align*}
    &1. f(G) = \max_{D}V(D,G) \\\
    &2. \min_{G} f(G)
\end{align*}
$$

{{< math.inline >}}
<p>
The solution for inner problem is given by:
</p>
{{</ math.inline >}}

$$
f(G) = \max_{D}V(D,G) \rArr \frac{\partial }{\partial D}V(D,G)=0 \\\
\begin{align*}
\frac{\partial }{\partial D}V(D,G) &= \frac{\partial }{\partial D} \left[\mathbb{E}\_{z\sim p(z|\theta_z)}\left[\log\left(1-D(G(z;\theta_g);\theta_d)\right)\right] + \mathbb{E}\_{x\sim P_{\text{data}}}\left[ \log D(x;\theta_d) \right] \right] \\\
&= \frac{\partial }{\partial D} \left[ \mathbb{E}\_{x\sim p(x|\theta_g)}\left[\log\left(1-D(x;\theta_d)\right)\right] + \mathbb{E}\_{x\sim P_{\text{data}}}\left[ \log D(x;\theta_d) \right] \right] \\\
&\text{Let } D = D(x;\theta_d) \\\
&= \frac{\partial }{\partial D} \left[\int P_{\text{data}}\log D dx + \int p(x|\theta_g)\log(1-D)dx \right] \\\
&\because\text{Leibnitz intergral rule} \\\
&= \int \frac{\partial }{\partial D}P_{\text{data}}\log D + \frac{\partial }{\partial D}p(x|\theta_g)\log(1-D) dx \\\
&= \int P_{\text{data}}\frac{1}{D} + p(x|\theta_g)\frac{1}{1-D} dx \\\
\\\
\frac{\partial }{\partial D}V(D,G) &= 0 \\\
P_{\text{data}}\frac{1}{D} &= p(x|\theta_g)\frac{1}{1-D} \\\
\frac{1-D}{D} &= \frac{p(x|\theta_g)}{P_{\text{data}}} \\\
\hat{D} &= \frac{P_{\text{data}}}{p(x|\theta_g)+P_{\text{data}}}
\end{align*}
$$

The solution for outer problem is given by:

$$
\min_{G}f(G) \rArr \min_{G}V(\hat{D},G) \\\
\begin{align*}
\min_{G}V(\hat{D},G) &= \min_{G} \mathbb{E}\_{x\sim p(x|\theta_g)}\left[\log\left(1-\hat{D}\right)\right] + \mathbb{E}\_{x\sim P_{\text{data}}}\left[ \log \hat{D} \right] \\\
&= \min_{G} \mathbb{E}\_{x\sim p(x|\theta_g)}\left[\log\frac{p(x|\theta_g)}{p(x|\theta_g)+P_{\text{data}}}\right] + \mathbb{E}\_{x\sim P_{\text{data}}}\left[ \log \frac{P_{\text{data}}}{p(x|\theta_g)+P_{\text{data}}} \right] \\\
&= \min_{G} \mathbb{E}\_{x\sim p(x|\theta_g)}\left[\log\frac{p(x|\theta_g)}{(p(x|\theta_g)+P_{\text{data}})\frac{1}{2}}\frac{1}{2}\right] + \mathbb{E}\_{x\sim P_{\text{data}}}\left[ \log \frac{P_{\text{data}}}{(p(x|\theta_g)+P_{\text{data}})\frac{1}{2}}\frac{1}{2} \right] \\\
&\because \int \frac{1}{2}(p(x|\theta_g)+P_{\text{data}}) dx = \frac{1}{2}\int p(x|\theta_g)dx + \frac{1}{2}\int P_{\text{data}}dx= 1 \\\
&\because \mathbb{E}\_{x\sim p(x|\theta_g)}\left[\log f(x)\frac{1}{2}\right] = \mathbb{E}\_{x\sim p(x|\theta_g)}\left[\log f(x)\right]+\mathbb{E}\_{x\sim p(x|\theta_g)}\left[\log\frac{1}{2}\right] \\\
&= \min_{G} \text{KL}(p(x|\theta_g)||\frac{p(x|\theta_g)+P_{\text{data}}}{2}) + \text{KL}(P_{\text{data}}||\frac{p(x|\theta_g)+P_{\text{data}}}{2}) - 2\log2\\\
&\geq -2\log2 \\\
\\\
V(\hat{D},G) &= -2\log2,\text{ when } \\\
p(x|\theta_g) &=\frac{p(x|\theta_g)+P_{\text{data}}}{2} = P_{\text{data}} \\\
\hat{p}(x|\theta_g) &= P_{\text{data}}
\end{align*}
$$

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
