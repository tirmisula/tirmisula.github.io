---
author: "X. Wang"
title: "Confronting the Partition Function"
date: "2023-09-25"
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

## Introduction to Partition Function

<cite>[^1]</cite>

Any Directed Acyclic Graph(DAG) has a topological sort, that is to say our [directed probalistic graphical model is factorable by a topological ordering](https://tirmisula.github.io/posts/probabilistic-graphical-model/#definition):

$$
p(x_1,\cdots,x_p)=\prod_{i=1}^p p(x_i|x_{pa(i)}), \text{node $x_{pa}(i)\rarr$node $x_i$}
$$

Local structures in Bayesian Networks(BN) are still probalistic models. However it is not applicable for [undirected probalistic graphical model](https://tirmisula.github.io/posts/probabilistic-graphical-model/#factorization-of-mrf):

$$
p(x) = \frac{1}{z} \prod_{i=1}^K \psi(x_{C_i})
$$

Local structures of MRF are not probabilties, only the whole graph is considered as probability model. Thus partition function is introduced.

Partition function exists in:

$$
\begin{cases}
    \text{learning} : \hat{\theta} = \argmax_{\theta} p(O|\theta) \\\
    \text{evaluation} : p(O=o_1,\cdots,o_t|\theta)
\end{cases}
$$

## The Log-likelihood Gradient of Undirected graph

Given data:

$$
X = \lbrace x_1,\cdots,x_N \rbrace \\\
x_i \in \mathbb{R}^p, x_i \in \lbrace 0,1 \rbrace
$$

Suppose it is undirected graph model, we have likelihood function:

$$
\begin{align*}
p(x|\theta) &= \frac{1}{z(\theta)} \hat{p}(x|\theta) \\\
z(\theta) &= \int \hat{p}(x|\theta)dx
\end{align*}
$$

The learning problem by maximum likelihood estimation is:

$$
\begin{align*}
\hat{\theta} &= \argmax_{\theta} p(x|\theta) \\\
&= \argmax_{\theta} \prod_{i=1}^N p(x_i|\theta) \\\
&= \argmax_{\theta} \sum_{i=1}^N \log\left( p(x_i|\theta)  \right) \\\
&= \argmax_{\theta} \sum_{i=1}^N\left[\log\left( \hat{p}(x_i|\theta)\right) - \log\left(z(\theta)  \right)\right] \\\
&= \argmax_{\theta} \frac{1}{N}\sum_{i=1}^N\log(\hat{p}(x_i|\theta)) - \log(z(\theta)) \\\
\\\
&\text{Let } \mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^N\log(\hat{p}(x_i|\theta)) - \log(z(\theta)) \\\
&= \argmax_{\theta} \mathcal{L}(\theta)
\end{align*}
$$

The log-likelihood gradient is:

$$
\begin{align*}
\nabla_{\theta}\mathcal{L}(\theta) &= \frac{1}{N}\sum_{i=1}^N\nabla_{\theta}\log(\hat{p}(x_i|\theta)) - \nabla_{\theta}\log(z(\theta)) \\\
&\because\frac{1}{N}\sum_{i=1}^N\nabla_{\theta}\log(\hat{p}(x_i|\theta)) \text{ is computable} \\\
&\therefore \text{focus on } \nabla_{\theta}\log(z(\theta)) \\\
\\\
\nabla_{\theta}\log(z(\theta)) &= \frac{1}{z(\theta)}\nabla_{\theta}z(\theta) \\\
&= \frac{p(x|\theta)}{\hat{p}(x|\theta)} \nabla_{\theta}\int \hat{p}(x|\theta)dx \\\
&\because\text{Leibniz integral rule} \\\
&= \frac{p(x|\theta)}{\hat{p}(x|\theta)} \int\nabla_{\theta} \hat{p}(x|\theta)dx \\\
&= \int \frac{p(x|\theta)}{\hat{p}(x|\theta)}\nabla_{\theta}\hat{p}(x|\theta)dx \\\
&= \int p(x|\theta)\nabla_{\theta}\log\hat{p}(x|\theta)dx \\\
&= \mathbb{E}\_{x \sim p(x | \theta)} \left[ \nabla_{\theta}\log\hat{p}(x|\theta) \right]
\end{align*}
$$

{{< math.inline >}}
<p>
We avoid computing intractable \( z(\theta) \), expectation approximation method like MCMC is available for the distribution \( x \sim p(x|\theta) \), so computing integral is also avoided.
</p>
{{</ math.inline >}}

## Gradient Ascent based on MCMC for MLE

The **empirical distribution** associated with a vector of numbers x = (x1,...,xn) is the probability distribution with expectation operator<cite>[^2]</cite>:

$$
\mathbb{E}[g(x)] = \frac{1}{n}\sum_{i=1}^ng(x_i)
$$

{{< math.inline >}}
<p>
So \( \frac{1}{N}\sum_{i=1}^N\nabla_{\theta}\log(\hat{p}(x_i|\theta)) \) can be considered as sampling from empirical distribution \( p(\text{data}) \):
</p>
{{</ math.inline >}}

$$
\frac{1}{N}\sum_{i=1}^N\nabla_{\theta}\log(\hat{p}(x_i|\theta)) = \mathbb{E}\_{x\sim p(\text{data})}[\nabla_{\theta}\log(\hat{p}(x|\theta))] 
$$

So the log-likelihood gradient becomes:

$$
\nabla_{\theta}\mathcal{L}(\theta) = \mathbb{E}\_{x\sim p(\text{data})}[\nabla_{\theta}\log(\hat{p}(x|\theta))] - \mathbb{E}\_{x\sim p(x|\theta)} \left[ \nabla_{\theta}\log\hat{p}(x|\theta) \right] \\\
\\\
\begin{cases}
    p(\text{data}) &: \text{empirical distribution} \\\
    p(x|\theta) &: \text{assumption distribution}
\end{cases}
$$

{{< math.inline >}}
<p>
The essence of MLE is to let proposed distribution \( p(x|\theta) \) to approximate empirical distribution \( p(\text{data}) \).
</p>
{{</ math.inline >}}

We use [Gibbs sampling](https://tirmisula.github.io/posts/markov-chain-monte-carlo/#gibbs-algorithm)(MCMC method) to approximate expectation of real and assumption model, suppose we sample M particles:

$$
\begin{rcases}
&\tilde{x}_1 &\sim p(\text{data}) \\\
&&\cdots \\\
&\tilde{x}_M &\sim p(\text{data})
\end{rcases} : \mathbb{E}\_{x\sim p(\text{data})}[\nabla_{\theta}\log(\hat{p}(x|\theta))] \approx \frac{1}{M}\sum_{i=1}^M\nabla_{\theta}\log(\hat{p}(\tilde{x}_i|\theta))
$$

$$
\begin{rcases}
&\hat{x}_1 &\sim p(x|\theta) \\\
&&\cdots \\\
&\hat{x}_M &\sim p(x|\theta)
\end{rcases} : \mathbb{E}\_{x\sim p(x|\theta)}[\nabla_{\theta}\log(\hat{p}(x|\theta))] \approx \frac{1}{M}\sum_{i=1}^M\nabla_{\theta}\log(\hat{p}(\hat{x}_i|\theta))
$$

Combining with gradient ascent algorithm, we have:

$$
\text{Gradient Ascent by Gibbs} \\\
\begin{align*}
&\text{For each turn } t=1\cdots T \\\
&1. \text{ sampling directly from real data: } \\\
    &\quad \tilde{x}_{1:M} \sim p(\text{data}) \\\
&2. \text{ sampling by gibbs algorithm : } \\\
    &\quad \hat{x}_{1:M} \sim p(x|\theta^{(t)}) \\\
&3. \text{update parameters until converge: } \\\
    &\quad \theta^{(t+1)} = \theta^{(t)} + \eta\left( \sum_{i=1}^M\nabla_{\theta}\log(\hat{p}(\tilde{x}_i|\theta^{(t)}))-\sum_{i=1}^M\nabla_{\theta}\log(\hat{p}(\tilde{x}_i|\theta^{(t)})) \right)
\end{align*}
$$

## Contrastive Divergence Algorithm
### Problem in Gibbs sampling
{{< math.inline >}}
<p>
Gibbs sampling involves multiple steps for it's Markov chain to reach steady distribution. For each single step k before steady, it has unsteady distribution, denoted as:
</p>
{{</ math.inline >}}

$$
\underset{\text{k-th step}}{p^{(k)}(x|\theta)}
$$

{{< math.inline >}}
<p>
On the other hand, since \( x \) has \( p \) dimensions, \( p \) state transitions happened in it's Markov chain during step k to step k+1:
</p>
{{</ math.inline >}}

$$
\underset{\text{k-th step}}{p^{(k)}(x|\theta)} \longrightarrow \underset{\text{(k+1)th step}}{p^{(k)}(x|\theta)} : \begin{array}{c}
    \text{state}_{k*p} \\\
    \darr \\\
    \cdots \\\
    \darr \\\
    \text{state}_{(k+1)*p}
\end{array}
$$

{{< math.inline >}}
<p>
The whole sampling process is:
</p>
{{</ math.inline >}}

$$
\underset{\text{initial}}{p^{(0)}} \rarr \underset{\text{1-step}}{p^{(1)}(x|\theta)} \rarr \cdots \rarr \underset{\text{$\infty$-step}}{p^{(\infty)}(x|\theta)} \rarr \hat{x}_i
$$

{{< math.inline >}}
<p>
However the mixing time(refers to \( p^{(0)}\rarr p^{(\infty)} \)) for each data sample could be huge, which requires plenty of computation resources.
</p>
{{</ math.inline >}}

### CD-k
{{< math.inline >}}
<p>
To shorten mixing time, consider two options:
    1. select the initial distribution of \( \hat{x} \) close to \( p(\text{data}) \) 2. sample earlier before distribution is steady
</p>
{{</ math.inline >}}

The CD-k algorithm is defined as:

$$
\text{The k step Contrastive Divergence algorithm(CD-k)} \\\
\begin{align*}
&\text{For each turn } t=1\cdots T \\\
&1. \text{ sampling directly from real data: } \\\
    &\quad \tilde{x}_{1:M} \sim p(\text{data}) \\\
&2. \text{ initialize $\hat{x}$ : } \\\
    &\quad \hat{x}_{1:M} = \tilde{x}_{1:M} \\\
&3. \text{ sampling k steps by gibbs: } \\\
    &\quad \hat{x}_{1:M} \sim p^{(k)}(x|\theta^{(t)}) \\\
&4. \text{update parameters until converge: } \\\
    &\quad \theta^{(t+1)} = \theta^{(t)} + \eta\left( \sum_{i=1}^M\nabla_{\theta}\log(\hat{p}(\tilde{x}_i|\theta^{(t)}))-\sum_{i=1}^M\nabla_{\theta}\log(\hat{p}(\tilde{x}_i|\theta^{(t)})) \right)
\end{align*}
$$


## Contrative divergence vs KL divergence
### KL divergence in MLE
In original maximum likelihood estimation we have:

$$
\begin{align*}
\hat{\theta} &= \argmax_{\theta} p(x|\theta) \\\
&= \argmax_{\theta} \prod_{i=1}^N p(x_i|\theta) \\\
&= \argmax_{\theta} \sum_{i=1}^N \log\left( p(x_i|\theta)  \right) \\\
&= \argmax_{\theta} \frac{1}{N}\sum_{i=1}^N \log\left( p(x_i|\theta)  \right) \\\
&= \argmax_{\theta}\mathbb{E}\_{x\sim p(\text{data})}[\log(p(x|\theta))] \\\
&= \argmax_{\theta} \int p(\text{data})\log(p(x|\theta)) dx \\\
&= \argmax_{\theta} \int p(\text{data})\log p(x|\theta) dx - \int p(\text{data})\log p(\text{data}) dx \\\
&= \argmax_{\theta} \int p(\text{data})\log\frac{ p(x|\theta)}{ p(\text{data})} dx \\\
&= \argmax_{\theta} -\text{KL}(p(\text{data}) || p(x|\theta)) \\\
&= \argmin_{\theta} \space\text{KL}(p(\text{data}) || p(x|\theta)) \\\
\end{align*}
$$

Using a partition function, the MLE object is given by [graident ascent](#gradient-ascent-algorithm-based-on-mcmc) from above section:

$$
\begin{align*}
\nabla_{\theta}\mathcal{L}(\theta) &= \mathbb{E}\_{x\sim p(\text{data})}[\nabla_{\theta}\log(\hat{p}(x|\theta))] - \nabla_{\theta}\log(z(\theta)) \\\
&= \mathbb{E}\_{x\sim p(\text{data})}[\nabla_{\theta}\log(\hat{p}(x|\theta))] - \mathbb{E}\_{x\sim p(x|\theta)} \left[ \nabla_{\theta}\log\hat{p}(x|\theta) \right] \\\
&\because \text{Leibniz integral rule} \\\
&= \nabla_{\theta}\mathbb{E}\_{x\sim p(\text{data})}[\log\hat{p}(x|\theta)] - \nabla_{\theta}\mathbb{E}\_{x\sim p(x|\theta)} \left[ \log\hat{p}(x|\theta) \right] \\\
\mathcal{L}(\theta) &= \mathbb{E}\_{x\sim p(\text{data})}[\log\hat{p}(x|\theta)] - \mathbb{E}\_{x\sim p(x|\theta)} \left[ \log\hat{p}(x|\theta) \right] \\\
&\text{by Gibbs sampling} \\\
&\approx \mathbb{E}\_{x\sim p(\text{data})}[\log\hat{p}(x|\theta)] - \mathbb{E}\_{x\sim p^{(\infty)}(x|\theta)} \left[ \log\hat{p}(x|\theta) \right] \\\
&\approx -\text{KL}(p^{(0)} || \hat{p}(x|\theta)) + \text{KL}(p^{(\infty)}(x|\theta) || \hat{p}(x|\theta)) \\\
\\\
\argmax_{\theta}\mathcal{L}(\theta) &\approx \argmin_{\theta} \text{KL}(p^{(0)} || \hat{p}(x|\theta)) - \text{KL}(p^{(\infty)}(x|\theta) || \hat{p}(x|\theta))
\end{align*}
$$


In conlusion, MLE can be expressed by KL divergence with the following form:

$$
\begin{cases}
    \mathcal{L}(\theta) = -\text{KL}(p^{(0)} || p(x|\theta)) \\\
    \mathcal{L}(\theta) \approx -\text{KL}(p^{(0)} || \hat{p}(x|\theta)) + \text{KL}(p^{(\infty)}(x|\theta) || \hat{p}(x|\theta))
\end{cases}
$$

$$
\begin{cases}
    \hat{\theta}_{\text{MLE}} = \argmin_{\theta} \space\text{KL}(p^{(0)} || p(x|\theta)) \\\
    \hat{\theta}_{\text{MLE}} \approx \begin{cases}
        \argmin_{\theta} \space\text{KL}(p^{(0)} || \hat{p}(x|\theta)) - \text{KL}(p^{(\infty)}(x|\theta) || \hat{p}(x|\theta)) \\\
        \text{OR} \\\
        \argmin_{\theta} \space\text{KL}(p^{(0)} || \hat{p}(x|\theta)) + \log z(\theta)
    \end{cases}
\end{cases}
$$

<cite>[^3]</cite>

{{< math.inline >}}
<p>
we can directly perceive that MLE is equivalent to finding a model that minimizes difference from empirical distribution. If model is unnormalized, minimize difference from empirical distribution and minimize the log-partition function \( \log z(\theta) \).
</p>
{{</ math.inline >}}

### CD learning object

{{< math.inline >}}
<p>
In ML(maximum likelihood) learning, computing \( p^{(\infty)} \) could be costly, we learn it on k-th step \( p^{(\infty)} \) which is called Contrastive learning(CD learning):
</p>
{{</ math.inline >}}

$$
\hat{\theta}_{\text{CDL}} = \argmin_{\theta} \space\text{KL}(p^{(0)} || \hat{p}(x|\theta)) - \text{KL}(p^{(k)}(x|\theta) || \hat{p}(x|\theta)) \\\
$$

## RBM Learning
### Review RBM model
We have introduced [RBM model](https://tirmisula.github.io/posts/restricted-boltzman-machine/#rbm-model-definition) in previous chapter:

$$
\text{Given: }
x = \begin{bmatrix}
    x_1 \\\
    \vdots \\\
    x_p
\end{bmatrix}=\begin{bmatrix}
    h \\\
    o
\end{bmatrix}, h = \begin{bmatrix}
    h_1 \\\
    \vdots \\\
    h_m
\end{bmatrix}, o = \begin{bmatrix}
    o_1 \\\
    \vdots \\\
    o_n
\end{bmatrix}, m+n=p \\\
\alpha = \begin{bmatrix}
    \alpha_1 \\\
    \vdots \\\
    \alpha_n
\end{bmatrix}, \beta = \begin{bmatrix}
    \beta_1 \\\
    \vdots \\\
    \beta_m
\end{bmatrix}, w = \begin{bmatrix}
    w_{11} & \cdots & w_{1n} \\\
    \vdots & \ddots & \vdots \\\
    w_{m1} & \cdots & w_{mn}
\end{bmatrix}, \theta = \lbrace \alpha,\beta,w  \rbrace \\\
$$

$$
\begin{align*}
p(x|\theta)=p(o,h) &= \frac{1}{z}\exp(-E(o,h)) \\\
&= \frac{1}{z}\exp(h^Two+\alpha^To+\beta^Th) \\\
&= \frac{1}{z} \prod_{i=1}^m\prod_{j=1}^n\exp(h_iw_{ij}o_j) \prod_{i=1}^m\exp(\beta_i h_i) \prod_{j=1}^n\exp(\alpha_j o_j)
\end{align*}
$$

### The Log-likelihood gradient of energy-based model

A more general form is the energy-based model with latent variables:

$$
\begin{align*}
p(o,h) = \frac{1}{z}\exp\left(-E(o,h)\right)
\end{align*}
$$

{{< math.inline >}}
<p>
For energy-based model (suppose N samples), the average log-likelihood of observations is given by:
</p>
{{</ math.inline >}}


$$
\begin{align*}
    \mathcal{L}(\theta) &= \frac{1}{N}\log\prod_{i=1}^N p(o^{(i)}) \\\
    &= \frac{1}{N} \sum_{i=1}^N \log \sum_h\frac{1}{z}\exp\left(-E(o^{(i)},h^{(i)})\right)
\end{align*}
$$

The log-likellihood gradient is given by:

$$
\begin{align*}
\nabla_{\theta}\mathcal{L}(\theta) &= \frac{1}{N} \sum_{i=1}^N\nabla_{\theta}\log\sum_h\frac{1}{z}\exp\left( -E(o^{(i)}, h^{(i)}) \right) \\\
&\text{Let } \nabla_{\theta}\mathcal{L}_i(\theta) = \nabla_{\theta}\log\sum_h\frac{1}{z}\exp\left( -E(o^{(i)}, h^{(i)}) \right) \\\
\\\
\nabla_{\theta}\mathcal{L}_i(\theta) &= \nabla_{\theta}\log\sum_h\exp\left(-E(o^{(i)},h^{(i)})\right) - \nabla_{\theta}\log\sum_h z \\\
&= \nabla_{\theta}\log\sum_h\exp\left(-E(o^{(i)},h^{(i)})\right) - \nabla_{\theta}\log z \\\
&\because \int\frac{1}{z}\exp\left( -E(o,h) \right)dodh = 1 \\\
&= \nabla_{\theta}\log\sum_h\exp\left(-E(o^{(i)},h^{(i)})\right) - \nabla_{\theta}\log \sum_{o,h}\exp\left(-E(o,h)\right) \\\
&\text{Let } \text{part1} = \nabla_{\theta}\log\sum_h\exp\left(-E(o^{(i)},h^{(i)})\right) \\\
&\text{Let } \text{part2} = \nabla_{\theta}\log \sum_{o,h}\exp\left(-E(o,h)\right) \\\
\text{part1} &= \frac{1}{\sum_h\exp\left( -E(o^{(i)},h^{(i)}) \right)}\nabla_{\theta}\sum_h\exp\left( -E(o^{(i)},h^{(i)}) \right) \\\
&= \frac{1}{\sum_h\exp\left( -E(o^{(i)},h^{(i)}) \right)}\sum_h\nabla_{\theta}\exp\left( -E(o^{(i)},h^{(i)}) \right) \\\
&= \frac{-1}{\sum_h\exp\left( -E(o^{(i)},h^{(i)}) \right)}\sum_h\exp\left( -E(o^{(i)},h^{(i)}) \right)\nabla_{\theta}E(o^{(i)},h^{(i)}) \\\
&= -\sum_h\frac{\exp\left( -E(o^{(i)},h^{(i)}) \right)}{\sum_h \exp\left( -E(o^{(i)},h^{(i)}) \right)} \nabla_{\theta}E(o^{(i)},h^{(i)}) \\\
&= -\sum_h\frac{\frac{1}{z}\exp\left( -E(o^{(i)},h^{(i)}) \right)}{\frac{1}{z}\sum_h \exp\left( -E(o^{(i)},h^{(i)}) \right)} \nabla_{\theta}E(o^{(i)},h^{(i)}) \\\
&= -\sum_h\frac{p(o^{(i)},h^{(i)})}{p(o^{(i)})} \nabla_{\theta}E(o^{(i)},h^{(i)}) \\\
&= -\sum_h p(h^{(i)}|o^{(i)}) \nabla_{\theta}E(o^{(i)},h^{(i)}) \\\
\text{part2} &= \frac{1}{\sum_{o,h}\exp\left(-E(o,h)\right)}\nabla_{\theta}\sum_{o,h}\exp\left(-E(o,h)\right) \\\
&= \frac{1}{\sum_{o,h}\exp\left(-E(o,h)\right)}\sum_{o,h}\nabla_{\theta}\exp\left(-E(o,h)\right) \\\
&= \frac{-1}{\sum_{o,h}\exp\left(-E(o,h)\right)}\sum_{o,h}\exp\left(-E(o,h)\right)\nabla_{\theta}E(o,h) \\\
&= -\sum_{o,h}\frac{\exp\left(-E(o,h)\right)}{\sum_{o,h}\exp\left(-E(o,h)\right)}\nabla_{\theta}E(o,h) \\\
&= -\sum_{o,h}\frac{\exp\left(-E(o,h)\right)}{\sum_{o,h}\exp\left(-E(o,h)\right)}\nabla_{\theta}E(o,h) \\\
&= -\sum_{o,h}\frac{p(o,h)}{\sum_{o,h}p(o,h)}\nabla_{\theta}E(o,h) \\\
&= -\sum_{o,h} p(o,h)\nabla_{\theta}E(o,h)
\end{align*}
$$

In conlusion, we have the gradient:

$$
\begin{align*}
\nabla_{\theta}\mathcal{L}(\theta) &= \frac{1}{N}\sum_{i=1}^N\nabla_{\theta}\mathcal{L}_i(\theta) \\\
&= \frac{1}{N}\sum_{i=1}^N\left( \sum_{o,h}p(o,h)\nabla_{\theta}E(o,h)-\sum_{h}p(h^{(i)}|o^{(i)})\nabla_{\theta}E(o^{(i)},h^{(i)}) \right)
\end{align*}
$$

### The Log-likelihood gradient of RBM
Baed on the conlusion from [last section](#the-log-likelihood-gradient-of-energy-based-model), the partial derivative of RBM can be given by:

$$
\begin{align*}
\frac{\partial}{\partial w_{ij}} \mathcal{L}_k(\theta) &= \sum_{o,h}p(o,h)\frac{\partial}{\partial w_{ij}}E(o,h) - \sum_{h}p(h^{(k)}|o^{(k)})\frac{\partial}{\partial w_{ij}}E(o^{(k)},h^{(k)}) \\\
&\because E(o,h) = -(h^Two+\alpha^To+\beta^Th) \\\
&\therefore \frac{\partial}{\partial w_{ij}}E(o,h) = -\frac{\partial}{\partial w_{ij}}\sum_{i=1}^m\sum_{j=1}^n h_iw_{ij}o_j = -h_io_j \\\
&= -\sum_{o,h}p(o,h)h_io_j + \sum_{h}p(h^{(k)}|o^{(k)})h^{(k)}_io^{(k)}_j \\\
&\text{Let } \text{part1}=\sum_{o,h}p(o,h)h_io_j \\\
&\text{Let } \text{part2}=\sum_{h}p(h^{(k)}|o^{(k)})h^{(k)}_io^{(k)}_j \\\
\text{part1} &= \sum_{o_1\cdots o_n}\sum_{h_1\cdots h_m}p(o)p(h|o)h_io_j \\\
&= \sum_{o_1\cdots o_n}p(o)\sum_{h_1\cdots h_m}p(h_{1:m}|o)h_io_j \\\
&= \sum_{o_1\cdots o_n}p(o)\sum_{h_i}p(h_i|o)h_io_j \\\
&\because h\in\lbrace 0,1 \rbrace \text{ in RBM} \\\
&= \sum_{o_1\cdots o_n}p(o)p(h_i=1|o)o_j \\\
\text{part2} &= \sum_{h_1\cdots h_m}p(h^{(k)}_{1:m}|o^{(k)})h^{(k)}_io^{(k)}_j \\\
&= \sum_{h_i}p(h^{(k)}_{i}|o^{(k)})h^{(k)}_io^{(k)}_j \\\
&= p(h^{(k)}_{i}=1|o^{(k)})o^{(k)}_j \\\
\\\
&\text{so we have} \\\
\frac{\partial}{\partial w_{ij}} \mathcal{L}_k(\theta) &= -\sum_{o_1\cdots o_n}p(o)p(h_i=1|o)o_j + p(h^{(k)}_{i}=1|o^{(k)})o^{(k)}_j
\end{align*}
$$

{{< math.inline >}}
<p>
Similarly, the partial derivatives of \(\alpha_j\) and \(\beta_i\) are given by:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial \alpha_{j}} \mathcal{L}_k(\theta) &= -\sum_{o,h}p(o,h)o_j + \sum_{h}p(h^{(k)}|o^{(k)})o^{(k)}_j \\\
&= -\sum_{o_1\cdots o_n}p(o)\sum_{h_1\cdots h_m}p(h_{1:m}|o)o_j + \sum_{h_1\cdots h_m}p(h^{(k)}_{1:m}|o^{(k)})o^{(k)}_j \\\
&= -\sum_{o_1\cdots o_n}p(o)o_j + o^{(k)}_j \\\
\frac{\partial}{\partial \beta_{i}} \mathcal{L}_k(\theta) &= -\sum_{o,h}p(o,h)h_i + \sum_{h}p(h^{(k)}|o^{(k)})h^{(k)}_i \\\
&= -\sum_{o_1\cdots o_n}p(o)\sum_{h_1\cdots h_m}p(h_{1:m}|o)h_i + \sum_{h_1\cdots h_m}p(h^{(k)}_{1:m}|o^{(k)})h^{(k)}_i \\\
&= -\sum_{o_1\cdots o_n}p(o)\sum_{h_i}p(h_i|o)h_i + \sum_{h_i}p(h^{(k)}_{i}|o^{(k)})h^{(k)}_i \\\
&= -\sum_{o_1\cdots o_n}p(o)p(h_i=1|o) + p(h^{(k)}_{i}=1|o^{(k)})
\end{align*}
$$

In conlusion, we have the gradient for RBM:

$$
\begin{align*}
\frac{\partial}{\partial w_{ij}}\mathcal{L}(\theta) &= \frac{1}{N}\sum_{k=1}^N\left( -\sum_{o_1\cdots o_n}p(o)p(h_i=1|o)o_j + p(h^{(k)}_{i}=1|o^{(k)})o^{(k)}_j \right) \\\
\frac{\partial}{\partial \alpha_{j}} \mathcal{L}_k(\theta) &= \frac{1}{N}\sum_{k=1}^N\left( -\sum_{o_1\cdots o_n}p(o)o_j + o^{(k)}_j \right) \\\
\frac{\partial}{\partial \beta_{i}}\mathcal{L}(\theta) &= \frac{1}{N}\sum_{k=1}^N\left( -\sum_{o_1\cdots o_n}p(o)p(h_i=1|o) + p(h^{(k)}_{i}=1|o^{(k)}) \right)
\end{align*}
$$

### CD-k for RBM

{{< math.inline >}}
<p>
Suppose we are finding \( \frac{\partial}{\partial w_{ij}}\mathcal{L}(\theta) \), denote:
</p>
{{</ math.inline >}}

$$
o^{(s)<k>}_{j} : \text{$s$-th observation, $j$-th dimension, sampled at $k$-th step by Gibbs}
$$

{{< math.inline >}}
<p>
We can see that \( \sum_{o_1\cdots o_n}p(o)p(h_i=1|o)o_j \) is intractable due to integration, while \( p(h^{(k)}_{i}=1|o^{(k)})o^{(k)}_j \) can be solved directly from observation tranining set:
</p>
{{</ math.inline >}}

$$
\begin{cases}
\sum_{o_1\cdots o_n}p(o)p(h_i=1|o)o_j &= \mathbb{E}\_{o^{<k>}}[p(h_i=1|o)o_j] \\\
&\approx \frac{1}{N}\sum_{s=1}^N p(h_{i}=1|o^{(s)<k>})o^{(s)<k>}_j \\\
\\\
p(h_{i}=1|o^{(s)})o^{(s)}_j &= p(h_{i}=1|o^{(s)<0>})o^{(s)<0>}_j
\end{cases}
$$

The CD-k algorithm for RBM is:

$$
\text{The Contrastive Divergence algorithm for RBM(CD-k)} \\\
\begin{align*}
&\text{For each turn } t=1\cdots T \\\
&1. \text{ sample N observations from training set $\lbrace O \rbrace$: } \\\
    &\quad  o^{(1)}_{1:n},\cdots,o^{(N)}_{1:n} \sim \lbrace O \rbrace \\\
&2. \text{ initialize $o^{(1)<0>}_{1:n},\cdots,o^{(N)<0>}_{1:n}$ : } \\\
    &\quad o^{(1)<0>}_{1:n},\cdots,o^{(N)<0>}_{1:n} = o^{(1)}_{1:n},\cdots,o^{(N)}_{1:n} \\\
&3. \text{ sampling from $p(h|o),p(o|h)$ alternatively, stop at k-step: } \\\
    &\quad \text{For } l=0\cdots k-1 \\\
    &\quad\quad \text{For }i=1\cdots m \\\
    &\quad\quad\quad h^{(1)<l>}_i,\cdots,h^{(N)<l>}_i \sim p(h_i|o^{(1)<l>}),\cdots,p(h_i|o^{(N)<l>}) \\\
    &\quad\quad \text{For }j=1\cdots n \\\
    &\quad\quad\quad o^{(1)<l+1>}_j,\cdots,o^{(N)<l+1>}_j \sim p(o_j|h^{(1)<l>}),\cdots,p(o_j|h^{(N)<l>}) \\\
&4. \text{ cumulate $\Delta w_{ij}$ from all samples: } \\\
    &\quad \text{For }i=1\cdots m,j=1\cdots n \\\
    &\quad\quad \text{For }s=1\cdots N \\\
    &\quad\quad\quad \Delta w_{ij} = \Delta w_{ij} + \left( p(h_{i}=1|o^{(s)<0>})o^{(s)<0>}_j - p(h_{i}=1|o^{(s)<k>})o^{(s)<k>}_j \right) \\\
&4. \text{ update parameters until converge: } \\\
    &\quad w^{(t+1)} = w^{(t)} + \eta\left( \frac{1}{N} \begin{bmatrix}
    \Delta w_{11} \\\
    \vdots \\\
    \Delta w_{mn}
\end{bmatrix} \right)
\end{align*}
$$

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=117).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^3]: [Confronting the PartitionFunction. Ian Goodfellow and Yoshua Bengio and Aaron Courville](https://www.deeplearningbook.org/contents/partition.html).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^2]: - [Empirical Distributions, Exact Sampling Distributions, Asymptotic Sampling Distributions. Charles J. Geyer](https://www.stat.umn.edu/geyer/5102/slides/s1.pdf).
