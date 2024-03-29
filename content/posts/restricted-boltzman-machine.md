---
author: "X. Wang"
title: "Ristricted Boltzman Machine"
date: "2023-09-24"
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

## Review Markov Random Field

<cite>[^1]</cite>

Markov Random Field or Markov Random Network is a kind of undirected probalistic graphical model. The key content is [factorization of MRF](https://tirmisula.github.io/posts/probabilistic-graphical-model/#factorization-of-mrf), which we have introduced before. According to Hammersley Clifford theorem<cite>[^2]</cite>, we can express MRF with potential function:

$$
\begin{align*}
p(x) &= \frac{1}{z} \prod_{i=1}^K \psi(x_{C_i})=\frac{1}{z}\prod_{i=1}^K\exp(-E(x_{C_i})) \\\
C_i &: \text{i-th maximum clique} \\\
x_{C_i} &: \text{variable nodes in $C_i$} \\\
\psi &: \text{potential function, $\psi>0$} \\\
\psi(x_{C_i}) &= \exp(-E(x_{C_i})) \\\
E(x) &: \text{energy function} \\\
z &: \text{nomalize factor, $z=\sum_{x_1\cdots x_p}\prod_{i=1}^K \psi(x_{C_i})$} 
\end{align*}
$$

{{< math.inline >}}
<p>
MRF probability model \( p(x) \) belongs to exponential family distribution, we call it Gibbs distribution or Boltzman distribution:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(x) &= \frac{1}{z} \exp( -\sum_{i=1}^K E(x_{C_i})) \\\
&\triangleq \frac{1}{z} \exp( -\mathrm{E}(x)) \\\
&= \frac{1}{z(\eta)}h(x)\exp( \eta^T\phi(x) )
\end{align*}
$$

**Boltzman machine** named after this Boltzman distribution which is equivalent to MRF + hidden nodes.

## RBM Model
### Inference trouble in Boltzman machine
Based on the statement above, nodes in Boltzman machine have 2 classes:

$$
\text{Nodes }x : \begin{cases}
    \text{observed } r.v. & o \\\
    \text{hidden } r.v. & h
\end{cases}
$$

We have:

$$
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
\end{bmatrix} \\\
p = m+n
$$

Inference problem in PGM is finding posterier, and we have introduced exact inference[(variable elimination)](https://tirmisula.github.io/posts/probabilistic-graphical-model/#variable-elimination) and approximate inference[(variational inference)](https://tirmisula.github.io/posts/variational-inference/#mean-field-vi-derivation)[(MCMC)](https://tirmisula.github.io/posts/markov-chain-monte-carlo/) in previous chapter::
$$
p(h|o) : \begin{cases}
    \text{exact inference} &: \text{variable elimination} \\\
    \text{approximate inference} &: \begin{cases}
            \text{variational method} \\\
            \text{MCMC}
        \end{cases}
\end{cases}
$$

However exact inference is intractable because the overall complexity is determined by the number of largest
elimination clique[3] and finding optimal elimination order is NP-hard.

Variational inference face the same problem when integrating:

$$
\int_{q_1\cdots q_N}q_1\cdots q_N\log p(o,h)dq_1\cdots dq_N
$$

MCMC method takes long time to converge which leads to high computation cost.

### Prediction

{{< math.inline >}}
<p>
For the new coming data \( x^{\ast} \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
f(x^{\ast}) &= \phi(x^{\ast})^Tw \\\
f(x^{\ast})|X,Y,x^{\ast} &\sim \mathcal{N}\left(\phi(x^{\ast})^T\mu_w, \phi(x^{\ast})^T\Sigma_{w}\phi(x^{\ast})\right)
\end{align*}
$$

$$
\begin{align*}
    \Sigma_w^{-1}\Sigma_q\Phi^T &= (\sigma^{-2}\Phi^T\Phi+\Sigma_q^{-1})\Sigma_q\Phi^T \\\
    &= \sigma^{-2}\Phi^T\Phi\Sigma_q\Phi^T + \Phi^T \\\
    &\text{Let } K = \Phi\Sigma_q\Phi^T \\\
    &= \sigma^{-2}\Phi^TK + \Phi^T \\\
    &= \sigma^{-2}\Phi^T(K+\sigma^2I)
\end{align*}
$$

For the mean we have:

$$
\begin{align*}
    \Sigma_{q}\Phi^T &= \sigma^{-2}\Sigma_w\Phi^T(K+\sigma^2I) \\\
    \Sigma_{q}\Phi^T(K+\sigma^2I)^{-1} &= \sigma^{-2}\Sigma_w\Phi^T \\\
    \Sigma_{q}\Phi^T(K+\sigma^2I)^{-1}Y &= \sigma^{-2}\Sigma_w\Phi^TY \\\
    \Sigma_{q}\Phi^T(K+\sigma^2I)^{-1}Y &= \mu_w \\\
    \phi(x^{\ast})^T\Sigma_{q}\Phi^T(K+\sigma^2I)^{-1}Y &= \phi(x^{\ast})^T\mu_w \\\
\end{align*}
$$

On the other hand, we have introduced [Woodbury formula](https://tirmisula.github.io/posts/gaussian-network/#marginal-pdf-in-gmrf) for calculating matrix inversion in previous chapter:

$$
A^{-1} + A^{-1}B(D - CA^{-1}B)^{-1}CA^{-1} = (A-BD^{-1}C)^{-1} \\\
    \dArr \\\
\Sigma_w = (\sigma^{-2}\Phi^T\Phi+\Sigma_q^{-1})^{-1} : \begin{cases}
    A = \Sigma_q^{-1} \\\
    B = \Phi^T \\\
    D = -\sigma^2I \\\
    C = \Phi
\end{cases} \\\
\dArr \\\
$$

$$
\begin{align*}
    \Sigma_w &= \Sigma_q + \Sigma_q\Phi^T(-\sigma^2I-\Phi\Sigma_q\Phi^T)^{-1}\Phi\Sigma_q \\\
    &= \Sigma_q - \Sigma_q\Phi^T(\sigma^2I+\Phi\Sigma_q\Phi^T)^{-1}\Phi\Sigma_q \\\
    &= \Sigma_q - \Sigma_q\Phi^T(\sigma^2I+K)^{-1}\Phi\Sigma_q \\\
\end{align*}
$$

So for the covariance we have:

$$
\begin{align*}
    \phi(x^{\ast})^T\Sigma_w\phi(x^{\ast}) &= \phi(x^{\ast})^T(\Sigma_q - \Sigma_q\Phi^T(\sigma^2I+K)^{-1}\Phi\Sigma_q)\phi(x^{\ast}) \\\
    &= \phi(x^{\ast})^T\Sigma_q\phi(x^{\ast}) - \phi(x^{\ast})^T\Sigma_q\Phi^T(\sigma^2I+K)^{-1}\Phi\Sigma_q\phi(x^{\ast})
\end{align*}
$$

In conclusion we have:

$$
f(x^{\ast})|X,Y,x^{\ast} \sim \mathcal{N}\begin{cases}
    \mu^{\ast} = \phi(x^{\ast})^T\Sigma_{q}\Phi^T(K+\sigma^2I)^{-1}Y \\\
    \sigma^{\ast} = \phi(x^{\ast})^T\Sigma_q\phi(x^{\ast}) - \phi(x^{\ast})^T\Sigma_q\Phi^T(\sigma^2I+K)^{-1}\Phi\Sigma_q\phi(x^{\ast})
\end{cases}
$$

$$
y^{\ast}|X,Y,x^{\ast} \sim \mathcal{N}\begin{cases}
    \mu_y^{\ast} = \phi(x^{\ast})^T\Sigma_{q}\Phi^T(K+\sigma^2I)^{-1}Y \\\
    \sigma_y^{\ast} = \phi(x^{\ast})^T\Sigma_q\phi(x^{\ast}) - \phi(x^{\ast})^T\Sigma_q\Phi^T(\sigma^2I+K)^{-1}\Phi\Sigma_q\phi(x^{\ast})+\sigma^2
\end{cases}
$$

### Find kernel function
{{< math.inline >}}
<p>
By observing \( \mu^{\ast},\sigma^{\ast} \), it exists a common pattern \( \phi\Sigma_q\phi \) which helps us to define the kernel function \( k(x,x') \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{Define } k(x,x') &= \phi(x)^T\Sigma_q\phi(x') \\\
&\because \text{$\Sigma_q$ is positive definite}\\\
&= \phi(x)^T\Sigma_q^{\frac{1}{2}}\Sigma_q^{\frac{1}{2}}\phi(x') \\\
&= \left(\Sigma_q^{\frac{1}{2}}\phi(x)\right)^T\Sigma_q^{\frac{1}{2}}\phi(x') \\\
&\text{Let }\psi(x)=\Sigma_q^{\frac{1}{2}}\phi(x) \\\
&= \langle\psi(x),\psi(x')\rangle
\end{align*}
$$

{{< math.inline >}}
<p>
\( k(x,x') \) is the inner product of high-dimensional function \(\psi(x)\), and we can represents common patterns in \( \mu^{\ast},\sigma^{\ast} \) with kernel function:
</p>
{{</ math.inline >}}

$$
\begin{align*}
K &= \begin{bmatrix}
    \phi(x_1)^T \\\
    \cdots \\\
    \phi(x_N)^T
\end{bmatrix}\Sigma_q\begin{bmatrix}\phi(x_1)\cdots\phi(x_N)\end{bmatrix} \\\
&= \begin{bmatrix}
    \phi(x_1)^T\Sigma_q\phi(x_1) & \cdots & \phi(x_1)^T\Sigma_q\phi(x_N) \\\
    & \vdots & \\\
    \phi(x_N)^T\Sigma_q\phi(x_1) & \cdots & \phi(x_N)^T\Sigma_q\phi(x_N)
\end{bmatrix} \\\
&= \begin{bmatrix}
    k(x_1,x_1) & \cdots & k(x_1,x_N) \\\
    & \vdots & \\\
    k(x_N,x_1) & \cdots & k(x_N,x_N)
\end{bmatrix} \\\
\\\
\phi(x^{\ast})^T\Sigma_q\Phi^T &= \phi(x^{\ast})^T\Sigma_q\begin{bmatrix}\phi(x_1)\cdots\phi(x_N)\end{bmatrix} \\\
&= \begin{bmatrix}\phi(x^{\ast})^T\Sigma_q\phi(x_1)\cdots\phi(x^{\ast})^T\Sigma_q\phi(x_N)\end{bmatrix} \\\
&= \begin{bmatrix}k(x^{\ast},x_1)\cdots k(x^{\ast},x_N)\end{bmatrix} \\\
\\\
\Phi\Sigma_q\phi(x^{\ast}) &= \begin{bmatrix}
    \phi(x_1)^T \\\
    \cdots \\\
    \phi(x_N)^T
\end{bmatrix}\Sigma_q \phi(x^{\ast}) \\\
&= \begin{bmatrix}
    \phi(x_1)^T\Sigma_q\phi(x^{\ast}) \\\
    \cdots \\\
    \phi(x_N)^T\Sigma_q\phi(x^{\ast})
\end{bmatrix} \\\
&= \begin{bmatrix}
    k(x_1,x^{\ast}) \\\
    \cdots \\\
    k(x_N,x^{\ast})
\end{bmatrix}
\end{align*}
$$

So we have prediction result in kernel form:

$$
f(x^{\ast}) \sim \mathcal{N}(\mu^{\ast}, \sigma^{\ast}) \\\
\\\
\begin{cases}
    \mu^{\ast} = \begin{bmatrix}k(x^{\ast},x_1) \\\
    \cdots \\\
    k(x^{\ast},x_N)\end{bmatrix}^T\begin{bmatrix}
    k(x_1,x_1)+\sigma^2 & \cdots & k(x_1,x_N) \\\
    & \ddots & \\\
    k(x_N,x_1) & \cdots & k(x_N,x_N)+\sigma^2
\end{bmatrix}^{-1}Y \\\
    \sigma^{\ast} = k(x^{\ast},x^{\ast}) - \begin{bmatrix}k(x^{\ast},x_1) \\\
    \cdots \\\
    k(x^{\ast},x_N)\end{bmatrix}^T\begin{bmatrix}
    k(x_1,x_1)+\sigma^2 & \cdots & k(x_1,x_N) \\\
    & \ddots & \\\
    k(x_N,x_1) & \cdots & k(x_N,x_N)+\sigma^2
\end{bmatrix}^{-1}\begin{bmatrix}k(x^{\ast},x_1) \\\
    \cdots \\\
    k(x^{\ast},x_N)\end{bmatrix}
\end{cases}
$$

## Gaussian Process Regression
### GPR definition
{{< math.inline >}}
<p>
Given a Gaussian process \( f(x) \), we have:
</p>
{{</ math.inline >}}

$$
\lbrace f(x) \rbrace_{x\in\mathbb{R}^p} \sim \text{GP}(m(x),k(x,x')) \\\
\\\
\begin{cases}
    m(x) &= \mathbb{E}[f(x)] \\\
    k(x,x') &= \mathbb{E}\left[ (f(x)-m(x))(f(x')-m(x')) \right] 
\end{cases}
$$

The Gaussian process regression is given by:

$$
\text{Data: }\lbrace x_i,y_i \rbrace_{i=1}^N, X = \begin{bmatrix}
    x_{11} & x_{12} & \dots & x_{1p}\\\
    x_{21} & x_{22} & \dots & x_{2p}\\\
    \dots \\\
    x_{N1} & x_{N2} & \dots & x_{Np}
\end{bmatrix}, Y = \begin{bmatrix}
    y_1 \\\
    y_2 \\\
    \dots \\\
    y_N
\end{bmatrix} \\\
\text{Model: }\begin{cases}
    y = f(x)+\epsilon, \quad\epsilon\sim\mathcal{N}(0,\sigma^2) \\\
    f(X) \sim \mathcal{N}(m(X),k(X,X)) \\\
    Y \sim \mathcal{N}(m(x), k(X,X)+\sigma^2I)
\end{cases}
$$

### Prediction in GPR
For the new data in GP, we have predictions:

$$
X^{\ast} = \begin{bmatrix}x_1^{\ast}\cdots x_M^{\ast}\end{bmatrix} \\\
Y^{\ast} = f(X^{\ast}) + \epsilon
$$

{{< math.inline >}}
<p>
We can easily write down joint distribution of \( Y \) and \( f(X^{\ast}) \), since they are both Gaussian:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\because \text{Cov}(f(X)+\epsilon,f(X^{\ast}))&=\text{Cov}(f(X),f(X^{\ast}))+\text{Cov}(\epsilon,f(X^{\ast})) \\\
&=\text{Cov}(f(X),f(X^{\ast})), \epsilon\perp f(X^{\ast})
\end{align*} \\\
\dArr
$$

$$
\begin{bmatrix}
    Y \\\
    f(X^{\ast})
\end{bmatrix} \sim \mathcal{N}(
    \begin{bmatrix}
        m(X) \\\
        m(X^{\ast})
    \end{bmatrix}, \begin{bmatrix}
        k(X,X)+\sigma^2I & k(X,X^{\ast}) \\\
        k(X^{\ast},X) & k(X^{\ast},X^{\ast})
    \end{bmatrix}
)
$$

{{< math.inline >}}
<p>
On the other hand, the conditional distribution \( f(X^{\ast})|Y \) is exactly the prediction problem:
</p>
{{</ math.inline >}}


$$
p(f(X^{\ast})|\text{Data},X^{\ast}) = p(f(X^{\ast})|Y,X,X^{\ast})
$$

We have learned how to calculate a conditional Gaussian distribution from a joint Gaussian distribution in [previous chapter](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/#solve-the-conditonal-pdf):

$$
\begin{cases}
x = \begin{bmatrix}
    x_a \\\
    x_b
\end{bmatrix} \\\
x \sim \mathcal{N}(\mu,\Sigma) \\\
\mu = \begin{bmatrix}
    \mu_a \\\
    \mu_b
\end{bmatrix} \\\
\Sigma = \begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab} \\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix}
\end{cases} \implies 
\begin{cases}
x_a \sim \mathcal{N}(\mu_a,\Sigma_{aa}) \\\
x_b \sim \mathcal{N}(\mu_b,\Sigma_{bb}) \\\
x_b|x_a \sim \mathcal{N}(\Sigma_{ba}\Sigma_{aa}^{-1} (x_a-\mu_{a}) + \mu_{b}, {-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb} } ) \\\
x_a|x_b \sim \mathcal{N}(\Sigma_{ab}\Sigma_{bb}^{-1} (x_b-\mu_{b}) + \mu_{a}, {-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}+\Sigma_{aa} } )
\end{cases}
$$

So we have:

$$
f(X^{\ast})|Y,X,X^{\ast} \sim \mathcal{N}(\mu^{\ast}, \Sigma^{\ast}) \\\
\begin{cases}
    \mu^{\ast} = k(X^{\ast},X)(k(X,X)+\sigma^2I)^{-1}(Y-m(X)) + m(X^{\ast}) \\\
    \Sigma^{\ast} = -k(X^{\ast},X)(k(X,X)+\sigma^2I)^{-1}k(X,X^{\ast})+k(X^{\ast},X^{\ast})
\end{cases}
$$

$$
Y^{\ast}|Y,X,X^{\ast} \sim \mathcal{N}(\mu_Y^{\ast}, \Sigma_Y^{\ast}) \\\
\begin{cases}
    \mu_Y^{\ast} = k(X^{\ast},X)(k(X,X)+\sigma^2I)^{-1}(Y-m(X)) + m(X^{\ast}) \\\
    \Sigma_Y^{\ast} = -k(X^{\ast},X)(k(X,X)+\sigma^2I)^{-1}k(X,X^{\ast})+k(X^{\ast},X^{\ast}) + \sigma^2I
\end{cases}
$$

## Kernel BLR and GPR comparison
### Kernel function is covariance function
By observing the prediction results in Kernel BLR and GPR we have:

$$
\text{BLR} : \begin{cases}
    \mu^{\ast} = \phi(x^{\ast})^T\Sigma_{q}\Phi^T(K+\sigma^2I)^{-1}Y \\\
    \sigma^{\ast} = \phi(x^{\ast})^T\Sigma_q\phi(x^{\ast}) - \phi(x^{\ast})^T\Sigma_q\Phi^T(\sigma^2I+K)^{-1}\Phi\Sigma_q\phi(x^{\ast}) \\\
    \mu_y^{\ast} = \phi(x^{\ast})^T\Sigma_{q}\Phi^T(K+\sigma^2I)^{-1}Y \\\
    \sigma_y^{\ast} = \phi(x^{\ast})^T\Sigma_q\phi(x^{\ast}) - \phi(x^{\ast})^T\Sigma_q\Phi^T(\sigma^2I+K)^{-1}\Phi\Sigma_q\phi(x^{\ast})+\sigma^2
\end{cases}
$$

$$
\text{GPR} : \begin{cases}
    \mu^{\ast} = k(X^{\ast},X)(k(X,X)+\sigma^2I)^{-1}(Y-m(X)) + m(X^{\ast}) \\\
    \Sigma^{\ast} = -k(X^{\ast},X)(k(X,X)+\sigma^2I)^{-1}k(X,X^{\ast})+k(X^{\ast},X^{\ast}) \\\
    \mu_Y^{\ast} = k(X^{\ast},X)(k(X,X)+\sigma^2I)^{-1}(Y-m(X)) + m(X^{\ast}) \\\
    \Sigma_Y^{\ast} = -k(X^{\ast},X)(k(X,X)+\sigma^2I)^{-1}k(X,X^{\ast})+k(X^{\ast},X^{\ast}) + \sigma^2I
\end{cases}
$$

{{< math.inline >}}
<p>
We can conclude that these two expressions of prediction between Kernel BLR and GPR are identically the same, if Gaussian process \( f(x) \) has attributes:
</p>
{{</ math.inline >}}

$$
\begin{cases}
    m(X) &= 0 \\\
    k(X,X) &\triangleq k(x,x) = \langle \Sigma_q^{\frac{1}{2}}\phi(x),\Sigma_q^{\frac{1}{2}}\phi(x) \rangle
\end{cases}
$$

<!-- {{< math.inline >}}
<p>
Furthermore kernel function \( k(x,x) \) in BLR is exactly the covariance function \( k(X,X) \) in GP.
</p>
{{</ math.inline >}} -->

### f(x) in kernel BLR is a Gaussian process

{{< math.inline >}}
<p>
On the other hand, in kernel BLR:
</p>
{{</ math.inline >}}

$$
\begin{align*}
    f(x) &= \phi(x)^Tw \\\
    \\\
    \mathbb{E}[f(x)] &= \mathbb{E}[ \phi(x)^Tw] \\\
    &= \phi(x)^T\mathbb{E}[w] \\\
    &= \phi(x)^T\mu_w \\\
    \\\
    \text{Cov}(f(x),f(x')) &= \mathbb{E}\left[ (f(x)-\mathbb{E}[f(x)])(f(x‘)-\mathbb{E}[f(x')]) \right] \\\
    &= \mathbb{E}\left[\phi(x)^Tw\phi(x')^Tw - \phi(x)^Tw\phi(x')^T\mu_w - \phi(x')^Tw\phi(x)^T\mu_w + \phi(x)^T\mu_w\phi(x')^T\mu_w \right] \\\
    &= \mathbb{E}\left[ \phi(x)^Tww^T\phi(x') - \phi(x)^Tw\mu_w^T\phi(x') - \phi(x)^T\mu_ww^T\phi(x') + \phi(x)^T\mu_w\mu_w^T\phi(x') \right] \\\
    &= \mathbb{E}\left[ \phi(x)^T(ww^T-w\mu_w^T-\mu_ww^T+\mu_w\mu_w^T)\phi(x') \right] \\\
    &= \phi(x)^T\mathbb{E}\left[(w-\mu_w)(w-\mu_w)^T\right]\phi(x') \\\
    &= \phi(x)^T\Sigma_w\phi(x') = \langle\psi(x),\psi(x')\rangle \\\
    &= k(x,x')
\end{align*}
$$

Based on the definition of GP we provided in the [above section](#formulation-of-gaussian-process), we can accordingly write:

$$
\begin{cases}
    t\rarr\xi_t, &\lbrace \xi_t \rbrace_{t\in T} \sim \text{GP} \\\
    x\rarr f(x), &\lbrace f(x) \rbrace_{x\in \mathbb{R}^{p}} \sim \text{GP}
\end{cases}
$$

{{< math.inline >}}
<p>
So \( f(x) \) in kernel BLR is the expected Gaussian process.
</p>
{{</ math.inline >}}

### Summary

$$
\text{GPR}: \begin{cases}
    \text{weight space} : \phi(x)^Tw\sim\mathcal{N}\left(\phi(x^T\mu_w, \phi(x)^T\Sigma_{w}\phi(x)\right) \\\
    \text{function space} : f(x)\sim \text{GP}(m,k)
\end{cases} \rArr \phi(x)^Tw=f(x)
$$

Thus, 

$$
\text{Gaussian process regression} = \text{Bayesian linear regression} + \text{kernel trick}
$$

<!-- If you found any mistakes, please contact me via email. -->

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=117).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^3]: From [Probabilistic Graphical Models (II) Inference & Leaning. Jun Zhu](https://ml.cs.tsinghua.edu.cn/~jun/courses/statml-fall2015/8-PGM-Inference.pdf).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^2]: - [Hammersley–Clifford theorem](http://www.statslab.cam.ac.uk/~grg/books/hammfest/hamm-cliff.pdf).
