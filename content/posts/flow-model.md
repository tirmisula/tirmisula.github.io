---
author: "X. Wang"
title: "Flow Model"
date: "2023-10-06"
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

{{< math.inline >}}
<p>
For any latent variable model, \( p(x) \) is hard to integrate:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(x) &= \int p(z)p(x|z,\theta)dz
\end{align*}
$$

{{< math.inline >}}
<p>
Thus the posterier \( p(z|x) \) is intractable:
</p>
{{</ math.inline >}}

$$
p(z|x) = \frac{p(z)p(x|z)}{p(x)}
$$

{{< math.inline >}}
<p>
The alternative way is to use approximate inference \( q(z|x) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\log p(x) &= \int_{z}q(z|x)\log\frac{p(x,z)}{q(z|x)}\space dz - \int_{z}q(z|x)\log\frac{p(z|x)}{q(z|x)}\space dz \\\ \\\
&= \text{ELBO} + \text{KL}(q(z|x,\phi)||p(z|x,\theta))
\end{align*}
$$

{{< math.inline >}}
<p>
Another motivation is to think \( x \) comes from an evolution where the begining state is a simple normal distribution: 
</p>
{{</ math.inline >}}

$$
\underset{\mathcal{N}(0,I)}{z_0} \xmapsto{f_0} z_1 \xmapsto{f_1} \cdots \xmapsto{f_{k-1}} z_k \xmapsto{f_k} x
$$

## Flow-based Model

### Definition

Flow models are likelihood-based models which use change of variable method:

$$
\text{Given } x=f(z), x,z \in \mathbb{R}^p \\\
z \sim p_z(z), x \sim p_x(x) \\\
f(\cdot) \text{ is continous and invertible} \\\
\\\
$$

$$
\text{change of variable theorem indicates: } \\\
\begin{align*}
\int_z p_z(z)dz = \int_x p_x(x)dx &\rArr \lvert p_z(z)dz \rvert = \lvert p_x(x)dx \rvert \\\
&\rArr p_x(x) = \lvert \frac{dz}{dx} \rvert p_z(z) \\\
&\rArr p_x(x) = \lvert \frac{df^{-1}(x)}{dx} \rvert p_z(z) \\\
&\because \text{it's p-dimentional} \\\
&\rArr p_x(x) = \lvert \frac{df^{-1}(x)}{dx} \rvert p_z(z) \\\
\end{align*}
$$

{{< math.inline >}}
<p>
Since \( x,z \) are p-dimentional, \( \frac{df^{-1}(x)}{dx} \) should be a Jacobian matrix:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p_x(x) &= \lvert \begin{bmatrix} 
    \frac{\partial f^{-1}(x_1)}{\partial x_1} & \cdots & \frac{\partial f^{-1}(x_1)}{\partial x_p} \\\
    \vdots & \ddots & \vdots \\\
    \frac{\partial f^{-1}(x_p)}{\partial x_1} & \cdots & \frac{\partial f^{-1}(x_p)}{\partial x_p}
\end{bmatrix} \rvert p_z(z) \\\
&= \lvert det( \left[\frac{\partial f^{-1}(x_i)}{\partial x_j}\right] ) \rvert p_z(z) \\\
&\because \frac{dz}{dx}\frac{dx}{dz} = 1 \\\
&\therefore \frac{d f^{-1}(x)}{d x}\frac{df(z)}{dz} = 1 \\\

&= \lvert det( \frac{\partial f(z)}{\partial z} ) \rvert^{-1} p_z(z) \\\
\end{align*}
$$

It's becasue the chain rule for Jacobians states that:

$$
\begin{align*}
J_{f^{-1}}(x)J_{f}(z) &= \frac{\partial}{\partial x}(f^{-1}\circ f)(x) \\\
&= I \\\
det(J_{f^{-1}}(x)J_{f}(z)) &= det(I) \\\
&= 1
\end{align*}
$$

The determinant of a product of matrices is the product of their determinants, so we have:

$$
\begin{align*}
det(J_{f^{-1}}(x))\cdot det(J_{f}(z)) &= 1 \\\
|det(\frac{\partial f^{-1}(x)}{\partial x})|\cdot|det(\frac{\partial f(z)}{\partial z})| &= 1
\end{align*}
$$

{{< math.inline >}}
<p>
It can also be proved by geometric way, suppose we have three points \( A,B,C \), \( A,B \) are symmetric along line \( y=x \), \( C \) is intersection point of two tangents passing through \( A \) and \( B \), we have:
</p>
{{</ math.inline >}}

$$
\text{Let } \begin{cases}
    A : (a,b) \\\
    B : (b,a) \\\
    C : (c,c)
\end{cases} \text{ where }
\begin{cases}
    A\in x=f^{-1}(y) \\\
    B\in y=f(x)
\end{cases}
\\\
\begin{align*}
\frac{b-c}{a-c}\cdot\frac{a-c}{b-c} &= 1 \\\
\tan\theta_A\cdot\tan\theta_B &=1 \\\
\frac{df^{-1}(y)}{dy}\cdot\frac{df(x)}{dx} &= 1
\end{align*}
$$



## Reference

[^1]: - [video](https://www.bilibili.com/video/BV17D42177Au).
[^2]: - [Attention Is All You Need. Ashish Vaswani, Llion Jones, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Aidan N. Gomez, Łukasz Kaiser](https://arxiv.org/pdf/1706.03762).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^4]: - [Denoising Diffusion Probabilistic Models. Jonathan Ho, Ajay Jain, Pieter Abbee](https://arxiv.org/pdf/2006.11239).
[^7]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
[^3]: - [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, Franc ̧ois Fleuret](https://arxiv.org/pdf/2006.16236).
