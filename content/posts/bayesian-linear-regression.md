---
author: "X. Wang"
title: "Bayesian Linear Regression"
date: "2023-09-18"
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

## Review of Linear Regression

<cite>[^1]</cite>

$$
\begin{cases}
    \underset{\text{MLE}}{\text{LS method: }}\begin{cases}
            y = w^Tx + \epsilon\\\
            \epsilon \sim \mathcal{N}(0, \sigma^2) \\\ 
             \argmax_w \sum_{i=1}^N \log\left( \frac{1}{ \sqrt{2\pi} \sigma} \mathrm{e}^{-\frac{(y_i-w^Tx_i)^2}{2\sigma^2}} \right)\\\
            \argmin_w \sum_{i=1}^N (y_i-w^Tx_i)^2
    \end{cases} \\\
    \underset{\text{MAP}}{\text{Regularized LS: }}\begin{cases}
        \text{Ridge: }   \begin{cases}
            y = w^Tx + \epsilon\\\
            y|w^Tx \sim \mathcal{N}(w^Tx, \sigma^2) \\\
            w \sim \mathcal{N}(0, \sigma_0^2) \\\ 
            \argmax_w \log\left( \prod_{i=1}^Np(y_i|w)p(w) \right) \\\
            \argmin_w \sum_{i=1}^N (y_i-w^Tx_i)^2 + \frac{\sigma^2}{\sigma_0^2}{\lVert w \rVert}^2
            \end{cases} \\\
        \text{Lasso: }   \begin{cases}
            y = w^Tx + \epsilon\\\
            y|w^Tx \sim \mathcal{N}(w^Tx, \sigma^2) \\\
            w \sim \text{Laplace}(0,\frac{1}{\lambda}) = \frac{\lambda}{2}\exp(-\lambda \Vert w \rVert) \\\ 
            \argmax_w \log\left( \prod_{i=1}^Np(y_i|w)p(w) \right) \\\
            \argmin_w \sum_{i=1}^N (y_i-w^Tx_i)^2 + \lambda{\lVert w \rVert}
            \end{cases} 
    \end{cases} \\\
    \text{\color{red}{Bayesian Linear Regresion}} \text{: } \begin{cases}
        \text{$w\in$ a distribution rather than a constant}  \\\
        \text{Inference } p(w|y)
    \end{cases}
\end{cases}
$$

## Bayesian Linear Regression
### Inference
#### Problem Statement
{{< math.inline >}}
<p>
Given that dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\):
</p>
{{</ math.inline >}}

$$
X = \begin{bmatrix}
    x_{11} & x_{12} & \dots & x_{1p}\\\
    x_{21} & x_{22} & \dots & x_{2p}\\\
    \dots \\\
    x_{N1} & x_{N2} & \dots & x_{Np}
\end{bmatrix}_{N \times p},
Y = \begin{bmatrix}
    y_1 \\\
    y_2 \\\
    \dots \\\
    y_N
\end{bmatrix}_{N \times 1}
$$

And the model:

$$
y = w^Tx + \epsilon\\\
\epsilon \sim \mathcal{N}(0, \sigma^2)
$$

We want to inference the posterier distribution:

$$
p(w|\text{Data})
$$

That is:

$$
\begin{align*}
    p(w|\text{Data}) &= p(w|X,Y) \\\
    &= \frac{p(w,Y|X)}{p(Y|X)} \\\
    &= \frac{p(Y|w,X)p(w|X)}{p(Y|X)} \\\
    &= \frac{p(Y|w,X)p(w)}{p(Y|X)} \\\
    &= \frac{p(Y|w,X)p(w)}{\int p(Y|w,X)p(w)dw} \\\
    &\propto p(Y|w,X)p(w) \\\
    &= \prod_{i=1}^N p(y_i|w,x_i)p(w) \\\
    &\text{Let prior distribution $p(w)$ be Gaussian} \\\
    &= \prod_{i=1}^N\mathcal{N}(y_i|w^Tx_i,\sigma^2)\mathcal{N}(0,\Sigma_p)
\end{align*}
$$

#### Posterier is Gaussian
From previous [Exponential Family Distribution Chapter](https://tirmisula.github.io/posts/exponential-family-distribution/#conjugate-distribution), we know that if likelihood is exponential family distribution then the prior and posterier distributions are conjugate distributions. So we have:

$$
p(w|X,Y) \sim \mathcal{N}(\mu_{?},\Sigma_{?})
$$

Proof:

$$
\begin{align*}
p(Y|w,X) &= \prod_{i=1}^N\mathcal{N}(y_i|w^Tx_i,\sigma^2) \\\
&= \frac{1}{(2\pi)^{\frac{N}{2}}\sigma^N}\prod_{i=1}^N\exp\left(-\frac{1}{2\sigma^2}(y_i-w^Tx_i)^2\right) \\\
&= \frac{1}{(2\pi)^{\frac{N}{2}}\sigma^N} \exp\left(-\frac{1}{2\sigma^2}\sum_{i=1}^N(y_i-w^Tx_i)^2\right) \\\
&= \frac{1}{(2\pi)^{\frac{N}{2}}\sigma^N} \exp\left(-\frac{1}{2\sigma^2}\begin{bmatrix} y_1-w^Tx_1 \cdots y_N-w^Tx_N \end{bmatrix}\begin{bmatrix} y_1-w^Tx_1 \\\ \cdots \\\ y_N-w^Tx_N \end{bmatrix}\right) \\\
&= \frac{1}{(2\pi)^{\frac{N}{2}}\sigma^N} \exp\left(-\frac{1}{2\sigma^2} (Y^T-w^TX^T)(Y-Xw)\right) \\\
&= \frac{1}{(2\pi)^{\frac{N}{2}}\sigma^N} \exp\left(-\frac{1}{2} (Y^T-w^TX^T)\sigma^2I_{N\times N}(Y-Xw)\right) \\\
&= \mathcal{N}(Y|Xw, \sigma^{-2}I_{N\times N})
\end{align*}
$$

$$
\begin{align*}
p(w|X,Y) &\propto \mathcal{N}(Xw, \sigma^{-2}I)\mathcal{N}(0,\Sigma_p) \\\
&\propto \exp\left(-\frac{1}{2\sigma^2}(Y^T-w^TX^T)(Y-Xw)\right)\exp\left(-\frac{1}{2}w^T\Sigma_p^{-1}w\right) \\\
&= \exp\left(-\frac{1}{2\sigma^2}(Y^TY-2Y^TXw+w^TX^TXw)-\frac{1}{2}(w^T\Sigma_p^{-1}w)\right) \\\
&= \exp\left( \frac{1}{\sigma^2}Y^TXw-\frac{1}{2}w^T(\frac{1}{\sigma^{2}}X^TX+\Sigma_p^{-1})w-\frac{1}{2\sigma^2}Y^TY \right) \\\
&\triangleq \exp\left( -\frac{1}{2}(w-\mu_{?})^T \Sigma_{?}^{-1} (w-\mu_{?}) \right) \\\
&\triangleq \exp(-\frac{1}{2}(w^T\Sigma_{?}^{-1}w-2\mu_{?}^T\Sigma_{?}^{-1}w+\mu_{?}^T\Sigma_{?}^{-1}\mu_{?}))
\end{align*}
$$

Thus we can conclude:

$$
\begin{cases}
-\frac{1}{2}w^T\Sigma_{?}^{-1}w = -\frac{1}{2}w^T(\frac{1}{\sigma^{2}}X^TX+\Sigma_p^{-1})w \\\
\mu_{?}^T\Sigma_{?}^{-1}w = \sigma^{-2}Y^TXw
\end{cases} \\\
\dArr \\\
\begin{cases}
\Sigma_{?} = (\sigma^{-2}X^TX+\Sigma_p^{-1})^{-1} \\\
\mu_{?} = \sigma^{-2}(\sigma^{-2}X^TX+\Sigma_p^{-1})^{-1}X^TY
\end{cases}
$$

$$
w|X,Y \sim \mathcal{N}(\sigma^{-2}(\sigma^{-2}X^TX+\Sigma_p^{-1})^{-1}X^TY, (\sigma^{-2}X^TX+\Sigma_p^{-1})^{-1})
$$

### Prediction
#### Problem statement
Given new data and the posterier parameter:

$$
\begin{cases}
x_{new}, y_{new} \\\
f(x) = w^Tx \\\
y = f(x) + \epsilon\\\
\epsilon \sim \mathcal{N}(0, \sigma^2) \\\
w|\text{Data} \sim \mathcal{N}(\mu_w,\Sigma_w)
\end{cases}
$$

We want to predict:

$$
\begin{cases}
f(x_{new}) & \text{without noise}\\\
f(x_{new})+\epsilon & \text{with noise}
\end{cases}
$$

#### Without Noise
We have the conclusion of linear gaussian system from [previous chapter](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/#prerequisite):

$$
\begin{cases}
x \sim \mathcal{N}(\mu,\Sigma),
x \in \mathbb{R}^p\\\
y = Ax+B,
y \in \mathbb{R}^q
\end{cases}
\\\
\dArr \\\
y \sim \mathcal{N}(A\mu+B,A\Sigma A^T)
$$

For current prediction problem:

$$
\begin{align*}
f(x_{new}) &= w^Tx_{new} = x_{new}^Tw \\\
f(x_{new})|\text{Data},x_{new} &\sim \mathcal{N}(x^T_{new}\mu_w, x_{new}^T\Sigma_{w}x_{new}) \\\
p(f(x_{new})|\text{Data},x_{new}) &= \int_w \mathcal{N}(x^T_{new}\mu_w, x_{new}^T\Sigma_{w}x_{new})p(w|\text{Data}) dw
\end{align*}
$$

#### With Noise
We have the conclusion of linear gaussian system with random noise from [previous chapter](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/#joint-pdf-problem-clarification-cite2cite):

$$
\begin{cases}
x \sim \mathcal{N} (\mu, \Lambda^{-1})\\\
\epsilon \sim \mathcal{N}(0, L^{-1}), \epsilon \perp x \\\
y=Ax+B+\epsilon
\end{cases}
$$



$$
y \sim \mathcal{N}(A\mu+B, A \Lambda^{-1} A^T + L^{-1})
$$

For current prediction problem:

$$
\begin{align*}
f(x_{new})+\epsilon &= x_{new}^Tw + \epsilon \\\
y_{new}|\text{Data},x_{new}&\sim \mathcal{N}(x^T_{new}\mu_w, x_{new}^T\Sigma_{w}x_{new}+\sigma^2) \\\
p(y_{new}|\text{Data},x_{new}) &= \int_w \mathcal{N}(x^T_{new}\mu_w, x_{new}^T\Sigma_{w}x_{new}+\sigma^2)p(w|\text{Data}) dw
\end{align*}
$$

<!-- If you found any mistakes, please contact me via email. -->

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=108).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^3]: From [Tzon-Tzer, Lu; Sheng-Hua, Shiou (2002). "Inverses of 2 × 2 block matrices"](https://doi.org/10.1016%2FS0898-1221%2801%2900278-4).
[^2]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
