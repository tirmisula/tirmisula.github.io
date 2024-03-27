---
author: "X. Wang"
title: "Gaussian Process Regression"
date: "2023-09-23"
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

## Review Linear Regression

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
    \text{{Bayesian Linear Regresion}} \text{: } \begin{cases}
        \text{$w\in$ dist}  \\\
        \text{Inference } p(w|y) \\\
        \text{Predict } p(y^{\ast}|w,\text{Data})
    \end{cases} \\\
    \text{\color{red}{Gaussian Process Regression}} : \text{Bayesian LR} + \text{kernel trick}
\end{cases}
$$


## Formulation of Gaussian Process

{{< math.inline >}}
<p>
\( \lbrace \xi_{t} \rbrace_{t \in T} \) is a <b>Gaussian Process</b>, where:
</p>
{{</ math.inline >}}

$$
\begin{cases}
T : \text{continous domain} \\\
t_1 \cdots t_n \in T, \forall n \in \mathbb{N} : \text{any finite subset in $T$} \\\
r.v.\space\lbrace \xi_{t_1}\cdots\xi_{t_n} \rbrace : \text{$\forall \xi_{t_i\cdots\xi_{j}}\in$ multivariate gaussian}
\end{cases}
$$

{{< math.inline >}}
<p>
We can say stochastic function \( \xi \) distributed through Gaussian Process(GP):
</p>
{{</ math.inline >}}

$$
\xi \sim \text{GP}(m(t),k(t,t')) \\\
\begin{cases}
m(t) &= \mathbb{E}[\xi_t] & \text{mean function} \\\
k(t,t') &= \mathbb{E}[(\xi_t-\mathbb{E}[\xi_t])(\xi_{t'}-\mathbb{E}[\xi_{t'}])] & \text{covariance function}
\end{cases}
$$

<!-- ## Gaussian Process Regression -->
## Kernel Bayesian LR
### Definition
Recall that in [bayesian linear regression chapter](https://tirmisula.github.io/posts/bayesian-linear-regression/), we have the conclusion of posterier inference:

$$
p(w|\text{Data}) = \mathcal{N}(\mu_w, \Sigma_w) \\\
\begin{cases}
    \mu_w = \sigma^{-2}(\sigma^{-2}X^TX+\Sigma_p^{-1})^{-1}X^TY \\\
    \Sigma_w =  \sigma^{-2}X^TX+\Sigma_p^{-1})^{-1}
\end{cases}
$$

And the conlusion of new data prediction:

$$
\begin{cases}
    y = f(x) + \epsilon \\\
    f(x) = x^Tw \\\
    p(f(x_{new})|w,\text{Data},x_{new}) = \mathcal{N}(x^T_{new}\mu_w, x_{new}^T\Sigma_{w}x_{new}) \\\
    p(y_{new}|w,\text{Data},x_{new}) = \mathcal{N}(x^T_{new}\mu_w, x_{new}^T\Sigma_{w}x_{new}+\sigma^2)
\end{cases}
$$

{{< math.inline >}}
<p>
Kernel Bayesian LR indicates that \( f(x) \) is not a linear function. In this case \( f(x) \) usually has a new form which we have introduced in <a href="https://tirmisula.github.io/posts/support-vector-machine/#background-of-kernel-method">Kernel SVM chapter</a> before, a non-linear tranformation on \(x\) + a linear regression model:
</p>
{{</ math.inline >}}

$$
\begin{cases}
    f(x) &= \phi(x)^Tw \\\
    \phi(x) &= z, \quad\phi : x \mapsto z, x\in\mathbb{R}^p, z\in\mathbb{R}^q, p<q
\end{cases}
$$

{{< math.inline >}}
<p>
Like <a href="https://tirmisula.github.io/posts/support-vector-machine/#background-of-kernel-method">Kernel SVM chapter</a> mentioned, we want to use the kernel trick to find a kernel function, so that we don't directly solve \( \phi(x) \):
</p>
{{</ math.inline >}}

$$
\text{Given: } X = \begin{bmatrix}
    x_1 \cdots x_N
\end{bmatrix}^T,
Y = \begin{bmatrix}
    y_1 \cdots y_N
\end{bmatrix}^T, \lbrace x^{\ast},y^{\ast} \rbrace\in\lbrace x_{new},y_{new}\rbrace
$$

$$
\text{Let } \Phi = \phi(X) = \begin{bmatrix}
    \phi(x_1) \cdots \phi(x_N)
\end{bmatrix}^T_{N\times q}
$$

{{< math.inline >}}
<p>
Now the posterier \( p(w|\text{Data}) \) is calculated based on transformed data \( \Phi \), so we do the mapping \( X\mapsto\Phi \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
&\text{Let } p(w)\sim\mathcal{N}(0,\Sigma_q) \\\
&\text{Let } \epsilon\sim\mathcal{N}(0,\sigma^2)
\end{align*}
$$

$$
\text{We have } \begin{cases}
    \mu_w &= \sigma^{-2}(\sigma^{-2}X^TX+\Sigma_p^{-1})^{-1}X^TY \\\
    \Sigma_w &=  \sigma^{-2}X^TX+\Sigma_p^{-1})^{-1} \\\
    &\dArr \\\
    \mu_w &= \sigma^{-2}(\sigma^{-2}\Phi^T\Phi+\Sigma_q^{-1})^{-1}\Phi^TY \\\
    \Sigma_w &=  (\sigma^{-2}\Phi^T\Phi+\Sigma_q^{-1})^{-1}
\end{cases}
$$

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
\phi(x^{\ast})^T\Sigma_q\Phi^T &= \phi(x^{\ast})^T\Sigma_q\begin{bmatrix}\phi(x_1)\cdots\phi(x_N)\end{bmatrix} \\\
&= \begin{bmatrix}\phi(x^{\ast})^T\Sigma_q\phi(x_1)\cdots\phi(x^{\ast})^T\Sigma_q\phi(x_N)\end{bmatrix} \\\
&= \begin{bmatrix}k(x^{\ast},x_1)\cdots k(x^{\ast},x_N)\end{bmatrix} \\\
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

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=113).
[^4]: From [Higham, Nicholas (2002). Accuracy and Stability of Numerical Algorithms](https://archive.org/details/accuracystabilit00high_878).
[^5]: From [The Multivariate Gaussian. Michael I. Jordan](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter13.pdf).
[^3]: From [Tzon-Tzer, Lu; Sheng-Hua, Shiou (2002). "Inverses of 2 × 2 block matrices"](https://doi.org/10.1016%2FS0898-1221%2801%2900278-4).
[^2]: - [GAUSS-MARKOV MODELS, JONATHAN HUANG AND J. ANDREW BAGNELL](https://www.cs.cmu.edu/~16831-f14/notes/F14/gaussmarkov.pdf).
[^6]: - [Gaussian Processes and Gaussian Markov Random Fields](https://folk.ntnu.no/joeid/MA8702/jan16.pdf)
