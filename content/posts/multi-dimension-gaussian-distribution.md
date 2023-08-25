---
author: "X. Wang"
title: " Introduction to Multidimension Gaussian Distribution"
date: "2023-07-13"
description: "A brief introduction."
tags: ["machine learning"]
categories: ["themes", "syntax"]
# series: ["Themes Guide"]
aliases: ["migrate-from-jekyl"]
math: true
ShowBreadCrumbs: false
ShowToc: true
TocOpen: true
---

To simplify the calculations, we take the logarithm of the likelihood function:                                                         

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}

<!-- KaTeX -->
<!-- <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script> -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css" integrity="sha384-GvrOXuhMATgEsSwCs4smul74iXGOixntILdUW9XmUC6+HX0sLNAK3q71HotJqlAn" crossorigin="anonymous">

<!-- The loading of KaTeX is deferred to speed up page rendering -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js" integrity="sha384-cpW21h6RZv/phavutF+AuVYrr+dA8xD9zs6FwLpaCct6O9ctzYFfFr4dgmgccOTx" crossorigin="anonymous"></script>

<!-- To automatically render math in text elements, include the auto-render extension: -->
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js" integrity="sha384-+VBxd3r6XgURycqtZ117nYw44OOcIax56Z4dCRWbxyPt0Koah1uHoK0o4+/RRE05" crossorigin="anonymous"
    onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}

<style>
    /* Set the font size of all math elements to 16px */
    .katex {
        font-size: 16px !important;
    }
</style>

## Definition <cite>[^1]</cite>

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{x_1, x_2, ..., x_n\}\), where each data point \(x_i\) is drawn from a 1-D Gaussian distribution with unknown mean \(\mu\) and variance \(\sigma^2\).
</p>

{{</ math.inline >}}

$$p(x_i|\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)$$

{{< math.inline >}}
<p>
When it comes to N-D Gaussian distribution, we have:
</p>
{{</ math.inline >}}

$$p(x_i|\mu,\Sigma) = \frac{1}{(2\pi)^\frac{p}{2} |\Sigma|^\frac{1}{2}} \exp\left( -\frac{1}{2}(x_i-\mu) \Sigma^{-1} (x_i-\mu)^T \right)$$

{{< math.inline >}}
<p>
If variable \(x\) has p dimensions, we have:
</p>
{{</ math.inline >}}

$$
\begin{cases}
x \in \mathbb{R}^p\\

x = \begin{bmatrix}
    x_{1} \\
    x_{2} \\
    \vdots \\
    x_{p} \\
\end{bmatrix},
\mu = \begin{bmatrix}
    \mu_{1} \\
    \mu_{2} \\
    \vdots \\
    \mu_{p} \\
\end{bmatrix},
\Sigma =\begin{bmatrix}
    \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\
    \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\
    \vdots & \vdots & \ddots & \vdots \\
    \sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp} \\
\end{bmatrix}
\end{cases}
$$

{{< math.inline >}}
<p>
The likelihood function for the entire dataset \(\mathcal{D}\) is the product of the likelihoods of individual data points:
</p>
{{</ math.inline >}}

$$\mathcal{L}(\mu,\sigma^2|\mathcal{D}) = \prod_{i=1}^{n} p(x_i|\mu,\sigma^2)$$

$$

$$

{{< math.inline >}}
<p>
To find the maximum likelihood estimates of \(\mu\) and \(\sigma^2\), we need to maximize the likelihood function \(\mathcal{L}(\mu,\sigma^2|\mathcal{D})\) with respect to these parameters. Taking the logarithm of the likelihood function, we get:
</p>
{{</ math.inline >}}

<!-- {{< math.inline >}}
<p align="center">
\(\begin{align*}
\ln \mathcal{L}(\mu,\sigma^2|\mathcal{D}) &= \sum_{i=1}^{n} (\ln p(x_i|\mu,\sigma^2)) \\
a &= \sum_{i=1}^{n} (\ln \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)) \\
a &= \sum_{i=1}^{n} (\ln \frac{1}{\sqrt{2\pi\sigma^2}} + \ln \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)) \\
a &= n\ln \frac{1}{\sqrt{2\pi\sigma^2}}  + \sum_{i=1}^{n} \ln \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right) \\
a &= n\ln \frac{1}{\sqrt{2\pi\sigma^2}}  + \sum_{i=1}^{n} \left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right) \\
a &= -\frac{n}{2} \ln (2\pi) - \frac{n}{2} \ln \sigma^2 - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2
\end{align*}\)
</p>
{{</ math.inline >}} -->

$$
\begin{align*}
\ln \mathcal{L}(\mu,\sigma^2|\mathcal{D}) &= \sum_{i=1}^{n} \left(\ln p\left(x_i|\mu,\sigma^2\right)\right) \\\
&= \sum_{i=1}^{n} (\ln \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)) \\\
&= \sum_{i=1}^{n} (\ln \frac{1}{\sqrt{2\pi\sigma^2}} + \ln \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right)) \\\
&= n\ln \frac{1}{\sqrt{2\pi\sigma^2}}  + \sum_{i=1}^{n} \ln \exp\left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right) \\\
&= n\ln \frac{1}{\sqrt{2\pi\sigma^2}}  + \sum_{i=1}^{n} \left(-\frac{(x_i-\mu)^2}{2\sigma^2}\right) \\\
&= -\frac{n}{2} \ln (2\pi) - \frac{n}{2} \ln \sigma^2 - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2
\end{align*}
$$
<!-- $$\ln \mathcal{L}(\mu,\sigma^2|\mathcal{D}) &= -\frac{n}{2} \ln (2\pi) - \frac{n}{2} \ln \sigma^2 - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (x_i - \mu)^2$$ -->

{{< math.inline >}}
<p>
To find the maximum likelihood estimates, we differentiate the log-likelihood function with respect to the parameters \(\mu\) and \(\sigma^2\), set the derivatives equal to zero, and solve for the parameters. 
</p>
<p>
First, we differentiate with respect to \(\mu\):
</p>
{{</ math.inline >}}

$$\frac{\partial}{\partial\mu} \ln \mathcal{L}(\mu,\sigma^2|\mathcal{D}) = \frac{1}{\sigma^2} \sum_{i=1}^{n} (x_i - \mu) = 0$$

{{< math.inline >}}
<p>
Solving for \(\mu\), we get:
</p>
{{</ math.inline >}}

$$\hat{\mu}^1_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

This is the sample mean of the data, which is an unbiased estimator of the population mean.

{{< math.inline >}}
<p>
Next, we differentiate with respect to \(\sigma^2\):
</p>
{{</ math.inline >}}

$$\frac{\partial}{\partial\sigma^2} \ln \mathcal{L}(\mu,\sigma^2|\mathcal{D}) = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2} \sum_{i=1}^{n} (x_i - \mu)^2 = 0$$

{{< math.inline >}}
<p>
Solving for \(\sigma^2\), we get:
</p>
{{</ math.inline >}}

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu}_{\text{MLE}})^2$$

This is the sample variance of the data, which is an unbiased estimator of the population variance.

{{< math.inline >}}
<p>
Therefore, the maximum likelihood estimates of the parameters \(\mu\) and \(\sigma^2\) are given by the sample mean and sample variance of the data, respectively:
</p>
{{</ math.inline >}}

$$\hat{\mu}^1_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
{{< math.inline >}}
<p align="center">
\(\hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} x_i\)
</p>
{{</ math.inline >}}
<!-- ```math
    \hat{\mu}_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} x_i
``` -->

$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu}_{\text{MLE}})^2$$
{{< math.inline >}}
<p align="center">
\(\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n} \sum_{i=1}^{n} (x_i - \hat{\mu}_{\text{MLE}})^2\)
</p>
{{</ math.inline >}}

{{< math.inline >}}
<p>
To prove that the MLE of \(\hat{\mu}_{\text{MLE}}\) is unbiased, we need to compute the expected value of \(\hat{\mu}_{\text{MLE}}\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
E(\hat{\mu}^1_{\text{MLE}}) &= E\left(\frac{1}{n} \sum_{i=1}^n x_i\right) \\\
&= \frac{1}{n} E\left(\sum_{i=1}^n x_i\right)
\end{align*}
$$
{{< math.inline >}}
<p>
Since the observations \(x_1, x_2, \ldots, x_n\) are independent and identically distributed (i.i.d.), their expectations are equal:
</p>
{{</ math.inline >}}

$$
\begin{align*}
E(\hat{\mu}^1_{\text{MLE}}) &= \frac{1}{n} \sum_{i=1}^n E(x_i) \\\
&= \frac{1}{n} \sum_{i=1}^n \mu \\\
&= \frac{1}{n} (n\mu) \\\
&= \mu
\end{align*}
$$

{{< math.inline >}}
<p>
To prove that the MLE of \(\sigma^2\) is biased, we need to compute the expected value of \(\sigma^2\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
E(\hat{\sigma}^2_{\text{MLE}}) &= E\left(\frac{1}{n} \sum_{i=1}^n (x_i - \hat{\mu}^1_{\text{MLE}})^2\right) \\\
&= E\left(\frac{1}{n} \sum_{i=1}^n (x_i^2-2x_i\hat{\mu}^1_{\text{MLE}}+\hat{\mu}^2_{\text{MLE}})\right) \\\
&= E\left(\frac{1}{n} \sum_{i=1}^n x_i^2 - \frac{1}{n} \sum_{i=1}^n 2x_i\hat{\mu}^1_{\text{MLE}} + \frac{1}{n} \sum_{i=1}^n \hat{\mu}^2_{\text{MLE}} \right) \\\
&= E\left(\frac{1}{n} \sum_{i=1}^n x_i^2 - 2\hat{\mu}^1_{\text{MLE}}\frac{1}{n} \sum_{i=1}^n x_i + \hat{\mu}^2_{\text{MLE}} \right) \\\
&= E\left(\frac{1}{n} \sum_{i=1}^n x_i^2 - 2\hat{\mu}^2_{\text{MLE}} + \hat{\mu}^2_{\text{MLE}} \right) \\\
&= E\left(\frac{1}{n} \sum_{i=1}^n x_i^2 - \hat{\mu}^2_{\text{MLE}} \right) \\\
&= E\left((\frac{1}{n} \sum_{i=1}^n x_i^2 - \mu^2) - (\hat{\mu}^2_{\text{MLE}}-\mu^2) \right) \\\
&= E\left((\frac{1}{n} \sum_{i=1}^n x_i^2 - \frac{1}{n} \sum_{i=1}^n \mu^2) - (\hat{\mu}^2_{\text{MLE}}-\mu^2) \right) \\\
&= E\left(\frac{1}{n}\sum_{i=1}^n \left(x_i^2 - \mu^2\right) - \left(\hat{\mu}^2_{\text{MLE}}-\mu^2 \right)\right)\\\
&= E\left(\frac{1}{n}\sum_{i=1}^n \left(x_i^2 - \mu^2\right) \right) - E\left(\hat{\mu}^2_{\text{MLE}}-\mu^2 \right)\\\
&= \frac{1}{n}E\left(\sum_{i=1}^n \left(x_i^2 - \mu^2\right) \right) - E\left(\hat{\mu}^2_{\text{MLE}}-\mu^2 \right)\\\
&= \frac{1}{n}\sum_{i=1}^n E\left( x_i^2 - \mu^2 \right) - E\left(\hat{\mu}^2_{\text{MLE}}-\mu^2 \right)\\\
&= \frac{1}{n}\sum_{i=1}^n \left(E\left( x_i^2 \right)-E\left( \mu^2 \right)\right) - E\left(\hat{\mu}^2_{\text{MLE}}-\mu^2 \right)\\\
\end{align*}
$$

{{< math.inline >}}
<p>
Since \(\mu=E(x_i)\) and \(\sigma^2=Var(X) = E(X^2)-E(X)^2\) , we have:
</p>
{{</ math.inline >}}

$$
\begin{align*}
E(\hat{\sigma}^2_{\text{MLE}}) &= \frac{1}{n}\sum_{i=1}^n \left(E\left( x_i^2 \right)-E\left( x_i \right)^2\right) - E\left(\hat{\mu}^2_{\text{MLE}}-\mu^2 \right)\\\
&= \frac{1}{n}\sum_{i=1}^n \left(\sigma^2\right) - E\left(\hat{\mu}^2_{\text{MLE}}-\mu^2 \right)\\\
&= \frac{1}{n}\sum_{i=1}^n \left(\sigma^2\right) - E\left(\hat{\mu}^2_{\text{MLE}}-\mu^2 \right)\\\
\end{align*}
$$

{{< math.inline >}}
<p>
Recall that we have unbiased parameter \(E(\hat{\mu}^1_{\text{MLE}}) = \mu\) so we can get:
</p>
{{</ math.inline >}}

$$
\begin{align*}
E(\hat{\sigma}^2_{\text{MLE}}) &= \frac{1}{n}\sum_{i=1}^n \left(\sigma^2\right) - E\left(\hat{\mu}^2_{\text{MLE}}-E(\hat{\mu}^1_{\text{MLE}})^2 \right)\\\
&= \frac{1}{n}\sum_{i=1}^n \left(\sigma^2\right) - \left(E\left(\hat{\mu}^2_{\text{MLE}}\right) -E(\hat{\mu}^1_{\text{MLE}})^2 \right)\\\
&= \frac{1}{n}\sum_{i=1}^n \left(\sigma^2\right) - Var(\hat{\mu}^1_{\text{MLE}})\\\
&= \frac{1}{n}\sum_{i=1}^n \left(\sigma^2\right) - Var(\frac{1}{n} \sum_{i=1}^{n} x_i)\\\
&= \frac{1}{n}\sum_{i=1}^n \left(\sigma^2\right) - \frac{1}{n^2}\sum_{i=1}^{n}Var(x_i)\\\
&= \frac{1}{n}\sum_{i=1}^n \left(\sigma^2\right) - \frac{1}{n^2}\sum_{i=1}^{n}\sigma^2\\\
&= \frac{1}{n}n\sigma^2 - \frac{1}{n^2}n\sigma^2\\\
&= \frac{n-1}{n}\sigma^2\\\
\end{align*}
$$

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=4).