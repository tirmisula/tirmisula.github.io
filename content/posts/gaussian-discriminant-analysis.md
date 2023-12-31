---
author: "X. Wang"
title: "Linear Classification - Gaussian Discriminant Analysis"
date: "2023-09-03"
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
{{</ math.inline >}}

<style>
    /* Set the font size of all math elements to 16px */
    .katex {
        font-size: 16px !important;
    }
</style>

## Prerequisite
### Matrix trace derivation

$$
\begin{align}
\frac{\partial}{\partial A}Tr(AB) &= B^T \\\
\frac{\partial}{\partial A}\lvert A \rvert &= \lvert A \rvert A^{-1} \\\
Tr(AB) &= Tr(BA) \\\
Tr(ABC) &= Tr(CAB) = Tr(BCA) \\\
Tr(A) &= A \iff A \text{ is a scalar}
\end{align}
$$

## Definition

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\), where each data point \( (x_i,y_i) \) has attribution \(x_i \in \mathbb{R}^p \), \( y_i \in \{ 0,1 \} \). We define 2 classes C1 and C2:
</p>
{{</ math.inline >}}

$$
x_{C1} = \{ x_i|y_i=1 \} \\\
x_{C2} = \{ x_i|y_i=0 \} \\\
|x_{C1}| = N_1 \\\
|x_{C2}| = N_2 \\\
N = N_1+N_2
$$

{{< math.inline >}}
<p>
In gaussian discriminant analysis(GDA) we don't directly model \(p(y|x)\) like <a href="https://tirmisula.github.io/posts/logistic-regression/">logistic regression</a>, we model \(p(x|y)\) instead and then solve \(p(y|x)\) by maximum posterier estimation(MAP).
<br/>
<br/>
MAP is shown as:<cite>[^1]</cite>
</p>
{{</ math.inline >}}

$$
\underset{posterier}{p(y|x)} \propto \underset{likelihood}{p(x|y)}\underset{prior}{p(y)}
$$

{{< math.inline >}}
<p>
Since it is porpotional value, we don't directly solve \(p(y=1|x)\) and \(p(y=0|x)), but we can determine which one is larger:
</p>
{{</ math.inline >}}

$$
p(y=1|x) \triangleq p(y=0|x)
$$

{{< math.inline >}}
<p>
We assume \(p(y)\) subject to Bernoulli distribution, which is:
</p>
{{</ math.inline >}}

$$
y \sim \mathcal{Bernoulli}(\phi) \\\
p(y) = \phi^y (1-\phi)^{1-y} \implies
\begin{cases}
p(y=1) = \phi \\\
p(y=0) = 1-\phi
\end{cases}
$$

{{< math.inline >}}
<p>
We assume \(p(x|y=0)\) and \(p(x|y=1)\) subject to Gaussian distribution, which is:
</p>
{{</ math.inline >}}

$$
x|{y=1} \sim \mathcal{N}(\mu_1, \Sigma) \\\
x|{y=0} \sim \mathcal{N}(\mu_2, \Sigma) \\\
\dArr \\\
p(x|y) = \mathcal{N}(\mu_1, \Sigma)^y \mathcal{N}(\mu_2, \Sigma)^{1-y}
$$

{{< math.inline >}}
<p>
Let \(\theta\) be the parameters to be estimated, which is:
</p>
{{</ math.inline >}}

$$
\theta = (\mu_1, \mu_2, \Sigma, \phi)
$$

Then we can define the likelihood function:

$$
L(\theta) = \prod_{i=1}^N p(x_i|y_i)p(y_i)
$$

{{< math.inline >}}
<p>
The problem is find \( \hat{\theta} \) by maximizing \(L(\theta)\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{\theta} &= \argmax_{\theta} L(\theta) \\\
&= \argmax_{\theta} \prod_{i=1}^N p(x_i|y_i)p(y_i) \\\
&= \argmax_{\theta} \sum_{i=1}^N \log\left( p(x_i|y_i)p(y_i) \right) \\\
&= \argmax_{\theta} \sum_{i=1}^N \left[ \log p(x_i|y_i) + \log p(y_i) \right] \\\
&= \argmax_{\theta} \sum_{i=1}^N \left[ \log \mathcal{N}(\mu_1, \Sigma)^{y_i} \mathcal{N}(\mu_2, \Sigma)^{1-y_i} + \log \phi^{y_i} (1-\phi)^{1-y_i} \right] \\\
&= \argmax_{\theta} \sum_{i=1}^N \left[ \log \mathcal{N}(\mu_1, \Sigma)^{y_i} + \log\mathcal{N}(\mu_2, \Sigma)^{1-y_i} + \log \phi^{y_i} (1-\phi)^{1-y_i} \right]
\end{align*}
$$

## Solve parameters

{{< math.inline >}}
<p>
We solve \(\phi\) first:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial \phi} L(\theta) &= 0 \\\
\frac{\partial}{\partial \phi} \sum_{i=1}^N \left[ \log \mathcal{N}(\mu_1, \Sigma)^{y_i} + \log\mathcal{N}(\mu_2, \Sigma)^{1-y_i} + \log \phi^{y_i} (1-\phi)^{1-y_i} \right] &=  0 \\\
\frac{\partial}{\partial \phi} \sum_{i=1}^N \left[ \log \phi^{y_i} (1-\phi)^{1-y_i} \right] &= 0 \\\
\frac{\partial}{\partial \phi} \sum_{i=1}^N \left[ \log \phi^{y_i}  + \log (1-\phi)^{1-y_i} \right] &= 0 \\\
\frac{\partial}{\partial \phi} \sum_{i=1}^N \left[ y_i\log\phi + (1-y_i)\log(1-\phi) \right] &= 0 \\\
\sum_{i=1}^N \left[ y_i\frac{1}{\phi} - (1-y_i)\frac{1}{1-\phi} \right] &= 0 \\\
\sum_{i=1}^N \left[ y_i(1-\phi) - (1-y_i)\phi \right] &= 0 \\\
\sum_{i=1}^N \left[ y_i-\phi \right] &= 0 \\\
\sum_{i=1}^N y_i - N\phi &= 0 \\\
\phi &= \frac{1}{N}\sum_{i=1}^N y_i \\\
\phi &= \frac{N_1}{N}
\end{align*}
$$

{{< math.inline >}}
<p>
Next, we solve \(\mu_1\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial \mu_1} L(\theta) &= 0 \\\
\frac{\partial}{\partial \mu_1} \sum_{i=1}^N \left[ \log \mathcal{N}(\mu_1, \Sigma)^{y_i} + \log\mathcal{N}(\mu_2, \Sigma)^{1-y_i} + \log \phi^{y_i} (1-\phi)^{1-y_i} \right] &=  0 \\\
\frac{\partial}{\partial \mu_1} \sum_{i=1}^N \left[ \log \mathcal{N}(\mu_1, \Sigma)^{y_i} \right] &= 0 \\\
\frac{\partial}{\partial \mu_1} \sum_{i=1}^N \left[ y_i \log\frac{1}{(2\pi)^{\frac{p}{2}}\lvert\Sigma\rvert^{\frac{1}{2}}}\mathrm{e}^{-\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)} \right] &= 0 \\\
\frac{\partial}{\partial \mu_1} \sum_{i=1}^N \left[ y_i \log\frac{1}{(2\pi)^{\frac{p}{2}}\lvert\Sigma\rvert^{\frac{1}{2}}} + y_i\log\mathrm{e}^{-\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)} \right] &= 0 \\\
\frac{\partial}{\partial \mu_1} \sum_{i=1}^N \left[ {y_i(-\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1))} \right] &= 0 \\\
\frac{\partial}{\partial \mu_1} -\frac{1}{2}\sum_{i=1}^N \left[ {y_i(x_i^T\Sigma^{-1}-\mu_1^T\Sigma^{-1})(x_i-\mu_1)} \right] &= 0 \\\
\frac{\partial}{\partial \mu_1} \sum_{i=1}^N \left[ {y_i(x_i^T\Sigma^{-1}x_i-x_i^T\Sigma^{-1}\mu_1-\mu_1^T\Sigma^{-1}x_i+\mu_1^T\Sigma^{-1}\mu_1)} \right] &= 0 \\\
\frac{\partial}{\partial \mu_1} \sum_{i=1}^N \left[ {y_i(-2\mu_1^T\Sigma^{-1}x_i+\mu_1^T\Sigma^{-1}\mu_1)} \right] &= 0 \\\
\sum_{i=1}^N \left[ {-2y_i\Sigma^{-1}x_i+2y_i\Sigma^{-1}\mu_1} \right] &= 0 \\\
\sum_{i=1}^N \left[ {y_i(\Sigma^{-1}\mu_1-\Sigma^{-1}x_i}) \right] &= 0 \\\
\Sigma \cdot \sum_{i=1}^N \left[ {y_i(\Sigma^{-1}\mu_1-\Sigma^{-1}x_i}) \right] &= 0 \\\
\sum_{i=1}^N \left[ {y_i\Sigma(\Sigma^{-1}\mu_1-\Sigma^{-1}x_i}) \right] &= 0 \\\
\sum_{i=1}^N y_i\mu_1 &= \sum_{i=1}^N y_ix_i \\\
\mu_1 &= \frac{\sum_{i=1}^N y_ix_i}{\sum_{i=1}^N y_i} \\\
\mu_1 &= \frac{\sum_{i=1}^N y_ix_i}{N_1}
\end{align*}
$$

{{< math.inline >}}
<p>
Similarly, we get \(\mu_2\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial \mu_2} L(\theta) &= 0 \\\
\sum_{i=1}^N \left[ {(1-y_i)\Sigma(\Sigma^{-1}\mu_2-\Sigma^{-1}x_i}) \right] &= 0 \\\
\sum_{i=1}^N (1-y_i)\mu_2 &= \sum_{i=1}^N y_ix_i \\\
\mu_2 &= \frac{\sum_{i=1}^N y_ix_i}{\sum_{i=1}^N (1-y_i)} \\\
\mu_2 &= \frac{\sum_{i=1}^N y_ix_i}{N_2}
\end{align*}
$$

{{< math.inline >}}
<p>
Finally we solve \(\Sigma\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial \Sigma} L(\theta) &= 0 \\\
\frac{\partial}{\partial \Sigma} \sum_{i=1}^N \left[ \log \mathcal{N}(\mu_1, \Sigma)^{y_i} + \log\mathcal{N}(\mu_2, \Sigma)^{1-y_i} + \log \phi^{y_i} (1-\phi)^{1-y_i} \right] &=  0 \\\
\frac{\partial}{\partial \Sigma} \sum_{i=1}^N \left[ \log \mathcal{N}(\mu_1, \Sigma)^{y_i} + \log \mathcal{N}(\mu_2, \Sigma)^{1-y_i} \right] &= 0 \\\
\frac{\partial}{\partial \Sigma} \sum_{x_i\in C1} \log \mathcal{N}(\mu_1, \Sigma) + \sum_{x_i\in C2} \log \mathcal{N}(\mu_2, \Sigma) &= 0
\end{align*}
$$

For C1 Part:

$$
\begin{align*}
\frac{\partial}{\partial \Sigma} \sum_{x_i\in C1} \log \mathcal{N}(\mu_1, \Sigma) &= \frac{\partial}{\partial \Sigma} \sum_{x_i \in C1}  \left[\log\frac{1}{(2\pi)^{\frac{p}{2}}\lvert\Sigma\rvert^{\frac{1}{2}}} + \log\mathrm{e}^{-\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)}\right] \\\
&= \frac{\partial}{\partial \Sigma} \sum_{x_i \in C1}  \left[\log\frac{1}{(2\pi)^{\frac{p}{2}}} + \log\lvert\Sigma\rvert^{-\frac{1}{2}} + \log\mathrm{e}^{-\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)}\right] \\\
&= \frac{\partial}{\partial \Sigma} \sum_{x_i \in C1}  \left[\log\lvert\Sigma\rvert^{-\frac{1}{2}} - {\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)}\right] \\\
&= \frac{\partial}{\partial \Sigma} \sum_{x_i \in C1}  \left[-\frac{1}{2}\log\lvert\Sigma\rvert\right] - \sum_{x_i \in C1}  \left[{\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)}\right] \\\
&= -\frac{1}{2}\sum_{x_i \in C1} \frac{1}{\lvert\Sigma\rvert}\cdot\frac{\partial \lvert\Sigma\rvert}{\partial \Sigma} - \frac{\partial}{\partial \Sigma}\sum_{x_i \in C1}  \left[{\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)}\right] \\\
\because \text{ According to \textbf{Prerequisite} } \frac{\partial}{\partial A}\lvert A \rvert = \lvert A \rvert A^{-1} \\\
&= -\frac{1}{2}\sum_{x_i \in C1} \frac{1}{\lvert\Sigma\rvert}\cdot \lvert\Sigma\rvert\Sigma^{-1} - \frac{\partial}{\partial \Sigma}\sum_{x_i \in C1}  \left[{\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)}\right] \\\
&= -\frac{1}{2}\sum_{x_i \in C1} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}\sum_{x_i \in C1}  \left[{\frac{1}{2}(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)}\right] \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}\sum_{x_i \in C1}  \left[{\frac{1}{2}\underset{\color{red}{scalar}}{(x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)}}\right] \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}\sum_{x_i \in C1}  \left[\frac{1}{2}Tr\left((x_i-\mu_1)^T\Sigma^{-1}(x_i-\mu_1)\right)\right] \\\
\because \text{ Circular shifts of trace shown in \textbf{Prerequisite }} Tr(ABC) &= Tr(CAB) = Tr(BCA)\\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}Tr\left(\sum_{x_i \in C1}  \left[\frac{1}{2}(x_i-\mu_1)(x_i-\mu_1)^T\Sigma^{-1}\right]\right) \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}Tr\left(\sum_{x_i \in C1}  \left[\frac{1}{2}(x_i-\mu_1)(x_i-\mu_1)^T\right]\cdot\Sigma^{-1}\right) \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}Tr\left(\frac{1}{2}\sum_{x_i \in C1}  \left[(x_i-\mu_1)(x_i-\mu_1)^T\right]\cdot\Sigma^{-1}\right) \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}Tr\left(\frac{1}{2}N_1Var(x_{C1})\cdot\Sigma^{-1}\right) \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}\frac{1}{2}N_1Tr\left(Var(x_{C1})\cdot\Sigma^{-1}\right) \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}\frac{1}{2}N_1Tr\left(\Sigma^{-1}Var(x_{C1})\right) \\\
\because \text{ According to \textbf{Prerequisite} } \frac{\partial}{\partial A}Tr(AB) = B^T \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} - \frac{1}{2}N_1Var(x_{C1})^T \cdot \frac{\partial \Sigma^{-1}}{\partial \Sigma} \\\
&= -\frac{N_1}{2} \frac{1}{\Sigma} + \frac{1}{2}N_1Var(x_{C1}) \Sigma^{-2}
\end{align*}
$$

For C2 Part, Similarly:

$$
\begin{align*}
\frac{\partial}{\partial \Sigma} \sum_{x_i\in C2} \log \mathcal{N}(\mu_2, \Sigma) &= -\frac{N_2}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}Tr\left(\frac{1}{2}\sum_{x_i \in C2}  \left[(x_i-\mu_2)(x_i-\mu_2)^T\right]\cdot\Sigma^{-1}\right) \\\
&= -\frac{N_2}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}Tr\left(\frac{1}{2}N_2Var(x_{C2})\cdot\Sigma^{-1}\right) \\\
&= -\frac{N_2}{2} \frac{1}{\Sigma} - \frac{\partial}{\partial \Sigma}\frac{1}{2}N_2Tr\left(Var(x_{C2})\cdot\Sigma^{-1}\right) \\\
&= -\frac{N_2}{2} \frac{1}{\Sigma} + \frac{1}{2}N_2Var(x_{C2}) \Sigma^{-2}
\end{align*}
$$

Back to original equation:

$$
\begin{align*}
\frac{\partial}{\partial \Sigma} \sum_{x_i\in C1} \log \mathcal{N}(\mu_1, \Sigma) + \frac{\partial}{\partial \Sigma} \sum_{x_i\in C2} \log \mathcal{N}(\mu_2, \Sigma) &= 0 \\\
-\frac{N_1}{2} \frac{1}{\Sigma} + \frac{1}{2}N_1Var(x_{C1}) \Sigma^{-2} -\frac{N_2}{2} \frac{1}{\Sigma} + \frac{1}{2}N_2Var(x_{C2}) \Sigma^{-2} &= 0 \\\
-\frac{1}{2} \frac{N_1+N_2}{\Sigma} + \frac{1}{2}(N_1Var(x_{C1})+N_2Var(x_{C2}))\Sigma^{-2} &= 0 \\\
(N_1Var(x_{C1})+N_2Var(x_{C2}))\Sigma^{-2} &= (N_1+N_2)\Sigma^{-1} \\\
N_1Var(x_{C1})+N_2Var(x_{C2}) &= N\Sigma \\\
\Sigma &= \frac{N_1Var(x_{C1})+N_2Var(x_{C2})}{N} \\\
\Sigma &= \frac{N_1\Sigma_{x_{C1}}+N_2\Sigma_{x_{C2}}}{N}
\end{align*}
$$

## Conclusion

GDA likelihood is defined as:

$$
L(\theta) = \prod_{i=1}^N p(x_i|y_i)p(y_i)
$$

We assume:

$$
y \sim \mathcal{Bernoulli}(\phi) \\\
x|{y=1} \sim \mathcal{N}(\mu_1, \Sigma) \\\
x|{y=0} \sim \mathcal{N}(\mu_2, \Sigma) \\\
$$

{{< math.inline >}}
<p>
The final result of \( \theta=(\mu_1, \mu_2, \Sigma, \phi) \) is :
</p>
{{</ math.inline >}}

$$
\begin{align*}
\hat{\phi} &= \frac{N_1}{N} \\\
\hat{\mu_1} &= \frac{\sum_{i=1}^N y_ix_i}{N_1} \\\
\hat{\mu_2} &= \frac{\sum_{i=1}^N y_ix_i}{N_2} \\\
\hat{\Sigma} &= \frac{N_1\Sigma_{x_{C1}}+N_2\Sigma_{x_{C2}}}{N}
\end{align*}
$$

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=18).
[^2]: From [source](https://en.wikipedia.org/wiki/Logistic_regression).