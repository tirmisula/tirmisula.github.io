---
author: "X. Wang"
title: "Support Vector Machine"
date: "2023-09-06"
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

## Background

Support vector machine(SVM) has 3 parts:

$$
\begin{cases}
\text{Hard-margin SVM} \\\
\text{Soft-margin SVM} \\\
\text{Kernel SVM}
\end{cases}
$$

## Hard-margin SVM
### Definition of hard-margin SVM

Given a dataset:

$$
\mathcal{D} = \lbrace(x_i,y_i)\rbrace_{i=1,\cdots,N} \\\
x_i \in \mathbb{R}^p \\\
y_i = \lbrace1,-1\rbrace
$$

Similar to [perceptron model](https://tirmisula.github.io/posts/perceptron/), hard-margin SVM tries to find a decision hyperplane that 100% correctly classifies each data sample and the <mark>closest</mark> point to decision hyperplane should have it's distance as long as possible:

$$
\max_{w,b}\min_{\forall x_i, i=1,\cdots,N} dist(x_i, w^Tx+b) \\\
\text{subject to } y_i(w^Tx_i+b) > 0
$$

{{< math.inline >}}
<p>
\(w\) is the normal vector of hyperplane \(w^Tx+b\), because given 2 points \(a_1,a_2\) on hyperplane, it has:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(w^Ta_1+b) - (w^Ta_2+b) &= 0 \\\
w^T(a_1-a_2) &= 0 \\\
w &\perp a_1-a_2
\end{align*}
$$

{{< math.inline >}}
<p>
Assuming we have \(x\)'s projection point \(h\), then \(x-h\) must be perpendicular to normal vector \(w\), then we can define distance function:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(x-h) &= kw \\\
h &= x-kw \\\
&\dArr \\\
w^T(x-kw)+b &= 0 \\\
w^Tx - k\lVert w\rVert^2 +b &= 0 \\\
k &= \frac{w^Tx+b}{\lVert w\rVert^2} \\\
&\dArr \\\
dist(x,h) = \lvert x-h \rvert &= \frac{\lvert w^Tx+b \rvert}{\lVert w\rVert^2} \lVert w \rVert \\\
&= \frac{\lvert w^Tx+b \rvert}{\lVert w\rVert}
\end{align*}
$$

The original problem becomes:

$$
\begin{align*}
\max_{w,b}\min_{\forall x_i, i=1,\cdots,N} dist(x_i, w^Tx+b) &= \max_{w,b}\min_{\forall x_i, i=1,\cdots,N} \frac{\lvert w^Tx_i+b \rvert}{\lVert w\rVert} \\\
&= \max_{w,b}\frac{1}{\lVert w\rVert}\min_{\forall x_i, i=1,\cdots,N} \lvert w^Tx_i+b \rvert \\\
&= \max_{w,b}\frac{1}{\lVert w\rVert}\min_{\forall x_i, i=1,\cdots,N} y_i(w^Tx_i+b)
\end{align*}
$$

{{< math.inline >}}
<p>
Since hyperplane \(w^Tx+b \implies 2w^Tx+2b \) is scalable which doesn't affect which one is the closest point, we can normalize \( \min y_i(w^Tx_i+b) \) to a certain scale:
</p>
{{</ math.inline >}}

$$
\min_{\forall x_i, i=1,\cdots,N} y_i(w^Tx_i+b) = \gamma
$$

Then above max-min problem becomes:

$$
\begin{align*}
\max_{w,b}\frac{1}{\lVert w\rVert }\gamma &= \max_{w,b}\frac{1}{\lVert w\rVert } \\\
&= \min_{w,b}\lVert w\rVert \\\
&= \min_{w,b} \frac{1}{2}w^Tw
\end{align*}
$$

And the constriant part becomes:

$$
\begin{align*}
y_i(w^Tx_i+b) > 0 &\implies \exist \gamma>0, \min_i y_i(w^Tx_i+b) = \gamma \\\
&\implies \exist \gamma>0, y_i(w^Tx_i+b) \geq \gamma
\end{align*}
$$

{{< math.inline >}}
<p>
Let \( \gamma=1 \), we have:
</p>
{{</ math.inline >}}

$$
\min_{w,b} \frac{1}{2}w^Tw \\\
\text{subject to } y_i(w^Tx_i+b) \geq 1, i=1,2,\cdots,N
$$

SVM is transformed to a convex optimization problem which can be solved like quadratic programming.

### Introduce non-constraint and dual problem

#### non-constraint problem
{{< math.inline >}}
<p>
The original problem can be transformed into a non-constraint problem with lagrange multiplier:
</p>
{{</ math.inline >}}

$$
L(w,b,\lambda) = \frac{1}{2}w^Tw+\sum_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)) \\\
\begin{cases}
\min_{w,b} \frac{1}{2}w^Tw \\\
\text{subject to } y_i(w^Tx_i+b) \geq 1, i=1,2,\cdots,N
\end{cases}
\iff
\begin{cases}
\min_{w,b} \max_{\lambda} L(w,b,\lambda) \\\
\text{subject to } \lambda_i \geq 0, i=1,2,\cdots,N
\end{cases}
$$

{{< math.inline >}}
<p>
This can be proved:
</p>
{{</ math.inline >}}

$$
\begin{cases}
\text{If } 1-y_i(w^Tx_i+b)>0, \max_{\lambda}L=\frac{1}{2}w^Tw+\infty=\infty \\\
\text{if } 1-y_i(w^Tx_i+b)\leq 0, \max_{\lambda}L=\frac{1}{2}w^Tw+0=\frac{1}{2}w^Tw
\end{cases} \\\
\therefore \min_{w,b} \max_{\lambda} L(w,b,\lambda) = \min_{w,b} (\infty, \frac{1}{2}w^Tw)= \min_{w,b} \frac{1}{2}w^Tw
$$

#### dual problem
{{< math.inline >}}
<p>
The dual problem is:
</p>
{{</ math.inline >}}

$$
\begin{cases}
\min_{w,b} \max_{\lambda} L(w,b,\lambda) \\\
\text{subject to } \lambda_i \geq 0, i=1,2,\cdots,N
\end{cases}
\iff
\begin{cases}
\max_{\lambda} \min_{w,b} L(w,b,\lambda) \\\
\text{subject to } \lambda_i \geq 0, i=1,2,\cdots,N
\end{cases}
$$

By weak duality theorem<cite>[^2]</cite>:

$$
\min_{w,b} \max_{\lambda} L(w,b,\lambda) \geq \max_{\lambda} \min_{w,b} L(w,b,\lambda)
$$

Because primal problem has a convex function and linear constraints it satisfies [KKT condition](https://tirmisula.github.io/posts/support-vector-machine/#kkt-conditions), which means a strong duality exsits in between primal and dual problem<cite>[^3]</cite>:

$$
\min_{w,b} \max_{\lambda} L(w,b,\lambda) = \max_{\lambda} \min_{w,b} L(w,b,\lambda)
$$

Solve primal problem is equivalent to solve it's dual problem.

#### solve dual problem

{{< math.inline >}}
<p>
We take the partial derivatives of \(L(w,b,\lambda)\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\frac{\partial}{\partial b}L(w,b,\lambda) &= 0 \\\
\sum_{i=1}^N \lambda_iy_i &= 0
\end{align*}
$$

{{< math.inline >}}
<p>
We put this condition \(\sum_{i=1}^N \lambda_iy_i = 0\) back to \(L(w,b,\lambda)\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
L(w,b,\lambda) &= \frac{1}{2}w^Tw+\sum_{i=1}^N\lambda_i(1-y_i(w^Tx_i+b)) \\\
&= \frac{1}{2}w^Tw+\sum_{i=1}^N\lambda_i-\sum_{i=1}^N\lambda_iy_iw^Tx_i-\sum_{i=1}^N\lambda_iy_ib \\\
&= \frac{1}{2}w^Tw+\sum_{i=1}^N\lambda_i-\sum_{i=1}^N\lambda_iy_iw^Tx_i
\end{align*}
$$

Next:

$$
\begin{align*}
\frac{\partial}{\partial w}L(w,b,\lambda) &= 0 \\\
w-\sum_{i=1}^N\lambda_iy_ix_i &= 0 \\\
w &= \sum_{i=1}^N\lambda_iy_ix_i
\end{align*}
$$

{{< math.inline >}}
<p>
We put this condition \(w = \sum_{i=1}^N\lambda_iy_ix_i\) back to \(L(w,b,\lambda)\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\min_{w,b}L(w,b,\lambda) &= \frac{1}{2}w^Tw+\sum_{i=1}^N\lambda_i-\sum_{i=1}^N\lambda_iy_iw^Tx_i \\\
&= \frac{1}{2}\left(\sum_{i=1}^N\lambda_iy_ix_i\right)^T\sum_{j=1}^N\lambda_jy_jx_j + \sum_{i-1}^N\lambda_i - \sum_{i=1}^N\lambda_iy_i\left(\sum_{j=1}^N\lambda_jy_jx_j\right)^Tx_i \\\
&= \frac{1}{2}\sum_{i=1}^N\lambda_iy_ix_i^T\sum_{j=1}^N\lambda_jy_jx_j + \sum_{i-1}^N\lambda_i- \sum_{i=1}^N\lambda_iy_i\left(\sum_{j=1}^N\lambda_jy_jx_j^T\right)x_i \\\
&= \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j + \sum_{i-1}^N\lambda_i - \sum_{i=1}^N\left(\sum_{j=1}^N\lambda_iy_i\lambda_jy_jx_j^T\right)x_i \\\
&= \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j + \sum_{i-1}^N\lambda_i - \sum_{i=1}^N\sum_{j=1}^N\lambda_iy_i\lambda_jy_jx_j^Tx_i \\\
&= \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j + \sum_{i-1}^N\lambda_i - \sum_{i=1}^N\sum_{j=1}^N\lambda_iy_i\lambda_jy_jx_i^Tx_j \\\
&= -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j + \sum_{i=1}^N\lambda_i
\end{align*}
$$

Finally the dual optimization problem becomes:

$$
\max_{\lambda} -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j + \sum_{i=1}^N\lambda_i \\\
\text{subject to } \lambda_i \geq 0 , i=1,2,\cdots,N \\\
\text{subject to } \sum_{i=1}^N \lambda_iy_i=0 , i=1,2,\cdots,N \\\
\dArr \\\
\min_{\lambda} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j - \sum_{i=1}^N\lambda_i \\\
\text{subject to } \lambda_i \geq 0 , i=1,2,\cdots,N \\\
\text{subject to } \sum_{i=1}^N \lambda_iy_i=0 , i=1,2,\cdots,N \\\
$$

#### KKT conditions

Karush-Kuhn-Tucker Conditions(KKT) for the primal problem with constraints are listed as follows:

$$
\begin{cases}
\frac{\partial L(w,b,\lambda)}{\partial w} = 0,\frac{\partial L(w,b,\lambda)}{\partial b} = 0 \implies \text{satisfies because }w^*,b^* \text{ solvable} \\\
\lambda_i(1-y_i(w^Tx_i+b)) = 0  \implies \text{complementary slackness}\\\
\lambda_i \geq 0, i=1,2,\cdots,N \implies \text{satisfies because mentioned in no constraints primal problem} \\\
1-y_i(w^Tx_i+b) \leq 0 \implies \text{satisfies because mentioned in constraints primal problem}
\end{cases}
$$

{{< math.inline >}}
<p>
\(w^*\) is solved in last section:
</p>
{{</ math.inline >}}

$$
\begin{align*}
w^* &= \sum_{i=1}^N\lambda_iy_ix_i \\\
\exist (x_k,y_k), 1-y_k((w^*)^Tx_k+b^*) &= 0 \\\
y_k^2((w^*)^Tx_k+b^*) &= y_k \\\
\sum_{i=1}^N\lambda_iy_ix_i^Tx_k+b^* &= y_k \\\
b^* &= y_k - \sum_{i=1}^N\lambda_iy_ix_i^Tx_k
\end{align*}
$$

<mark>complementary slackness</mark> can be divided into 2 situations:

$$
\lambda_i = 0 , y_i(w^Tx_i+b) > 1 \\\
\lambda_i \not= 0 , y_i(w^Tx_i+b) = 1 \\\
$$

{{< math.inline >}}
<p>
Because primal problem maximize distance between closest point to decision boundary \( w^Tx+b=0 \), it must have 2 dual boundaries \( w^Tx+b=1 \), \( w^Tx+b=-1 \) where the closest points for each class locate at.

So the first situation of satifying complementary slackness indicates other data samples that is far away from dual boundaries and have their weight \(\lambda_i\) determined to be zero, they play no rule in optimization.

And the second situation indicates only the closest points from 2 class have effective weight \(\lambda_i\). 

In conclusion, since \( w^* \), \( b^* \) is the linear combination of all data samples, complementary slackness contrained most of \(\lambda_i\) to zero, thus only a few points closest to class border determined the decision boundary.
</p>
{{</ math.inline >}}


## P-PCA
### PPCA model definition

Probabilistic principal component analysis (PPCA) is a probalistic generation model like gaussian discriminant analysis, it has following assumptions:<cite>[^1]</cite>

$$
\begin{align*}
\text{observed data } x &\in \mathbb{R}^p \\\
\text{latent variable } z &\in \mathbb{R}^q, q<p \\\
x &= Wz+b+\epsilon \\\
z &\sim \mathcal{N}(0_q, I_q) \\\
\epsilon &\sim \mathcal{N}(0_p, \sigma^2I_p) \\\
z,\epsilon &\text{ are isotropic gaussian variables} \\\
z &\perp \epsilon
\end{align*}
$$

It represents a linear gaussian model, and the object is to solve:

$$
\text{PPCA}
\begin{cases}
\text{probability: } p(z|x) \\\
\text{parameters: } W, b, \sigma^2
\end{cases}
$$

### Inference conditional probabilty

{{< math.inline >}}
<p>
\(p(z)\) is known, \(p(x|z)\) is solvable, so we can compute joint probability \(p(x,z)\) first then conditional probability \(p(z|x)\).
</p>
{{</ math.inline >}}

Deduction process of thhis part is mentioned in [previous article](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/):

$$
\begin{align*}
E[x|z] &= E[Wz+b+\epsilon] \\\
&=Wz+b+E[\epsilon] \\\
&= Wz+b
\end{align*}
$$

$$
\begin{align*}
Var[x|z] &= Var[Wz+b+\epsilon] \\\
&= Var[\epsilon] \\\
&= \sigma^2I
\end{align*}
$$

$$
x|z \sim \mathcal{N}(Wz+b,\sigma^2I)
$$

$$
\begin{align*}
E[x] &= E[Wz+b+\epsilon] \\\
&= WE[z]+b+E[\epsilon] \\\
&= b
\end{align*}
$$

$$
\begin{align*}
Var[x] &= Var[Wz+b+\epsilon] \\\
&= Var[Wz] + Var[\epsilon] \\\
&= WIW^T + \sigma^2I \\\
&= WW^T +\sigma^2I
\end{align*}
$$

$$
x \sim \mathcal{N}(b,WW^T+\sigma^2I)
$$

$$
\begin{align*}
E\left[\begin{bmatrix}
x \\\
z
\end{bmatrix}\right] &= \begin{bmatrix}
E[x] \\\
E[z]
\end{bmatrix} \\\
&= \begin{bmatrix}
b \\\
0
\end{bmatrix}
\end{align*}
$$

$$
\begin{align*}
Var\left[\begin{bmatrix}
x \\\
z
\end{bmatrix}\right] &= \begin{bmatrix}
Cov[x,x] & Cov[x,z] \\\
Cov[z,x] & Cov[z,z]
\end{bmatrix} \\\
&= \begin{bmatrix}
WW^T+\sigma^2I & Cov[x,z] \\\
Cov[z,x] & I
\end{bmatrix} \\\
\because Cov[x,z] &= E\left[ (x-b) (z-0)^T \right] \\\
&= E\left[ (Wz+\epsilon) z^T \right] \\\
&= E\left[Wzz^T\right] + E\left[\epsilon z^T\right] \\\
&= WE\left[(z-0)(z-0)^T\right] + E[\epsilon]E[z^T] \\\
&= WI \\\
&= W \\\
\therefore Var\left[\begin{bmatrix}
x \\\
z
\end{bmatrix}\right] &= \begin{bmatrix}
WW^T+\sigma^2I & W \\\
W^T & I
\end{bmatrix}
\end{align*}
$$

$$
\begin{bmatrix}
x \\\
z
\end{bmatrix} \sim \mathcal{N}(\begin{bmatrix}
b \\\
0
\end{bmatrix}, \begin{bmatrix}
WW^T+\sigma^2I & W \\\
W^T & I
\end{bmatrix})
$$

Just use conclusion from [previous article](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/):

$$
\begin{align*}
E[z|x] &= \Sigma_{zx}\Sigma_{xx}^{-1}(x-E[x])+E[z] \\\
&= W^T(WW^T+\sigma^2I)^{-1}(x-b)
\end{align*}
$$

$$
\begin{align*}
Var[z|x] &= -\Sigma_{zx}\Sigma_{xx}^{-1}\Sigma_{xz}+\Sigma_{zz} \\\
&= -W^T(WW^T+\sigma^2I)^{-1}W + I
\end{align*}
$$


## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=27).
[^2]: From [Weak duality](https://en.wikipedia.org/wiki/Weak_duality).
[^3]: From [Strong duality](https://en.wikipedia.org/wiki/Strong_duality).
