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
\frac{\partial L(w,b,\lambda)}{\partial w} = 0,\frac{\partial L(w,b,\lambda)}{\partial b} = 0 \implies \text{satisfies because }w^{\ast},b^{\ast} \text{ solvable} \\\
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
w^{\ast} &= \sum_{i=1}^N\lambda_iy_ix_i \\\
\exist (x_k,y_k), 1-y_k((w^{\ast})^Tx_k+b^{\ast}) &= 0 \\\
y_k^2((w^{\ast})^Tx_k+b^{\ast}) &= y_k \\\
\sum_{i=1}^N\lambda_iy_ix_i^Tx_k+b^{\ast} &= y_k \\\
b^{\ast} &= y_k - \sum_{i=1}^N\lambda_iy_ix_i^Tx_k
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
<br/>
So the first situation of satifying complementary slackness indicates other data samples that is far away from dual boundaries and have their weight \(\lambda_i\) determined to be zero, they play no rule in optimization.
<br/>
And the second situation indicates only the closest points from 2 class have effective weight \(\lambda_i\). 
<br/>
In conclusion, since \( w^* \), \( b^* \) is the linear combination of all data samples, complementary slackness contrained most of \(\lambda_i\) to zero, thus only a few points closest to class border determined the decision boundary.
</p>
{{</ math.inline >}}


## Soft-margin SVM
### Definition of soft-margin SVM

{{< math.inline >}}
<p>
Soft-margin SVM tolerates minor error that some points surpass dual boundaries \( w^Tx+b=\pm 1 \) : 
</p>
{{</ math.inline >}}

$$
\min_{w,b} \frac{1}{2}w^Tw+loss \\\
\text{subject to } y_i(w^Tx_i+b) \geq 1, i=1,2,\cdots,N
$$

#### inidicator loss function

The first choice of loss function is counting the number of surpassing points, which is:

$$
\min_{w,b} \frac{1}{2}w^Tw+\sum_{i=1}^NI\lbrace y_i(w^Tx_i+b)<1 \rbrace
$$

{{< math.inline >}}
<p>
But \(I(\cdot)\) is not continous, so we consider let the surpassing distance be the loss.
</p>
{{</ math.inline >}}

#### hinge loss function

Loss function is designed as:

$$
\begin{cases}
\text{If } y_i(w^Tx_i+b) \geq 1, loss=0 \\\
\text{If } y_i(w^Tx_i+b) < 1, loss=1-y_I(w^Tx_i+b)
\end{cases}
$$

Combine together we get:

$$
\min_{w,b} \frac{1}{2}w^Tw+\sum_{i=1}^N \max(0, 1-y_i(w^Tx_i+b))
$$

{{< math.inline >}}
<p>
Let \( \xi_i \) be the surpass distance, and add weight for the loss, we have:
</p>
{{</ math.inline >}}

$$
\text{Let } \xi_i = 1-y_i(w^Tx_i+b), \xi_i \geq 0 \\\
\dArr \\\
\min_{w,b} \frac{1}{2}w^Tw+C\sum_{i=1}^N \max(0,\xi_i) \\\
\text{subject to } y_i(w^Tx_i+b) \geq 1-\xi_i, i=1,2,\cdots,N \\\
\dArr \\\
\min_{w,b} \frac{1}{2}w^Tw+C\sum_{i=1}^N\xi_i \\\
\text{subject to } y_i(w^Tx_i+b) \geq 1-\xi_i, i=1,2,\cdots,N
$$

### Dual problem of soft-margin SVM

## Supplementation

### Constrained problem to non-constrained

Given a primal problem:

$$
\min_{x\in\mathbb{R}^p} f(x) \\\
\text{subject to } m_i(x) \leq 0, i=1,\cdots,M  \\\
\text{subject to } n_j(x) = 0, j=1,\cdots,N
$$

It's equivalent non-constrained problem is :

$$
\min_{x}\max_{\lambda,\eta} f(x)+\sum_{i=1}^M\lambda_im_i(x)+\sum_{j=1}^N\eta_jn_j(x) \\\
\text{subject to } \lambda_i \geq 0, i=1,\cdots,M
$$

{{< math.inline >}}
<p>
It is because if \( m_i(x)>0 \) given a \( x \), it reaches infinite:
</p>
{{</ math.inline >}}

$$
\max_{\lambda,\eta} f(x)+\underset{\rarr\infty}{\sum_{i=1}^M\lambda_im_i(x)}+\sum_{j=1}^N\eta_jn_j(x) = \infty
$$

{{< math.inline >}}
<p>
if \( m_i(x) \leq 0 \), it has normal value, then we can conclude:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\min_{x}\max_{\lambda,\eta} f(x)+\sum_{i=1}^M\lambda_im_i(x)+\sum_{j=1}^N\eta_jn_j(x)
&=
\min_{x}\lbrace \max_{\lambda,\eta} f(x)+\sum_{i=1}^M\lambda_im_i(x)+\sum_{j=1}^N\eta_jn_j(x),\infty \rbrace \\\
&= \min_{x,m(x)\leq0}\max_{\lambda,\eta} f(x)+\sum_{i=1}^M\lambda_im_i(x)+\sum_{j=1}^N\eta_jn_j(x)
\end{align*}
$$

It is the same thing primal problem described.

### proof of weak duality

The dual problem is :

$$
\max_{\lambda,\eta}\min_{x} f(x)+\sum_{i=1}^M\lambda_im_i(x)+\sum_{j=1}^N\eta_jn_j(x) \\\
\text{subject to } \lambda_i \geq 0, i=1,\cdots,M
$$

We want to prove:

$$
\min_{x}\max_{\lambda,\eta} f(x)+\sum_{i=1}^M\lambda_im_i(x)+\sum_{j=1}^N\eta_jn_j(x) \geq \max_{\lambda,\eta}\min_{x} f(x)+\sum_{i=1}^M\lambda_im_i(x)+\sum_{j=1}^N\eta_jn_j(x)
$$

it is because:

$$
\text{Let } L(x,\lambda,\eta) = f(x)+\sum_{i=1}^M\lambda_im_i(x)+\sum_{j=1}^N\eta_jn_j(x) \\\
\max_{\lambda,\eta} L(x,\lambda,\eta) \geq L(x,\lambda,\eta) \geq \min_{x} L(x,\lambda,\eta) \\\
\therefore \max_{\lambda,\eta} L(x,\lambda,\eta) \geq \min_{x} L(x,\lambda,\eta)
$$

$$
\begin{align*}
\text{Let }A(x)&=\max_{\lambda,\eta}L(x,\lambda,\eta)\\\
B(\lambda,\eta) &= \min_{x}L(x,\lambda,\eta)
\end{align*}
$$

{{< math.inline >}}
<p>
The largest value of \(B(\lambda,\eta)\) is still smaller than the smallest value of \(A(x)\):
</p>
{{</ math.inline >}}

$$
\min_x A(x) \geq \max_{\lambda,\eta} B(\lambda,\eta) \\\
\therefore \min_x \max_{\lambda,\eta}L(x,\lambda,\eta) \geq \max_{\lambda,\eta}\min_{x}L(x,\lambda,\eta)
$$

### Geometric explanation of duality



## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=27).
[^2]: From [Weak duality](https://en.wikipedia.org/wiki/Weak_duality).
[^3]: From [Strong duality](https://en.wikipedia.org/wiki/Strong_duality).
