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

Because primal problem has a convex function and linear constraintsa and it satisfies slater condition, we can check whether it satisfies [KKT condition](https://tirmisula.github.io/posts/support-vector-machine/#kkt-conditions) to prove a strong duality exsits in between primal and dual problem<cite>[^3]</cite>:

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
\text{subject to } \sum_{i=1}^N \lambda_iy_i=0 , i=1,2,\cdots,N
$$

#### property of strong duality

From previous subsection we know SVM has strong dual problem, then it must satisfies [Karush-Kuhn-Tucker Conditions](https://tirmisula.github.io/posts/support-vector-machine/#karush-kuhn-tucker-conditions)(KKT). The KKT conditions are listed as follows:

$$
\begin{cases}
\frac{\partial L(w,b,\lambda)}{\partial w} = 0,\frac{\partial L(w,b,\lambda)}{\partial b} = 0 \implies \text{satisfies because }w^{\ast},b^{\ast} \text{ solvable} \\\
\lambda_i(1-y_i(w^Tx_i+b)) = 0  \implies \text{satisfies complementary slackness}\\\
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
w^{\ast} &= \sum_{i=1}^N\lambda_iy_ix_i
\end{align*}
$$

{{< math.inline >}}
<p>
Then we can solve \(b^*\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\exist (x_k,y_k), 1-y_k((w^{\ast})^Tx_k+b^{\ast}) &= 0 \\\
y_k^2((w^{\ast})^Tx_k+b^{\ast}) &= y_k \\\
\sum_{i=1}^N\lambda_iy_ix_i^Tx_k+b^{\ast} &= y_k \\\
b^{\ast} &= y_k - \sum_{i=1}^N\lambda_iy_ix_i^Tx_k
\end{align*}
$$

{{< math.inline >}}
<p>
\( w^* \), \( b^* \) are the linear combinations of all data samples.
</p>
{{</ math.inline >}}

The second equation called <mark>complementary slackness</mark> can be divided into 2 situations:

$$
\lambda_i = 0 , y_i(w^Tx_i+b) > 1 \\\
\lambda_i \not= 0 , y_i(w^Tx_i+b) = 1 \\\
$$

{{< math.inline >}}
<p>
Because primal problem maximize distance between closest point to decision boundary \( w^Tx+b=0 \), it must have 2 dual boundaries \( w^Tx+b=1 \), \( w^Tx+b=-1 \) for the closest point for each class.
<br/>
<br/>
So the first situation of satifying complementary slackness indicates other data samples that is far away from dual boundaries and have their weight \(\lambda_i\) determined to be zero, they play no rule in optimization.
<br/>
<br/>
And the second situation indicates only the closest points from 2 class have effective weight \(\lambda_i\). 
<br/>
<br/>
In conclusion, <b>strong duality</b> implies complementary slackness which indicates dual decision borders, <mark>only those points on dual decision borders are support vectors</mark> which determined the final decision boundary for SVM.
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

## Kernel SVM

### Background of kernel method

kernel method is introduced when solving nonlinear separable data samples:

| Linear separable   | Separable but minor error     | nonlinear separable   |
| --------- | -------- | ------ |
| perceptron(PLA) | pocket algorithm | multilayer perceptron |
| hard-margin SVM | soft-margin SVM | kernel SVM |

The thought of kernel method is transforming data to higher dimension which based on the following truth:

$$
\text{Cover Theoreom: high dimentional data are easier to separate than low dimentional data}
$$

For example given a XOR problem:

$$
a,b,c,d \in \mathbb{R}^2 \\\
a,d \in \text{Class 1} \\\
b,c \in \text{Class 2} \\\
a=(0,0) \\\
b=(1,0) \\\
c=(0,1) \\\
d=(1,1)
$$

{{< math.inline >}}
<p>
It is obvious \(a,b,c,d\) are nonlinear separable, but we can design a kernel function to transform samples to higher dimension to make it separable:
</p>
{{</ math.inline >}}

$$
(x_1,x_2) \underset{\phi(x)}{\rarr} \left(x_1,x_2,(x_1-x_2)^2)\right) \\\
a \underset{\phi(x)}{\rarr} (0,0,0) \\\
b \underset{\phi(x)}{\rarr} (1,0,1) \\\
c \underset{\phi(x)}{\rarr} (0,1,1) \\\
d \underset{\phi(x)}{\rarr} (1,1,0) \\\
a,b,c,d \text{ are now linear separable by panel } (*,*,0.5), *\in\mathbb{R}
$$

Given SVM and it's lagrange dual problem:

$$
\min_{w,b} \frac{1}{2}w^Tw \\\
\text{subject to } y_i(w^Tx_i+b) \geq 1, i=1,2,\cdots,N
$$

<br/>

$$
\min_{\lambda} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_jx_i^Tx_j - \sum_{i=1}^N\lambda_i \\\
\text{subject to } \lambda_i \geq 0 , i=1,2,\cdots,N \\\
\text{subject to } \sum_{i=1}^N \lambda_iy_i=0 , i=1,2,\cdots,N
$$

{{< math.inline >}}
<p>
We can replace \(x_i,x_j\) in dual problem with kernel function:
</p>
{{</ math.inline >}}

$$
\min_{\lambda} \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\lambda_i\lambda_jy_iy_j\phi(x_i)^T\phi(x_j) - \sum_{i=1}^N\lambda_i \\\
\text{subject to } \lambda_i \geq 0 , i=1,2,\cdots,N \\\
\text{subject to } \sum_{i=1}^N \lambda_iy_i=0 , i=1,2,\cdots,N
$$

It draw forth the definiton of kernel function:

$$
\forall x_i,x_j \in \mathcal{D}, \exist\space\phi(x):x\rarr z \\\
K(x_i,x_j) = \phi(x_i)^T\phi(x_j) = \left< \phi(x_i),\phi(x_j) \right> \\\
K(x_i,x_j) \text{ is a kernel function} \\\
\text{e.g. } K(x_i,x_j) = \mathrm{e}^{-\frac{(x_i-x_j)^2}{2\sigma^2}}
$$

{{< math.inline >}}
<p>
Finding kernel function rather than finding \( \phi(x) \) and calculating inner product reduces the amount of computation, which is called <mark>kernel tricks</mark>.
</p>
{{</ math.inline >}}

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

### Proof of weak duality

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

Given a primal problem and it's dual problem:

$$
\min_{x\in\mathbb{R}^p} f(x) \\\
\text{subject to } m_1(x) \leq 0
$$

$$
\max_{\lambda}\min_{x} f(x)+\lambda m_1(x) \\\
\text{subject to }\lambda \geq 0
$$

{{< math.inline >}}
<p>
Let \(p^*\) and \(d^*\) be the optimization result of primal problem and dual problem:
</p>
{{</ math.inline >}}

$$
p^{\ast} = \min_{x}f(x), m_1(x)\leq 0 \\\
d^{\ast} = \max_{\lambda}\min_{x} f(x)+\lambda m_1(x), \lambda\geq 0
$$

{{< math.inline >}}
<p>
Define domain \(D\) which represents the intersection of domain \(f(\cdot)\) and domain \(m_1(\cdot)\), and assuming it locates in a 2D cartesian axis, each domain represents horizontal axis \(u\) or vertical axis \(v\) and restricted area is defined as set \(G\): 
</p>
{{</ math.inline >}}

$$
\begin{align*}
D &= f(\cdot) \cap m_1(\cdot) \\\
t &= f(x) \\\
u &= m_1(x) \\\
G &= \lbrace (u,t)|x\in D \rbrace
\end{align*}
$$

{{< math.inline >}}
<p>
Then we can define \(p^*\) and \(d^*\) in this \(D\) space by inifimum of set:
</p>
{{</ math.inline >}}

$$
p^{\ast} = inf\lbrace t|(u,t)\in G,u\leq 0 \rbrace \\\
d^{\ast} = \max_{\lambda} inf\lbrace t+\lambda u|(u,t)\in G,\lambda\geq 0 \rbrace
$$

{{< math.inline >}}
<p>
The geometry meaning of \(p^*\) is the smallest projection point on axis t of left half part of \(G\).
</p>
{{</ math.inline >}}

{{< math.inline >}}
<p>
Let \(t+\lambda u = \gamma \) , it represents the line with slope equals \(-lambda\) has intercept on axis t equals \(\gamma\). Then we can rewrite \(d^*\):
</p>
{{</ math.inline >}}

$$
d^{\ast} = \max_{\lambda} inf\lbrace \gamma |(u,t)\in G,\lambda\geq 0 \rbrace \\\
inf\lbrace \gamma |(u,t)\in G,\lambda\geq 0 \rbrace \implies 
\begin{cases}
\text{intercept of the line} \\\
\text{it's a tangent line below G with negetive slope}
\end{cases} \\\
d^{\ast} = \max_{\lambda} \text{intercept} \implies 
\text{line has }\begin{cases}
\text{tangent position on left half part of G} \\\
\text{AND} \\\
\text{tangent position on right half part of G}
\end{cases}
$$

{{< math.inline >}}
<p>
Recall that the geometric meaning of \(d^*\) happened only when \(\lambda \geq 0\). If maximize intercept by tangent to both sides requires \(\lambda < 0\), it reduced to \(\lambda=0\), which means \(p^*\) and \(d^*\) have the same geometric definition:
</p>
{{</ math.inline >}}

$$
\text{intercept of tangent line with 0 slope}  \iff \text{lowest point's projection coordinate on axis t}
$$

{{< math.inline >}}
<p>
Then we can conclude, if \(G\) is concave:
</p>
{{</ math.inline >}}

$$
p^{\ast} > d^{\ast}
$$

{{< math.inline >}}
<p>
If \(G\) is convex:
</p>
{{</ math.inline >}}

$$
p^{\ast} = d^{\ast}
$$

### Slater condition

Given a primal problem:

$$
\min_{x\in\mathbb{R}^p} f(x) \\\
\text{subject to } m_i(x) \leq 0, i=1,\cdots,M  \\\
\text{subject to } n_j(x) = 0, j=1,\cdots,N
$$

{{< math.inline >}}
<p>
Define the domain \(D\) and set \(G\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
D &= f(\cdot) \cap \lbrace m_i(\cdot)\rbrace_{i=1,\cdots,M} \cap \lbrace n_j(\cdot)\rbrace_{j=1,\cdots,N} \\\
G &= \lbrace (u_1,\cdots,u_{M+N},t|x\in D) \rbrace
\end{align*}
$$

Slater condition is described as:

$$
\exist \hat{x}\in relint(D), (relint\rarr \text{relative interior})\\\
\text{subject to } \forall i=1,\cdots,M, m_i(\hat{x})<0
$$

{{< math.inline >}}
<p>
Slater condition determined the leftmost point of \(G\) is not on axis \(t\) so that tangent line is restricted to be non-vertical. In this way slater condition becomes unnecessary but sufficient condition for strong duality.
</p>
{{</ math.inline >}}

There are two attributes for slater condition:
1. Most of the convex optimization problem satisfy slater condition.
2. Relaxed slater: if M constrained functions contain K affine functions, only the rest of M-K functions are ensured to be less than zero.

Because SVM is a quadratic convex optimization problem, it natrually satisfies slater condition and thus has strong duality problem. Because SVM has strong duality problem and belongs to convex optimization problem, it can be solved by KKT condition.

### Karush-Kuhn-Tucker Conditions

Given a primal problem and it's dual problem:

$$
p^{\ast} = \min_{x\in\mathbb{R}^p} f(x) \\\
\text{subject to } m_i(x) \leq 0, i=1,\cdots,M  \\\
\text{subject to } n_j(x) = 0, j=1,\cdots,N
$$

<br/>

$$
d^{\ast} = \max_{\lambda,\eta}g(\lambda,\eta) \\\
L(x,\lambda,\eta) = f(x)+\sum_{i=1}^M\lambda_{i}m_i(x)+\sum_{j=1}^N\eta_{j}n_j(x) \\\
g(\lambda,\eta) = \min_{x} L(x,\lambda,\eta) \\\
\text{subject to } \lambda_i \geq 0
$$

{{< math.inline >}}
<p>
\(p^*\) corresponds to \(x^*\), \(d^*\) corredponds to \(\lambda^*\) and \(\eta^*\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
p^{\ast} &= f(x^{\ast}) \\\
d^{\ast} &= g(\lambda^{\ast}, \eta^{\ast})
\end{align*}
$$

The relationships among slater condition, KKT conditions and strong duality are listed as follows:

$$
\text{convex} + \text{slater} \implies \text{strong duality} \iff \text{KKT conditions}
$$

KKT conditions for this problem are:

$$
\text{KKT: }
\begin{cases}
\text{restrictions: }
    \begin{cases}
    m_i(x^{\ast}) \leq 0 & (1)\\\
    n_j(x^{\ast}) = 0 & (2)\\\
    \lambda^{\ast} \geq 0 & (3)
    \end{cases} \\\
\text{complementary slackness: } 
    \begin{cases}
    \lambda_i^{\ast}m_i(x^{\ast}) = 0, i=1,\cdots,N & (4)
    \end{cases}\\\
\text{zero gradient: }
    \begin{cases}
    \frac{\partial}{\partial x}L(x,\lambda^{\ast},\eta^{\ast}) = 0 & (5)
    \end{cases}
\end{cases}
$$

Conditions (1),(2),(3) natrually satisfy because they exist in problem clarifications. Next we demonstrate conditions (4) and (5) are necessary and sufficient for strong duality<cite>[^4]</cite>:

$$
\begin{align*}
\because d^{\ast} &= \max_{\lambda,\eta}g(\lambda,\eta) \\\
&= g(\lambda^{\ast}, \eta^{\ast}) \\\
&= \min_{x} L(x,\lambda^{\ast},\eta^{\ast}) \\\
&\leq L(x^{\ast},\lambda^{\ast},\eta^{\ast}) \\\
&\leq f(x^{\ast}) + \sum_{i=1}^M\lambda_{i}^{\ast}m_i(x^{\ast})+\underset{\color{red}{=0}}{\sum_{j=1}^N\eta_{j}^{\ast}n_j(x^{\ast})} \\\
&\leq  f(x^{\ast}) + \underset{\color{red}{\leq 0}}{\sum_{i=1}^M\lambda_{i}^{\ast}m_i(x^{\ast})} \\\
&\leq f(x^{\ast}) \\\
&\leq p^{\ast} \\\
\therefore d^{\ast} = p^{\ast} &\iff \sum_{i=1}^M\lambda_{i}^{\ast}m_i(x^{\ast}) = 0 \\\
&\iff \lambda_{i}^{\ast}m_i(x^{\ast}) = 0 \space, i =1,\cdots,M \\\
\therefore d^{\ast} = p^{\ast} &\iff d^{\ast} = f(x^{\ast}) + \underset{\color{red}{}}{\sum_{i=1}^M\lambda_{i}^{\ast}m_i(x^{\ast})} \\\
&\iff d^{\ast} = L(x^{\ast},\lambda^{\ast},\eta^{\ast}) \\\
&\iff \min_{x} L(x,\lambda^{\ast},\eta^{\ast}) = L(x^{\ast},\lambda^{\ast},\eta^{\ast}) \\\
&\iff \exist x=x^{\ast}, \nabla_{x}L(x,\lambda^{\ast}, \eta^{\ast}) = 0
\end{align*}
$$


## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=27).
[^2]: From [Weak duality](https://en.wikipedia.org/wiki/Weak_duality).
[^3]: From [Strong duality](https://en.wikipedia.org/wiki/Strong_duality).
[^4]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=35).