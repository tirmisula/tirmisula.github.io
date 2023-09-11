---
author: "X. Wang"
title: "Linear Classification - LDA"
date: "2023-09-01"
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

LDA algorithm:                                                         

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

## Prerequisite
### Geometric meaning of vector inner product
Given that,
$$
w \in \mathbb{R}^p, \lvert w \rvert = 1\\\
x \in \mathbb{R}^p
$$

We have the inner product by definition:

$$
\begin{align*}
w^Tx &= \lvert w \rvert \lvert x \rvert \cos(\alpha) \\\
&= \lvert x \rvert \cos(\alpha)
\end{align*}
$$
{{< math.inline >}}
<p>
So \( w^Tx \) means the \(x\)'s projection point on aixs \( w \), it is  a scalar.
</p>
{{</ math.inline >}}


## LDA model

Linear discriminant analysis(LDA) is a linear classification method similar to perceptron model.

{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\), it is described as:
</p>
{{</ math.inline >}}

$$
\mathcal{D} = \{(x_1,y_1), (x_2,y_2), ..., (x_N,y_N)\}\\\
x_i \in \mathbb{R}^p, y_i \in \lbrace 1,-1 \rbrace
$$

We define 2 classes C1 and C2, it has following attributes:

$$
x_{C1} = \{ x_i|y_i=1 \} \\\
x_{C2} = \{ x_i|y_i=-1 \} \\\
|x_{C1}| = N_1 \\\
|x_{C2}| = N_2 \\\
N = N_1+N_2
$$

{{< math.inline >}}
<p>
The core idea of LDA method is reducing dimensions. By projecting \(x_i\) to a 1-dimensional axis, LDA tries to minimize variance of samples inside a class and maximize distance between 2 classes simutaneously.
</p>
{{</ math.inline >}}

From [Prerequisite](#prerequisite) we can define projection first:

$$
\begin{align*}
z_i = w^Tx_i
\end{align*}
$$

{{< math.inline >}}
<p>
\(z_i\) is the projection point <mark>scalar</mark> of \(x_i\) on \(w\) axis. We can still split 2 classes for \(z_i\):
</p>
{{</ math.inline >}}

$$
z_{C1} = \{ z_i|x_i \in C1 \}\\\
z_{C2} = \{ z_i|x_i \in C2 \}
$$

Then for each class we can define centroid and sample variance:

$$
\begin{align*}
\bar{z_{C1}} &= \frac{1}{N_1} \sum_{z_i \in z_{C1}} z_i \\\
&= \frac{1}{N_1} \sum_{x_i \in x_{C1}} w^Tx_i \\\
&= w^T \bar{x_{C1}}
\end{align*}
$$

$$
\begin{align*}
\bar{z_{C2}} &= \frac{1}{N_2} \sum_{z_i \in z_{C2}} z_i \\\
&= \frac{1}{N_2} \sum_{x_i \in x_{C2}} w^Tx_i \\\
&= w^T \bar{x_{C2}}
\end{align*}
$$

$$
\begin{align*}
\sigma_{z_{C1}}^2 &= Cov(z_{C1},z_{C1})\\\
&= E\left[ (w^Tx_i - \bar{z_{C1}})(w^Tx_i - \bar{z_{C1}})^T \right]\\\
&= \frac{1}{N_1} \sum_{x_i \in x_{C1}} (w^Tx_i - \bar{z_{C1}})(w^Tx_i - \bar{z_{C1}})^T \\\
&= \frac{1}{N_1} \sum_{x_i \in x_{C1}} (w^Tx_i - w^T \bar{x_{C1}})(w^Tx_i - w^T \bar{x_{C1}})^T \\\
&= \frac{1}{N_1} \sum_{x_i \in x_{C1}} w^T(x_i - \bar{x_{C1}})(x_i - \bar{x_{C1}})^Tw \\\
&= w^T \left[\frac{1}{N_1} \sum_{x_i \in x_{C1}} (x_i - \bar{x_{C1}})(x_i - \bar{x_{C1}})^T\right] w \\\
&= w^T \sigma_{x_{C1}}^2 w
\end{align*}
$$

$$
\begin{align*}
\sigma_{z_{C2}}^2 &= Cov(z_{C2},z_{C2})\\\
&= E\left[ (w^Tx_i - \bar{z_{C2}})(w^Tx_i - \bar{z_{C2}})^T \right]\\\
&= \frac{1}{N_2} \sum_{x_i \in x_{C2}} (w^Tx_i - \bar{z_{C2}})(w^Tx_i - \bar{z_{C2}})^T \\\
&= \frac{1}{N_2} \sum_{x_i \in x_{C2}} (w^Tx_i - w^T \bar{x_{C2}})(w^Tx_i - w^T \bar{x_{C2}})^T \\\
&= \frac{1}{N_2} \sum_{x_i \in x_{C2}} w^T(x_i - \bar{x_{C2}})(x_i - \bar{x_{C2}})^Tw \\\
&= w^T \left[\frac{1}{N_1} \sum_{x_i \in x_{C2}} (x_i - \bar{x_{C2}})(x_i - \bar{x_{C1}})^T\right] w \\\
&= w^T \sigma_{x_{C2}}^2 w
\end{align*}
$$

Then we can design the object function:

$$
\begin{align*}
J(w) &= \frac{(\bar{z_{C1}}-\bar{z_{C2}})^2}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 }\\\
&= \frac{(w^T \bar{x_{C1}}-w^T \bar{x_{C2}})^2}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 } \\\
&= \frac{(w^T (\bar{x_{C1}}-\bar{x_{C2}}))^2}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 } \\\
&= \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}})(w^T (\bar{x_{C1}}-\bar{x_{C2}}))^T}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 } \\\
&= \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^T w}{ \sigma_{z_{C1}}^2 + \sigma_{z_{C2}}^2 } \\\
&= \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^T w}{ w^T \sigma_{x_{C1}}^2 w + w^T \sigma_{x_{C2}}^2 w } \\\
&= \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^T w}{ w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w }
\end{align*}
$$

$$
\hat{w} = \argmax_w J(w)
$$

 <cite>[^1]</cite>

## Solve LDA object function

<!-- \frac{\partial}{\partial w} w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw  \left(w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w\right)^{-1}  &= 0,uv \implies u'v+uv'\\\ 
2(\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw
\left(w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w\right)^{-1}
+
w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw
\cdot(-1)\cdot \left(w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w\right)^{-2} \cdot 2(\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)w
= 0 \\\ -->

$$
\begin{align*}
\frac{\partial}{\partial w}J(w) &= 0\\\
\frac{\partial}{\partial w} \frac{w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^T w}{ w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w } &= 0\\\
(\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw
\left(w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w\right)^{-1}
&= 
w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw
 \left(w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w\right)^{-2} (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)w
\end{align*}
$$

{{< math.inline >}}
<p>
Notice that \( w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w \) and \( w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw \) are <mark>scalars</mark>, we can continue the simplification:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw
w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w
&= 
w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw
 (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)w \\\
(\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw 
&=
\frac{w^T (\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw}
{w^T (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2) w} 
(\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)w \\\
(\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw 
&\propto
(\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)w \\\
&\Darr\\\
(\bar{x_{C1}}-\bar{x_{C2}}) &\propto (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)w
\end{align*}
$$

{{< math.inline >}}
<p>
Since \( (\bar{x_{C1}}-\bar{x_{C2}})^Tw \) is a scalar:
</p>
{{</ math.inline >}}

$$
\begin{align*}
(\bar{x_{C1}}-\bar{x_{C2}}) (\bar{x_{C1}}-\bar{x_{C2}})^Tw 
&\propto
(\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)w \\\
&\Darr\\\
(\bar{x_{C1}}-\bar{x_{C2}}) &\propto (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)w \\\
w &\propto (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)^{-1}(\bar{x_{C1}}-\bar{x_{C2}})
\end{align*}
$$

{{< math.inline >}}
<p>
Furthermore, if \( (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)^{-1} \) is diagnal:
</p>
{{</ math.inline >}}

$$
(\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)^{-1} \propto I \implies w \propto (\bar{x_{C1}}-\bar{x_{C2}})
$$

## Conclusion

In general situation:

$$
\hat{w} \propto (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)^{-1}(\bar{x_{C1}}-\bar{x_{C2}})
$$

{{< math.inline >}}
<p>
If \( (\sigma_{x_{C1}}^2 + \sigma_{x_{C2}}^2)^{-1} \propto I \):
</p>
{{</ math.inline >}}

$$
\hat{w} \propto (\bar{x_{C1}}-\bar{x_{C2}})
$$

{{< math.inline >}}
<p>
\( \hat{w} \) is the normal to the discrimant hyperplane.
</p>
{{</ math.inline >}}

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=16).
[^2]: From [source](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).