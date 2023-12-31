---
author: "X. Wang"
title: "Marginal/Conditional/Joint Gaussian probability density function"
date: "2023-08-28"
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

Before we started, two lemmas will be given first:                                                         

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
### Lemma 1
If we have the following,
$$
\begin{cases}
x \sim \mathcal{N}(\mu,\Sigma),
x \in \mathbb{R}^p\\\
y = Ax+B,
y \in \mathbb{R}^q
\end{cases}
$$

{{< math.inline >}}
<p>
Then \(y\) is subject to Gaussian distribution,
</p>
{{</ math.inline >}}

$$
y \sim \mathcal{N}(A\mu+B,A\Sigma A^T)
$$
Because,
$$
\begin{align*}
E[y] &= E[Ax+B]\\\
&= AE[x]+B\\\
&= A\mu+B
\end{align*}\\\
\begin{align*}
Var[y] &= Var[Ax+B] \\\
&= AVar[x]A^T \\\
&= A\Sigma A^T
\end{align*}
$$

### Lemma 2
If we have the following,
$$
\begin{cases}
x \sim \mathcal{N}(\mu,\Sigma),
x \in \mathbb{R}^p\\\
y_1 = A_1 x\\\
y_2 = A_2 x
\end{cases}
$$

Then,
$$
y_1 \perp y_2 \iff A_1 \Sigma A_2^T=0
$$

Because,
$$
\begin{rcases}
y_1 \sim \mathcal{N}(A_1\mu, A_1 \Sigma A_1^T)\\\
y_2 \sim \mathcal{N}(A_2\mu, A_2 \Sigma A_2^T)
\end{rcases}
\implies \text{Lemma 1}
$$

$$
\begin{align*}
Cov(y_1,y_2) &= E\left[ (y_1 - E[y_1])(y_2 - E[y_2])^T \right]\\\
&= E\left[ (A_1 x - A_1 \mu)(A_2 x - A_2 \mu)^T \right]\\\
&= E\left[ A_1(x - \mu)(x - \mu)^TA_2^T \right]\\\
&= A_1 E\left[ (x - \mu)(x - \mu)^T \right] A_2^T\\\
&= A_1 Cov(x,x) A_2\\\
&= A_1 \Sigma A_2\\\
\end{align*}
$$

$$
\therefore Cov(y_1,y_2) =  A_1 \Sigma A_2 =0 \iff y_1 \perp y_2
$$

## Marginal and Conditional PDF Problem clarification <cite>[^1]</cite>
{{< math.inline >}}
<p>
Given a dataset \(\mathcal{D} = \{x_1, x_2, ..., x_n\}\), where each data point \(x_i\) is drawn from a p-D Gaussian distribution with unknown mean \(\mu\) and variance \(\Sigma\), we have:
</p>
{{</ math.inline >}}

$$p(x|\mu,\Sigma) = \frac{1}{(2\pi)^\frac{p}{2} |\Sigma|^\frac{1}{2}} \exp\left( -\frac{1}{2}(x-\mu)^T \Sigma^{-1} (x-\mu) \right)$$

$$
x = \begin{bmatrix}
    x_{1} \\\
    x_{2} \\\
    \vdots \\\
    x_{p}
\end{bmatrix},
\mu = \begin{bmatrix}
    \mu_{1} \\\
    \mu_{2} \\\
    \vdots \\\
    \mu_{p}
\end{bmatrix},
\Sigma =\begin{bmatrix}
    \sigma_{11} & \sigma_{12} & \cdots & \sigma_{1p} \\\
    \sigma_{21} & \sigma_{22} & \cdots & \sigma_{2p} \\\
    \vdots & \vdots & \ddots & \vdots \\\
    \sigma_{p1} & \sigma_{p2} & \cdots & \sigma_{pp}
\end{bmatrix}
$$

{{< math.inline >}}
<p>
If we consider x is the <b>joint pdf</b> of two parts \(x_a\) and \(x_b\), we can rewrite \(x\), \(\mu\), \(\Sigma\) above with block matrix:
</p>
{{</ math.inline >}}

$$
x=\begin{bmatrix}
    x_1\\\
    x_2\\\
    \vdots\\\
    x_m\\\
    x_{m+1}\\\
    x_{m+2}\\\
    \vdots\\\
    x_{m+n}
\end{bmatrix}
=\begin{bmatrix}
    x_a\\\
    x_b
\end{bmatrix}, m+n=p
$$

$$
\mu = \begin{bmatrix}
    \mu_{a}\\\
    \mu_{b}
\end{bmatrix}
$$

$$
\Sigma = \begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab}\\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix}
$$

{{< math.inline >}}
<p>
Because joint pdf \( p(x) \) is known, we want to obtain the marginal pdf \(p(x_a)\) and \(p(x_b)\), the conditional pdf \(p(x_b|x_a)\) and \(p(x_a|x_b)\)
</p>
{{</ math.inline >}}

### Solve the marginal pdf

{{< math.inline >}}
<p>
Based on **Lemma 1** we can construct \(x_a\) by \(Ax+B\):
</p>
{{</ math.inline >}}

$$
x_a = \underset{
    \color{red}{A}
}{\begin{bmatrix}
    1_m & 0_n
\end{bmatrix}}
\underset{
    \color{red}{x}
    }{\begin{bmatrix}
    x_{a}\\\
    x_{b}
\end{bmatrix}}
$$

Then we can get:
$$
\begin{align*}
E[x_a] &= E\left[\begin{bmatrix}
    1_m & 0_n
\end{bmatrix}\begin{bmatrix}
    x_{a}\\\
    x_{b}
\end{bmatrix}\right]\\\
&= \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}E\left[\begin{bmatrix}
    x_{a}\\\
    x_{b}
\end{bmatrix}\right]\\\
&= \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}\begin{bmatrix}
    \mu_{a}\\\
    \mu_{b}
\end{bmatrix}\\\
&= \mu_{a}
\end{align*}
$$

$$
\begin{align*}
Var[x_a] &= \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}
\Sigma
\begin{bmatrix}
    1_m\\\
    0_n
\end{bmatrix}\\\
&= \begin{bmatrix}
    1_m & 0_n
\end{bmatrix}
\begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab}\\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix}
\begin{bmatrix}
    1_m\\\
    0_n
\end{bmatrix}\\\
&= \begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab}
\end{bmatrix}
\begin{bmatrix}
    1_m\\\
    0_n
\end{bmatrix}\\\
&= \Sigma_{aa}
\end{align*}
$$

{{< math.inline >}}
<p>
Deduction of \(x_b\) is similar, so the conclusion is:
</p>
{{</ math.inline >}}

$$
x_a \sim \mathcal{N}(\mu_{a},\Sigma_{aa})
$$
$$
x_b \sim \mathcal{N}(\mu_{b},\Sigma_{bb})
$$

### Solve the conditonal pdf
{{< math.inline >}}
<p>
Before dig into \(p(x_b|x_a)\), let's define a new variable \(x_{b \cdot a}\)
</p>
{{</ math.inline >}}

$$
\begin{align*}
x_{b \cdot a} &= x_b - \Sigma_{ba} \Sigma_{aa}^{-1} x_a\\\
&= \underset{
    \color{red}{A}
}{\begin{bmatrix}
    -\Sigma_{ba} \Sigma_{aa}^{-1} & 1_n
\end{bmatrix}}
\underset{
    \color{red}{x}
    }{\begin{bmatrix}
    x_{a}\\\
    x_{b}
\end{bmatrix}}
\end{align*}
$$


{{< math.inline >}}
<p>
According to Lemma 1:
</p>
{{</ math.inline >}}

$$
\begin{align*}
E[x_{b \cdot a}] &= \begin{bmatrix}
    -\Sigma_{ba}\Sigma_{aa}^{-1} & 1_n
\end{bmatrix}
\begin{bmatrix}
    \mu_{a}\\\
    \mu_{b}
\end{bmatrix}\\\
&= -\Sigma_{ba}\Sigma_{aa}^{-1} \mu_{a} + \mu_{b}
\end{align*}
$$

$$
\begin{align*}
Var[x_{b \cdot a}] &= \begin{bmatrix}
    -\Sigma_{ba}\Sigma_{aa}^{-1} & 1_n
\end{bmatrix}
\Sigma
\begin{bmatrix}
    -\Sigma_{ba}\Sigma_{aa}^{-1}\\\
    1_{n}
\end{bmatrix}\\\
&= \begin{bmatrix}
    -\Sigma_{ba}\Sigma_{aa}^{-1} & 1_n
\end{bmatrix}
\begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab}\\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix}
\begin{bmatrix}
    -\Sigma_{ba}\Sigma_{aa}^{-1}\\\
    1_{n}
\end{bmatrix}\\\
&= \begin{bmatrix}
    0 & -\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb}
\end{bmatrix}
\begin{bmatrix}
    -\Sigma_{ba}\Sigma_{aa}^{-1}\\\
    1_{n}
\end{bmatrix}\\\
&= -\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb}
\end{align*}
$$

{{< math.inline >}}
<p>
Based on Lemma 2, we can prove that \(x_{b \cdot a}\) is independent from \(x_a\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
Cov(x_{b \cdot a}, x_a) &= \begin{bmatrix}
    -\Sigma_{ba}\Sigma_{aa}^{-1} & 1_n
\end{bmatrix}
\begin{bmatrix}
    \Sigma_{aa} & \Sigma_{ab}\\\
    \Sigma_{ba} & \Sigma_{bb}
\end{bmatrix}
\begin{bmatrix}
    1_m \\\
    0_n
\end{bmatrix}\\\
&= \begin{bmatrix}
    0_m & -\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab} + \Sigma_{bb}
\end{bmatrix}
\begin{bmatrix}
    1_m \\\
    0_n
\end{bmatrix}\\\
&= 0
\end{align*}
\\\
\therefore 
x_{b \cdot a} \perp x_a \implies p(x_{b \cdot a}) = p(x_{b \cdot a}|x_a)
$$

{{< math.inline >}}
<p>
It's clear \( x_{b \cdot a} \) is a well-structured variable that links \(x_a\) and \(x_b|x_a\) together, so that \( p(x_b|x_a) \) can be derived from \( p(x_a) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\therefore p(x_b|x_a) &= p((x_{b \cdot a} + \Sigma_{ba}\Sigma_{aa}^{-1}x_a) | x_a)\\\
&= p(x_{b \cdot a}|x_a + \Sigma_{ba}\Sigma_{aa}^{-1}x_a|x_a)\\\
&= p(x_{b \cdot a} + \Sigma_{ba}\Sigma_{aa}^{-1}x_a)
\end{align*}
$$
$$
\begin{align*}
\therefore x_b|x_a &= x_{b \cdot a} + \Sigma_{ba}\Sigma_{aa}^{-1}x_a\\\
x_b|x_a &= \underset{\color{red}{A}}{
\begin{bmatrix}
    1_m \\\
    1_n
\end{bmatrix}
}
x_{b \cdot a}+
\underset{\color{red}{B}}{\Sigma_{ba}\Sigma_{aa}^{-1}x_a}\\\
E[x_b|x_a] &= -\Sigma_{ba}\Sigma_{aa}^{-1} \mu_{a} + \mu_{b} + \Sigma_{ba}\Sigma_{aa}^{-1}x_a\\\
&= \Sigma_{ba}\Sigma_{aa}^{-1} (x_a-\mu_{a}) + \mu_{b}\\\
Var[x_b|x_a] &= I (-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb}) I^T\\\
&= -\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb}
\end{align*}
$$

{{< math.inline >}}
<p>
Deduction of \(x_a|x_b\) is similar, so the conclusion is :
</p>
{{</ math.inline >}}

$$
x_b|x_a \sim \mathcal{N}(\Sigma_{ba}\Sigma_{aa}^{-1} (x_a-\mu_{a}) + \mu_{b}, -\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb})
$$
$$
x_a|x_b \sim \mathcal{N}(\Sigma_{ab}\Sigma_{bb}^{-1} (x_b-\mu_{b}) + \mu_{a}, -\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}+\Sigma_{aa})
$$

### Conclusion

Marginal PDF:

$$
x_a \sim \mathcal{N}(\mu_{a},\Sigma_{aa})
$$
$$
x_b \sim \mathcal{N}(\mu_{b},\Sigma_{bb})
$$

Conditional PDF:

$$
x_b|x_a \sim \mathcal{N}(\Sigma_{ba}\Sigma_{aa}^{-1} (x_a-\mu_{a}) + \mu_{b}, \underset{\color{red}{\text{Schur complementary of }\Sigma_{aa}}}{-\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb} } )
$$
$$
x_a|x_b \sim \mathcal{N}(\Sigma_{ab}\Sigma_{bb}^{-1} (x_b-\mu_{b}) + \mu_{a}, \underset{\color{red}{\text{Schur complementary of }\Sigma_{bb}}}{-\Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}+\Sigma_{aa} } )
$$

## Joint PDF Problem clarification <cite>[^2]</cite>
{{< math.inline >}}
<p>
Assuming we have variable \(x\) from a p-dimensional Gaussian distribution with mean \(\mu\) and covariance matrix \(\Sigma\):
</p>
{{</ math.inline >}}

$$
x \sim \mathcal{N} (\mu, \Lambda^{-1})\\\
\Lambda \text{ is precision matrix, }\Lambda=\Sigma^{-1}
$$

{{< math.inline >}}
<p>
And a linear transformation \(y\):
</p>
{{</ math.inline >}}

$$
y=Ax+B+\epsilon\\\
\epsilon \sim \mathcal{N}(0, L^{-1}), \epsilon \perp x
$$

{{< math.inline >}}
<p>
Let z be the joint variable:
</p>
{{</ math.inline >}}

$$
z = \begin{bmatrix}
    x\\\
    y
\end{bmatrix}
$$

{{< math.inline >}}
<p>
Because marginal pdf \( p(x) \) is known, we want to obtain the joint pdf \(p(z)\) and conditional pdf \(p(x|y)\)
</p>
{{</ math.inline >}}

### Solve the joint pdf
{{< math.inline >}}
<p>
Since \(x\) is known, we can solve \(y\) first:
</p>
{{</ math.inline >}}

$$
\begin{align*}
E[y] &= E[Ax+B+\epsilon]\\\
&= E[Ax+B] + E[\epsilon]\\\
&= AE[x]+B+0\\\
&= A\mu + B
\end{align*}
$$

$$
\begin{align*}
Var[y] &= Var[Ax+B+\epsilon]\\\
&= Var[Ax+B] + Var[\epsilon] + 2Cov[Ax+B, \epsilon]\\\
&= Var[Ax+B] + Var[\epsilon], \because \epsilon \perp x\\\
&= A \Lambda^{-1} A^T + L^{-1}
\end{align*}
$$

$$
y \sim \mathcal{N}(A\mu+B, A \Lambda^{-1} A^T + L^{-1})
$$

{{< math.inline >}}
<p>
Then we solve \(z\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
E[z] &= E\left[
    \begin{bmatrix}
    x \\\
    y
\end{bmatrix}
\right]\\\
&= \begin{bmatrix}
    E[x] \\\
    E[y]
\end{bmatrix}\\\
&= \begin{bmatrix}
    \mu \\\
    A\mu + B
\end{bmatrix}
\end{align*}
$$

$$
\begin{align*}
Var[z] &= Var\left[
    \begin{bmatrix}
    x \\\
    y
\end{bmatrix}
\right]\\\
&= \begin{bmatrix}
    Cov[x,x] & Cov[x,y]\\\
    Cov[y,x] & Cov[y,y]
\end{bmatrix}\\\
&= \begin{bmatrix}
    \Lambda^{-1} & Cov[x,y]\\\
    Cov[y,x] & A \Lambda^{-1} A^T + L^{-1}
\end{bmatrix}\\\
&= \begin{bmatrix}
    \Lambda^{-1} & Cov[x,y]\\\
    Cov[x,y]^T & A \Lambda^{-1} A^T + L^{-1}
\end{bmatrix}
\end{align*}
$$

{{< math.inline >}}
<p>
For the \(Cov[x,y]\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
Cov[x,y] &= E\left[
    \left( x-E[x] \right) \left( y-E[y] \right)^T
\right]\\\
&= E\left[
    \left( x-\mu \right) \left( y-A\mu-B \right)^T
\right]\\\
&= E\left[
    \left( x-\mu \right) \left( Ax+B+\epsilon-A\mu-B \right)^T
\right]\\\
&= E\left[
    \left( x-\mu \right) \left( Ax-A\mu+\epsilon \right)^T
\right]\\\
& = E\left[
    \left( x-\mu \right) \left( Ax-A\mu \right)^T + \left( x-\mu \right) \epsilon^T
\right]\\\
&= E\left[
    \left( x-\mu \right) \left( Ax-A\mu \right)^T \right] + E\left[\left( x-\mu \right) \epsilon^T
\right]\\\
&= E\left[
    \left( x-\mu \right) \left( Ax-A\mu \right)^T \right] + E\left[x-\mu
\right] E\left[ \epsilon^T \right], \because x \perp \epsilon\\\
&= E\left[
    \left( x-\mu \right) \left( x-\mu \right)^T A^T
\right]\\\
&= E\left[
    \left( x-\mu \right) \left( x-\mu \right)^T
\right] A^T\\\
&= Var[x] A^T\\\
&= \Lambda^{-1} A^T
\end{align*}
$$

{{< math.inline >}}
<p>
So continue solving \(Var[z]\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
Var[z] &= \begin{bmatrix}
    \Lambda^{-1} & Cov[x,y]\\\
    Cov[x,y]^T & A \Lambda^{-1} A^T + L^{-1}
\end{bmatrix}\\\
&= \begin{bmatrix}
    \Lambda^{-1} & \Lambda^{-1} A^T\\\
    (\Lambda^{-1} A^T)^T & A \Lambda^{-1} A^T + L^{-1}
\end{bmatrix}
\end{align*}
$$

{{< math.inline >}}
<p>
So the conclusion is:
</p>
{{</ math.inline >}}

$$
y \sim \mathcal{N}(A\mu+B, A \Lambda^{-1} A^T + L^{-1})
$$

$$
z = \begin{bmatrix}
    x\\\
    y
\end{bmatrix} \sim \mathcal{N}(\begin{bmatrix}
    \mu \\\
    A\mu + B
\end{bmatrix}, \begin{bmatrix}
    \Lambda^{-1} & \Lambda^{-1} A^T\\\
    (\Lambda^{-1} A^T)^T & A \Lambda^{-1} A^T + L^{-1}
\end{bmatrix})
$$

### Solve the conditonal pdf
{{< math.inline >}}
<p>
Based on previous conclusion,
</p>
{{</ math.inline >}}

$$
x_b|x_a \sim \mathcal{N}(\Sigma_{ba}\Sigma_{aa}^{-1} (x_a-\mu_{a}) + \mu_{b}, -\Sigma_{ba}\Sigma_{aa}^{-1}\Sigma_{ab}+\Sigma_{bb})
$$

We have:
$$
\begin{align*}
E[x|y] &= 
    \Lambda^{-1} A^T (A \Lambda^{-1} A^T + L^{-1})^{-1} (y - A\mu-B) + \mu 
\end{align*}
$$

$$
\begin{align*}
Var[x|y] &= 
    -\Lambda^{-1} A^T (A \Lambda^{-1} A^T + L^{-1})^{-1}  (\Lambda^{-1} A^T)^T + \Lambda^{-1} \\\
&= -\Lambda^{-1} A^T (A \Lambda^{-1} A^T + L^{-1})^{-1}  A (\Lambda^{-1})^{T} + \Lambda^{-1}
\end{align*}
$$

$$
\begin{align*}
x|y \sim \mathcal{N} \left( \Lambda^{-1} A^T (A \Lambda^{-1} A^T + L^{-1})^{-1} (y - A\mu-B) + \mu , 
    -\Lambda^{-1} A^T (A \Lambda^{-1} A^T + L^{-1})^{-1}  (\Lambda^{-1} A^T)^T + \Lambda^{-1}
    \right)
\end{align*}
$$

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=7).
[^2]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=8).