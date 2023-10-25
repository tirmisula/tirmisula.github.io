---
author: "X. Wang"
title: "Markov chain Monte Carlo"
date: "2023-09-11"
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

### The object of Monte Carlo

<cite>[^1]</cite>

$$
\begin{align*}
p(z|x) \rarr E_{z|x\sim p(z|x)}\left[ f(z) \right] &= \int_{z}f(z)p(z|x)dz \\\
&\approx \frac{1}{N} \sum_{i=1}^N f(z_i) \\\
z_1,\cdots,z_N&\text{ are randomly sampled from } p(z|x)
\end{align*}
$$

There are serveral ways to do sampling.

### Sampling from cdf

{{< math.inline >}}
<p>
Suppose we have cumulative density function of \(p(z|x)\), CDF:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{CDF}(t) &= \int_{-\infty}^{t}p(z|x)dz \\\
&t\in [-\infty,\infty] \\\
&\text{CDF}(t)\in [0,1]
\end{align*}
$$

{{< math.inline >}}
<p>
And we have \( u_1,\cdots,u_N \) sampled from uniform distribution \( \mathcal{U}(0,1) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
u_1,\cdots,u_N\sim \mathcal{U}(0,1)
\end{align*}
$$

{{< math.inline >}}
<p>
\(z_i\) is given by the inverse of cdf:
</p>
{{</ math.inline >}}

$$
\begin{align*}
&\forall 1,\cdots,N \\\
&z_i = \text{CDF}^{-1}(u_i) \\\
\end{align*}
$$

{{< math.inline >}}
<p>
Apparently CDF may not be solvable for most of complex \(p(z|x)\).
</p>
{{</ math.inline >}}

### Rejection sampling

{{< math.inline >}}
<p>
We purpose a pdf \(q(z)\) to approximate \(p(z|x)\), define acceptance rate \(\alpha\):
</p>
{{</ math.inline >}}

$$
\alpha = \frac{p(z|x)}{Mq(z)}
$$

Then the sampling procedures are:

$$
\begin{align*}
&\forall i=1,\cdots,N \\\
&z_i \sim q(z) \\\
&u_i \sim \mathcal{U}(0,1) \\\
&\begin{cases}
\text{If } u\leq \alpha, \text{accept $z_i$} \\\
\text{Else }, \text{reject $z_i$} 
\end{cases}
\end{align*}
$$

{{< math.inline >}}
<p>
The core idea is the larger difference between \(p(z_i|x)\) and \(q(z_i)\) the more likely \(z_i\) will be rejected. The problem is finding such \(q(z)\) is difficult.
</p>
{{</ math.inline >}}

### Importance sampling

$$
\begin{align*}
E_{z|x\sim p(z|x)}\left[ f(z) \right] &= \int_{z}f(z)p(z|x)dz \\\
&= \int_{z}f(z)\frac{p(z|x)}{q(z)}q(z)dz \\\
&\approx \frac{1}{N} \sum_{i=1}^N f(z_i)\frac{p(z_i|x)}{q(z_i)}, \frac{p(z_i|x)}{q(z_i)}\text{ is weight} \\\
z_1,\cdots,z_N&\text{ are randomly sampled from } q(z)
\end{align*}
$$

{{< math.inline >}}
<p>
It works when \( p(z|x) \) and \( q(z) \) are close, if the weight is not close to 1 we need to do resampling based on the importance sampling result by weights which is called Importance-Sampling-Resampling.
</p>
{{</ math.inline >}}

## Markov chain
### Description
Given preset:

$$
\begin{align*}
&\lbrace x_t \rbrace : \text{discrete time series} \\\
&S_i : \text{state i} \\\
&\pi_t(x) : \text{probability distribution of state at time $t$} \\\
&\sum_j \pi_t(x=S_j) = 1
\end{align*}
$$

The first order Markov chain is described as:

$$
\begin{align*}
p(x_{t+1}|x_1,x_2,\cdots,x_t) = p(x_{t+1}|x_t)
\end{align*}
$$

{{< math.inline >}}
<p>
The state of \(x_{t+1}\) is only relevant to the state of last time \(x_t\)
</p>
{{</ math.inline >}}

### State transition matrix

{{< math.inline >}}
<p>
The state transition probability can be described as a transition matrix \(P_{ij}\) and has following attribute:
</p>
{{</ math.inline >}}

$$
\begin{align*}
P_{ij} &= p(x_{t+1}=S_j|x_t=S_i) \\\
\sum_j P_{ij} &= 1
\end{align*}
$$

{{< math.inline >}}
<p>
Then we can easily derive the distribution \(\pi_{t+1}\) from \(\pi_t\) by transition matrix:
</p>
{{</ math.inline >}}

$$
\pi_{t+1}(x=S_{j}) = \sum_i \pi_t(x=S_i)P_{ij} \\\
\text{for any state $\ast$} \\\
\pi_{t+1}(x=S_{\ast}) = \sum_i \pi_t(x=S_i)P_{i\ast}
$$

### Stationary distribution and detailed balance

{{< math.inline >}}
<p>
We call \( \pi(x) \) a <mark>stationary distribution</mark> if \( \pi_t(x) \) and \( \pi_{t+1}(x) \) are indentically equal after transition:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\pi_{t+1}(x=S_{\ast}) = \pi_{t}(x=S_{\ast}) &= \sum_i \pi_t(x=S_i)P_{i\ast} \\\
&\dArr \\\
\pi(x=S_{\ast}) &= \sum_i \pi(x=S_i)P_{i\ast}
\end{align*}
$$

{{< math.inline >}}
<p>
If the chain is reversible w.r.t. \(\pi(x)\), we call it <mark>detailed balance</mark> equation:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\pi(x=S_i)P_{ij} = \pi(x=S_j)P_{ji}, \forall i,j
\end{align*}
$$

{{< math.inline >}}
<p>
Next we can prove that detailed balance is sufficient but not necessary condition for stationary distribution:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{detailed balance} &\implies \text{stationary distribution} \\\
&\dArr\\\
\text{Let } j&=\ast, \text{we sum all over $j$} \\\
\sum_j \pi(x=S_i)P_{ij} &= \sum_j \pi(x=S_{j})P_{ji} \\\
\pi(x=S_i)\sum_j P_{ij} &= \sum_j \pi(x=S_{j})P_{ji} \\\
\pi(x=S_i) &= \sum_j \pi(x=S_{j})P_{ji} \\\
\text{Let } i&=\ast \\\
\pi(x=S_{\ast}) &= \sum_{j}\pi(x=S_j)P_{j\ast} \\\
\text{Q.E.D.}
\end{align*}
$$

## Metropolis-Hastings algorithm

Based on the introduction from previous sections, our ideas of how to sample are listed as follows:

$$
\begin{align*}
&E_{z|x\sim p(z|x)}\left[ f(z) \right] \approx \frac{1}{N} \sum_{i=1}^N f(z_i) \\\
&z_1,\cdots,z_N\text{ cannot sample from } p(z|x) \\\
&\dArr \\\
&\text{construct markov chain, let } \pi(z)=p(z|x) \\\
&\text{$z_i$ sampled by: } z_1 \sim \pi(z_1),z_2\sim \pi(z_2|z_1),\cdots,z_N\sim \pi(z_N|z_{N-1}) \\\
&\dArr \\\
&\text{requires $\pi(z)$ is stationary distribution} \\\
&\dArr \\\
&\text{requires detailed balance: } \pi(z=S_i)P_{ij} = \pi(z=S_j)P_{ji} \\\
&\dArr \\\
&\text{find satisfied transition matrix $P$}
\end{align*}
$$

{{< math.inline >}}
<p>
Suppose we have a unsatisfied transition matrix \(Q_{ij}\), we can multiply \(Q\) by an acceptance factor \(\alpha(\cdot)\) for balancing the equation.
</p>
{{</ math.inline >}}

$$
\begin{align*}
\text{Let } p_z(S_i)&=\pi(z=S_i) \\\
p_z(S_i) Q_{ij} &\neq p_z(S_j)Q_{ji} \\\
p_z(S_i) Q_{ij}\alpha_{ij} &= p_z(S_j)Q_{ji}\alpha_{ji}
\end{align*}
$$

$$
\begin{align*}
\text{Let }\alpha_{ij} &= \min (1, \frac{p_z(S_j)Q_{ji}}{p_z(S_i)Q_{ij}}), \text{we have:} \\\
p_z(S_i) Q_{ij}\alpha_{ij} &= p_z(S_i) Q_{ij}\min (1, \frac{p_z(S_j)Q_{ji}}{p_z(S_i)Q_{ij}}) \\\
&= \min( p_z(S_i) Q_{ij}, p_z(S_j) Q_{ji} ) \\\
&= p_z(S_j) Q_{ji}\min (\frac{p_z(S_i)Q_{ij}}{p_z(S_j)Q_{ji}}, 1) \\\
&= p_z(S_j) Q_{ji} \alpha_{ji} \\\
\therefore P_{ij}&=Q_{ij}\alpha_{ij} \text{ is the satisfied transition matrix}
\end{align*}
$$

{{< math.inline >}}
<p>
Then we can write down the whole <b>Metropolis-Hastings algorithm</b>:
</p>
{{</ math.inline >}}

$$
\begin{align}
&\text{Sample }u_1,\cdots,u_N\sim \mathcal{U}(0,1) \\\
&\text{For $t=1\cdots N$} \\\
&\hspace{1em}z_{\ast} \sim Q_{z_{t-1}z_{\ast}} \space,\space \text{sampled candidate state for $z_t$} \\\
&\hspace{1em}\alpha_{z_{t-1}z_{\ast}} = \min (1, \frac{p({z_{\ast}})Q_{z_{\ast}z_{t-1}}}{p({z_{t-1}})Q_{z_{t-1}z_{\ast}}}) \\\
&\hspace{1em}\begin{cases}
\text{if } u_t \leq  \alpha_{z_{t-1}z_{\ast}} \space , \text{accept $z_t=z_{\ast}$} & \text{the higher probability of $p(z_{\ast})Q_{z_{\ast}z_{t-1}}$, the more likely $z_{\ast}$ is accepted}\\\
\text{else} \space, z_t=z_{t-1}
\end{cases} \\\
&\text{EndFor} \\\
&\text{Return $z_1,\cdots,z_N$ after N loops}
\end{align}
$$

The demonstration is as follows:

$$
\begin{align*}
&\because z_{\ast} \sim p(z|z_{t-1}), \space p(z=z_{\ast}|z_{t-1}) = p(z_{t-1})Q_{z_{t-1}z_{\ast}}\alpha_{z_{t-1}z_{\ast}}=Q_{z_{t-1}z_{\ast}}\alpha_{z_{t-1}z_{\ast}} \\\
&\therefore z_{\ast}\sim Q_{z_{t-1}z_{\ast}} \alpha_{z_{t-1}z_{\ast}} \\\
&\text{we want } Q_{z_{t-1}z_{\ast}} \approx Q_{z_{t-1}z_{\ast}}\alpha_{z_{t-1}z_{\ast}} \iff 1 \approx \alpha_{z_{t-1}z_{\ast}} \\\
&\dArr \\\
&\because \alpha_{z_{t-1}z_{\ast}} \in [0,1] \text{ and } \alpha_{z_{t-1}z_{\ast}}= 1 \text{ means acceptance} \\\
&\therefore \text{given }\forall z_{\ast},  \begin{cases}
\text{Pr}(z_{\ast}\text{ is accepted}) = \alpha_{z_{t-1}z_{\ast}} \\\
\text{Pr}(z_{\ast}\text{ is rejected}) = 1-\alpha_{z_{t-1}z_{\ast}}
\end{cases}
\end{align*}
$$

{{< math.inline >}}
<p>
Notice that we sampled from \(p(z)\)'s likelihood which is propotional to \(p(z)\):
</p>
{{</ math.inline >}}

$$
\begin{align*}
\alpha_{z_{t-1}z_{\ast}} &= \min (1, \frac{p({z_{\ast}})Q_{z_{\ast}z_{t-1}}}{p({z_{t-1}})Q_{z_{t-1}z_{\ast}}}) \\\
&= \min (1, \frac{ \frac{\hat{p}({z_{\ast}})}{\int\hat{p}({z_{\ast}})dz} Q_{z_{\ast}z_{t-1}}}{ \frac{\hat{p}({z_{t-1}})}{\int\hat{p}({z_{t-1}})dz} Q_{z_{t-1}z_{\ast}}}) \\\
&= \min (1, \frac{\hat{p}({z_{\ast}})Q_{z_{\ast}z_{t-1}}}{\hat{p}({z_{t-1}})Q_{z_{t-1}z_{\ast}}}) \\\
&\dArr \\\
z_{\ast}\sim p(z) &\equiv z_{\ast}\sim \hat{p}(z)
\end{align*}
$$

## Gibbs algorithm

{{< math.inline >}}
<p>
Gibbs algorithm is a special Metropolis-Hastings algorithm that considers \(p(z)\) the joint probability of different dimensions \( p(z)=p(z_1,\cdots,z_p) \), and we sample each dimension by fixing other dimensions, here are some prerequisite notations:
</p>
{{</ math.inline >}}

$$
\begin{align*}
z_{ij} &: \text{i-th sample, j-th dimension of $z$} \\\
i&=1,\cdots,N \\\
j&=1,\cdots,p \\\
z_{i\xcancel{j}} &: \text{i-th sample, dimensions of $z$ without j-th} \\\
z_{i<j} &: \text{i-th sample, dimensions of $z$ lower than j-th} \\\
z_{i>j} &: \text{i-th sample, dimensions of $z$ higher than j-th}
% p(z_i) &= \prod_{j=1}^p p(z_{ij})
\end{align*}
$$

{{< math.inline >}}
<p>
<b>Gibbs algorithm</b>:
</p>
{{</ math.inline >}}

$$
\begin{align*}
&\text{Sample }u_1,\cdots,u_N\sim \mathcal{U}(0,1) \\\
&\text{Initialize }z_1=(z_{11},z_{12},\cdots,z_{1p}) \\\
&\text{For $t=2\cdots N$} \\\
    &\hspace{1em}\text{For $j=1\cdots p$} \\\
        &\hspace{2em}z_{tj}\sim p(z|z_{t-1 >j},z_{t<j})  \\\
        &\hspace{2em}\text{accept $z_{ij}$} \\\
    &\hspace{1em}\text{EndFor} \\\
    &\hspace{1em}z_t = (z_{t1},z_{t2},\cdots,z_{tp})  \\\
&\text{EndFor} \\\
&\text{Return $z_1,\cdots,z_N$}
\end{align*}
$$

{{< math.inline >}}
<p>
Next, we prove the accept ratio is always 1 in Gibbs algorithm:
</p>
{{</ math.inline >}}

$$
\begin{align*}
\alpha &= \frac{ p(z_{tj}|z_{t\xcancel{j}})p(z_{t\xcancel{j}})p(z_{t-1}|z_{t}) }{ p(z_{t-1j}|z_{t-1\xcancel{j}})p(z_{t-1\xcancel{j}})p(z_t|z_{t-1}) } \\\
&\because \xcancel{j}\text{ is fixed, transition probability becomes transition probability to specific dimension $j$} \\\
&\therefore p(z_{t-1}|z_{t})=p(z_{t-1j}|z_t)=p(z_{t-1j}|z_{t\xcancel{j}}) \\\
&= \frac{ p(z_{tj}|z_{t\xcancel{j}})p(z_{t\xcancel{j}})p(z_{t-1j}|z_{t\xcancel{j}}) }{ p(z_{t-1j}|z_{t-1\xcancel{j}})p(z_{t-1\xcancel{j}})p(z_{tj}|z_{t-1\xcancel{j}}) } \\\
&\because p(z_{t\xcancel{j}}) = p(z_{t-1\xcancel{j}}) \text{ when fix $\xcancel{j}$} \\\
&= \frac{ p(z_{tj}|z_{t\xcancel{j}})p(z_{t-1j}|z_{t\xcancel{j}}) }{ p(z_{t-1j}|z_{t-1\xcancel{j}})p(z_{tj}|z_{t-1\xcancel{j}}) } \\\
&= \frac{ p(z_{tj}|z_{t\xcancel{j}})p(z_{t-1j}|z_{t\xcancel{j}}) }{ p(z_{t-1j}|z_{t-1\xcancel{j}})p(z_{tj}|z_{t\xcancel{j}}) } \\\
&= \frac{ p(z_{t-1j}|z_{t\xcancel{j}}) }{ p(z_{t-1j}|z_{t-1\xcancel{j}}) } \\\
&= \frac{ p(z_{t-1j}|z_{t\xcancel{j}}) }{ p(z_{t-1j}|z_{t\xcancel{j}}) } \\\
&= 1
\end{align*}
$$

## Supplementation

Some common transformation in reparamerterization trick:

![homeinfo](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAmoAAAHLCAIAAAD3GCP+AAAAA3NCSVQICAjb4U/gAAAAEHRFWHRTb2Z0d2FyZQBTaHV0dGVyY4LQCQAAIABJREFUeNrsnX9cE1fW/48a681jWiYrXSdbuibVron6LIl1C0H7NcZ1hbitgI9VQltDUwUaLdLYAoUqWlSwKGJB0IKi+4CiS8B2QawLxrXyY1v58diSuCqJqzbjis1QcTMu0Xz/4Ff4KShaxPt+8YdmJjfJPeeez7ln7twZ4XA4AIPBYDAYzEAYibsAg8FgMBgsnxgMBoPBYPnEYDAYDAbLJwaDwWAwWD4xGAwGg8HyicFgMBgMBssnBoPBYDBYPjEYDAaDwfKJwWAwGAyWTwwGg8FgsHxiMBgMBoPpGVYfxyiKwh2EwWAwmCcWkiR7OzQCbxmPwWAwGMxAwcVbDAaDwWCwfGIwGAwGg+UTg8FgMBgsnxgMBoPBYPnEYDAYDAbLJwaDwWAwmIcln8zhQPaI7rAX/Y+STQCAQCAQinwQCgUAgEPkkEAgEAmECMlyNMcuy5AURCAQC4YGFpofcxWq4zFsCgUAgEAiDQoK3BAKBQCAQ+SQQCAQCgcgngUAgEAhEPgkEAoFAIPJJIBAIBAKRTwKBQCAQCEQ+CQQCgUAg8kkgEAgEApFPAoFAIBCIfBIIBAKB8KDw/wFJuim74GF+VgAAAABJRU5ErkJggg==
)

## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=69).
[^3]: From [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
[^2]: From [Mean field variational inference](https://mbernste.github.io/files/notes/MeanFieldVariationalInference.pdf).
[^4]: From [Ross, Sheldon M. (2019). Introduction to probability models](https://doi.org/10.1016%2FC2017-0-01324-1).
[^5]: From [Ross, Sheldon M. (2019). Introduction to probability models](https://www.cnblogs.com/RyanXing/p/M-H.html).