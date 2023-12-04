---
author: "X. Wang"
title: "Particle Filter"
date: "2023-09-15"
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

## Background

### Overview of PGM

<cite>[^1]</cite>

$$
\text{PGM}: \begin{cases}
    \text{Representation} \begin{cases}
        \text{directed graph}\rarr  \text{Bayesian network} \\\
        \text{undirected graph}\rarr \text{Markov network(MRF)} \\\
        \text{continous variable}\rarr \text{Gaussian BN/Gaussian MRF} \\\
        \text{time$\rarr$} \underset{\text{$x_i$ not i.i.d.}}{\text{ Dynamic model}} \begin{cases}
            \text{discrete state$\rarr$Hidden Markov Model} \\\
            \text{continous state} \begin{cases}
                \text{Linear model$\rarr$Karman Filter} \\\
                \text{Nonlinear model$\rarr$\color{red}{Particle Filter}}
            \end{cases}
        \end{cases}
    \end{cases} \\\
    \text{Inference} \begin{cases}
        \text{MAP inference$\rarr \hat{x_A}=\argmax_{x_A}p(x_A|x_B)\propto\argmax p(x_A,x_B)$} \\\
        \text{exact inference} \begin{cases}
          \text{Variable elimination(VE)} \\\
          \text{Belief propagation(BP)$\rarr$sum-product algorithm(Tree)} \\\
          \text{Junction tree algorithm(Normal graph)}
        \end{cases} \\\
        \text{approximate inference} \begin{cases}
            \text{Loop belief propagation(Cyclic graph)} \\\
            \text{Variational inference} \\\
            \text{MCMC: importance sampling}
        \end{cases} 
    \end{cases} \\\
    \text{Learning} \begin{cases}
        \text{parameter learning} \begin{cases}
            \text{complete data: $(x,z)$} \\\
            \text{hidden variable: $z$}
        \end{cases} \\\
        \text{structure learning}
    \end{cases}
\end{cases}
$$


## Particle Filter

### Definition

<div class="graph" style="text-align: center;">

```mermaid
%%{
  init: {
    'theme': 'base',
    'themeVariables': {
      'primaryColor': 'white',
      'primaryTextColor': '#000',
      'primaryBorderColor': '#7C0200',
      'lineColor': '#F8B229',
      'secondaryColor': 'red',
      'tertiaryColor': '#fff'
    }
  }
}%%
flowchart LR
    id1((State z_1)) --> id2((State z_2))
    id2((State z_2)) --> id3((State z_i))
    id3((State z_i)) --> id9(((...)))
    id3((State z_i)) --> id4((State z_t))
    id4((State z_t)) --> id5((State z_{t+1}))
    id1((State z_1)) --> id6(((Observation x_1)))
    id2((State z_2)) --> id7(((Observation x_2)))
    id4((State z_t)) --> id8(((Observation x_t)))
```

</div>

Particle filter, like the Kalman filter, is a state-space model. However, it is designed to handle non-linear and non-Gaussian scenarios:

$$
\begin{cases}
z_t = g(z_{t-1}, u, \epsilon) \\\
x_t = h(z_{t}, u, \delta)
\end{cases}
$$

Then we have unknown parameters:

$$
\theta = (  )
$$

It has 2 steps for filtering:

$$
\begin{cases}
\text{Prediction: $p(z_t|x_1,\cdots,x_{t-1})=\int_{z_{t-1}}p(z_t,z_{t-1}|x_1,\cdots,x_{t-1})dz_{t-1}$} \\\
\text{Update: $p(z_t|x_1,\cdots,x_{t})\propto p(x_t|z_t)p(z_t|x_1,\cdots,x_{t-1})$}
\end{cases}
$$

{{< math.inline >}}
<p>
Since \( h(\cdot) \) and \( g(\cdot) \) are not linear, \( \epsilon \) and \( \delta \) are not gaussian noise, we can't find a analytic expression for filtering problem like Kalman filter, sampling method is used on the other hand.
</p>
{{</ math.inline >}}

### Review Monte Carlo method

$$
\begin{align*}
p(z|x) \rarr E_{z|x\sim p(z|x)}\left[ f(z) \right] &= \int_{z}f(z)p(z|x)dz \\\
&\approx \frac{1}{N} \sum_{i=1}^N f(z^{(i)}) \\\
z^{(1)},\cdots,z^{(N)}&\text{ are randomly sampled from } p(z|x)
\end{align*}
$$

Monte Carlo method resolves difficulty in integration part, then the problem comes to how to do sampling.

#### Review importance sampling

{{< math.inline >}}
<p>
Since posterior distribution is not easy to solve we want to find a close distribution \( q(z) \), and sampling on \( q(z) \):
</p>
{{</ math.inline >}}

$$
\begin{align*}
E_{z|x\sim p(z|x)}\left[ f(z) \right] &= \int_{z}f(z)p(z|x)dz \\\
&= \int_{z}f(z)\frac{p(z|x)}{q(z)}q(z)dz \\\
&\approx \frac{1}{N} \sum_{i=1}^N f(z^{(i)})\frac{p(z^{(i)}|x)}{q(z^{(i)})} \\\
&\frac{p(z^{(i)}|x)}{q(z^{(i)})}\text{ is weight $w^{(i)}$} \\\
z^{(1)},\cdots,z^{(N)}&\text{ are randomly sampled from } q(z)
\end{align*}
$$

### Apply importance sampling to filtering problem

Define weight at time t as:

$$
w_t^{(i)} = \frac{p(z_t^{(i)}|x_{1},\cdots,x_t)}{q(z_t^{(i)}|x_{1},\cdots,x_t)} \\\
i=1\cdots N, \space t=1\cdots T
$$

{{< math.inline >}}
<p>
Intead of computing weights \( N\times T \) times, we can find a recurrence relation between \( w_t^{(i)} \) and \( w_{t-1}^{(i)} \) which brings us to sequential importance sampling(SIS).
</p>
{{</ math.inline >}}

### Sequential importance sampling

{{< math.inline >}}
<p>
Denote \( x_{1},\cdots,x_t \) as \( x_{1:t} \), sequential importance sampling(SIS) assumes that the weight is porpotional to the ratio of \( p(z_{1:t}|x_{1:t}) \) and \( q(z_{1:t}|x_{1:t}) \), so we have:
</p>
{{</ math.inline >}}

$$
\begin{align*}
w_t^{(i)} &= \frac{p(z_t^{(i)}|x_{1:t})}{q(z_t^{(i)}|x_{1:t})} \\\
&\propto \frac{p(z_{1:t}^{(i)}|x_{1:t})}{q(z_{1:t}^{(i)}|x_{1:t})}
\end{align*}
$$

{{< math.inline >}}
<p>
For \( p(z_{1:t}^{(i)}|x_{1:t}) \) we have:
</p>
{{</ math.inline >}}

$$
\begin{align*}
p(z_{1:t}^{(i)}|x_{1:t}) &= \frac{p(z_{1:t}^{(i)},x_{1:t})}{p(x_{1:t})} \\\
&\propto p(z_{1:t}^{(i)},x_{1:t})
\end{align*}
$$

## Conclusion

work in progress

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=92).
[^3]: From [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
[^5]: From [Mean field variational inference](https://mbernste.github.io/files/notes/MeanFieldVariationalInference.pdf).
[^4]: From [Ross, Sheldon M. (2019). Introduction to probability models](https://doi.org/10.1016%2FC2017-0-01324-1).
[^2]: - [Hammersley–Clifford theorem](http://www.statslab.cam.ac.uk/~grg/books/hammfest/hamm-cliff.pdf).
