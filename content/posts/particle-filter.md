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
    id3((State z_i)) --> id9(((...))) // Continue the sequence
    id3((State z_i)) --> id4((State z_t)) // Current state
    id4((State z_t)) --> id5((State z_{t+1})) // Next state
    id1((State z_1)) --> id6(((Observation x_1))) // Observation for state z_1
    id2((State z_2)) --> id7(((Observation x_2))) // Observation for state z_2
    id4((State z_t)) --> id8(((Observation x_t))) // Observation for current state
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



Then we can give the update order:

$$
\underset{update1}{p(z_1|x_1)} \rarr \underset{prediction2}{p(z_2|x_1)} \rarr \underset{update2}{p(z_2|x_1,x_2)} \rarr \underset{prediction3}{p(z_3|x_1,x_2)} \cdots \underset{predictionT}{p(z_{T}|x_1,\cdots,x_{T-1})} \rarr \underset{updateT}{p(z_{T}|x_1,\cdots,x_{T})}
$$

### Solving filtering problem

From previous section we know filtering problem can be written as 2 steps: update, prediction:

$$
\begin{cases}
\text{Prediction: $p(z_t|x_1,\cdots,x_{t-1})=\int_{z_{t-1}}p(z_t,z_{t-1}|x_1,\cdots,x_{t-1})dz_{t-1}$} \\\
\text{Update: $p(z_t|x_1,\cdots,x_{t})\propto p(x_t|z_t)p(z_t|x_1,\cdots,x_{t-1})$}
\end{cases}
$$

It is obvious that results from both steps are still gaussian distributions:

$$
p(z_t|x_1,\cdots,x_{t-1}) \sim \mathcal{N}({\mu_{t}}^{\ast},{\Sigma_{t}}^{\ast}) \\\
p(z_t|x_1,\cdots,x_{t}) \sim \mathcal{N}({\mu_{t}},{\Sigma_{t}})
$$

#### Lemma

Besides, we have conclusions for [conditional pdf](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/#solve-the-joint-pdf) and [marginal pdf](https://tirmisula.github.io/posts/marginal-and-joint-gaussian-probability/#solve-the-conditonal-pdf-1) of Gaussian from previous artice:

$$
x \sim \mathcal{N} (\mu, \Lambda^{-1}) \\\
y=Ax+B+\epsilon\\\
\epsilon \sim \mathcal{N}(0, L^{-1}), \epsilon \perp x \\\
\dArr \\\
\begin{cases}
y|x \sim \mathcal{N}(Ax+B, L^{-1}) \\\
y \sim \mathcal{N}(A\mu+B, A \Lambda^{-1} A^T + L^{-1}) \\\
\begin{bmatrix}
    x\\\
    y
\end{bmatrix} \sim \mathcal{N}(\begin{bmatrix}
    \mu \\\
    A\mu + B
\end{bmatrix}, \begin{bmatrix}
    \Lambda^{-1} & \Lambda^{-1} A^T\\\
    (\Lambda^{-1} A^T)^T & A \Lambda^{-1} A^T + L^{-1}
\end{bmatrix}) \\\
x|y \sim \mathcal{N}(\Sigma_{xy}\Sigma_{yy}^{-1} (x_y-\mu_{y}) + \mu_{x}, -\Sigma_{xy}\Sigma_{yy}^{-1}\Sigma_{yx}+\Sigma_{xx})
\end{cases}
$$

#### Prediction step
For the prediction part:

$$
\begin{align*}
p(z_t|x_1,\cdots,x_{t-1}) &= \int_{z_{t-1}}p(z_t,z_{t-1}|x_1,\cdots,x_{t-1})dz_{t-1} \\\
&= \int_{z_{t-1}} p(z_t|z_{t-1})p(z_{t-1}|x_1,\cdots,x_{t-1}) \\\
&\text{Consider $y=z_t, x=z_{t-1}$, Then $p(z_t)=p(y)$} \\\
p(y) &= \int_{x}p(y|x)p(x) \\\
\mathcal{N}({\mu_{t}}^{\ast},{\Sigma_{t}}^{\ast}) &= \int_{z_{t-1}} p(z_t|z_{t-1}) \mathcal{N}(\mu_{t-1},\Sigma_{t-1})
\end{align*}
$$

Following the [rules](#lemma) above we can write down:

$$
\begin{cases}
z_{t-1} \sim \mathcal{N}(\mu_{t-1}, \Sigma_t) \\\
z_t|z_{t-1} \sim \mathcal{N}(Az_{t-1}+B, Q) \\\
z_t \sim \mathcal{N}(A\mu_{t-1}+B, A\Sigma_{t-1}A^T+Q)
\end{cases}
$$

The predicting process is:

$$
\begin{cases}
{\mu_{t}}^{\ast} = A\mu_{t-1}+B \\\
{\Sigma_{t}}^{\ast} = A\Sigma_{t-1}A^T+Q
\end{cases}
$$

#### Update step
For the update part:

$$
\begin{align*}
p(z_t|x_1,\cdots,x_{t}) &\propto p(x_t|z_t)p(z_t|x_1,\cdots,x_{t-1}) \\\
\mathcal{N}(\mu_t, \Sigma_t) &\propto \mathcal{N}(x_t|Cz_{t}+D, R) \mathcal{N}({\mu_{t}}^{\ast},{\Sigma_{t}}^{\ast}) \\\
&\text{Consider $x=z_t, y=x_{t}$, Then $p(z_t|x_t)=p(x|y)$} \\\
p(x|y) &\propto p(y|x)p(x)
\end{align*}
$$

Following the [rules](#lemma) above we can write down:

$$
\begin{cases}
z_t \sim \mathcal{N}({\mu_{t}}^{\ast},{\Sigma_{t}}^{\ast}) \\\
x_t|z_t \sim \mathcal{N}(Cz_{t}+D, R) \\\
x_t \sim \mathcal{N}(C{\mu_{t}}^{\ast}+D, C{\Sigma_{t}}^{\ast}C^T+R) \\\
\begin{bmatrix}
    z_t\\\
    x_t
\end{bmatrix} \sim \mathcal{N}(\begin{bmatrix}
    {\mu_{t}}^{\ast} \\\
    C{\mu_{t}}^{\ast}+D
\end{bmatrix}, \begin{bmatrix}
    {\Sigma_{t}}^{\ast} & {\Sigma_{t}}^{\ast}C^T \\\
    ({\Sigma_{t}}^{\ast}C^T)^T & C{\Sigma_{t}}^{\ast}C^T+R
\end{bmatrix}) \\\
z_t|x_t \sim \mathcal{N}(\Sigma_{z_tx_t}\Sigma_{x_tx_t}^{-1} (x_t-\mu_{x_t}) + \mu_{z_t}, -\Sigma_{z_tx_t}\Sigma_{x_tx_t}^{-1}\Sigma_{x_tz_t}+\Sigma_{z_tz_t})
\end{cases}
$$

The updating process is:

$$
\begin{cases}
{\mu_{t}} = {\Sigma_{t}}^{\ast}C^T (C{\Sigma_{t}}^{\ast}C^T+R)^{-1} (x_t-C{\mu_{t}}^{\ast}-D)+{\mu_{t}}^{\ast} \\\
{\Sigma_{t}} = -{\Sigma_{t}}^{\ast}C^T (C{\Sigma_{t}}^{\ast}C^T+R)^{-1} ({\Sigma_{t}}^{\ast}C^T)^T + {\Sigma_{t}}^{\ast}
\end{cases}
$$

## Conclusion

work in progress

## Reference

[^1]: - [video](https://www.bilibili.com/video/BV1aE411o7qd?p=92).
[^3]: From [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
[^5]: From [Mean field variational inference](https://mbernste.github.io/files/notes/MeanFieldVariationalInference.pdf).
[^4]: From [Ross, Sheldon M. (2019). Introduction to probability models](https://doi.org/10.1016%2FC2017-0-01324-1).
[^2]: - [Hammersley–Clifford theorem](http://www.statslab.cam.ac.uk/~grg/books/hammfest/hamm-cliff.pdf).
