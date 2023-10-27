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
&\text{construct markov chain, with stationary dsitribution } \pi(z) \\\
&\text{$z_i$ sampled by: } z_1 \sim \pi(z_1),z_2\sim \pi(z_2|z_1),\cdots,z_N\sim \pi(z_N|z_{N-1}) \\\
&\dArr \\\
&\text{requires $\pi(z)$ is stationary distribution} \\\
&\dArr \\\
&\text{requires detailed balance: } \pi(z=S_i)P_{ij} = \pi(z=S_j)P_{ji} \\\
&\dArr \\\
&\text{find satisfied transition matrix $P$ so that $\pi(z)=p(z)$}
\end{align*}
$$

{{< math.inline >}}
<p>
Suppose we have a unsatisfied transition matrix \(Q_{ij}\), we can multiply \(Q\) by an acceptance ratio \(\alpha(\cdot)\) for balancing the equation.
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
&\hspace{1em}z_{\ast} \sim Q_{z_{t-1}z_{\ast}} \space,\space \text{candidate state sampled from \colorbox{brown}{\color{white}{simple distribution}} } Q \\\
&\hspace{1em}\alpha_{z_{t-1}z_{\ast}} = \min (1, \frac{p({z_{\ast}})Q_{z_{\ast}z_{t-1}}}{p({z_{t-1}})Q_{z_{t-1}z_{\ast}}}), Q \text{ is also the state transition distribution} \\\ \\\
&\hspace{1em}\begin{cases}
\text{if } u_t \leq  \alpha_{z_{t-1}z_{\ast}} \space , \text{accept $z_t=z_{\ast}$} & \text{the larger $p(z_{\ast})Q_{z_{\ast}z_{t-1}}$, the more likely $z_{\ast}$ is accepted}\\\
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
&\text{we want } Q_{z_{t-1}z_{\ast}} \approx P_{z_{t-1}z_{\ast}} \approx Q_{z_{t-1}z_{\ast}}\alpha_{z_{t-1}z_{\ast}} \iff 1 \approx \alpha_{z_{t-1}z_{\ast}} \\\
&\dArr \\\
&\text{ $\alpha_{z_{t-1}z_{\ast}}$ is the likeliness of $Q_{z_{t-1}z_{\ast}} = P_{z_{t-1}z_{\ast}}$ } \\\
&\text{ $\alpha_{z_{t-1}z_{\ast}}$ is the probability of $z_{\ast}$ being accepted by rejection sampling thought } \\\
% &\because \alpha_{z_{t-1}z_{\ast}} \in [0,1] \text{ and } \alpha_{z_{t-1}z_{\ast}}= 1 \text{ means acceptance} \\\
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

### Experiment

```python {linenos=true}
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def target_distribution(x):
    # Define your target distribution here
    return 2.5*np.exp(-2.5 * ((x - 0) ))

def proposal_distribution(x,mu):
    # Define your proposal distribution here (e.g., normal distribution)
    return (1/np.sqrt(2*np.pi))*np.exp( -((x-mu)**2) / 2 )

def metropolis_hastings(num_samples, initial_state):
    samples = [initial_state]
    current_state = initial_state

    for _ in tqdm.tqdm(range(num_samples)):
        # Propose a new state from the proposal distribution
        values = np.linspace(-0, 50, 1000)
        probs = proposal_distribution(values, current_state)
        probs /= np.sum(probs)
        proposed_state = np.random.choice(values, size=1, p=probs)[0]

        # Calculate the acceptance ratio
        acceptance_ratio =min(1, (
            (target_distribution(proposed_state)*proposal_distribution(current_state, proposed_state)) / (target_distribution(current_state)*proposal_distribution(proposed_state, current_state))
        ))
        print(f'proposed_state: {proposed_state}, current_state: {current_state}')
        # Accept or reject the proposed state based on the acceptance ratio
        if np.random.uniform() < acceptance_ratio:
            current_state = proposed_state

        samples.append(current_state)

    return samples

# Parameters
num_samples = 10000
initial_state = 0.0

# Run Metropolis-Hastings
samples = metropolis_hastings(num_samples, initial_state)

# Plot the samples
plt.hist(samples, bins=50, density=True, alpha=0.7, label='Sampled Distribution')
x = np.linspace(0, 20, 1000)
plt.plot(x, target_distribution(x), 'r-', label='Target Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.show()
```

![homeinfo](data:image/png;base64,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17VkyRKtWbNGCQkJGjJkiFavXq3Vq1frvffe04IFC/TJJ58Ume+VV15RZGSkdu7cqaeffloTJ05UXFxciXXk5uaqX79+8vX11caNG5WYmCgfHx/1799fOTk5kqRZs2YpNjZW7777rr755hudPn1ay5Ytq9TPp3bt2nrkkUeUmJio48ePF3s/NTVV999/vx566CHt27dPCQkJGjp0qAzD0JQpUzRs2DD1799fqampSk1NVffu3a3zFq7rvn371K9fvxI//8CBA/roo4/0+eefa82aNdq5c6cee+yxcte/dOlSNW3aVM8//7y1hpIsW7ZMEydO1OOPP669e/fq4Ycf1ujRo7Vu3boi42bMmKFhw4bp+++/18CBAzVixAidPn263PXgd06xC/hK+fn5mjRpknr06KEbb7yx1HERERF699131b59e6Wnp+vVV19V9+7d9cMPP6hp06YlzjNz5kzNqIlQdvq0lJQkHT1a/Z8FACaqVauWYmNjNW7cOM2fP1+dOnVSz549dd9996l9+/bWcVd2mcLCwvT3v/9djzzyiN58803r9NzcXGsXSZLuuecevffeezp27Jh8fHx0/fXX65ZbbtG6des0fPhw63w9evTQ008/LUlq3bq1EhMT9dprr6lPnz7F6v3www+Vn5+vt99+29rNWrhwoQICApSQkKC+fftqzpw5mjp1qoYOHSpJmj9/vr788stK/4zatGkjSfr111/VqFGjIu+lpqbq0qVLGjp0qJo3by5J1l3FUkGAzM7OVnBwcLHlTpo0yVpjaS5evKjFixerSZMmkqQ33nhDt99+u2bNmlXiMq9Wv359ubu7y9fXt8zxr776qqKjo63hMiYmRt9++61effVV67GRUkH38/7775ck/fOf/9Trr7+uLVu2qH///tesBUU5XQdw/Pjx2rt3r5YsWVLmuKioKI0cOVIdOnRQz549tXTpUgUGBmrBggWlzjN16lSlp6dbH4cPH7Z1+QW8vAr+LeP/hgHAWdx99906evSoPvvsM/Xv318JCQnq1KmTYmNjrWPWrl2r2267TU2aNJGvr68efPBBnTp1SufPn7eOqVOnjjX8SVJQUJDCwsLk4+NTZNrVnbSoqKhir/ft21dirbt379aBAwfk6+tr7V7Wr19fFy9e1MGDB5Wenq7U1FR169bNOk+tWrV00003VepnI8m6+7ik49QjIyN12223qV27drr33nv1n//8R2fOnCnXcstTU7NmzazhTyr42eTn5xc5ft4W9u3bV+yEnx49ehTbDlf+T0HdunXl5+dXYmcU1+ZUAXDChAlauXKl1q1bV2oXrzQeHh7q2LGjDhw4UOoYLy8v+fn5FXlUCwIgABfj7e2tPn366Nlnn9WmTZsUHR2t6dOnSyrofN1xxx1q3769Pv30U23fvt16nF7hblep4L/jV7JYLCVOy8/Pr3SdmZmZ6ty5s3bt2lXk8fPPP+uBBx6o9HLLUhiCwsLCir3n7u6uuLg4ffHFF7r++uv1xhtvKCIiQocOHbrmcss6SbK83Nzcih3fmJubW+XllsbW29OVOUUANAxDEyZM0LJly/T1118rPDy8wsvIy8vTnj17ipxqbxoCIAAXd/311ysrK0uStH37duXn52vWrFm6+eab1bp1ax214SEy3377bbHXbdu2LXFsp06dtH//fjVq1EgtW7Ys8vD395e/v79CQkL03XffWee5dOnSNU9KLM2FCxf01ltv6U9/+pMCAwNLHGOxWNSjRw/NmDFDO3fulKenp/WYQ09PzyInvVRUSkpKkZ/1t99+Kzc3N+tl1gIDA4sc15eXl6e9e/cWWUZ5amjbtq0SExOLTEtMTNT1119f6dpRNqcIgOPHj9f777+vDz74QL6+vkpLS1NaWpouXLhgHTNy5EhNnTrV+vr555/XV199pV9++UU7duzQ//3f/yk5OVljx441YxWKKrxnIQEQgJM7deqUbr31Vr3//vv6/vvvdejQIX388cd6+eWXNWjQIElSy5YtlZubqzfeeEO//PKL3nvvPc2fP99mNSQmJurll1/Wzz//rLlz5+rjjz/WxIkTSxw7YsQINWzYUIMGDdLGjRt16NAhJSQk6K9//at+++03SdLEiRP14osvavny5frpp5/02GOP6ezZs+Wq5fjx40pLS9P+/fu1ZMkS9ejRQydPntS8efNKHP/dd9/pn//8p7Zt26aUlBQtXbpUJ06csAbYsLAwff/990pKStLJkycr3J3z9vbWqFGjtHv3bm3cuFF//etfNWzYMOvxfLfeeqtWrVqlVatW6aefftKjjz5abF3DwsK0YcMGHTlyRCdPnizxc5544gnFxsZq3rx52r9/v2bPnq2lS5dqypQpFaoX5ecUJ4EUfjF69epVZPrChQutp6KnpKQUuS/gmTNnNG7cOKWlpalevXrq3LmzNm3aZB//t0EHEICL8PHxUbdu3fTaa6/p4MGDys3NVWhoqMaNG6dnnnlGUsFxbrNnz9ZLL72kqVOn6k9/+pNmzpypkSNH2qSGxx9/XNu2bdOMGTPk5+en2bNnl3pWbJ06dbRhwwY99dRTGjp0qM6dO6cmTZrotttusx4W9Pjjjys1NVWjRo2Sm5ubHnroIQ0ZMqTI5VNKExERIYvFIh8fH7Vo0UJ9+/ZVTExMqSdQ+Pn5acOGDZozZ44yMjLUvHlzzZo1SwMGDJAkjRs3TgkJCbrpppuUmZmpdevWlbgruTQtW7bU0KFDNXDgQJ0+fVp33HFHkRNvHnroIe3evVsjR45UrVq1NHny5CInbUgFDZeHH35Y1113nbKzs0u8JM7gwYP1r3/9S6+++qomTpyo8PBwLVy4sNjfddiOxSjvxYlQTEZGhvz9/ZWenm7b4wE//VS65x7pD3+QNm603XIBOLWLFy/q0KFDCg8Pl3fhngSUqaS7VMA5lPV9qLa/3w7EKXYBOx06gAAAoBoRAO0RARAAAFQjpzgG0OkQAAGgRlx5CzfAldABtEcEQAAAUI0IgPaIAAgAAKoRAdAeEQABAEA1IgDaIwIgAACoRgRAe1QYAC9eNLcOAADglAiA9qgwAOblFTwAAABsiABojwoDoMRuYABwErGxsQoICLD5cn/99VdZLBbt2rVLkpSQkCCLxVLu+w9X5bPguAiA9ogACMBFWCyWMh/PPfecqbUtX768XOMKH3Xr1lWrVq0UHR2t7du3Fxk3fPhw/fzzz+X67IqExdDQUKWmpurGG28s1/jyio6O1uDBg2vks1DzCID2yMPj9+cEQABOLDU11fqYM2eO/Pz8ikybMmVKhZaXk5NTTZWWbeHChUpNTdUPP/yguXPnKjMzU926ddPixYutY2rXrq1GjRrZ9HNzcnLk7u6u4OBg1apV/fd2qMnPQvUiANoji4UzgQG4hODgYOvD399fFovF+jorK0sjRoxQUFCQfHx81KVLF61du7bI/GFhYXrhhRc0cuRI+fn56c9//rMk6T//+Y9CQ0NVp04dDRkyRLNnzy7WUVuxYoU6deokb29vtWjRQjNmzNClS5esy5WkIUOGyGKxWF+XJiAgQMHBwQoLC1Pfvn31ySefaMSIEZowYYLOnDkjqXhXb/fu3brlllvk6+srPz8/de7cWdu2bVNCQoJGjx6t9PT0Yp3Qkta3tN2yiYmJat++vby9vXXzzTdr79691veee+45dejQocj4OXPmWNfzueee06JFi7RixQprDQkJCSV+1vr169W1a1d5eXkpJCRETz/9tPXnKEm9evXSX//6Vz355JOqX7++goODTe3sogAB0F55exf8SwAEUFmGIWVlmfMwjCqXn5mZqYEDByo+Pl47d+5U//79deeddyolJaXIuFdffVWRkZHauXOnnn32WSUmJuqRRx7RxIkTtWvXLvXp00f/+Mc/isyzceNGjRw5UhMnTtSPP/6oBQsWKDY21jpu69atkn7v7BW+rojJkyfr3LlziouLK/H9ESNGqGnTptq6dau2b9+up59+Wh4eHurevXuxbuiVndCr17c0TzzxhGbNmqWtW7cqMDBQd955p3Jzc8tV+5QpUzRs2DD179/fWkP37t2LjTty5IgGDhyoLl26aPfu3Zo3b57eeecd/f3vfy8ybtGiRapbt66+++47vfzyy3r++edL/bmgZtDDtVd0AAFU1fnzko+POZ+dmSnVrVulRURGRioyMtL6+oUXXtCyZcv02WefacKECdbpt956qx5//HHr6//3//6fBgwYYA1NrVu31qZNm7Ry5UrrmBkzZujpp5/WqFGjJEktWrTQCy+8oCeffFLTp09XYGCgpN87e5XRpk0bSaXfbzglJUVPPPGEdVyrVq2s713ZDb3a1etb2vKnT5+uPn36SCoIYE2bNtWyZcs0bNiwa9bu4+Oj2rVrKzs7u8z1f/PNNxUaGqp///vfslgsatOmjY4ePaqnnnpK06ZNk5tbQZ+pffv2mj59unU9//3vfys+Pt5aH2oeHUB7RQAE4OIyMzM1ZcoUtW3bVgEBAfLx8dG+ffuKdQBvuummIq+TkpLUtWvXItOufr179249//zz8vHxsT7GjRun1NRUnT9/3ib1G5e7oBaLpcT3Y2JiNHbsWPXu3VsvvviiDh48WK7lXr2+pYmKirI+r1+/viIiIrRv375yzVte+/btU1RUVJF17NGjhzIzM/Xbb79Zp7Vv377IfCEhITp+/LhNa0HF0AG0VwRAAFVVp05BJ86sz66iKVOmKC4uTq+++qpatmyp2rVr65577il2okfdSnQaMzMzNWPGDA0dOrTYe96Fh+BU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HVk/VIXTXAcQAAAA5eMUxwACAACg/AiAAAAALoYACAAA4GIIgAAAAC6GAGiSuXPnKiwsTN7e3urWrZu2bNlS5viPP/5Ybdq0kbe3t9q1a6fVq1fXUKUVN3PmTHXp0kW+vr5q1KiRBg8erKSkpDLniY2NlcViKfLw9vauoYor5rnnnitWa5s2bcqcx5G2nySFhYUVW0eLxaLx48eXON7et9+GDRt05513qnHjxrJYLMXu+W0YhqZNm6aQkBDVrl1bvXv31v79+6+53Ip+j6tTWeuYm5urp556Su3atVPdunXVuHFjjRw5UkePln1v7sr8rleXa23D6OjoYrX279//msu1l214rfUr6ftosVj0yiuvlLpMe9p+5fm7cPHiRY0fP14NGjSQj4+P7r777mI3bLhaZb+7IACa4sMPP1RMTIymT5+uHTt2KDIyUv369dPx48dLHL9p0ybdf//9GjNmjHbu3KnBgwdr8ODB2rt3bw1XXj7r16/X+PHj9e233youLk65ubnq27evsrKyypzPz89Pqamp1kdycnINVVxxN9xwQ5Fav/nmm1LHOtr2k6StW7cWWb+4uDhJ0r333lvqPPa8/bKyshQZGam5c+eW+P7LL7+s119/XfPnz9d3332nunXrql+/frp48WKpy6zo97i6lbWO58+f144dO/Tss89qx44dWrp0qZKSknTXXXddc7kV+V2vTtfahpLUv3//IrX+73//K3OZ9rQNr7V+V65Xamqq3n33XVksFt19991lLtdetl95/i5MnjxZn3/+uT7++GOtX79eR48e1dChQ8tcbmW+u7jMQI3r2rWrMX78eOvrvLw8o3HjxsbMmTNLHD9s2DDj9ttvLzKtW7duxsMPP1ytddrK8ePHDUnG+vXrSx2zcOFCw9/fv+aKqoLp06cbkZGR5R7v6NvPMAxj4sSJxnXXXWfk5+eX+L4jbT9JxrJly6yv8/PzjeDgYOOVV16xTjt79qzh5eVl/O9//yt1ORX9Htekq9exJFu2bDEkGcnJyaWOqejvek0paf1GjRplDBo0qELLsddtWJ7tN2jQIOPWW28tc4y9bj/DKP534ezZs4aHh4fx8ccfW8fs27fPkGRs3ry5xGVU9ruLAnQAa1hOTo62b9+u3r17W6e5ubmpd+/e2rx5c4nzbN68uch4SerXr1+p4+1Nenq6JKl+/fpl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
)

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
<b>Gibbs algorithm</b> solves joint distribution by known conditional distribution:
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
\text{like M-H algorithm, conditional distribution $p(z_i|z_j)$ is the same as state transition distribution $Q$} \\\
\dArr \\\
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



## Reference

[^1]: From [video](https://www.bilibili.com/video/BV1aE411o7qd?p=69).
[^3]: From [The Matrix Cookbook](https://www.math.uwaterloo.ca/~hwolkowi/matrixcookbook.pdf).
[^2]: From [Mean field variational inference](https://mbernste.github.io/files/notes/MeanFieldVariationalInference.pdf).
[^4]: From [Ross, Sheldon M. (2019). Introduction to probability models](https://doi.org/10.1016%2FC2017-0-01324-1).
[^5]: From [Ross, Sheldon M. (2019). Introduction to probability models](https://www.cnblogs.com/RyanXing/p/M-H.html).