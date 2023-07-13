---
author: "Xuan Wang"
title: " Gaussian MLE Guide"
date: "2023-07-13"
description: "A simple deduction of Gaussian MLE."
tags: ["machine learning"]
categories: ["themes", "syntax"]
# series: ["Themes Guide"]
aliases: ["migrate-from-jekyl"]
math: true
ShowBreadCrumbs: false
ShowToc: true
TocOpen: true
---

To simplify the calculations, we take the logarithm of the likelihood function. This is a common practice
Certainly! Here's a detailed step-by-step explanation of the Maximum Likelihood Estimation (MLE) for a   
Gaussian distribution using Markdown with KaTeX:                                                         

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}

## Step 1: Define the Gaussian Distribution                                                                 

We start by defining the Gaussian distribution, also known as the Normal distribution. The probability   
density function (PDF) of a Gaussian distribution is given by:                                           

$$
[f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)]
$$

where $$(x)$$ is the observed value, $$(\mu)$$ is the mean, and $$(\sigma^2)$$ is the variance.                      

## Step 2: Formulate the Likelihood Function                                                                

Next, we formulate the likelihood function, which represents the joint probability of observing the      
dataset given the parameters. Since the observations are assumed to be independent and identically       
distributed (i.i.d.), the likelihood function is the product of the individual probabilities:            

$$
[L(\mu, \sigma^2) = f(x_1; \mu, \sigma^2) \cdot f(x_2; \mu, \sigma^2) \cdot ... \cdot f(x_n; \mu,        
\sigma^2)]               
$$                                                                                

where (x_1, x_2, ..., x_n) are the observed values.                                                      

## Step 3: Take the Logarithm of the Likelihood Function                                                    

To simplify the calculations, we take the logarithm of the likelihood function. This is a common practice
and does not affect the location of the maximum:                                                         

$$
[\log L(\mu, \sigma^2) = \log f(x_1; \mu, \sigma^2) + \log f(x_2; \mu, \sigma^2) + ... + \log f(x_n; \mu,
\sigma^2)]                                                                                               
$$

## Step 4: Differentiate the Log-Likelihood Function                                                        

To find the maximum likelihood estimates for (\mu) and (\sigma^2), we differentiate the log-likelihood   
function with respect to each parameter and set the derivatives equal to zero.                           

4.1. Differentiate with respect to (\mu):                                                                

$$[\frac{d}{d\mu} (\log L(\mu, \sigma^2)) = 0]$$                                                   

4.2. Differentiate with respect to (\sigma^2):                                                           

$$[\frac{d}{d(\sigma^2)} (\log L(\mu, \sigma^2)) = 0]$$                                       

## Step 5: Solve the Equations                                                                              


Solve the equations obtained in Step 4 to find the values of $$(\mu)$$ and $$(\sigma^2)$$ that maximize the      
log-likelihood function. This can be done by solving the equations analytically or using numerical       
optimization techniques.                                                                                 

## Step 6: Interpret the Results                                                                            

Interpret the estimated values of (\mu) and (\sigma^2) in the context of the problem. These estimates    
represent the maximum likelihood estimates for the mean and variance of the Gaussian distribution based  
on the observed data.                                                                                    

Note: This is a detailed step-by-step explanation of the MLE deduction for a Gaussian distribution. The  
actual derivation involves more mathematical steps and calculations.        