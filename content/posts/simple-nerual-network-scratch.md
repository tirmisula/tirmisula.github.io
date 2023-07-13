---
author: "Xuan Wang"
title: " Machine Learning Guide"
date: "2023-07-13"
description: "A simple neural network from scratch."
tags: ["machine learning"]
categories: ["themes", "syntax"]
# series: ["Themes Guide"]
aliases: ["migrate-from-jekyl"]
math: true
ShowToc: true
TocOpen: true
---

This article offers a basic knowledge of machine learning.

<!--more-->


## Block math

MSE Loss: 

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)})^2
$$

## Paragraph

A simple NN from scratch.

## Blockquotes

Use `footer` or `cite` element.

## Reference
> Neural Network.
>
> — <cite>Josh Starmer[^1]</cite>

[^1]: The above quote is excerpted from [video](https://www.bilibili.com/video/BV1Au411m7bM) 2021.

<!-- ## Tables

Tables aren't part of the core Markdown spec, but Hugo supports them out-of-the-box.

| Name  | Age |
| ----- | --- |
| Bob   | 27  |
| Alice | 23  |

#### Inline Markdown within tables

| Italics   | Bold     | Code   |
| --------- | -------- | ------ |
| _italics_ | **bold** | `code` | -->

## Code Blocks

#### Code block with backticks and language specified

```python {linenos=true}
 import numpy as np                                                                                      
                                                                                                                                                                                                                   
 class NeuralNetwork:                                                                                    
                                                                                                         
     def __init__(self, input_size, hidden_size1, hidden_size2, output_size):                            
                                                                                                         
         self.input_size = input_size                                                                    
                                                                                                         
         self.hidden_size1 = hidden_size1                                                                
                                                                                                         
         self.hidden_size2 = hidden_size2                                                                
                                                                                                         
         self.output_size = output_size     

        # Initialize weights and biases for the first hidden layer                                      
                                                                                                         
         self.W1 = np.random.randn(self.input_size, self.hidden_size1)                                   
                                                                                                         
         self.b1 = np.zeros((1, self.hidden_size1))                                                      
                                                                                                         
                                                                                                         
                                                                                                         
         # Initialize weights and biases for the second hidden layer                                     
                                                                                                         
         self.W2 = np.random.randn(self.hidden_size1, self.hidden_size2)                                 
                                                                                                         
         self.b2 = np.zeros((1, self.hidden_size2))                                                      
                                                                                                         
                                                                                                         
                                                                                                         
         # Initialize weights and biases for the output layer                                            
                                                                                                         
         self.W3 = np.random.randn(self.hidden_size2, self.output_size)                                  
                                                                                                         
         self.b3 = np.zeros((1, self.output_size))  
```

<!-- #### Code block with Hugo's internal highlight shortcode

{{< highlight html >}}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Example HTML5 Document</title>
</head>
<body>
  <p>Test</p>
</body>
</html>
{{< /highlight >}} -->

<!-- #### Gist

{{< gist spf13 7896402 >}} -->

<!-- ## List Types

#### Ordered List

1. First item
2. Second item
3. Third item

#### Unordered List

-   List item
-   Another item
-   And another item

#### Nested list

-   Fruit
    -   Apple
    -   Orange
    -   Banana
-   Dairy
    -   Milk
    -   Cheese

## Other Elements — abbr, sub, sup, kbd, mark

<abbr title="Graphics Interchange Format">GIF</abbr> is a bitmap image format.

H<sub>2</sub>O

X<sup>n</sup> + Y<sup>n</sup> = Z<sup>n</sup>

Press <kbd><kbd>CTRL</kbd>+<kbd>ALT</kbd>+<kbd>Delete</kbd></kbd> to end the session.

Most <mark>salamanders</mark> are nocturnal, and hunt for insects, worms, and other small creatures. -->
