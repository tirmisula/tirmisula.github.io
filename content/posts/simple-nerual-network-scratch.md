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
ShowBreadCrumbs: false
ShowToc: true
TocOpen: true
---

This article offers a basic knowledge of machine learning.

<!-- ```bash
{{ if or .Params.math .Site.Params.math }}
{{ partial "math.html" . }}
{{ end }}
``` -->

{{< math.inline >}}
{{ if or .Page.Params.math .Site.Params.math }}

<!-- KaTeX -->
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.css" integrity="sha384-zB1R0rpPzHqg7Kpt0Aljp8JPLqbXI3bhnPWROx27a9N0Ll6ZP/+DiW/UqRcLbRjq" crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/katex.min.js" integrity="sha384-y23I5Q6l+B6vatafAwxRu/0oK/79VlbSz7Q9aiSZUvyWYIYsd+qj+o24G5ZU2zJz" crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.11.1/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body);"></script>
{{ end }}
{{</ math.inline >}}

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
> — <cite>[^1]</cite>

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

      def forward(self, X):
          # Compute the activations of the first hidden layer
          self.z1 = np.dot(X, self.W1) + self.b1
          self.a1 = self.sigmoid(self.z1)

          # Compute the activations of the second hidden layer
          self.z2 = np.dot(self.a1, self.W2) + self.b2
          self.a2 = self.sigmoid(self.z2)

          # Compute the output
          self.z3 = np.dot(self.a2, self.W3) + self.b3
          self.a3 = self.sigmoid(self.z3)

          return self.a3

      def backward(self, X, y, output):
          # Compute the gradients of the output layer
          delta3 = (output - y) * self.sigmoid_derivative(self.z3)
          dW3 = np.dot(self.a2.T, delta3)
          db3 = np.sum(delta3, axis=0, keepdims=True)

          # Compute the gradients of the second hidden layer
          delta2 = np.dot(delta3, self.W3.T) * self.sigmoid_derivative(self.z2)
          dW2 = np.dot(self.a1.T, delta2)
          db2 = np.sum(delta2, axis=0, keepdims=True)

          # Compute the gradients of the first hidden layer
          delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.z1)
          dW1 = np.dot(X.T, delta1)
          db1 = np.sum(delta1, axis=0, keepdims=True)

          return dW1, db1, dW2, db2, dW3, db3

      def train(self, X, y, num_epochs, learning_rate):
          for epoch in range(num_epochs):
              # Forward pass
              output = self.forward(X)

              # Backward pass
              dW1, db1, dW2, db2, dW3, db3 = self.backward(X, y, output)

              # Update weights and biases
              self.W1 -= learning_rate * dW1
              self.b1 -= learning_rate * db1
              self.W2 -= learning_rate * dW2
              self.b2 -= learning_rate * db2
              self.W3 -= learning_rate * dW3
              self.b3 -= learning_rate * db3

              # Print loss every 100 epochs                                                              
              if epoch % 100 == 0:                                                                       
                  loss = self.mse_error(y, output)                                                  
                  print(f"Epoch {epoch}: Loss = {loss}")

      def predict(self, X):
          return self.forward(X)

      def sigmoid(self, x):
          return 1 / (1 + np.exp(-x))

      def sigmoid_derivative(self, x):
          return self.sigmoid(x) * (1 - self.sigmoid(x))

      def mse_error(self, y_true, y_pred):
          return np.mean((y_true - y_pred) ** 2)


  # Create a neural network with 2 input units, 3 hidden units, and 1 output unit
  nn = NeuralNetwork(2, 3, 4, 1)

  # Generate some dummy data for training
  X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
  y = np.array([[0], [1], [1], [0]])

  # Train the neural network
  nn.train(X, y, num_epochs=10000, learning_rate=0.1)

  # Make predictions
  predictions = nn.predict(X)
  print("Predictions:", predictions)
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
