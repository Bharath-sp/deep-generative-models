---
layout: post
title: Energy-Based Models
categories: [Generative AI]
toc: true
---

We are given a set of images. That is, we are given a set of samples from some underlying distribution. How can we sample more images from this (unknown) underlying distribution?

* TOC
{:toc}

## Energy-Based Models

How do we predict a probability distribution $p(\mathbf{x})$ where $\mathbf{x}$ is of high-dimension using a simple neural network? The problem is that we cannot just predict a score between 0 and 1, because a probability distribution over data needs to fulfill two properties:

- The probability distribution needs to assign any possible value of $\mathbf{x}$ a non-negative value: $p(x) \geq 0$
- The probability density must sum/integrate to 1 over all possible inputs: $\int p(\mathbf{x}) d\mathbf{x} = 1$.

There are actually many approaches for this, and one of them is energy-based models. The fundamental idea of energy-based models is that we can turn any function that predicts values larger than zero into a probability distribution by dividing by its volume. Imagine we have a neural network, which has as output a single neuron, like in regression. We can call this network $f_{\theta}(\mathbf{x})$, where $\theta$ are the parameters of the network and $\mathbf{x}$ is the input data (for e.g., an image). The output of $f_{\theta}$ is a scalar between $-\infty$ and $\infty$. Now, we can use basic probability theory to normalize the scores of all possible inputs:

$$
\begin{split}p_{\theta}(\mathbf{x}) = \frac{\exp\left(-f_{\theta}(\mathbf{x})\right)}{Z_{\theta}} \hspace{5mm}\text{where}\hspace{5mm}
Z_{\theta} = \begin{cases}
    \int_{\mathbf{x}}\exp\left(-f_{\theta}(\mathbf{x})\right) d\mathbf{x} & \text{if }x\text{ is continuous}\\
    \sum_{\mathbf{x}}\exp\left(-f_{\theta}(\mathbf{x})\right) & \text{if }x\text{ is discrete}
\end{cases}\end{split}
$$

The $\text{exp}$-function ensures that we assign a probability greater than zero to any possible input. We use a negative sign in front of $f$ because we call $f_{\theta}$ to be the energy function: data points with high likelihood have a low energy, while data points with low likelihood have a high energy. $Z_{\theta}$ is our normalization term that ensures that the density integrates/sums to 1.

Reference: [Taken from here](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial8/Deep_Energy_Models.html)