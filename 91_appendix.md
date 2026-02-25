---
layout: post
title: Appendix
categories: [Generative AI]
toc: true
---

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

## Rate of Convergence
Let $x_k$ be an iterative sequence converging to $x^*$. Define the error:

$$
e_k := \| x_k -x^* \|
$$

We describe convergence by how fast $e_k \to 0$ as $k \to \infty$.

We say $e_k = O(g(k))$ if there exists a constant $C>0$ and $k_0$ such that

$$
e_k \leq C\, g(k), \hspace{1cm} \forall k \geq k_0
$$

This means the error decreases at most at the rate of $g(k)$ asymptotically. For example

$$
e_k = O\left(\frac{1}{k}\right) \implies e_k \leq \frac{C}{k} \hspace{1cm} \forall k \geq k_0
$$

That is, $O(g(k))$ means that the error decays no slower than $g(k)$. Or, the error shrinks proportionally to $g(k)$ asymptotically, up to a constant factor.