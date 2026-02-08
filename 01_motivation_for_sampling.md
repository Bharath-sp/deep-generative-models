---
layout: post
title: Motivation for Sampling
categories: [Generative AI]
toc: true
---

We are given a set of images. That is, we are given a set of samples from some underlying distribution. How can we sample more images from this (unknown) underlying distribution?

* TOC
{:toc}

## Introduction

There are two ways to solve the generation problem. Diffusion models and sampling-based models are indirect methods of generative models. In these methods, we learn the likelihood from the given samples, and then produce more samples using some sampling algorithms. GANs are direct methods. In these methods, we directly learn the sampler without learning the likelihood or score function.

GANs and VAE are classical methods, now people have moved towards diffusion models. If we want to generate images along with their likelihoods, we can use normalizing flows. These methods learn the likelihood and sampler simultaneously.

## Problem Statements in Generative Modelling

In the classical machine learning, generative modelling refers to supervised learning problems where we build a model for the likelihood of $(\mathbf{x},y)$, then we build a Bayes classifier or regressor based on the estimate of the likelihood. In modern days, generative modelling refers to a problem where we are given a set of samples and the objective is to produce more samples that are similar.

1. Generation Problem (learning problem because from the given few samples we need to generalize and produce more samples):
    We have an underlying distribution $p^*$ (could be over $ \mathbf{x}, (\mathbf{x},y), (y \, | \, \mathbf{x})$). We are given samples from it.

    Let the underlying distribution be $p^*(x)$ and the samples given are $x_1, x_2, \dots, x_m$. This forms our training set. Using only this training set (without the knowledge of $p^*(x)$), the objective is to create more samples from the same distribution $p^*(x)$.

    This problem often involves both likelihood estimation and sampling, but there are also generative algorithms which don't solve the problem of sampling (for e.g., GANs). In GANs, we don't estimate the likelihood explicitly, therefore no sampling problem involved; we directly generate the samples. <br><br/>


2. Sampling Problem (no learning involved, just inference):

    We are given a likelihood $p^*(x)$. We want to produce samples from it. We want an algorithm that takes the likelihood function $p^*(x)$ as input and produces **random** samples from it. Such algorithms are called as sampling algorithms.

    NOTE: Here likelihood refers to an unnormalized density. The normalizing constant (partition function) is unknown or intractable. We aim to sample from a probability distribution. In practice, we are often given an unnormalized density (sometimes derived from a likelihood or energy function), and sampling algorithms are used that do not require explicit normalization.

Most often, we solve the generation problem in two steps (but this is not the only way to solve generation problems):

1. Learn $\hat{p}^*(x)$ from the training set.
2. Sample from $\hat{p}^*(x)$

## Likelihood Estimation

Let the underlying distribution be $p^*(x)$ (not shown to us) and the samples given are $x_1, x_2, \dots, x_m$. Using these samples, we need to estimate $p^*(x)$. The problem of estimating $p^*(x)$ is known as likelihood estimation. Once we solve for this, we end up with a close proxy $\hat{p}^*(x)$.

### MLE Method

We assume the distribution to be $p(x)$, parameterized by $\theta$. The goal is

$$
\arg \max_{\theta} \frac{1}{N}\sum_{i=1}^N \log p_{\theta}(x_i)
$$

When we take the $\log$ outside the sum, it is the log of the product of the likelihoods which is the likelihood of the data. Therefore, we are maximizing the log-likelihood of the training data. In training, we need to solve this optimization problem. We typically use gradient descent algorithm to solve this. To use gradient descent, we need the gradient of the objective function (log-likelihood function)

$$
\begin{align*}
h(\theta) & = \frac{1}{N} \sum_{i=1}^N \log p_{\theta}(x_i) \\
\nabla h(\theta) & = \frac{1}{N} \sum_{i=1}^N \nabla \log p_{\theta}(x_i) \\
\end{align*}
$$

Unless we give a form for $p_{\theta}(x)$, we cannot further simplify it. We assume the distribution $p_{\theta}$ to be from exponential family of distributions.

$$
p_{\theta}(x) = \frac{e^{-f_{\theta}(x)}}{\int e^{-f_{\theta}(x)} dx} = \frac{e^{-f_{\theta}(x)}}{Z(\theta)}
$$

where $f_{\theta}(x)$ can be any neural network with any architecture (depending on the modality of $x$). It takes an $x$ and gives a real number. Raising it to the power of $e$ ensures that the result is non-negative. And normalizing it by a suitable constant makes $p_{\theta}(x)$ a valid probability distribution (a likelihood).

Such models for likelihood based on neural networks are called as **Energy-based** models. The neural network function $f_{\theta}(x)$ plays the role of energy in statistical physics interpretation. Electrons with high energy are less likely to be in lower orbits, and electrons with low energy are highly likely to be in lower orbits. In such sense, $f_{\theta}(x)$ can be interpreted as an energy function (that outputs energy for a given input).

* The higher the energy, the lower the likelihood.
* The lower the energy, the higher the likelihood.

The normalization constant $Z$ is a function of $\theta$, called as the partition function. Then, on substitution:

$$
\begin{align*}
\nabla h(\theta) & = \frac{1}{N} \sum_{i=1}^N \nabla \left(- f_{\theta}(x_i) - \log Z(\theta) \right) \\
& = -\frac{1}{N} \sum_{i=1}^N \nabla f_{\theta}(x_i) 
-\frac{1}{N} \sum_{i=1}^N \nabla \log Z(\theta) \\
& = -\frac{1}{N} \sum_{i=1}^N \nabla f_{\theta}(x_i) - \nabla \log Z(\theta) \\
\end{align*}
$$

We know that

$$
\begin{align*}
\nabla \log Z(\theta) & = \frac{\nabla Z(\theta)}{Z(\theta)} = \frac{\nabla \int e^{-f_{\theta}(x)} dx} {Z(\theta)} \\
\end{align*}
$$

The differentiation in gradient is with respect to $\theta$. The integral is not with respect to $\theta$. So, we can just differentiate the integrand.

$$
\begin{align*}
\frac{\nabla \int e^{-f_{\theta}(x)} dx} {Z(\theta)} & = \frac{ \int \nabla e^{-f_{\theta}(x)} dx} {Z(\theta)} \\
& = \frac{ - \int e^{-f_{\theta}(x)} \nabla f_{\theta}(x)  dx} {Z(\theta)} \\
& = - \int p_{\theta}(x) \, \nabla f_{\theta}(x)  dx  \\
& = - \mathbb{E}_{X \sim p_{\theta}(x)} \left[ \nabla f_{\theta}(x)  \right] \\
\end{align*}
$$

The gradient of the objective function in an MLE problem for training an energy-based model can then be given by:

$$
\begin{align*}
\nabla h(\theta) & = -\frac{1}{N} \sum_{i=1}^N \nabla f_{\theta}(x_i) + \mathbb{E}_{X \sim p_{\theta}(x)} \left[ \nabla f_{\theta}(x)  \right]  \\
\end{align*}
$$

* The first term is the expectation of the gradient of energy when the random variable is sampled from the distribution of our training points, $p^*(x)$.

* The second term is the expectation of the gradient of energy when the random variable is sampled from our learned distribution (model), $p_{\theta}(x)$. 

Therefore, we are essentially matching the expected values of gradient of $f$ between the model and the observations. When $\nabla h(\theta) = 0$:

$$
\frac{1}{N} \sum_{i=1}^N \nabla f_{\theta}(x_i) = \mathbb{E}_{X \sim p_{\theta}(x)} \left[ \nabla f_{\theta}(x)  \right]
$$

Here we match our model distribution to the training data distribution. Note: we are not matching the mean of the model distribution $p_{\theta}(x)$ to the mean of training samples $x_i$'s. We are matching the mean of gradient of $f$, thereby matching the model to $p^*$.

If $f$ is energy, then $\nabla f$ is rate of change of energy with displacement. It has energy divided by distance as unit. And we know that, force multiplied by distance calculates Work in physics, representing the energy transferred when a force causes displacement. Therefore, $\nabla f$ signifies some kind of force. Assume $x_i$'s to be particles. If we match the force applied on them according to our model to the force applied on them according to the evidence (given data), then the distribution of particle locations also match. That is, our model should be such that the forces at every point $(x,y)$ are matching in expectation with the actual forces that created these particles.

For any exponential family model, the MLE (for marginals or conditionals or joint) can be seen as this kind of moment matching condition.

Calculation of these gradients:

* The first term (or LHS) can be calculated easily as we have $x_i$'s and can find the gradient of the neural network's output with respect to its parameters using the autodiff procedure.
* The second term can be easily calculated the same way **if we know samples from $p_{\theta}$**. If we have samples from this distribution, say $z_1, \dots, z_M$, then we can calculate the expectation as:

$$
\frac{1}{M} \sum_{i=1}^M \nabla f_{\theta}(z_i)
$$

$p_{\theta}$ is the likelihood induced by some energy-based model. How can we produce samples from it? If we have a sampler (a sampling algorithm) that produces samples from any given distribution, then we can get samples using it. Therefore, a sampling algorithm is essential to train any energy-based model using the MLE approach.

MLE gives us the optimal parameters $\hat{\theta}$. The associated probability distribution (or likelihood) is $p_{\hat{\theta}}$. Then, we need to sample from this distribution to generate more samples. Again here we need a sampler.

### Energy-based Models for Conditionals
We assume the distribution to be $p(y \, | \, x)$, parameterized by $\theta$. The goal is

$$
\arg \max_{\theta} \frac{1}{N}\sum_{i=1}^N \log p_{\theta}(y_i \, | \, x_i)
$$

Suppose we assume that the distribution $p_{\theta}(y \, | \, x)$ is from the exponential family:

$$
p_{\theta}(y \, | \, x) = \frac{e^{-f_{\theta}(y,x)}}{\int e^{-f_{\theta}(y,x)} dy} = \frac{e^{-f_{\theta}(y,x)}}{Z(\theta, x)}
$$

On substitution, our objective is

$$
\begin{align*}
& \arg \max_{\theta} \frac{1}{N}\sum_{i=1}^N \log \left(
    \frac{e^{-f_{\theta}(y_i,x_i)}}{Z(\theta, x_i)}
 \right) \\
 & = \arg \max_{\theta} - \frac{1}{N}\sum_{i=1}^N f_{\theta}(y_i,x_i) - \frac{1}{N}\sum_{i=1}^N \log Z(\theta, x_i) \\
  & = \arg \max_{\theta} g(\theta) \\
\end{align*}
$$

On differentiating $g(\theta)$ with respect to $\theta$, we can see that the first term is

$$
\begin{align*}
\frac{1}{N}\sum_{i=1}^N \nabla f_{\theta}(y_i,x_i) & = \mathbb{E}[ \nabla f_{\theta}(Y, X) ] 
\end{align*}
$$

The expectation is over the distribution of $X, Y \sim p_{\mathcal{D}}(x_i, y_i)$. The second term is

$$
\begin{align*}
\nabla \log Z(\theta, x_i) & = \frac{\nabla Z(\theta, x_i) }{Z(\theta, x_i)} \\
& =  \frac{\nabla \int e^{-f_{\theta}(y,x_i)} dy}{\int e^{-f_{\theta}(y,x_i)} dy} \\
&= \frac{ \int \nabla e^{-f_{\theta}(y,x_i)} dy}{\int e^{-f_{\theta}(y,x_i)} dy}  \\
& = - \frac{ \int e^{-f_{\theta}(y,x_i)} \nabla f_{\theta}(y,x_i) \, dy }{\int e^{-f_{\theta}(y,x_i)} dy} \\
& = - \int p_{\theta}(y \, | \, x_i) \nabla f_{\theta}(y,x_i) \, dy \\
& = - \mathbb{E}[ \nabla f_{\theta}(Y,x_i) ] \\
\end{align*}
$$

Now, the expectation is over the distribution of $Y \, | \, x_i \sim p_{\theta}(y \, | \, x_i)$ (our model). So, the second term will be

$$
\frac{1}{N} \sum_{i=1}^N \nabla \log Z(\theta, x_i) = - \frac{1}{N} \sum_{i=1}^N \mathbb{E}[ \nabla f_{\theta}(Y,x_i) ] 
$$

To estimate this expectation efficiently, then we need samples from $p_{\theta}(y \, | \, x)$.

### Bayesian Method
The MLE method tries to find the best parameter $\theta$. Another way to do likelihood estimation is through Bayesian thinking. In Bayesian, $\theta$ is considered as a physical variable which cannot be measured, so we think about them as random variables. We start with a prior distribution over the parameters, $p(\theta)$. We then collect the data, and compute the likelihood $p(\mathcal{D} \, | \, \theta)$. The likelihood can come from any energy-based model.

In MLE, we maximized only this term $\max_{\theta} p(\mathcal{D} \, | \, \theta)$. In Bayesian, we compute the posterior:

$$
p(\theta \, | \, \mathcal{D}) \propto p(\mathcal{D} \, | \, \theta) \, p(\theta)
$$

The posterior is another distribution over parameters. After finding the posterior, we solve the general supervised learning problem by:

$$
p(y \, | \, x, \mathcal{D}) = \int p(y \, | \, \theta, x) \, p(\theta \, | \, \mathcal{D}) d\theta
$$

We take each likely parameter, get the likelihood value, and take the average opinion. The quantity $p(y \, | \, \theta, x)$ is computed from the same energy-based model, so it is known. But we don't have samples from $p(\theta \, | \, \mathcal{D})$. Therefore, to compute this integral, we need samples from the posterior. If we have samples from the posterior, say $\theta_1, \dots, \theta_L$, then we can approximate the integral efficiently as:

$$
p(y \, | \, x, \mathcal{D}) = \frac{1}{L} \sum_{i=1}^L p(y \, | \, x, \theta_i)
$$

In either case, be it modern generative modelling, or classical ML modelling using energy-based models via MLE, or Bayesian inference, we need to have sampling algorithms.

<div class="admonition tip">
  <p class="admonition-title">Note</p>
  <p>Few sampling strategies motivate/lead to novel generative modelling strategies</p>
</div>