---
layout: post
title: Explicit Generative Modelling Summary
categories: [Generative AI]
toc: true
---

* TOC
{:toc}

## Landscape
In explicit generative models, generation happens in a two-step process:

1. Given training data, we model the likelihood $p_{\theta}$ (which is the model distribution) and estimate $p_{\theta^*}$. We solve the optimization:

$$
\min_{\theta} D(p_{\theta}, p^*)
$$

to estimate $p_{\theta^*}$. If $D$ is KL, then we get the MLE estimate. If we do score matching, we get the score-matching objective. 

2. Sampling (LMC): sample from $p_{\theta^*}$.

These are equivalent to training and inference. But both are non-trivial because each has its own convergence issue.

**Classical Models:**
In explicit generative models such as energy-based models and score-based models, we typically do:

$X_k \sim p^*$ are samples from the target distribution $p^*$. We don't have access to $p^*$. With the given samples, we can only have the empirical distribution $p^*_{\text{emp}}$.

* In the first step,

  * In energy-based model, we learn $p_{\theta^*}$, which is an estimate of $p^*$ up to normalization constant. Assuming no model error, the estimate is the same as the $p^*_{\text{emp}}$.
  * In score-based models, we learn $S_{\theta^*}$, which is an estimate of the true score function $S^*$. Assuming no model error, the estimate is the same as the $S^*_{\text{emp}}$.

The error between the true and empirical distributions cannot be reduced; it will be low if we have a lot of training samples. It usually goes down at the rate of $\frac{1}{\sqrt{m}}$, where $m$ is the number of samples. This is the error, known as estimation error, incurred during training.

* In the second step, we run LMC to produce samples with the target $p_{\theta^*}$ or using the score function $S_{\theta^*}$. But as we run LMC only for finite steps, the distribution of the generated samples $\hat{X}_k$ will not be $p_{\theta^*}$. The error goes down as we take more steps; at the rate of $\frac{1}{\sqrt{k}}$ for likelihoods with strictly log-concave energy functions (such as Gaussian targets), where $k$ is the number of steps. The convergence rate for other targets will be slower. This is the error incurred during inference.

The overall error is the sum of these two in the classical models.

**Denoising Score-matching models:**
In denoising score matching models, we add noise to the training samples to get noisy samples.

$\tilde{X}_k \sim \tilde{p}_\sigma$ are the noisy samples obtained by adding a small Gaussian noise to the original samples. Note that this is the empirical noise distribution; we don't have access to the true noisy distribution. The score function associated with $\tilde{p}_\sigma$ is $\tilde{S}_\sigma$.

* As a first step, we learn $\tilde{S}_{\theta^*, \sigma}$ using a score-network, which is an estimate of $\tilde{S}_\sigma$. Assuming no model error, $\tilde{S}_{\theta^*, \sigma}$ is exactly same as $\tilde{S}_\sigma$. The error between the true distribution and empirical distribution cannot be reduced; it will be low if we have a lot of training samples.

* In the second step, we run LMC to produce samples using the score function $\tilde{S}_{\theta^*, \sigma}$. But as we run LMC only for finite steps, the distribution of the generated samples $\hat{\tilde{X}}_k$ will not exactly be $\tilde{p}_{\theta^*, \sigma}$. The error goes down as we take more steps; at the rate of $\frac{1}{\sqrt{k}}$, where $k$ is the number of steps.

We then denoise these samples to produce $\hat{X}_k$.

**Diffusion models:**
In diffusion models, we are given samples $X_k$ from $p^*$. We don't have access to $p^*$. With the given samples, we can only have the empirical distribution $p^*_{\text{emp}}$. We run LMC from this distribution with Gaussian as the target, and produce samples $\hat{X}_0$ from a Gaussian distribution. Since we run LMC for finite steps, the distribution of $\hat{X}_0$ may not be exactly Gaussian. So, we can do an LMC  from pure Gaussian to this distribution. During the forward LMC, we learn a score function at each time step.

Then, on $\hat{X}_0$ samples, we run a reverse LMC using the learnt score functions. The reverse Langevin sampling process is exactly the reversal of the forward Langevin sampling process. The forward process converges exponentially fast as the target is the Gaussian distribution, so the reverse process also converges exponentially faster regardless of the target. 

But as we discretize to carry out these processes in practice, there is an error introduced because of step sizes. Thus, when we carry out reverse LMC, the samples will not exactly follow the empirical distribution $p^*_{\text{emp}}$ we started with.

<figure markdown="0" class="figure zoomable">
<img src='./images/explicit_modelling.png' alt="" width=600><figcaption>
  <strong>Figure 1.</strong> Summary of explicit generative modelling.
  </figcaption>
</figure>

With the diffusion models, if we ignore the discretization (step size) error and consider the reverse **process** (instead of LMC), the convergence to the target is exponentially faster regardless of the target. Whereas with other models, the convergence depends on the target.

* For Gaussian, the convergence is exponentially fast. Even after discretization, the convergence is fast, at the rate of $\frac{1}{\sqrt{k}}$.
* For an arbitrary target distribution (not strongly convex), the convergence may not be exponential faster, especially after discretization.

<div class="admonition note">
  <p class="admonition-title">NOTE</p>
  <p>Pure Langevin sampling is the process of running the Langevin sampling process directly from the proposal to the target, without any learnings from the forward process.</p>
</div>


Therefore, in theory i.e., without discretization, diffusion models act as a new sampling algorithm that is exponentially fast for any target. The energy function of the target distribution doesn't need to be strongly convex for exponentially fast convergence. The reverse process converges exponentially fast regardless of the target. But in practice, we do discretized version of the reverse process whose convergence is not exponentially fast but certainly faster than the convergence of the discretized pure Langevin sampling process. This is because in diffusion models, we use different score function at each time step. And as we learn them jointly, we get a better estimation of score functions.

During inference, we use the estimated score functions to run the reverse LMC to transform noise to the target.

$$
x_{i(k-1)} = (1 + h)\,x_{ik} + 2h \, S_{\theta^*}(x_{ik}, k) + \sqrt{2h} \, n_k \\
$$

If the score functions were exact, if the step sizes are small $s \to 0$ and as the number of steps $k \to \infty$, we will reach $p^*$. In practice, our score functions are not exact, they have error at the rate $O(\frac{1}{\sqrt{m}})$. Since we are not using the exact score function, when we run through iterations in the reverse process, how does the error in the score functions propagate? Will the errors get accumulated and exploded?

In gradient descent, we take gradient steps to reach the target. Suppose if those gradients were not exact; they are erroneous, we may not reach convergence. Across iterations, how does the inaccuracy in gradients get accumulated? 

Compared to gradient descent, we know that SGD can handle noisy gradients better, because SGD is designed to handle noisy gradients whereas GD is not designed to handle them. So, when the gradients are erroneous, the GD is more affected than SGD.

The same thing happens with stochastic differential equations. The SDEs already have an explicit noise term $N_k$. So, the errors in the score functions can be swallowed by this explicit noise term. This is like transferring the error term in score function being transferred to the noise term. And as the coefficient of the noise term (which is $\sqrt{s}$) is bigger than the coefficient of the score function, the noise term can easily swallow small order noises.

* If $s=0.001$, then $\sqrt{2s} = 0.044$. Then, $s+\sqrt{s} = 0.045$. 

The error from the score functions which is of order $s$ can be easily consumed by the noise term. Therefore, the result is that the errors in the score functions don't propagate. And in fact, we can decay them by increasing the number of steps in the reverse process. That is, the error will be proportional to $\frac{n}{\sqrt{k}}$ regardless of the target. So, by taking more reverse steps, we can reach the empirical target distribution even on using the noisy estimated score functions. The error between the true and the empirical target distribution can be maintained at $O(\frac{1}{\sqrt{m}})$.

In all these methods, we first model either the likelihood or score function explicitly during training. During inference, we then run LMC or a diffusion process (backward denoising through reverse LMC) to obtain samples. The inference is non-trivial. This is because what is modelled and learned is not a sampler; it is a likelihood.

As of today, among all the explicit generative modelling methods, diffusion models are the best.






