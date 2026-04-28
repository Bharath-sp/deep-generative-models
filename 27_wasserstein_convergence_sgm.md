---
layout: post
title: Wasserstein Convergence of SGMs
categories: [Generative AI]
toc: true
---

* TOC
{:toc}


## Abstract
Score-based Generative Models (SGMs) such as denoising score matching and diffusion models approximate a data distribution by perturbing it with Gaussian noise and subsequently denoising it via a learned reverse diffusion process.

We saw that Langevin diffusion converges to $p^*$ for any distribution, but the rate of convergence will be exponentially slow. To prove that the convergence happens at an acceptable rate, we need some regularity conditions on $p^*$. We saw that under the assumption of $\mu$- strongly convex energy functions $f$, the convergence is exponentially fast.

$$
\begin{align*}
p^*(x) & \propto e^{-f(x)} \\
-\log p^*(x) & \propto f(x)
\end{align*}
$$

We did this convergence analysis under KL divergence, $\text{KL}(p_t || p^*)$. Existing Wasserstein-2 convergence analysis also assume strong regularity conditions such as smoothness (derivative doesn't change too rapidly) or convexity of the energy function of data distribution - that are rarely satisfied in practice.

In this paper, the authors have relaxed these assumptions and have established the Wasserstein-2 convergence guarantees for SGMs targeting **semi-convex distributions with potentially discontinuous gradients**. They have established explicit, non-asymptotic Wasserstein-2 convergence bounds under semi-convexity assumptions on the data distribution, accommodating potentials with discontinuous gradients.

<div class="admonition note">
  <p class="admonition-title">NOTE</p>
  <p>A distribution is semi-convex if its potential function $f(x)$ isn't necessarily perfectly convex, but can be made convex by adding a simple quadratic term like $\frac{\lambda}{2} \|x\|^2$.</p>
</div>

The work broadens the theoretical foundations of SGMs by accommodating a wide class of practically relevant distributions.

## Appendix

