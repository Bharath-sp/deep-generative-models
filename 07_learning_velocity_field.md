---
layout: post
title: Learning Velocity Field
categories: [Generative AI]
toc: true
---

* TOC
{:toc}

## Problem Statement
Consider the case where there is no random perturbation.

Suppose we start with $p_0$ at $t=0$ and we reach $p_1$ at $t=1$. We want $p_1$ to be our target distribution. We want to learn $v_t$ such that $p_1=p^*$ is achieved. In general, we are given only samples from $p^*$, so learning $v_t$ will be a learning problem. Here from the sampling aspect, we are assuming that $p^*$ is known, so finding an appropriate $v_t$ is an optimization problem.

## Optimization Problem
Suppose $p_0$ is our initial distribution (fixed) and we know $p^*$ which is also fixed. If $p_0$ and $p^*$ are known, we can compute $v_t$ which takes us from $p_0$ to $p^*$.

It is evident that there could be multiple solutions to this problem. There can be many velocity fields that can takes us from $p_0$ to $p^*$. Of all the possible velocities, we might be interested in:

* Simple particle paths: straight line paths in the input space.
* Simple likelihood paths: straight line paths in the likelihood space.
* Least energy paths

The particle flow ODE is:

$$
\frac{dx_t}{dt} = v_t(x_t)
$$

$v_t$ is a function that can be parameterized. On folding the ODE and suppose $\Delta t=1$, we get

<a name="eq:eq1"></a>
$$
\begin{align*}
x_1 & = x_0 + v_0(x_0) \\
x_2 & = x_1 + v_1(x_1) \\
\vdots \\
x_T & = x_{T-1} + v_{T-1}(x_{T-1}) \tag{1}\\
\end{align*}
$$

$x_T$ is a function of $x_0$, and this function depends on what is our velocity function. On looking at this unrolling of equations, we observe that the best way to parameterize $v_t$ is by an RNN. In RNN with skip connection, we typically do

$$
\begin{align*}
x_1 & = f(x_0) + x_0 \\
x_2 & = f(x_1) + x_1 \\
\vdots \\
x_T & = f(x_{T-1}) + x_{T-1} \\
\end{align*}
$$

In RNNs, we usually have the same network $f$ (with same parameters). This architecture can be used to model the function $v(x_t, t)$.

Equations in <a href="#eq:eq1">(1)</a> are discrete unfolding of the particle flow ODE. The particle ODE (which is the continuous version) has these unfoldings for every $t$. We can think of this as similar to RNNs (with skip connection) but with continuous set of layers. Such RNNs are known as Neural ODEs.

We learn a function $f_{\theta}(x_t,t)$ using a model (neural ODE NN + ODE solver) by imposing the condition that the output from the model should be distributed according to our target distribution $p^*$. So, our objective is to learn a parameter $\theta$ (out of many possible $\theta$s) such that $f_{\theta}(X_0) \sim p^*$. If $X_0$ follows a uniform distribution, then $f_{\theta}(X_0)$ is a RV which is some function of $X_0$. We want to choose $\theta$ such that the distribution of this RV is the same as $p^*$.

<div class="admonition tip">
  <p class="admonition-title">Note</p>
  <p>There can be many functions of a random variable having the same distribution. For example, if $X_0$ is Bernoulli distributed with $p=0.5$, then $1-X_0$ is also Bernoulli distributed with $p=0.5$, if $X_0$ is Gaussian, then $-X_0$ is also Gaussian.</p>
</div>

Therefore, there are many $\theta$s that satisfy our objective. So, we put some more conditions on $\theta$ to get a unique solution (i.e., in order to achieve simple paths).

## Two Paradigms of Sampling

* **No random perturbations**:
When $v_t$ is designed (or learned) and fixed, there is no concept of limiting distribution. So, we look at achieving the target distribution in finite time steps.
Suppose we design $v_t$ such that we achieve the target distribution at $t=1$. Even then, at time steps $t>1$, it is not necessary that we remain at $p^*$. But we don't care, once we reach $t=1$, we can start sampling. Thus, $p^*$ doesn't need to be a stationary distribution to the process under this setup.

* **With random perturbations**:
If we have fixed $v_t$ along with random perturbations, then we can look at the notion of limiting distribution, i.e., the convergence of the process to a distribution regardless of the initial distribution. Because of the presence of random noise, the initial conditions will be forgotten.
After large $t$, it doesn't matter where we sample from. All future distributions will remain the same, i.e., $p_T, p_{T+1}, \dots$ will all be the same.


