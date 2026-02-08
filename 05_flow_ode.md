---
layout: post
title: Flow ODEs
categories: [Generative AI]
toc: true
---

* TOC
{:toc}

## Particle Flow ODE
We know that the MCMC based sampling algorithms use the following stochastic equations:

<a name="eq:eq1"></a>
$$
\begin{align*}
X_0 & \sim p_0 \\
X_{k+1} & := X_k + s \cdot v_k(X_k) + \sqrt{s} \, \sigma_k(X_k) \, N_k \tag{1}
\end{align*}
$$

For time being, let's assume the noise component is absent, that is, $\sigma_k=0$. Then

$$
\begin{align*}
X_{k+1} & := X_k + s \cdot v_k(X_k) \\
\frac{X_{k+1} - X_k}{s} & = v_k(X_k)
\end{align*}
$$

This equation is similar to the gradient ascent if $v_k$ is the gradient of the log-likelihood. For the ease of understanding, let's assume the step size $s$ to be infinitesimally small. This way we don't have to deal with choosing the right step size, and focus only on the gradient term. Thus, we analyze the corresponding continuous time equation: when $s$ tends to 0, we have the ODE:

<a name="eq:eq2"></a>
$$
\frac{dX_t}{dt} =  v_t(X_t) \tag{2}
$$

where we have changed the discrete step $k$ to a continuous one $t$. Here $X_t$ is a function of $t$, explicitly it is $X(t)$. The step size $s$ actually represents a small displacement (the change in time). The LHS measures how the random variable $X$ changes for a small change in time.

Equation <a href="#eq:eq2">(2)</a> is a first-order ordinary differential equation (ODE). Note this is a differential equation on random variables. On writing the equation using the values of random variables:

<a name="eq:eq3"></a>
$$
\frac{dx_t}{dt} =  v_t(x_t) \tag{3}
$$

Equation <a href="#eq:eq3">(3)</a> is called the primal view or particle view.

When solved, this equation tells us how $x_0$ evolves with time, that is, the flow of the particle $x_0$, where $x_t$'s are the positions of the particle at time $t$. Thus, these are also called as flow ODEs, and $v_t$ is the velocity field.

## Examples of ODE

### Example 01

Suppose $v_t(x_t) = x_t$ be the identity function where $x_t \in \mathbb{R}$. Then the equation is:

$$
\frac{dx_t}{dt} =  x_t
$$

We need to solve this ODE for $x_t$. Solving this ODE means that given an initial condition, say the value of $x_0$, we want to find the value of $x$ (position) at some arbitrary $t$, that is we need to find the function $x(t)$.

We can use a method called separation of variables. This involves moving all terms with $x$ to one side and all terms with $t$ to the other. Now the equation looks like this:

$$
\frac{1}{x} dx = 1 dt
$$

* The LHS: We are looking for how the value $1/x$ changes as $x$ itself changes.
* The RHS: We are looking for how the value $1$ changes as $t$ changes.

The power of this method is that since both sides are equal, their integrals must also be equal. We can now integrate the left side with respect to $x$ and the right side with respect to $t$:

$$
\int \frac{1}{x} dx = \int 1 dt
$$

Assume we want to integrate from $t=0$ to $t=T$. So, on the LHS, when we change the variable we are integrating with respect to (from $dt$ to $dx$), the limits of integration must change to match that new variable.

* When $t=0$, $x(t)=x_0$
* When $t=T$, $x(t)=x_T$ 

$$
\begin{align*}
\int_{x_0}^{x_T} \frac{1}{x} dx & = \int_0^T 1 dt \\
[\ln |x|]_{x_0}^{x_T} & = T \\
\ln |x_T| - \ln |x_0| & = T \\
\ln \left( \frac{x_T}{x_0} \right) & = T && (\text{assuming } x_T \text{ and } x_0 \text{ are positive})\\ 
x_T & = x_0 \, e^T \\
x_t & = x_0 \, e^t \\
\end{align*}
$$

This shows that the value at time $t$ is the starting value multiplied by the growth factor $e^t$. The solution to the original ODE is: $X_t = X_0\, e^t$. Here the given ODE has solution for all $t$ from 0 to any $t>0$. The solution exists and there is only one solution to it. This is a valid Markov process.

There can be situations in which the given ODE doesn't have a solution or unique solution. When we design the ODE, we should avoid these situations.

### Example 02

Suppose $v_t(x_t) = x_t^2$ be the function where $x_t \in \mathbb{R}$. Then the equation is:

$$
\frac{dx_t}{dt} =  x_t^2
$$

On separating the variables

$$
\begin{align*}
\frac{dx}{dt} & =  x^2 \\
\frac{dx}{x^2} & =  dt \\
\int_{x_0}^{x_T} \frac{dx}{x^2} & =  T \\
\left[ \frac{-1}{x} \right]_{x_0}^{x_T} & = T \iff x_T = \frac{1}{\frac{1}{x_0} - T}
\end{align*}
$$

The solution is 

$$
x_t = \frac{1}{\frac{1}{x_0} - t}
$$

Here $x_0$ should not be 0. And when $t=\frac{1}{x_0}$ or close to this, the function value $x(t)$ blows up in finite time. The blowing up depends on $x_0$ (our initialization). This is a case where the solution doesn't exist, i.e., not a valid Markov process. These are the functions we should avoid for $v$.

### Choice of $v(x)$
The above two examples highlighted the need for insisting conditions on the velocity field, $v_t$, for the solution to uniquely exist for the ODE. In particular, if we assume $v_t$ is Lipschitz continuous (i.e., whenever $x$ changes, $v$ doesn't change too much), then we can guarantee that the ODE is uniquely solvable, defining a valid Markov process.

* For the function $v(x)=x$ or $v(x)=1000x, \sqrt{x}$: for a change in $x$, the $v(x)$ doesn't change an order of magnitude higher. So, these are good functions. So, for the flow ODEs to have a solution, we should choose these kinds of functions for velocity.

* For the function $v(x)=x^2$: for a change in $x$, the $v(x)$ changes an order of magnitude higher. So, these are bad functions.

The function $v(x(t),t)$ represents the vector (or direction) field that governs the flow of particles. The choice of this function is crucial to ensure that the ODE has a well-defined solution over time.

We model this vector field using a neural network, denoted as $v_{\theta}(x(t),t)$, where $\theta$ represents the parameters of the neural network. Refer to the [Neural ODE](04_neural_ode.md) section for more details on modeling ODEs with neural networks.

The goal in neural ODE is to learn the vector field $v_{\theta}$ by adjusting the parameters $\theta$ of the neural network so that the solution $x(t)$ (the resulting flow) has desired properties. When $v_t$ is fixed, the particle flow is deterministic. However, since $x_0 \sim p_0$ (the proposal likelihood), the particle flow also corresponds to a probability flow: that of the marginal likelihoods of $X_t$. In order to design $v_t$ such that the limiting likelihood is the target, we need to understand this probability flow explicitly.

<div class="admonition tip">
  <p class="admonition-title">Note</p>
  <p>Our goal is to parameterize $v_t$ by a neural ODE, and we want to find the parameters of that NN such that the distribution $p_t$ should match the data distribution $p_{data}$ as $t$ goes to $\infty$.</p>
</div>

The training of neural network involves optimizing the parameters $\theta$ to minimize a loss function that measures how well the final distribution matches the data distribution. 

## Probability Flow ODE
Our primary objective is to transport a simple base distribution $p_0$ (like Gaussian) to a complex data distribution $p_{data}$ using the flow induced by the learned vector field $v_{\theta}(x_t,t)$. But what is the objective function to optimize the parameters $\theta$ of the neural network representing the vector field? The criteria is that the flow induced by $v_{\theta}$ should satisfy the following continuity equation:

<a name="eq:eq4"></a>
$$
\frac{\partial p_t(x)}{\partial t} = - \nabla \cdot (p_t(x) v_{\theta}(x,t)) \tag{4}
$$

* Equation <a href="#eq:eq3">(3)</a> is called the primal view or particle view. This ODE governs the transformation of random variable $X_0$ to $X_t$. For example, an image with random pixels is transformed to a meaningful image.

* Equation <a href="#eq:eq4">(4)</a> is called the dual view or likelihood view. This ODE governs the transformation of likelihood $p_0$ to $p_t$. That is, it defines the probability flow (the evolution of the probability density function) $p_t(x)$ over time under the influence of the vector field $v_{\theta}(x,t)$. For example, likelihood of a random image to a likelihood of a meaningful image.

**Understanding the Continuity Equation:**

We need to connect the evolution of the random variable to the evolution of the likelihood. So, let's start with a connective equation that connects random variables to likelihoods. The natural choice is the expectation operation. The expectation of random variable $X$ under the distribution $p_t(x)$ is defined as:

<a name="eq:eq5"></a>
$$
\mathbb{E}_{X \sim p} [X] = \int x p(x) dx \tag{5}
$$

The expectation operator is bilinear, meaning it is linear in both $x$ and the distribution $p_t$. This bilinearity allows us to interchange differentiation and integration under certain conditions.

A function $B(u,v)$ is bilinear if:

* It is linear in $u$ when $v$ is fixed, and
* linear in $v$ when $u$ is fixed

Here $B(x,p):= \int x p(x) dx$.

**Linearity in $x$ for a fixed $p$:**

Fix a probability density $p$. Let the random variable be $X \sim p$. Suppose $x_1 = g_1(X)$ and $x_2=g_2(X)$ are two random variables, and $a,b$ be scalars.

$$
\begin{align*}
\mathbb{E}_p[ax_1 + bx_2] & = \int (ax_1 + bx_2) \, p(x) \, dx \\
& = a \int x_1 p(x) dx + b \int x_2 p(x) dx \\
& = a \mathbb{E}_p[x_1] + b \mathbb{E}_p[x_2]
\end{align*}
$$

**Linearity in $p$ for a fixed $x$:**

Now fix the function $X$. For densities $p_1, p_2$ and scalars $a, b$:

$$
\begin{align*}
\mathbb{E}_{ap_1 + bp_2}[X] & = \int x\, ap_1(x) + bp_2(x) \, dx \\
& = a \int x p_1(x) dx + b \int x p_2(x) \, dx \\
& = \mathbb{E}_{ap_1}[X] + \mathbb{E}_{bp_2}[X]
\end{align*}
$$

NOTE: If we restrict to probability densities, linearity holds for convex combinations $(a,b \ge 0, a+b=1)$. Full linearity holds when $p$ is viewed as a signed measure.

Equation <a href="#eq:eq5">(5)</a> gives us explicit relationship between the random variable (in terms of the expected value) and its likelihood, but it is not characterizing the likelihood $p_t$. Our objective is that, from random variables, we need to know something about its likelihood. Expectation of a random variable doesn't say much about its likelihood in the sense that we could have different likelihoods with the same mean. But if we know all the moments of the random variable, we can uniquely define its likelihood. Two different likelihoods cannot have all the moments same. So, this is used as a way of characterizing likelihoods in terms of random variables. The moments of a random variable are just functions of $X$, $f(X)$. Thus, we look at the expectation of $f(X)$ for all $f$. At time $t$, it will be:

<a name="eq:eq6"></a>
$$
\mathbb{E}_{X_t \sim p_t} [f(X_t)] = \int f(x) p_t(x) dx \hspace{1cm} \forall f  \tag{6}
$$

For all $f$, if $\mathbb{E}[f(X)]$ is the same for two distributions $p$ and $q$, then they both are the same.

But the only information that is available to us is the rate of change of the position of the particle $X_t$, $\frac{dX_t}{dt}$. If we know the rate of change of $f(X_t)$, that is $\frac{df(X_t)}{dt}$, then we can use equation <a href="#eq:eq6">(6)</a> to know the rate of change of $p_t$, $\frac{\partial p_t}{\partial t}$ ($p$ is a function of both $x$ and $t$).

On taking derivative with respect to $t$ on both the sides in <a href="#eq:eq6">(6)</a>:


$$
\frac{d}{dt}\mathbb{E}[f(X_t)] = \frac{d}{dt} \int f(x) p_t(x) dx
$$

As expectation and derivative are linear operators and they both exist in this case, we can interchange them. That is, it doesn't matter in which order we apply the linear transformations. Thus,

<a name="eq:eq7"></a>
$$
\mathbb{E}\left[\frac{d}{dt} f(X_t)\right] = \int f(x) \frac{\partial p_t(x)}{\partial t} dx \tag{7}
$$

We don't know the rate of change of $f(X_t)$, we just know how $X_t$ changes. The Taylor series helps us connect the change in $f(x)$ values with the change in $x$ values.

$$
\begin{align*}
f(x_{t + h}) & = f(x_t) + \nabla f(x_t)^\top (x_{t+h} - x_t) + o(h) \\
f(x_{t + h}) - f(x_t) & = \nabla f(x_t)^\top (x_{t+h} - x_t) + o(h) \\
\\
\frac{f(x_{t + h}) - f(x_t)}{h} & = \frac{\nabla f(x_t)^\top (x_{t+h} - x_t)}{h} + \frac{o(h)}{h} \\
\end{align*}
$$

Taking the limit as $h \to 0$ on both the sides:

$$
\frac{d f(x_t)}{dt} = \nabla f(x_t)^\top \left[ \frac{dx_t}{dt} \right]
$$

The error term $o(h)$ is quadratic in $h$ (or at least more than $h$), so the term is like $h^2$. On dividing it by $h$ and as we move $h \to 0$, the numerator goes to 0 faster than the denominator. Thus, the term becomes 0.

On substituting this in equation <a href="#eq:eq7">(7)</a>:

$$
\begin{align*}
\int f(x) \frac{\partial p_t(x)}{\partial t} dx & = \mathbb{E}\left[ \nabla f(X_t)^\top \left[ \frac{dX_t}{dt} \right] \right] \\
& = \mathbb{E}\left[ \nabla f(X_t)^\top v_t(X_t) \right]\\
& = \int \nabla f(x)^\top v_t(x) \, p_t(x) \, dx 
\end{align*}
$$

We use integration by parts in $x$ technique. Suppose we need to find the following where $f(x)$ and $g(x)$ are functions of $x$:

$$
\begin{align*}
\frac{d}{dx} \int_{-\infty}^{\infty} f(x)\, g(x) dx & = \int_{-\infty}^{\infty} \frac{d}{dx} \left( f(x)\, g(x) \right) dx \\
& =  \int_{-\infty}^{\infty} f'(x) g(x) \, dx + \int_{-\infty}^{\infty} f(x) \, g'(x) \, dx \\
f(x) g(x) \big|_{-\infty}^{\infty} & = \int f'(x) g(x) \, dx + \int f(x) \, g'(x) \, dx \\
\\
\int_{-\infty}^{\infty} f'(x) g(x) \, dx & = f(x) g(x) \big|_{-\infty}^{\infty} -\int_{-\infty}^{\infty} f(x) \, g'(x) \, dx \\
\end{align*}
$$

Assume $\lim_{\|x\| \to \infty} f(x)g(x) = 0$ (boundary terms vanish), then

$$
\int f'(x) g(x) \, dx = -\int f(x) \, g'(x) \, dx
$$

We can do the same with the double derivatives as well:

$$
\begin{align*}
\frac{d}{dx} \int f'(x)\, g(x) dx & = \int \frac{d}{dx} \left( f'(x)\, g(x) \right) dx \\
f'(x) g(x) \big|_{-\infty}^{\infty} & =  \int f''(x) g(x) \, dx + \int f'(x) \, g'(x) \, dx \\
\end{align*}
$$

Assume $\lim_{\|x\| \to \infty} f'(x)g(x) = 0$, then,

$$
\int f''(x) g(x) \, dx = - \int f'(x) \, g'(x) \, dx
$$

Similarly,

$$
\begin{align*}
\frac{d}{dx} \int f(x)\, g'(x) dx & = \int \frac{d}{dx} \left( f(x)\, g'(x) \right) dx \\
f(x) g'(x) \big|_{-\infty}^{\infty} & =  \int f'(x) g'(x) \, dx + \int f(x) \, g''(x) \, dx \\
\end{align*}
$$

Assume $\lim_{\|x\| \to \infty} f(x)g'(x) = 0$, then,

$$
\int f'(x) g'(x) \, dx = - \int f(x) \, g''(x) \, dx
$$

On comparing this with the previous case, we see

$$
\int f''(x) g(x) \, dx = \int f(x) \, g''(x) \, dx
$$

In such cases, it doesn't matter if we take the derivative of the first function or the second function to compute the integration.

Assume

$$
\lim_{\|x\| \to \infty} \nabla f(x)^\top \, v_t(x) p_t(x) = 0
$$

So we can apply the integration by parts technique to get

$$
\int \nabla f(x)^\top v_t(x) \, p_t(x) \, dx = - \int f(x) \nabla \cdot (v_t(x) \, p_t(x)) \, dx
$$

**Difference between grad and Div?**

* Gradient: $\nabla f(x) \in \mathbb{R}^d$. Gradient acts on a scalar $f$.
* Divergence

$$
\nabla \cdot g(x) = \sum_{i=1}^d \frac{\partial g_i(x)}{\partial x_i}
$$

for $g:\mathbb{R}^d \to \mathbb{R}^d$. Divergence acts on a vector field. The result from the divergence operator is a scalar.

**How grad in LHS changes to Div in RHS?**

<figure markdown="0" class="figure zoomable">
<img src='./images/grad_to_div.png' alt="Gradient to Div"><figcaption>
  <strong>Figure 2.</strong> How grad in LHS changes to Div in RHS
  </figcaption>
</figure>

So, our equation becomes:

$$
\begin{align*}
\int f(x) \frac{\partial p_t(x)}{\partial t} dx & = - \int f(x) \nabla \cdot (v_t(x) \, p_t(x)) \, dx 
\end{align*}
$$

This is true for all $f$, then it should be if and only if

$$
\frac{\partial p_t(x)}{\partial t} = -\nabla \cdot (v_t(x) \, p_t(x)) \hspace{1cm} \forall x
$$

Given the vector field $v_t$, this ODE tells us how the likelihood flows. We can solve this to find $p_t$. Here, $\nabla \cdot$ denotes the divergence operator, which measures how much the vector field is expanding or contracting at a point.
