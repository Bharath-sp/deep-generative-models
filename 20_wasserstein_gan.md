---
layout: post
title: Wasserstein GAN
categories: [Generative AI]
toc: true
---

* TOC
{:toc}

## Introduction
The dual form of the optimal transport problem is:

$$
\begin{align*}
& \max_{r,s \, \in \, \mathcal{C}(\mathcal{X})} \left[ \int r(x)\, p(x) \, dx + \int s(y)\, p^*(y) \, dy \right] \\
& \hspace{0.5cm} \text{s.t.  } r(x) + s(y) \leq  c(x,y) \,\, \forall x,y \in \mathcal{X}
\end{align*}
$$

Right now, we have uncountably infinite constraints:

$$
r(x) + s(y) \leq  c(x,y) \,\, \forall (x,y)
$$

This should be satisfied for all $(x,y)$. So, this is extremely challenging. When there are a lot of constraints, there is a ubiquitous way of reducing the constraints to one or fewer constraints. For example, say we have 3 inequality constraints $x_1 < 5, x_1 < 6$, and $x_1 < 2$. These 3 constraints can be reduced to one inequality constraint as $x_1 < \min(5,6,2)$. Similarly,

$$
\begin{align*}
r(x) + s(y) & \leq  c(x,y) \,\, \forall (x,y) \\
s(y) & \leq  c(x,y) - r(x) \,\, \forall (x,y)
\end{align*}
$$

can be written as

$$
\begin{align*}
s(y) & \leq  \min_x \{c(x,y) - r(x)\} \,\, \forall y \\
\text{or}\\
s(y) & \leq  - \max_x \{-c(x,y) - (- r(x))\} \,\, \forall y
\end{align*}
$$

The number of constraints is reduced to one Euclidean space, but still uncountably infinite. Let's look at some special cases of the cost function $c$, and observe how the constraints get simplified.

## When $c$ is any Distance
Let's assume $x$ and $y$ are Euclidean vectors. In general, the cost function $c$ can be any function, but let's consider $c$ to be any valid distance, i.e., $c(x,y)=d(x,y)$ is the distance between $x$ and $y$, i.e., it can be any distance such as L1, L2 norm, etc. (but not the squared distance). Then, the optimization problem in the constraint is

$$
\min_x \{d(x,y) - r(x)\} \equiv \bar{r}(y) \,\, \forall y
$$

At optimality, the objective value will be a function of $y$, let's call this as $\bar{r}(y)$. Our goal now is to characterize $\bar{r}$. What can we say about this function $\bar{r}$?.

**$\bar{r}(y)$ is 1-Lipschitz:**

Suppose $y$ changes a little bit to $y'$, we observe that $\bar{r}(y')$ wouldn't be very much from $\bar{r}(y)$. This is because $d$ is a valid distance, so $d(x,y) \approx d(x,y')$, and the second term doesn't involve $y$.

* The arg min of the problem $\{d(x,y) - r(x)\}$ will be a $x$ that is closer to $y$, as our goal is to minimize the objective.
* Similarly, the arg min of the problem $\{d(x,y') - r(x)\}$ will be some $x$ that is closer to $y'$ (but $y'$ is closer to $y$). So, $\bar{r}(y) \approx \bar{r}(y')$.

This leads to the intuition that $\bar{r}(y)$ is **1-Lipschitz** continuous under $d$. Here $d$ is the same distance that we are working with.

$$
|\bar{r}(y) - \bar{r}(y')| \leq d(y,y') \,\, \forall y,y' \in \text{dom}(f) \\
$$

**Proof:**

$$
\begin{align*}
\left|\min_x \{d(x,y) - r(x)\} - \min_{x'} \{d(x',y') - r(x')\}\right| & \leq d(y,y')\\
\end{align*}
$$

Let $x^*$ be the minimizer of the second optimization problem term. For any $x^*$, we know for the first term that:

$$
\min_x \{d(x,y) - r(x)\} \leq \{d(x^*,y) - r(x^*)\}
$$

The value we get for the first term by substituting $x^*$ will be

* Equal to $\min_x \{d(x,y) - r(x)\}$ if $x^*$ is the minimizer for the first term as well.
* Greater than $\min_x \{d(x,y) - r(x)\}$ if $x^*$ is not the minimizer. So, we can always write:

$$
\begin{align*}
& \left|\min_x \{d(x,y) - r(x)\} - \min_{x'} \{d(x',y') - r(x')\}\right| \\
& \leq \left|d(x^*,y) - r(x^*) - d(x^*,y') + r(x^*)\right| \\
& = \left|d(x^*,y) - d(x^*,y') \right| \\
& = \left|d(y, x^*) - d(x^*,y') \right| \\
& \leq  d(y,y')\\
\end{align*}
$$

by reverse triangle inequality. Thus the proof.

**$r(x)$ and $s(y)$ are 1-Lipschitz:**

The dual of the OT problem is now:

$$
\begin{align*}
& \max_{r,s} \left[ \mathbb{E}_{X \sim p}[r(X)] + \mathbb{E}_{Y \sim p^*}[s(Y)]\right] \\
& \hspace{0.5cm} \text{s.t.  } \, s(y) \leq  \bar{r}(y) \,\, \forall y
\end{align*}
$$

where $\bar{r}(y)$ is a 1-Lipschitz continuous function. The maximization is based on two variables $r$ and $s$. Let's fix the function $r(x)$, then $\bar{r}(y)$ which is also based on $r$ is fixed. The constraint is to have the function $s(y) \leq  \bar{r}(y)$ for all $y$. We are maximizing $\mathbb{E}[s(Y)]$, but from the constraint, $s(y)$ is upper bounded by $\bar{r}(y)$. Then, at optimality, the function $s(y)$ must be equal to $\bar{r}(y) \,\, \forall y$. Therefore, the problem can be reduced to:

$$
\begin{align*}
& \max_{r} \left[ \mathbb{E}_{X \sim p}[r(X)] + \mathbb{E}_{Y \sim p^*}[\bar{r}(Y)]\right] \\
\end{align*}
$$

The optimization is now wrt only one function $r$, and there are no explicit constraints. Note that $\bar{r}$ is a function of $r$.

At optimality, $s(y) = \bar{r}(y)$, which means $s(y)$ is 1-Lipschitz continuous at optimality.

Similarly, we could have done the same argument starting with the constraint

$$
r(x) \leq  c(x,y) - s(y) \,\, \forall (x,y) \implies
r(x) \leq  \min_y \{c(x,y) - s(y)\} \,\, \forall x
$$

and ended up with $r$ is 1-Lipschitz continuous at optimality. This proves that both $r$ and $s$ are 1-Lipschitz at optimality. Then, in our optimization, we can restrict ourselves to the space of 1-Lipschitz continuous function:

$$
\begin{align*}
& \max_{\|r\|_d \leq 1} \left[ \mathbb{E}_{X \sim p}[r(X)] + \mathbb{E}_{Y \sim p^*}[\bar{r}(Y)]\right] \\
\end{align*}
$$

Note $\|r\|_d \leq 1$ is read as norm of $r$ with respect to $d$ is less than or equal to 1. It means that the Lipschitz continuity constant of $r$ wrt $d$ is less than or equal to 1. The optimality happens $\|r\|_d=1$. So ideally, we can put $\|r\|_d=1$, but the set of functions in our search space won't be a convex set. So, for technical reason, we put $\|r\|_d \leq 1$. We are restricting our search space to the set of all functions that are 1-Lipschitz wrt. distance $d$.

**$\bar{r}(y) = -r(y)$ at optimality:**

We know that

<a name="eq:eq1"></a>
$$
\min_x \{d(x,y) - r(x)\} \equiv \bar{r}(y) \,\, \forall y \tag{1}
$$

In the LHS, we minimize over $x$. Suppose $x=y$, then (the below inequality is using the same argument that we used above in the proof).

$$
\begin{align*}
& \min_x \{d(x,y) - r(x)\} \leq d(y,y) - r(y) \\
& \implies \min_x \{d(x,y) - r(x)\} \leq - r(y) \\
& \implies \bar{r}(y) \leq - r(y)
\end{align*}
$$

This is true in general regardless of $r$ is 1-Lipschitz or not. But when $r(x)$ is 1-Lipschitz, that is,

$$
\begin{align*}
|r(x) - r(y)| & \leq d(x,y) \\
-d(x,y) \leq r(x) - r(y) & \leq d(x,y) \\
d(x,y) \geq -(r(x) - r(y)) & \geq -d(x,y) \\
\end{align*}
$$

for all $x,y$ in the domain of $r$, then <a href="#eq:eq1">(1)</a> becomes:

$$
\begin{align*}
& \min_x \{d(x,y) - (r(x) - r(y))\} = \bar{r}(y) + r(y) \\
\end{align*}
$$

The LHS is:

$$
\begin{align*}
\min_x \{d(x,y) - (r(x) - r(y))\} & \geq \min_x \{d(x,y) - d(x,y)\} \\
& \geq 0
\end{align*}
$$

which gives $\bar{r}(y) + r(y) \geq 0 \implies \bar{r}(y) \geq -r(y)$. We have both

* $\bar{r}(y) \leq -r(y)$ and 
* $\bar{r}(y) \geq -r(y)$

Then, it must be $\bar{r}(y) = -r(y)$. Thus, if $r$ is 1-Lipschitz, then at optimality, $\bar{r} = -r$. Then, our objective can be written as:

$$
\begin{align*}
& \max_{\|r\|_d \leq 1} \left[ \mathbb{E}_{X \sim p}[r(X)] - \mathbb{E}_{Y \sim p^*}[r(Y)]\right] \\
\end{align*}
$$

We are able to write in this form only because $r$ is 1-Lipschitz.

### Wasserstein Metric

If the cost function is a valid distance $d$ over data space $\mathcal{X}$, then the dual of the optimal transport problem simplifies to:

$$
\max_{\|r\|_d \leq 1} \hspace{0.5cm}  \mathbb{E}_{X \sim p}[r(X)] - \mathbb{E}_{Y \sim p^*}[r(Y)] \equiv W_1(p, p^*)
$$

where $\|r\|_d \leq 1$ is the set of all functions that are 1-Lipschitz wrt. distance $d$. At optimality, that is, the value of the objective with the maximizer $r$, turns out to be a valid distance between the distribution $p$ and $p^*$. It implies that the value of the optimal transport problem's objective defines a (new) valid distance, and this is called as **1-Wasserstein distance**. $W_1(p,p^*)$ satisfies all the properties of a distance.

1-Wasserstein helps us measure the distance between two distributions by comparing their moments. 1-Wasserstein is actually the worst-case discrepancy / deviation between the moments with respect to the given likelihoods, where the moments are computed over the set of 1-Lipschitz continuous functions.

Consider a function, say $r_1(x)=x$ (identity function), and check the deviation. Consider $r_2(x)=x^2$, and check the deviation. Repeat it for all such functions. Then, we choose that 1-Lipschitz function that induces the largest deviation between the moments with respect to the given likelihoods. More informally, among few witnesses $\{r \, | \, \|r\|_d \leq 1\}$, the one that provides the most significant evidence (critical witness) that the likelihoods are different is considered. The function $r$ that maximizes teh deviation is called as **adversary**. And, this corresponding worst-case deviation happens to be a valid metric, and is called the 1-Wasserstein distance.

The Wasserstein distance takes the distance between the data points $x$ and $y$ as $c(x,y)$ for each $x$ and $y$, and gives the corresponding distance between the distributions $p(x)$ and $p^*(y)$. The Wasserstein distance also implies the cost of the optimal transport plan to transform the source distribution $p$ into the target distribution $p^*$. If $p$ and $p^*$ are far, the value of this quantity will be more, and vice-versa.

<div class="admonition tip">
  <p class="admonition-title">TIP</p>
  <p> If we know how to measure the distance between points, WD tells us how to measure the distance between distributions of those points that respects the underlying distance between points.
  </p>
</div>

Since this is a distance, we can also write it as:

$$
W_1(p^*, p) = \max_{\|r\|_d \leq 1} \left[ \mathbb{E}_{Y \sim p^*}[r(Y)] - \mathbb{E}_{X \sim p}[r(X)] \right]
$$

because $W_1(p^*, p) = W_1(p, p^*)$ by the symmetry property of the distance. This can also be written as:

$$
\begin{align*}
W_1(p^*, p) & = \max_{\|r\|_d \leq 1} \left[ \int_{S_y} r(y) \, p^*(y) \, dy - \int_{S_x} r(x) \, p(x) \, dx \right] \\
& = \max_{\|r\|_d \leq 1} \left[ \int_{S_y \cup S_x} r(x) \, p^*(x) \, dx - \int_{S_y \cup S_x} r(x) \, p(x) \, dx \right] \\
& = \max_{\|r\|_d \leq 1} \left[ \int_{S_y \cup S_x} r(x) \, (p^*(x) - p(x))\, dx \right] \\
\end{align*}
$$

Step 2: $x$ is just a dummy variable covering the support of both the distributions.

Wasserstein explicitly involves the underlying distance over data points. So employing it as a loss is expected to lead to improved generation.

### Implicit Generation with WD
We can minimize the 1-Wasserstein distance between $p$ and $p^*$ instead of the KL divergence. Then, the objective of the implicit generative modelling becomes:

$$
\min_{\theta} W_1(p^*, p) \hspace{0.5cm} \text{ such that } g_{\theta}(X_0) \sim p
$$

On substituting the expression for WD:

$$
\begin{align*}
& \min_{\theta} \min_{\pi} \int \int d(x,y) \cdot \pi(x,y) \, dx \, dy \\
& \text{s.t.} \int \pi(x,y) \, dy =  p(x) \,\,\, \forall x \\
& \hspace{0.5cm} \int \pi(x,y)\, dx = p^*(y) \,\,\, \forall y \\
& \hspace{0.5cm} \pi(x,y) \geq 0 \,\,\, \forall (x,y), \,\, g_{\theta}(X_0) \sim p
\end{align*}
$$

This is the primal form of the problem. Using the duality result and if $c$ is a valid distance over the data space, we get:

$$
\begin{align*}
& \min_{\theta} \max_{\|r\|_d \leq 1} \left[ \mathbb{E}_{Y \sim p^*}[r(Y)] - \mathbb{E}_{X \sim p}[r(X)] \right] \,\, \text{ s.t. } \,\, g_{\theta}(X_0) \sim p
\end{align*}
$$

where $X$ is the random variable following $p$ and $Y$ is the RV following $p^*$. On eliminating $p$:

$$
\min_{\theta} \max_{\|r\|_d \leq 1} \left[ \mathbb{E}_{Y \sim p^*}[r(Y)] - \mathbb{E}_{X \sim g_{\theta}(X_0)}[r(X)] \right]
$$

The objective is to make the generator $g_{\theta}$ pass this critical witness $r^*$. If we pass our generator pass this, then the sample generated by the generator are close to $p^*$. 

We can use neural network to model $r$ parameterized by $\phi$ along with the constraint imposed on the weights to ensure that the function is 1-Lipschitz. Then,

$$
\begin{align*}
& \min_{\theta} \max_{\phi: \| r_{\phi} \|_d \leq 1} \left[ \mathbb{E}_{Y \sim p^*}[r_{\phi}(Y)] - \mathbb{E}_{X_0 \sim p_0}[r_{\phi}(g_{\theta}(X_0))] \right] \\
& \approx \min_{\theta} \max_{\phi} \frac{1}{m} \sum_i r_{\phi}(y_i) - \frac{1}{m} \sum_i r_{\phi}(g_{\theta}(x_i)) \\
\end{align*}
$$

* The parameters $\phi$ should be constrained to enforce the 1-Lipschitz condition on the function $r_{\phi}$ model by the network. Typically, we choose a constant $u$ (the clipping parameter), and enforce that each parameter of the critic network lies in the range $-u \leq \phi_i \leq u$.
* In addition, and more critically, we should also ensure that we explore all the 1-Lipschitz continuous functions. This is usually achieved by widening or deepening the network, so that the network (universal approximator) can cover all the 1-Lipschitz functions.

We can solve this optimization problem, which is of the form $\min_{\theta} L(\theta)$, using SGD. To find the gradient, $\nabla_{\theta} L(\theta)$, we need to solve the inner maximization (optimal transport) problem, which can be solved using the Danskin's theorem. Given a value of $\theta$, we solve the inner maximization problem, i.e., it is the same as computing the 1-Wasserstein distance between $p$ and $p^*$. We compute this distance at every iteration of SGD. But in practice, we don't solve it to optimality, for each outer iteration, we run only 1-10 inner iterations.

The discriminator is trained to estimate the 1-Wasserstein distance between $p$ and $p^*$, and the generator is trained to minimize this distance to make $p$ similar to $p^*$. The same objective the generator tries to minimize, and the critic tries to maximize. For this reason, the critic network is called as adversarial network.

Solving the implicit generation problem with 1-Wasserstein as the distance is called as **Wasserstein GAN** (WGAN).

## Types of Wasserstein Distance

* When we consider the cost function $c(x,y)$ as the distance between $x$ and $y$, that is $d$, we get 1-Wasserstein distance $W_1$ as the output.

* When we consider the cost function as the squared distance between $x$ and $y$, that is $d^2$, we get 2-Wasserstein distance raised to the power of 2, that is $W^2_2$.

$$ 
W_2^2(p,q) \equiv \min_{\pi} \mathbb{E}_{X,Y \sim \pi} [c(X,Y)^2]
$$


* When we consider the cost function as the distance between $x$ and $y$ to the power of $k$, that is $d^k$, we get $k$-Wasserstein distance raised to the power of $k$, that is $W^k_k$.

$$ 
W_k^k(p,q) \equiv \min_{\pi} \mathbb{E}_{X,Y \sim \pi} [c(X,Y)^k]
$$

So, there is a distance defined for every power of the base distance. Wasserstein-2 distance is the most commonly used one.











