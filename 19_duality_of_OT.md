---
layout: post
title: Duality of the optimal Transport Problem
categories: [Generative AI]
toc: true
---

* TOC
{:toc}

## Duality of the OT problem
One of the key ideas in Lagrange duality is to rewrite the constraints as terms in the objective. The objective of the optimal transport problem is:

$$
\begin{align*}
& \min_{\pi} \int \int c(x,y) \cdot \pi(x,y) \, dx \, dy \\
& \text{s.t.} \int \pi(x,y) \, dy = \pi_1(x) =  p(x) \,\,\, \forall x \\
& \hspace{0.5cm} \int \pi(x,y)\, dx = \pi_2(y) = q(y) \,\,\, \forall y \\
& \hspace{0.5cm} \pi(x,y) \geq 0 \,\,\, \forall x,y
\end{align*}
$$

Note here that the cost function $c$ can be any function. The objective can be rewritten as:

<a name="eq:eq1"></a>
$$
\begin{equation*}
\min_{\pi \geq 0} \left[ \begin{aligned} \int \int & c(x,y) \cdot \pi(x,y) \, dx \, dy \\
& + \max_r \int r(x) (\pi_1(x) - p(x)) \, dx \\
& + \max_s \int s(y) \, (\pi_2(y) - p^*(y)) \, dy \end{aligned} \right] \tag{1}
\end{equation*}
$$

where $r$ and $s$ can be any function.

* The term $\max_r \int r(x) (\pi_1(x) - p(x)) \, dx $ is an equivalent way of saying $\pi_1 =  p(x)$.

$$
\max_r \int r(x) (\pi_1(x) - p(x)) \, dx = \begin{cases} 0 & \text{if } \pi_1(x) = p(x) \, \, \forall x \in \mathcal{X} \\
\infty & \text{if } \pi_1(x) \ne p(x) \, \, \exists x \in \mathcal{X} \\
\end{cases}
$$

If there exists an $x$ for which $\pi_1(x) \ne p(x)$, say $\pi_1(x) - p(x) =3$ or $-3$, and the difference is 0 at all $x$s. Then, it boils down to:

$$
\max_r \int r(x) (\pi_1(x) - p(x)) \, dx = 3 \cdot \max_r r(x) \, dx
$$

$r$ can be pushed to $\infty$ and the maximum value will be $\infty$. Or,

$$
\max_r \int r(x) (\pi_1(x) - p(x)) \, dx = -3 \cdot \max_r r(x) \, dx
$$

$r$ can be pushed to $-\infty$ and the maximum value will still be $\infty$.

Our objective is to minimize <a href="#eq:eq1">(1)</a> with respect to $\pi$. If its marginal $\pi_1 \ne p$, then the second term will blow to $\infty$. Only for $\pi_1 = p$, the objective will be finite. Any finite number is always better than $\infty$ while minimizing. So, it is an equivalent way of saying $\pi$ should be $p$.

Similarly, the term $\max_s \int s(y) \, (\pi_2(y) - p^*(y)) \, dy$ is an equivalent way of saying $\pi_2 =  p^*(x)$.

We can write:

$$
\begin{align*}
\max_r \int r(x) (\pi_1(x) - p(x)) \, dx & = \max_r \int r(x) \left(\int \pi(x,y) \, dy - p(x) \right) \, dx \\
& = \max_r \left[ \int \int r(x) \, \pi(x,y) \, dy \, dx - \int r(x)\, p(x) \, dx \right]
\end{align*}
$$

Similarly,

$$
\begin{align*}
\max_s \int s(y) (\pi_2(y) - p^*(y)) \, dy & = \max_s \int s(y) \left(\int \pi(x,y) \, dx - p^*(y) \right) \, dy \\
& = \max_s  \left[ \int \int s(y) \, \pi(x,y) \, dx \, dy - \int s(y)\, p^*(y) \, dy \right]
\end{align*}
$$

The objective in <a href="#eq:eq1">(1)</a> becomes:

$$
\begin{align*}
& \min_{\pi \geq 0} \left[ \begin{aligned} \int \int & c(x,y) \cdot \pi(x,y) \, dx \, dy \\
& + \max_r \left[ \int \int r(x) \, \pi(x,y) \, dy \, dx - \int r(x)\, p(x) \, dx \right] \\
& + \max_s  \left[ \int \int s(y) \, \pi(x,y) \, dx \, dy - \int s(y)\, p^*(y) \, dy \right] \end{aligned} \right] \\
\\
& = \min_{\pi \geq 0} \max_{r,s} \left[ \begin{aligned} \int \int & c(x,y) \cdot \pi(x,y) \, dx \, dy \\
& + \int \int r(x) \, \pi(x,y) \, dy \, dx - \int r(x)\, p(x) \, dx \\
& + \int \int s(y) \, \pi(x,y) \, dx \, dy - \int s(y)\, p^*(y) \, dy \end{aligned} \right] \\
\\
& = \min_{\pi \geq 0} \max_{r,s} \left[ \begin{aligned} \int \int & \left( c(x,y) + r(x) + s(y) \right) \cdot \pi(x,y) \, dx \, dy \\
& - \int r(x)\, p(x) \, dx - \int s(y)\, p^*(y) \, dy \end{aligned} \right] \\
\end{align*}
$$

On exchanging the min and max, we get:

$$
\max_{r,s}  \left[ \min_{\pi \geq 0} \left[ \int \int \left( c(x,y) + r(x) + s(y) \right) \cdot \pi(x,y) \, dx \, dy \right] - \int r(x)\, p(x) \, dx - \int s(y)\, p^*(y) \, dy \right]
$$

Let's focus on the term $\min_{\pi \geq 0} \left[ \int \int \left( c(x,y) + r(x) + s(y) \right) \cdot \pi(x,y) \, dx \, dy \right]$.

At any $(x,y)$, we must have the density $\pi(x,y) \geq 0$, then

* When $c(x,y) + r(x) + s(y) \geq 0$, then we can make $\pi(x,y)=0$ for that $(x,y)$, then the minimum value of the term will be 0.
* When $c(x,y) + r(x) + s(y) < 0$, then we can blow $\pi(x,y)$ to a large value which gives us a minimum value of $-\infty$.

$$
\min_{\pi \geq 0} \left[ \int \int \left( c(x,y) + r(x) + s(y) \right) \cdot \pi(x,y) \, dx \, dy \right] = \begin{cases} 0 & \text{if } c(x,y) + r(x) + s(y) \geq 0 \, \, \forall (x,y) \\
-\infty & \text{if } c(x,y) + r(x) + s(y) < 0 \, \, \text{ for any } (x,y) \\
\end{cases}
$$

But we are looking for a maximum value with respect to $r,s$. Then, we can restrict our search in the space of

$$
c(x,y) + r(x) + s(y) \geq 0 \,\, \forall (x,y)
$$

When this condition is satisfied, the value of the term $\min_{\pi \geq 0} \left[ \int \int \left( c(x,y) + r(x) + s(y) \right) \cdot \pi(x,y) \, dx \, dy \right]$ will be 0 by making $\pi(x,y)=0$ for all $(x,y)$. Then, the terms in the objective can be brought down to the constraints and rewritten as:

$$
\begin{align*}
& \max_{r,s} \left[ - \int r(x)\, p(x) \, dx - \int s(y)\, p^*(y) \, dy \right] \\
& \hspace{0.5cm} \text{s.t.  } \, c(x,y) + r(x) + s(y) \geq 0
\end{align*}
$$

$r$ and $s$ can be any function, so on absorbing the negative sign:

<a name="eq:eq2"></a>
$$
\begin{align*}
& \max_{r,s} \left[ \int r(x)\, p(x) \, dx + \int s(y)\, p^*(y) \, dy \right] \\
& \hspace{0.5cm} \text{s.t.  } \, c(x,y) - r(x) - s(y) \geq 0 \implies r(x) + s(y) \leq  c(x,y) \,\, \forall (x,y) \tag{2}
\end{align*}
$$

This is the dual form of the optimal transport problem. This optimization problem gives a completely different perspective to the original problem. Note here that the primal variable $\pi$ is not involved. Now, the variables of optimization are $r,s$ which are called as dual variables.

## Kantorovich Duality
If $c$ is continuous, then the optimal transport problem is the same as:

$$
\begin{align*}
& \max_{r,s \, \in \, \mathcal{C}(\mathcal{X})} \left[ \int r(x)\, p(x) \, dx + \int s(y)\, p^*(y) \, dy \right] \\
& \hspace{0.5cm} \text{s.t.  } r(x) + s(y) \leq  c(x,y) \,\, \forall x,y \in \mathcal{X}
\end{align*}
$$

where $\mathcal{X}$ denotes the data space and $\mathcal{C}(\mathcal{X})$ is the set of all possible functions in the data space. The objective can also be written as:

$$
\begin{align*}
& \max_{r,s \, \in \, \mathcal{C}(\mathcal{X})}  \hspace{0.5cm} \mathbb{E}_{X \sim p}[r(X)] + \mathbb{E}_{Y \sim p^*}[s(Y)] \\
& \hspace{0.5cm} \text{s.t.  } r(x) + s(y) \leq  c(x,y) \,\, \forall x,y \in \mathcal{X}
\end{align*}
$$








