---
layout: post
title: IPM or Dual Norms
categories: [Generative AI]
toc: true
---

* TOC
{:toc}

## IPM or Dual Norms
When $c$ is a valid distance, the dual form of the optimal transport problem is:

$$
\begin{align*}
W(s,t)  \equiv \max_{\|r\|_d \leq 1} \left[ \mathbb{E}_{X \sim s}[r(X)] - \mathbb{E}_{Y \sim t}[r(Y)]\right] \\
\end{align*}
$$

where $r$ is 1-Lipschitz. And this is called as Kontorovich or Wasserstein metric.

Infact, we can consider the more general family, popularly known as **Integral Probability Metrics (IPMs) or dual norms**.

<a name="eq:eq1"></a>
$$
\begin{align*}
\text{IPM}_{\mathcal{G}}(s,t) \equiv \max_{r \in \mathcal{G}} \left[ \mathbb{E}_{X \sim s}[r(X)] - \mathbb{E}_{Y \sim t}[r(Y)]\right] \tag{1}\\
\end{align*}
$$

where IPM is a function of two distributions, and $\mathcal{G}$ is the so-called generating set (pool of witnesses). In <a href="#eq:eq1">(1)</a>, we are essentially comparing the moments of two distributions, and the solution (the IPM) is the worst-case deviation between moments across some functions. In simple words, it is matching moments where moments are defined by some functions. We can turn the IPM into a metric by considering appropriate functions in the set $\mathcal{G}$.

<div class="admonition note">
  <p class="admonition-title">NOTE</p>
  <p> If the set $\mathcal{G}$ has only the identity function, then IPM is simply $\text{IPM}_{\mathcal{G}}(s,t) = \mathbb{E}_{X \sim s}[X] - \mathbb{E}_{Y \sim t}[Y]$. Here we are just comparing the mean of the distributions.
  </p>
</div>

If we choose the set $\mathcal{G}$ badly, i.e., the included functions don't characterize a distribution (help us separate the distributions), then the corresponding IPM will not be a metric.

Let's consider $\mathcal{G}$ to be a symmetric set, that is, if $r \in \mathcal{G} \implies -r \in \mathcal{G}$. Under this constraint on $\mathcal{G}$, it can be shown that all IPMs are **almost** a metric (distance) in the space of probability distributions.

To prove that IPM is a distance, we should show that <a href="#eq:eq1">(1)</a> satisfies:

1. Non-negativity
2. Positive definiteness
3. Symmetry
4. Triangle inequality

IPM satisfies all these properties except the positive-definiteness.

**Non-negativity:**

IPM is always $\geq 0$.

At optimality, say $r^*$ is the chosen function, and suppose the quantity $\mathbb{E}_{X \sim s}[r^*(X)] - \mathbb{E}_{Y \sim t}[r^*(Y)]$ is negative. Say $5-10= -5$ Then, we can take $-r^*$, which gives $-5+10 = 5$. As we are maximizing the objective, -5 cannot be the max to this problem, it should be 5.

For every negative value of the objective, we can get a positive value of the same magnitude. And as we are maximizing, we always choose the positive value. At the worst case the maximum value will be 0. Thus, the value of the objective will always be $\geq 0$. 

**Positive definiteness:**

IPM is 0 if and only if $s=t$

Given $s=t$, we see that

$$
\max_{r \in \mathcal{G}} \left[ \mathbb{E}_{X \sim s}[r(X)] - \mathbb{E}_{Y \sim s}[r(Y)]\right]
$$

the IPM is 0. But how can we prove that given IPM is 0, then $s=t$?

In general, given IPM is zero, that is, the expectations $\mathbb{E}_{X \sim s}[r^*(X)] = \mathbb{E}_{Y \sim t}[r^*(Y)]$. Here we are matching the moments of the distributions over all functions in $\mathcal{G}$, and it is given that the worst-case deviation between the moments is zero. It doesn't necessarily mean that the distributions are the same, $s=t$, unless $\mathcal{G}$ is a special set.

If we want IPM to be positive-definiteness, we should put more constraints on the generating set $\mathcal{G}$. We should include those functions in $\mathcal{G}$ that characterize the distributions, i.e., their moments being the same implies that the distributions themselves are the same. Unless we put some non-trivial conditions such as 1-Lipschitz continuous functions, we cannot have positive-definiteness. 

<div class="admonition tip">
  <p class="admonition-title">TIP</p>
  <p> The set $\mathcal{G}$ is called as a generating set because given a set of functions, it generates a new distance over probability distributions.
  </p>
</div>

**Symmetry:**

$\text{IPM}(s,t) = \text{IPM}(t,s)$

If $\text{IPM}(s,t)$ is optimized at $r^*$:

$$
\begin{align*}
\text{IPM}_{\mathcal{G}}(s,t)& =  \mathbb{E}_{X \sim s}[r^*(X)] - \mathbb{E}_{Y \sim t}[r^*(Y)]\\
& = 10 - 2 = 8 \, \, (\text{say})
\end{align*}
$$

Then,

$$
\mathbb{E}_{Y \sim t}[r^*(Y)]  - \mathbb{E}_{X \sim s}[r^*(X)] = 2 - 10 = -8
$$

This can't be the maximum, i.e., this can't be the value of $\text{IPM}_{\mathcal{G}}(t,s)$. The maximum will occur at $-r^*$:

$$
\begin{align*}
\text{IPM}_{\mathcal{G}}(t,s) & = \max_{r \in \mathcal{G}} \left[\mathbb{E}_{Y \sim t}[r(Y)] - \mathbb{E}_{X \sim s}[r(X)] \right] \\
& = \mathbb{E}_{Y \sim t}[-r^*(Y)] - \mathbb{E}_{X \sim s}[-r^*(X)] \\ 
& =   \mathbb{E}_{X \sim s}[r^*(X)] - \mathbb{E}_{Y \sim t}[r^*(Y)] \\
& = \text{IPM}_{\mathcal{G}}(s,t)
\end{align*}
$$

Thus, $\text{IPM}(s,t)$ is symmetric.

**Triangle Inequality:**

$$
\begin{align*}
& \text{IPM}(s,u) + \text{IPM}(u,t) \\
& = \max_{r \in \mathcal{G}} \left[ \mathbb{E}_{X \sim s}[r(X)] - \mathbb{E}_{Y \sim u}[r(Y)]\right] + \max_{f \in \mathcal{G}} \left[ \mathbb{E}_{Y \sim u}[f(Y)] - \mathbb{E}_{Z \sim t}[f(Z)]\right] \\
\\
& = \max_{r \in \mathcal{G}} \max_{f \in \mathcal{G}} \left[ \mathbb{E}_{X \sim s}[r(X)] - \mathbb{E}_{Y \sim u}[r(Y)] +  \mathbb{E}_{Y \sim u}[f(Y)] - \mathbb{E}_{Z \sim t}[f(Z)]\right]
\end{align*}
$$

Now, let's put a constraint that $r=f$. Earlier we were allowed to move freely in the entire space, now with this constraint, we are allowed to move only in a restricted space. So, our result may be sub-optimal or optimal.

$$
\begin{align*}
& \geq \max_{r \in \mathcal{G}} \left[ \mathbb{E}_{X \sim s}[r(X)] - \mathbb{E}_{Y \sim u}[r(Y)] +  \mathbb{E}_{Y \sim u}[r(Y)] - \mathbb{E}_{Z \sim t}[r(Z)]\right] \\
& = \max_{r \in \mathcal{G}} \left[ \mathbb{E}_{X \sim s}[r(X)] - \mathbb{E}_{Z \sim t}[r(Z)]\right] \\
& = \text{IPM}(s,t)
\end{align*}
$$

Thus, $\text{IPM}(s,u) + \text{IPM}(u,t) \geq \text{IPM}(s,t)$, the triangle inequality is proved.

In other words, as long as the generating set is symmetric, that is, if $r \in \mathcal{G} \implies -r \in \mathcal{G}$, all IPMs are psuedo metrics. The only property needed to turn them into a full metric is positive-definiteness. Clearly, non-trivial conditions of $\mathcal{G}$ are necessary for IPM being positive-definite.  

Different non-trivial conditions on $\mathcal{G}$ in <a href="#eq:eq1">(1)</a> gives different (full) metrics (distances). In addition, these are not only distances, they all are in fact, more special, **norm-based distances** (thus the name dual norm). Then, we can get different implicit generators corresponding to each metric.

### 1-Wasserstein Metric
If $\mathcal{G}$ is the set of all 1-Lipschitz continuous functions, then we get the 1-Wasserstein or Kontorovich metric. It can be shown that 1-Wasserstein satisfies all the properties of distance.

**Non-negativity**:
The set of 1-Lipschitz functions is a symmetric set, that is, if $r$ is 1-Lipschitz, $-r$ is also 1-Lipschitz. So, $W_1(s,t)$ is always $\geq 0$. We could have also proved $W_1(s,t) \geq 0$ from the primal form easily. As $c$ is always $\geq 0$, then $\mathbb{E}[c(X,Y)]$ is always $\geq 0$.

<br>

**Positive definiteness**:
Given $s=t$, $W_1(s,t)=0$. This is evident from the dual form. To prove that if $W_1(s,t)=0$, then $s=t$, let's look at the primal form.

$$
\begin{align*}
W_1(s,t) = & \min_{\pi \geq 0} \int \int d(x,y) \cdot \pi(x,y) \, dx \, dy \\
& \text{s.t.} \int \pi(x,y) \, dy =  s(x) \,\,\, \forall x \\
& \hspace{0.5cm} \int \pi(x,y)\, dx = t(y) \,\,\, \forall y \\
\end{align*}
$$

Assume that the distributions $s$ and $t$ are non-zero in the entire space (like Gaussians). Then, their joint distribution can be zero for some $(x,y)$, but cannot be zero everywhere.

Given $W_1(s,t)=0$, that is, at optimality, $\int \int d(x,y) \cdot \pi^*(x,y) \, dx \, dy = 0$. Assume that at a particular point $(x',y')$, the density $\pi^*(x',y') \ne 0$, then $d(x', y')$ should be zero. As the cost is the distance here, it implies $x' = y'$.

* If $\pi^*(x,y)\ne0$, then $d(x,y)=0$ for any $x,y \implies x=y$.
* If $x\ne y \implies d(x,y)=0$. So, $\pi^*(x,y)$ should be 0.

This implies that, at optimality, $\pi^*$ can be non-zero only where $x=y$. So, the joint distribution should be diagonal.

$$
\begin{align*}
\int \pi(x,y) \, dy & =  \pi(x,x) = s(x) \,\,\, \forall x \\
\int \pi(x,y)\, dx & = \pi(y,y) = t(y) \,\,\, \forall y \\
s& =t
\end{align*}
$$

Therefore, we prove that the marginals should be the same, $s=t$. Positive definiteness is evident from the primal form. This gives us a theorem in dual that:

Matching moments of two distributions over all possible 1-Lipschitz continuous functions is enough to match the distributions themselves.

**Symmetry:**

$W_1(s,t) = W_1(t,s)$. This can be proved with the same argument as above for IPMs.

**Triangle Inequality:**

$W_1(s,u) + W_1(u,t) \geq W_1(s,t)$. This can also be proved with the same argument as above for IPMs.

Hence, we prove that 1-Wasserstein is indeed a full metric.

### Total Variation (TV)
If $\mathcal{G}$ is the set of all functions that are bounded, i.e., their values are between -1 and 1.

$$
\mathcal{G} \equiv \{r: \mathcal{X} \to \mathbb{R} \,\, \text{s.t. } |r(x)| \leq 1 \,\, \forall x \in \mathcal{X}\}
$$

The corresponding dual norm or IPM is a valid metric (it also satisfies the positive-definiteness property). And it is called as total variation distance (TVD).

An equivalent way of writing this generating set is:

$$
\mathcal{G} \equiv \{r: \mathcal{X} \to \mathbb{R} \,\, \text{s.t. } \|r\|_{\infty} \leq 1 \}
$$

where $\|r\|_{\infty} \equiv \max_{x \in \mathcal{X}} |r(x)|$.










