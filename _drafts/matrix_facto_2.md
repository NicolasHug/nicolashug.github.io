---
layout: post
title: Understanding matrix factorization for recommendation&#58; the model behind SVD (part 2)
categories: [general]
tags: [matrix factorization, PCA, SVD, recommender systems]
description:  TODO
comments: true
---

SVD of a (dense) rating matrix
==============================

Let's just recap a bit:

* PCA on the matrix $R$ will give us typical users. These typical users are of
  course represented as vectors (same length as the users, just like the creepy
  guys were of the same length as the original faces). As they are vectors, we
  can put them in the columns of a matrix that we will call $U$.
* PCA on the matrix $R^T$ will give us typical movies. These typical movies are
  also vectors (same length as the movies), and we can put them in the columns
  of a matrix that we will call $M$.

<h4>The matrix factorization</h4>

So what can SVD do for us?
**SVD is PCA on $R$ and $R^T$, in one shot**.

SVD will give you the two matrices $U$ **and** $M$, at the same time. You get
the typical users **and** the typical items in one shot. SVD gives you $U$ and
$M$ by **factorizing** $R$ into three matrices. Here is the **matrix
factorization**:

$$R = M \Sigma U^T$$

To be very clear: SVD is an algorithm that takes the matrix $R$ as an input,
and it gives you $M$, $\Sigma$ and $U$, such that:

* $R$ is equal to the product $M \Sigma U^T$.
* The columns of $M$ can build back all of the columns of $R$ (we already know
  this).
* The columns of $U$ can build back all of the rows of $R$ (we already know
  this).
* The columns of $M$ are orthogonal, as well as the columns of $U$. I haven't
  mentioned this before, so here it is: the principal components are always
  orthogonal. This is actually an extremely important feature of PCA (and SVD),
  but for our recommendation we actually don't care (we'll come to that).
* $\Sigma$ is a diagonal matrix (we'll also come to that).

We can basically sum up all of the above points by this statements: *the columns
of $M$ are an orthogonal basis that spans the column space of $R$, and the
columns of $U$ are an orthonormal basis that spans the row space of $R$*. If
this kind of phrases works for you, great. Personally, I prefer to talk about
creepy guys and typical potatoes ;)

<h4>The model behind SVD</h4>

When we compute and use the SVD of the rating matrix $R$, we are actually
**modeling** the ratings in a very specific, and meaningful way. We will
describe this modeling here.

For the sake of simplicity, we will forget about the matrix $\Sigma$: it is a
diagonal matrix, so it simply acts as a scaler on $M$ or $U^T$. Hence, we will
pretend that we have *merged* into one of the two matrices. Our matrix
factorization simply becomes:

$$R = MU^T$$

Now, with this factorization, let's consider the rating of user $u$ for item
$i$, that we will denote $r_{ui}$:

$$
\begin{pmatrix}
&&&&\\
&&r_{ui}&&\\
&&&&\\
\end{pmatrix}
=
\begin{pmatrix}
&&&&\\
&\horzbar&p_u& \horzbar&\\
&&&&\\
\end{pmatrix}
\begin{pmatrix}
&&\vertbar&&\\
&&q_i&&\\
&&\vertbar&&\\
\end{pmatrix}\\
$$

Because of the way a [matrix
product](https://en.wikipedia.org/wiki/Matrix_multiplication#Illustration) is
defined, the value of $r_{ui}$ is the result of a dot product between two
vectors: a vector $p_u$ which is a row of $M$ and which is specific to the user
$u$, and a vector $q_i$ which is a column of $U^T$ and which is specific to the
item $i$:

$$r_{ui} = p_u \cdot q_i,$$

where '$\cdot$' stands for the usual dot product. Now, remember how we can
describe our users and our items?

$$
\begin{array}{ll}
\text{Alice} & = 10\% \color{#048BA8}{\text{ Action fan}} &+ 10\%
\color{#048BA8}{\text{ Comedy fan}} &+
50\% \color{#048BA8}{\text{ Romance fan}} &+\cdots\\
\text{Bob} &= 50\% \color{#048BA8}{\text{ Action fan}}& + 30\%
\color{#048BA8}{\text{ Comedy fan}} &+ 10\%
\color{#048BA8}{\text{ Romance fan}}  &+\cdots\\
\text{Titanic} &= 20\% \color{#048BA8}{\text{ Action}}& + 00\%
\color{#048BA8}{\text{ Comedy}} &+
70\% \color{#048BA8}{\text{ Romance}} &+\cdots\\
\text{Toy Story} &= 30\% \color{#048BA8}{\text{ Action   }} &+ 60\%
\color{#048BA8}{\text{ Comedy}}&+ 00\%
\color{#048BA8}{\text{ Romance}}  &+\cdots\\
\end{array}
$$

Well, the values of the vectors $p_u$ and $q_i$ exactly correspond to the
coefficients that we have assigned to each latent factor:

$$
\begin{align*}
p_\text{Alice} &= (10\%,~~ 10\%,~~ 50\%,~~ \cdots)\\
p_\text{Bob} &= (50\%,~~ 30\%,~~ 10\%,~~ \cdots)\\
q_\text{Titanic} &= (20\%,~~ 00\%,~~ 70\%,~~ \cdots )\\
q_\text{Toy Story} &= (30\%,~~ 60\%,~~00\%,~~ \cdots )
\end{align*}
$$

The vector $p_u$ represents the **affinity** of user $u$ for each of the latent
factors. Similarly, the vector $q_i$ represents the **affinity** of the item
$i$ for the latent factors. Alice is represented as $(10\%, 10\%, 50\%,
\cdots)$, meaning that she's only slightly sensitive to action and  comedy
movies, but she seems to like romance. As for Bob, he seems to prefer action
movies above anything else. We can also see that Titanic is mostly a romance
movie and that it's not funny at all.

So, when we are using the SVD of $R$, we are modeling the rating of user $u$
for item $i$ as follows:

$$
\begin{align*}
r_{ui}= p_u \cdot q_i = \sum_{f \in \text{latent factors}} \text{affinity of } u
\text{ for } f \times \text{affinity of } i \text{ for }f
\end{align*}
$$

In other words, if $u$ has a taste for factors that are *endorsed* by $i$, then
the rating $r_{ui}$ will be high. Conversely, if $i$ is not the kind of items
that $u$ likes (i.e. the coefficient don't match well), the rating $r_{ui}$
will be low. In our case, the rating of Alice for Titanic will be high, while
that of Bob will be much lower because he's not so keen on romance movies. His
rating for Toy Story will, however, be higher than that of Alice.

