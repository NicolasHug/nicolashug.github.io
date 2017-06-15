---
layout: post
title: Understanding matrix factorization for recommendation&#58; the algorithm (part 3)
categories: [general]
tags: [matrix factorization, PCA, SVD, recommender systems]
description:  TODO
comments: true
---

SVD for recommendation
======================

Now that we have a good understanding of what SVD is and how it models the
ratings, we can get to the heart of the matter: using SVD for recommendation
purpose. Or rather, using SVD for **predicting** missing ratings. Let's go back
to our actual matrix $R$, which is sparse:

$$
R= \begin{pmatrix}
\checkmark & \color{#e74c3c}{?} & \checkmark & \color{#e74c3c}{?} & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & \checkmark\\
\checkmark & \color{#e74c3c}{?} & \checkmark & \checkmark & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & \color{#e74c3c}{?} & \checkmark & \color{#e74c3c}{?} & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & \checkmark & \color{#e74c3c}{?} & \checkmark & \color{#e74c3c}{?}\\
\checkmark & \color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & \checkmark\\
\end{pmatrix}
$$

We only know the values $\checkmark$, and we want to predict the
$\color{#ff2c2d}{?}$. 


<h4>Ooooops</h4>

Now guess what: **The SVD of $R$ is not defined.** It does not exist. Yup, it
is impossible to compute, it is not defined, it does not exist :). But don't
worry, all your efforts to read this article that far were not futile.

If $R$ was dense, we could compute $M$ and $U$ easily: the columns of $M$ are
the eigenvectors of $RR^T$, and the columns of $U$ are the eigenvectors of
$R^TR$. The associated eigenvalues make up the diagonal matrix $\Sigma$. There
are [very efficient
algorithms](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.linalg.svd.html)
that can do that.

But as $R$ is sparse, the matrices $RR^T$ and $R^TR$ do not exist, so their
eigenvectors do not exist either and we can't factorize $R$ as the product
$M\Sigma U^T$. However, there is a way around. A first option, that was taken
on until [Simon Funk](http://sifter.org/simon/) came around, is to fill the
missing entries of $R$ with some simple heuristic, e.g. the mean of the columns
(or the rows). This works OK, but results are usually highly biased. We
will rather use another way, based on a minimization problem.

<h4>The alternative</h4>

Computing the eigenvectors of $RR^T$ and $R^TR$ is not the only way of
computing the SVD of a dense matrix $R$. We can actually find the matrices $M$
and $U$ if we can find all the vectors $p_u$ and $q_i$ such that:

* $r_{ui} = p_u \cdot q_i$ for all $u$ and $i$
* All the vectors $p_u$ are mutually orthogonal, as well as the vectors $q_i$.

Finding such vectors $p_u$ and $q_i$ for all users and items can be done by
solving the following optimization problem (while respecting the orthogonality
constraints):

$$\min_{p_u, q_i}\sum_{r_{ui} \in R} (r_{ui} - p_u \cdot q_i)^2$$

The above one reads as *find vectors $p_u$ and $q_i$ that makes the sum
minimal.* In other words: we're trying to match as well as possible the values
$r_{ui}$ with what they are supposed to be: $p_u \cdot q_i$.

I'm abusing notation here and refering to $R$ as both a matrix and a set of
ratings. Once we know the values of the vectors $p_u$ and $q_i$ that make this
sum minimal (and here the minimum is zero), we can reconstruct $M$ and $U$ and
we get our SVD.

So what do we do when $R$ is sparse, i.e. when some ratings are missing from
the matrix? [Simon Funk's
answer](http://sifter.org/simon/journal/20061211.html) is that we should just
not give a crap. We **still** solve the same optimization problem:

$$\min_{p_u, q_i}\sum_{r_{ui} \in R} (r_{ui} - p_u \cdot q_i)^2.$$

The only difference is that this time, some ratings are missing, i.e. $R$ is
incomplete. Also, we will forget about the orthogonality constraints, because
even if they are useful for interpretation purpose, constraining the vectors
usually does not help us to obtain more accurate predictions.

Concretely, the fact that some ratings are missing has various implications


<a name="refs"></a>

resources, going further
========================

Here are a few resources that you may want to read if you want to dive further
on the various topics covered here:

- [Jeremy Kun](https://jeremykun.com/)'s posts on
  [PCA](https://jeremykun.com/2011/07/27/eigenfaces/) and
  [SVD](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/)
  are great. The first part of this article was inspired by them.
- [Aggarwal](http://charuaggarwal.net/)'s Textbook on recommender systems is
  the best RS resource out there. You'll find many details about the various
  matrix factorization variants.
- For background on linear algebra, the only book worth reading is Gilbert
  Strang's [Introduction to LA](http://math.mit.edu/~gs/linearalgebra/). His
  [MIT
  Course](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/index.htm)
  is also pure gold.
- Jonathon Shlens [Tutorial](https://arxiv.org/abs/1404.1100) provides great
  insights on PCA as a diagonalization process, and its link to SVD.
