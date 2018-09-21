---
layout: post
title: Understanding matrix factorization for recommendation (part 3) - SVD for recommendation 
description:  Third part of our series on matrix factorization for recommendation&#58; derivation of an algorithm for predicting ratings based on matrix factorization.
tag:
- matrix factorization
- PCA
- SVD
category: blog
author: nico
---

**Foreword**: this is the third part of a 4 parts series. Here are parts
[1]({% post_url 2017-06-14-matrix_facto_1%}), [2]({% post_url
2017-06-15-matrix_facto_2%}) and [4]({% post_url 2017-06-17-matrix_facto_4%}).
This series is an extended version of a [talk I
gave](https://www.youtube.com/watch?v=z0dx-YckFko&t=23m28s) at PyParis 17.

SVD for recommendation
======================

Now that we have a good understanding of what SVD is and how it models the
ratings, we can get to the heart of the matter: using SVD for recommendation
purpose. Or rather, using SVD for **predicting** missing ratings. Let's go back
to our actual matrix $R$, which is sparse:

$$
R= \begin{pmatrix}
1 & \color{#e74c3c}{?} & 2 & \color{#e74c3c}{?} & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & 4\\
2 & \color{#e74c3c}{?} & 4 & 5 & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & \color{#e74c3c}{?} & 3 & \color{#e74c3c}{?} & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & 1 & \color{#e74c3c}{?} & 3 & \color{#e74c3c}{?}\\
5 & \color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & 2\\
\end{pmatrix}
$$

By *sparse*, we here mean "*with missing entries*", not "*containing a lot of
zeros*". We don't replace the missing values with zeros. Remember that our goal
is to predict the $\color{#ff2c2d}{?}$. 

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
$M\Sigma U^T$. However, there is a way around. A first option that was used for
some time is to fill the missing entries of $R$ with some simple heuristic,
e.g. the mean of the columns (or the rows). Once the matrix is dense, we can
compute its SVD using the traditional algorithms. This works OK, but results are
usually highly biased. We will rather use another way, based on a minimization
problem.

<h4>The alternative</h4>

Computing the eigenvectors of $RR^T$ and $R^TR$ is not the only way of
computing the SVD of a dense matrix $R$. We can actually find the matrices $M$
and $U$ if we can find all the vectors $p_u$ and $q_i$ such that (the $p_u$
make up the rows of $M$ and the $q_i$ make up the columns of $U^T$):

* $r_{ui} = p_u \cdot q_i$ for all $u$ and $i$
* All the vectors $p_u$ are mutually orthogonal, as well as the vectors $q_i$.

Finding such vectors $p_u$ and $q_i$ for all users and items can be done by
solving the following optimization problem (while respecting the orthogonality
constraints):

$$\min_{p_u, q_i\\p_u \perp p_v\\ q_i \perp q_j}\sum_{r_{ui} \in R} (r_{ui} - p_u \cdot q_i)^2$$

The above one reads as *find vectors $p_u$ and $q_i$ that makes the sum
minimal.* In other words: we're trying to match as well as possible the values
$r_{ui}$ with what they are supposed to be: $p_u \cdot q_i$.

I'm abusing notation here and referring to $R$ as both a matrix and a set of
ratings. Once we know the values of the vectors $p_u$ and $q_i$ that make this
sum minimal (and here the minimum is zero), we can reconstruct $M$ and $U$ and
we get our SVD.

So what do we do when $R$ is sparse, i.e. when some ratings are missing from
the matrix? [Simon Funk's
answer](http://sifter.org/simon/journal/20061211.html) is that we should just
not give a crap. We **still** solve the same optimization problem:

$$\min_{p_u, q_i}\sum_{r_{ui} \in R} (r_{ui} - p_u \cdot q_i)^2.$$

The only difference is that this time, some ratings are missing, i.e. $R$ is
incomplete. Note that we are not treating the missing entries as zeros: we are
purely and simply ignoring them. Also, we will forget about the orthogonality
constraints, because even if they are useful for interpretation purpose,
constraining the vectors usually does not help us to obtain more accurate
predictions.

Thanks to his solution, Simon Funk ended up in the top 3 of the Netflix Prize
for some time. His algorithm was heavily used, studied and improved by the
other teams.

<h4>The algorithm</h4>

This optimization problem is not convex. That is, it will be very difficult to
find the values of the vectors $p_u$ and $q_i$  that make this sum minimal (and
the optimal solution may not even be unique). However, there are tons of
techniques that can find approximate solutions. We will here use **SGD**
(Stochastic Gradient Descent).

Gradient descent is a very classical technique for finding the (sometimes
local)  minimum of a function. If you have ever heard of back-propagation for
training neural networks, well backprop is just a technique to compute
gradients, which are later used for gradient descent. SGD is one of the
zillions variants of gradient descent. We won't detail too much how SGD works
(there are tons of good resources on the web) but the general pitch is as
follows.

When you have a function $f$ with a parameter $\theta$ that looks like this:

$$f(\theta) = \sum_k f_k(\theta),$$

the SGD procedue minimizes $f$ (i.e. finds the value of $\theta$ such that
$f(\theta)$ is as small as possible), with the following steps:

1. Randomly initialize $\theta$
2. for a given number of times, repeat:
    - for all $k$, repeat:
        * compute $\frac{\partial f_k}{\partial \theta}$
        * update $\theta$ with the following rule:
         $$\theta \leftarrow \theta - \alpha \frac{\partial f_k}{\partial
         \theta},$$, where $\alpha$ is the learning rate (a small value).

In our case, the parameter $\theta$ corresponds to all the vectors $p_u$ and
all the vectors $q_i$ (which we will denote by $$(p_*, q_*)$$), and the function
$f$ we want to minimize is

$$f(p_*, q_*) = \sum_{r_{ui} \in R} (r_{ui} - p_u \cdot q_i)^2 =\sum_{r_{ui}
\in R} f_{ui}(p_u, q_i),$$

where $f_{ui}$ is defined by $f_{ui}(p_u, q_i) = (r_{ui} - p_u \cdot
q_i)^2$.

So, **in order to apply SGD**, what we are looking for is the value of the
derivative of $f_{ui}$ with respect to any $p_u$ and any $q_i$.

- The derivative of $f_{ui}$ with respect to a given vector $p_u$ is given by:

  $$\frac{\partial f_{ui}}{\partial p_u} = \frac{\partial}{\partial p_u}  (r_{ui} - p_u \cdot
  q_i)^2 = - 2 q_i (r_{ui} - p_u \cdot q_i)$$

- Symmetrically, the derivative of $f_{ui}$ with respect to a given vector $q_i$
  is given by:

  $$\frac{\partial f_{ui}}{\partial q_i} = \frac{\partial}{\partial q_i}  (r_{ui} - p_u \cdot
  q_i)^2 = - 2 p_u (r_{ui} - p_u \cdot q_i)$$

Don't be scared, honestly this is highschool-level calculus.

**The SGD procedure then becomes**:

1. Randomly initialize all vectors $p_u$ and $q_i$
2. for a given number of times, repeat:
    - for all known ratings $r_{ui}$, repeat:
        * compute $\frac{\partial f_{ui}}{\partial p_u}$ and $\frac{\partial
          f_{ui}}{\partial q_i}$ (we just did)
        * update $p_u$ and $q_i$ with the following rule:
         $$p_u \leftarrow p_u + \alpha \cdot q_i (r_{ui} - p_u \cdot q_i)$$, and 
         $$q_i \leftarrow q_i + \alpha \cdot  p_u (r_{ui} - p_u \cdot q_i)$$.
         We avoided the multiplicative constant $2$ and merged it into the
         learning rate $\alpha$.

Notice how in this algorithm, the different factors in $p_u$ (and $q_i$) are all
updated at the same time. Funk's original algorithm was a bit different: he
actually trained the first factor, then the second, then the third, etc. This
gave his algorithm a more SVDesque flavor. A nice discussion about this can be
found in [Aggarwal](http://charuaggarwal.net/)'s Textbook on recommender
systems.

Once all the vectors $p_u$ and $q_i$ have been computed, we can estimate all
the ratings we want using the formula:

$$\hat{r}_{ui} = p_u \cdot q_i.$$

There's a hat on $\hat{r}_{ui}$ to indicate that it's an estimation, not its
real value.

<h4>Dimensionality reduction</h4>

Now before we can jump to the Python implementation of this algorithm, there's
one thing we need to decide: what should be the size of the vectors $p_u$ and
$q_i$? One thing we know for sure though, is that it has to be the same for all
vectors $p_u$ and $q_i$, else we could not compute dot products.

To answer this question, let's go back briefly to PCA and to our creepy guys:

<img src="{{ site.url }}/assets/mf_post/faces/eigenface_0.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_1.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_2.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_3.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_4.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_5.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_6.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_7.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_8.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/eigenface_9.jpg">

As you remember, these creepy guys can reconstruct all of the original faces:

<img src="{{ site.url }}/assets/mf_post/faces/face_0/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_1/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_2/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_3/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_4/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_5/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_6/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_7/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_8/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_9/anim.gif">

$$
\begin{align*}
\text{Face 1}~=~&\alpha_1 \cdot \color{#048BA8}{\text{Creepy guy #1}}\\ +~ &\alpha_2 \cdot \color{#048BA8}{\text{Creepy guy #2}}\\ +~ &\cdots\\ +~ &\alpha_{400} \cdot \color{#048BA8}{\text{Creepy guy #400}}
\end{align*}
~~~
$$
$$
\begin{align*}
\text{Face 2}~=~&\beta_1 \cdot \color{#048BA8}{\text{Creepy guy #1}}\\ +~ &\beta_2 \cdot \color{#048BA8}{\text{Creepy guy #2}}\\ +~ &\cdots\\ +~ &\beta_{400} \cdot \color{#048BA8}{\text{Creepy guy #400}}
\end{align*}
$$
$$
~~~\cdots
$$

But in fact, we don't need to use **all** the creepy guys to get a good
approximation of each original face. I actually lied to you: the gifs you see
above only use the first 200 creepy guys (instead of 400)! And you couldn't see
the difference, could you? To further illustrate this point, here is the
reconstruction of the first original face, using from 1 to 400 creepy guys,
each time adding 40 creepy guys into the reconstruction.

<img src="{{ site.url }}/assets/mf_post/faces/face_0/000.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/039.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/079.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/119.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/159.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/199.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/239.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/279.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/319.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/359.jpg">
<img src="{{ site.url }}/assets/mf_post/faces/face_0/399.jpg">

The last picture is the perfect reconstruction, and as you can see even using
only 80 creepy guys (third picture) is enough to recognize the original guy.

As a side note, you may wonder why the first picture does not look like the
first creepy guy, and why the creepy guys have a much higher contrast than the
original pictures: it's because PCA first [subtracts the
mean](https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/decomposition/pca.py#L385)
of all images. The first image you are seeing corresponds in fact to the
contribution of the first creepy guy **plus** the average face. If that doesn't
make sense to you, don't worry. We don't care when we're doing recommendation.

So the take-away message here is this: you don't need all the creepy/typical
guys to have a good approximation of your initial matrix. The same goes for SVD
and the recommendation problem: **you don't need all the typical movies or 
all the typical users to get a good approximation**.

This means that when we are representing our users and items like this (SVD
does it for us):

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

we can just use a small number of typical movies/users and get a good solution.
In our case, we will restrict the size of the $p_u$ and the $q_i$ to 10. That
is, we will only consider 10 latent factors.

You have the right to be skeptical about this, but we have in fact good
theoretical guaranties about this approximation. A fantastic result about SVD
(and PCA) is that when we're using only $k$ factors, we obtain the best
low-rank approximation (understand low number of factors) of the original
matrix.  Details are a bit technical and outside the scope of this article
(although very interesting), so I refer you to [this Stanford course
notes](http://theory.stanford.edu/~tim/s15/l/l9.pdf) (Fact 4.2). (Quick note:
in section 5 of the course notes, the author proposes a way to recover missing
entries from SVD. This heuristic technique is the one that we first suggested
above, but it's **not** what works best. What works best in recommendation is
to optimize on the known ratings, as we are doing with SGD ;)).

We now have all it takes to write a matrix factorization algorithm in the [next
(and last!) part]({% post_url 2017-06-17-matrix_facto_4%}) of this series.
We'll do that in Python (obviously ;)), using the
[Surprise](http://surpriselib.com) library. As you'll see, the algortihm is
surprisingly simple to write, yet fairly efficient.
