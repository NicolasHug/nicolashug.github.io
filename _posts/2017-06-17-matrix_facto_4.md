---
layout: post
title: Understanding matrix factorization for recommendation (part 4) - algorithm implementation
description:  Last part of our series on matrix factorization for recommendation&#58; implementation of our algorithm, and further notes.
tag:
- matrix factorization
- PCA
- SVD
category: blog
author: nico
---

**Foreword**: this is the last part of a 4 parts series. Here are parts
[1]({% post_url 2017-06-14-matrix_facto_1%}), [2]({% post_url
2017-06-15-matrix_facto_2%}) and [3]({% post_url 2017-06-16-matrix_facto_3%}).
This series is an extended version of a [talk I
gave](https://www.youtube.com/watch?v=z0dx-YckFko&t=23m28s) at PyParis 17.

Algorithm implementation in Python
==================================

In the [previous part]({% post_url 2017-06-16-matrix_facto_3%}), we have described how to find an approximate solution
to the SVD problem using Stochastic Gradient Descent. Here are the main lines
of this procedure:

1. Randomly initialize all vectors $p_u$ and $q_i$, each of size 10.
2. for a given number of times (i.e. number of **epochs**), repeat:
    - for all known ratings $r_{ui}$, repeat:
        * compute $\frac{\partial f_{ui}}{\partial p_u}$ and $\frac{\partial
          f_{ui}}{\partial q_i}$ (we just did)
        * update $p_u$ and $q_i$ with the following rule:
         $$p_u \leftarrow p_u + \alpha \cdot q_i (r_{ui} - p_u \cdot q_i)$$, and 
         $$q_i \leftarrow q_i + \alpha \cdot  p_u (r_{ui} - p_u \cdot q_i)$$.
         We avoided the multiplicative constant $2$ and merged it into the
         learning rate $\alpha$.

Without further ado, here is a Python translation of this algorithm:

{% highlight python%}
def SGD(data):
    '''Learn the vectors p_u and q_i with SGD.
       data is a dataset containing all ratings + some useful info (e.g. number
       of items/users).
    '''

    n_factors = 10  # number of factors
    alpha = .01  # learning rate
    n_epochs = 10  # number of iteration of the SGD procedure

    # Randomly initialize the user and item factors.
    p = np.random.normal(0, .1, (data.n_users, n_factors))
    q = np.random.normal(0, .1, (data.n_items, n_factors))

    # Optimization procedure
    for _ in range(n_epochs):
        for u, i, r_ui in data.all_ratings():
            err = r_ui - np.dot(p[u], q[i])
            # Update vectors p_u and q_i
            p[u] += alpha * err * q[i]
            q[i] += alpha * err * p[u]
{% endhighlight %}

This fits into less than 10 lines of code. Pretty neat, right?

Once we have run the SGD procedure, all vectors $p_u$ and $q_i$ will be
estimated. We can then predict all the ratings we want by simply using the dot
product between the vectors $p_u$ and $q_i$:

{% highlight python %}
def estimate(u, i):
    '''Estimate rating of user u for item i.'''
    return np.dot(p[u], q[i])
{% endhighlight %}

And that's it, we're done :). Want to try it out yourself? I made a small
[notebook](http://nbviewer.jupyter.org/github/NicolasHug/nicolashug.github.io/blob/master/assets/mf_post/Matrix%20factorization%20algorithm.ipynb)
where we can actually run this algorithm on a real dataset.  We will use the
library [surprise](http://surpriselib.com), which is a great library for
quickly implementing rating prediction algorithms (but I might be slightly
biased, as I'm the main dev of surprise!). To use it, you'll simply need to
install it first using pip:


{% highlight python%}
pip install scikit-surprise
{% endhighlight %}

<h4>How good is our algorithm?</h4>

As you can see on the
[notebook](http://nbviewer.jupyter.org/github/NicolasHug/nicolashug.github.io/blob/master/assets/mf_post/Matrix%20factorization%20algorithm.ipynb),
we obtain an average RMSE of about 0.98.  RMSE stands for Root Mean Squared
Error, and is computed as follows:

$$\text{RMSE} = \sqrt{\sum_{u,i} (\hat{r}_{ui} - r_{ui})^2}.$$

You can think of it as some sort of average error, where big errors are heavily
penalized (because they are squared). Having an RMSE of 0 means that all our
predictions are perfect. Having an RMSE of 0.5 means that in average, we are
approximately 0.5 off with each prediction.

So is 0.98 a good RMSE? It's actually really not that bad! As shown in the
notebook, a neighborhood algorithm only achieves an RMSE of 1. A [more
sophisticated MF
algorithm](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)
(the same as ours, except the ratings are unbiased and we use regularization)
can achieve an RMSE of about 0.96. This algorithm is called '*SVD*' in the
literature, but you know now that it can't be a real SVD, as there are missing
ratings ;). It's only (heavily) **inspired** by SVD.

<h4>Wrapping it up</h4>

All right, that's it! I hope you now understand how beautiful PCA and SVD are,
and how we can adapt SVD to a recommendation problem. If you want to play
around with [surprise](http://surpriselib.com), there are plenty of cool stuff
in the [docs](http://surprise.readthedocs.io/en/stable/FAQ.html).

If you found this series useful (or not), please let me know in the comments.
Any mistakes are my own, and **any** kind of criticism would be greatly
appreciated!

Thanks to Pierre Poulain for the valuable feedback!

<a name="refs"></a>

Resources, going further
========================

I tried to avoid theoretical considerations in this series, and rather give
visual, concrete examples to illustrate the use of PCA and SVD. But the
theoretical side should not be overlooked, and it is actually very interesting.
Here are a few resources that you may want to read if you want to dive further
into the various topics covered so far:

- [Jeremy Kun](https://jeremykun.com/)'s posts on
  [PCA](https://jeremykun.com/2011/07/27/eigenfaces/) and
  [SVD](https://jeremykun.com/2016/04/18/singular-value-decomposition-part-1-perspectives-on-linear-algebra/)
  are great. The first part of this article was inspired by these two posts.
- [Aggarwal](http://charuaggarwal.net/)'s Textbook on recommender systems is,
  IMHO, the best RS resource out there. You'll find many details about the
  various matrix factorization variants, plus tons of other subjects are
  covered.
- If you want to know more about the '*SVD*' algorithm and its possible
  extensions, check out [this
  paper](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)
  from the BellKor team ('*SVD*' corresponds to equation (5)). They are the
  guys who won the $1M of the Netflix Prize.
- PCA can be used for a lot of fun stuff, not just messing around with creepy
  faces. Jonathon Shlens' [Tutorial](https://arxiv.org/abs/1404.1100) provides
  great insights on PCA as a diagonalization process, and its link to SVD.
  Also, this [Stanford course
  notes](http://theory.stanford.edu/~tim/s15/l/l9.pdf) covers some of the
  topics we have presented (low-rank approximation, etc.) in a more
  theory-oriented way.
- For background on linear algebra, Gilbert Strang is your guy:
  [Introduction to LA](http://math.mit.edu/~gs/linearalgebra/). His [MIT
  Course](https://ocw.mit.edu/courses/mathematics/18-06-linear-algebra-spring-2010/index.htm)
  is also pure gold.
- And of course, [Surprise](http://surpriselib.com) is a great library for
  recommender systems (but again, I might be biased ;))
