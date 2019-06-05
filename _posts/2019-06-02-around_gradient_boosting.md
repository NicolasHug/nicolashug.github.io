---
layout: post
title: Around gradient boosting&#58; classification, missing values, second order derivatives, and line search.
description: Gradient boosting details
tag:
- Gradient boosting
category: blog
author: nico
---

This post is a sort of follow-up to this [introduction to gradient boosting
as a gradient descent]({% post_url 2019-06-01-gradient_boosting_descent%}).

It's a collection of notes about how gradient boosting works in practice.
We will cover:

- how to handle binary and multiclass classification tasks
- how to support missing values
- the use of second-order derivatives for appropriate losses
- how to implement some tricky losses like least absolute deviation

These sections are independent.

## Binary classification

In our [previous post]({% post_url 2019-06-01-gradient_boosting_descent%}), we
described gradient boosting for regression. In fact, **training a GBDT for
classification is exactly the same**. The only thing that changes is the
loss function.

It is quite analoguous to the linear regression / logistic regression thing.

In order to support the binary cross-entropy loss (or log loss, or logistic
loss, or negative log-likelihood), we only need to adapt the
`compute_gradient_of_loss()` function: instead of returning the gradient of
the least squares loss, we just return the gradient of the cross-entropy
loss.

With this loss, the trees do not predict a probability, they predict a
log-odds ratio (just like in logistic regression). To get the probability
that the sample belongs to the positive class, we just need to apply the
logistic sigmoid function to the raw values coming from the trees. There's
an implementation in this
[notebook](https://nbviewer.jupyter.org/github/NicolasHug/nicolashug.github.io/blob/master/assets/gradient_boosting_descent/GradientBoosting.ipynb).

Note that, **even for classification tasks, the trees that we build are
regression trees**! They are not classification trees. Indeed, we need the
trees to predict the gradients, and the gradients are continuous.

## Multiclass classification

Adding support for multiclass classification involves a few changes to the
base algorithm. The main one is that instead of building 1 tree per
iteration (like in binary classification and regression), we build `K` trees
per iteration, where `K` is the number of classes.

Each tree is a kind of OVR tree, but trees are not completely independent
because they influence each other when we compute the gradients (and the
hessians).

Concretely, the `K` trees of the `i`th iteration do not depend on each
other, but each tree at iteration `i` depends on **all** the `K` trees of
iteration `i - 1` (and before).

For a given sample, the probability that it belongs to class `k` is computed
as a regular softmax between `raw_predictions = [raw_predictions_0,
raw_predictions_1, ... raw_predictions_K-1]`,

where `raw_prediction_k` is the sum of the raw predictions coming from all
the trees of the `k`th class. The predicted class is then the argmax of the
`K` probabilities.

## Support for missing values

Decision trees (and thus GBDTs), have an elegant native support for missing
values.

It's deceptively simple, but this section requires a working knowledge of
how decision trees are trained.

When considering a potential split point (identified by a feature index and
a threshold value), some samples are mapped to the left child, and the rest
are mapped to the right child. This mapping is based on whether the feature
value of the samples at the node is lower (or greater) than the given
threshold.

If there are missing values, when computing the gain at a potential split
point, we additionally consider these 2 hypothetical scenarios:
- samples with missing values go to the left child?
- samples with missing values go to the right child?

We compute the gain of both these alternatives, and just keep the best one.

In practice, considering these two scenarios is performed very simply by
scanning the potential thresholds from left to right (ascending order),
**and then** from right to left (descending order). There's a neat
explaination in Alg. 3 of the [XGBoost
paper](https://arxiv.org/pdf/1603.02754).

When predicting, the samples with missing values for a given split point go to
the best alternative.

Another great feature of decision trees is that missing values can still be
supported at predict time, even if no missing values were encountered at fit
time. A common strategy is to map nodes with missing values to whichever
child has the most samples.

## Second order derivatives

Some losses, like the log-loss, have non-constant second order derivatives
(abusively called *hessians*). In that case, instead of performing a
gradient descent step, it is more efficient to perform a [Newton-Raphson
step](https://en.wikipedia.org/wiki/Newton%27s_method): concretely, instead of
predicting the gradients, the trees try to predict the ratio `gradients /
hessians`.

This corresponds to equation (5) of the [XGBoost
paper](https://arxiv.org/pdf/1603.02754). (This isn't really new, and was
already mentionned in [the original
paper](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) by Jerome
Friedman).


## Line search for Least Absolute Deviation loss

The least absolute deviation loss is defined as
$\mathcal{L} = \sum_i |y_i - \hat{y}_i|$.

Unlike other losses, this loss requires changing the values predicted by the
trees after they are trained.

We still build the trees by fitting the gradients (just like with
any other loss), **but once a tree is trained, we update its predicted
values with the median of the samples in each leave**.

Remember how gradient boosting is analoguous to gradient descent? Well
updating the tree values (again, only **after** the tree is trained)
corresponds to the *line search* procedure of gradient descent.

In gradient descent, the line search consists in computing an optimal value
for the learning rate. In gradient boosting, this translates in updating the
tree values.

If you don't replace the values, you'll have terrible predictions. This is not
surprising, since with this loss the gradients can only take the values -1
or 1 (and you can't correctly approximate a continuous function with a sum
of integers).

In fact, this line search is *not* specific to the LAD loss. We do apply
this line search for all losses, including least squares and log loss. The
thing is, for these losses, the line search dictates that the values of the
trees should be exactly what they were trained with, so we just don't need
to update them. With LAD, the line search dictates that the values should be
updated with a median.

I encourage you to refer to [the original
paper](https://statweb.stanford.edu/~jhf/ftp/trebst.pdf) by Jerome Friedman
for a more theoretically grounded analysis. You can also find more details
about line search for gradient descent in [Boyd and Vandenberghe's
book](https://web.stanford.edu/~boyd/cvxbook/).
