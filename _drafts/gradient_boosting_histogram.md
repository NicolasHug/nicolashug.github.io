---
layout: post
title: Gradient boosting&#58; Making it fast with histograms
description: Gradient boosting&#58; Making it fast with histograms
tag:
- Gradient boosting
category: blog
author: nico
---

I've been having a lot a fun lately with gradient boosting. With [Olivier
Grisel](http://ogrisel.com/), we worked on a [new GBDT
implementation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier)
for scikit-learn which is multiple orders of magnitude faster than the previous
one.

I'll describe here some of the optimizations that we used. We didn't come
up with these ideas: we took inspiration from
[LightGBM](https://github.com/microsoft/LightGBM). XGBoost also implemented
this as their ['hist'](https://github.com/dmlc/xgboost/issues/1950) method.

This post assumes a working knowledge of gradient boosting (refer to this
[previous post]({% post_url 2019-06-01-gradient_boosting_descent%}) for a
refresher!)

----

## What is slow: quick background on decision trees

As we explained [before]({% post_url
2019-06-01-gradient_boosting_descent%}), training a GBDT consists in
iteratively training decision trees. The trees are trained to predict
gradients (also sometimes incorporating info about second-order derivatives,
abusively called hessians, that we will ignore for simplicity). Each sample
has its own gradient (a scalar), and gradients are easy and cheap to
compute.

Trees are grown in a greedy fashion, starting from the root node to the
leaves. All the samples first belong to the root. We then search for the
best **split point**. A split point is identified by a feature index, and a
threshold value. Samples whose value for the given feature is less than the
threshold will be mapped to the left child, and the rest will be mapped to
the right child. This process of finding the best split point is repeated at
each node until some stopping criteria is met (e.g. maximum depth of the
tree).

For a given node, finding the best split point roughly looks like this[^1]:

[^1]:
    We're describing the case of continuous features here. Categorical
    features are trickier and out of scope.

```python
def find_best_split_point(samples_at_node, sum_gradients_node):
    for feature_idx in range(n_features):
        for threshold in sorted(samples_at_node[:, feature_idx]):  # slow

            samples_left, samples_right = split_samples_according_to_threshold(threshold)  # reasonably fast
            sum_gradients_left, sum_gradients_right = get_gradient_sums(samples_left, samples_right)  # reasonably fast

            potential_gain = compute_gain(sum_gradients_left, sum_gradients_right, sum_gradients_node)  # fast
            if potential_gain > best_gain:
                best_gain = potential_gain
                best_split_point = (feature_idx, threshold)

    return best_split_point

def compute_gain(sum_gradients_left, sum_gradients_right, sum_gradients_node):
    return sum_gradients_left**2 + sum_gradients_right**2 - sum_gradients_node**2
```

We can safely ignore the details of `split_samples_according_to_threshold()`
and `get_gradient_sums()`. We don't really care about them, the point being
that these are relatively fast, provided that the thresholds are ordered.

As you can see, at every node, for each feature, we need to sort the samples
belonging to that node. **That's slow.**

The `compute_gain()` function corresponds to equation (7) of the [XGBoost
paper](https://arxiv.org/pdf/1603.02754) (we ignored hessians and other
constant terms). You don't need to understand it in details. The important
thing to note is that it only depends on the sum of the gradients of the
samples belonging to three nodes: the current node, the potential left
child, and the potential right child.

`sum_gradient_nodes` is passed as a parameter (we just save the information
somewhere). If we can find a way to compute the two other gradient sums more
efficiently (i.e. without the sorting step), we'll have much faster GBDTs.

## Binning the data

Before the main boosting loop of the training procedure, the first thing you
do is bin your training data into integer-valued bins.

Binning simply consists in mapping your real-valued training data into a much
smaller integer range, typically `[0, 255]`. The training data can now be
represented as `uint8`:

```python
# (This part of the sklearn API is private, don't use it)
In [1]: from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
In [2]: import numpy as np
In [3]: X_train = np.random.normal(size=(4, 2))  # 4 samples, 2 features
In [4]: X_train
Out[4]:
array([[ 1.0,  0.4],
       [-0.1,  0.2],
       [-0.8,  2.0],
       [ 2.2, -0.1]])
In [5]: X_train_binned = _BinMapper(max_bins=3).fit_transform(X_train)
In [6]: X_train_binned
Out[6]:
array([[1, 1],
       [0, 0],
       [0, 2],
       [2, 0]], dtype=uint8)
```

Each feature is binned independently. Finding the bins is usually done by
computing the feature-wise quantiles, but there are other techniques (not
relevant for our discussion). Binning happens only once, at the very beginning
of the training procedure: we do **not** bin the data for each tree, nor for
each node.

Binning the data has an unvaluable advantage: we can now use the feature
values (i.e. the bins) as indexes. Specifically, we use the bin values to
index histograms.

## Histograms

```python
def find_best_split_point(sum_gradients_node, histograms):
    for feature in range(n_features):
        # histogram contains the sum of gradients of the samples in each bin
        historam = histograms[feature_idx]
        sum_gradients_left = 0
        for bin_idx in range(n_bins):  # no sorting step, and 256 iterations at most
            sum_gradients_left += histogram[bin_idx]
            sum_gradients_right = sum_gradients_node - sum_gradients_left

            potential_gain = compute_gain(sum_gradients_left, sum_gradients_right, sum_gradients_node)
            if potential_gain > best_gain:
                best_gain = potential_gain
                best_split_point = (feature_idx, threshold)

    return best_split_point
```


<!-- leave this here for better footnotes rendering -->
----