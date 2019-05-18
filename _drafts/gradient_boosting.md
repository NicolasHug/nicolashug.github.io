---
layout: post
title: Understanding Gradient Boosting
description: TODO
tag:
- Gradient boosting
- scikit-learn
category: blog
author: nico
---

This post is an attempt to explain gradient boosting as a (kinda weird)
gradient descent.

I'll assume zero previous knowledge of gradient boosting here, but you might
want to review gradient descent if you're not familiar with it. Let's get
started!

For a given sample $\mathbf{x}_i$, a gradient boosting regressor yields
predictions with the following form:

$$\hat{y}_i = \sum_{m = 1}^{\text{n_iter}} h_m(\mathbf{x}_i),$$

where each $h_m$ is an instance of a *base estimator* (often called *weak
learner*, since it usually does not need to be extremely accurate). Since the
base estimator is almost always a decision tree, I'll abusively use the term
*GBDT* (Gradient Boosting Decision Trees) to refer to gradient boosting in
general.

**This sum $\sum_{m = 1}^{\text{n_iter}} h_m(\mathbf{x}_i)$ is actually
performing a gradient descent.**

Specifically, it's a gradient descent in a **functional space**. This is in
contrast to what we're used to in many other machine learning algorithms
(typically neural networks or linear regression), where gradient descent is
instead performed in the **parameter space**.

## Gradient descent in parameter space: linear regression

Recipe for :

1. define a parametrized model
2. define a loss
3. optimize loss on training data with gradient descent on the parameter:
   update the **parameters** of your model to better fit the data


TODO:
- classif vs regression
- make clear that all trees are regression trees



