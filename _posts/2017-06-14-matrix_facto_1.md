---
layout: post
title: Understanding matrix factorization for recommendation (part 1) - preliminary insights on PCA
categories: [general]
tags: [matrix factorization, PCA, SVD, recommender systems]
description:  First part of our series on matrix factorization for recommendation&#58; problem definition, and visual examples of what can PCA do.
comments: true
---

**Foreword**: this is the first part of a 4 parts series. Here are parts [2]({%
post_url 2017-06-15-matrix_facto_2%}), [3]({% post_url
2017-06-16-matrix_facto_3%}) and [4]({% post_url 2017-06-17-matrix_facto_4%}).

About ten years ago, Netflix launched the [Netflix
Prize](https://en.wikipedia.org/wiki/Netflix_Prize): an open contest where the
goal was to design state-of-the-art algorithms for predicting ratings. During
3 years, research teams developed many different prediction algorithms,
among which matrix factorization techniques stood out by their efficiency.

**The goal of this series of posts is twofold**:
- Give some insights on how matrix factorization **models** the ratings. To
  this end, we will illustrate how PCA and SVD work, using concrete examples.
  You may have already read explanations like *users and items are represented
  as vectors in a latent factors vector space, and ratings are defined as a dot
  product between two vectors*. If it makes no sense to you, I hope that you
  will understand what it means by the end of this article.
- Derive and implement an algorithm for predicting ratings, based on matrix
  factorization. In its simplest form, this algorithm fits in 10 lines of
  Python. We will use this algorithm and evaluate its performances on real
  datasets.

I tried to keep the math level of the article as accessible as possible, but
without trying to over-simplify things, and avoiding dull statements. My hope
is that this article is accessible to ML beginners, while still being
insightful to the more experienced.

**This article is divided into 4 parts**: in this first part, we will
clearly define the problem we plan to address, and provide some insights about
PCA. In the [second part]({% post_url 2017-06-15-matrix_facto_2%}), we will
review SVD and see how it models the ratings. In the [third part]({% post_url 2017-06-16-matrix_facto_3%}), we will
see how to apply our knowledge of SVD to the rating prediction task, and derive
an implementation of a matrix-factorization-based algorithm. In the [last
part]({% post_url 2017-06-17-matrix_facto_4%}), we will implement a matrix
factorization algorithm in Python using the [Surprise](http://surpriselib.com)
library.

The problem
===========

The problem we propose to address here is that of **rating prediction**. The
data we have  is a rating history: ratings of users for items in the interval
$[1, 5]$.  We can put all this data into a sparse matrix called $R$:

$$
R= \begin{pmatrix}
1 & \color{#e74c3c}{?} & 2 & \color{#e74c3c}{?} & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & 4\\
2 & \color{#e74c3c}{?} & 4 & 5 & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & \color{#e74c3c}{?} & 3 & \color{#e74c3c}{?} & \color{#e74c3c}{?}\\
\color{#e74c3c}{?} & 1 & \color{#e74c3c}{?} & 3 & \color{#e74c3c}{?}\\
5 & \color{#e74c3c}{?} & \color{#e74c3c}{?} & \color{#e74c3c}{?} & 2\\
\end{pmatrix}
\begin{matrix}
\text{Alice}\\
\text{Bob}\\
\text{Charlie}\\
\text{Daniel}\\
\text{Eric}\\
\text{Frank}\\
\end{matrix}
$$

Each row of the matrix corresponds to a given user, and each column corresponds
to a given item. For instance here, Alice has rated the first item with a
rating of $1$, and Charlie has rated the third item with a rating of $4$. The
matrix $R$ is sparse (more than 99% of the entries are missing), and **our goal
is to predict the missing entries**, i.e. predict the $\color{#e74c3c}{?}$.

To predict ratings, we will **factorize** the matrix $R$. This matrix
factorization is fundamentally linked to **SVD**, which stands for Singular
Value Decomposition.  SVD is one of the highlights of linear algebra. It's a
beautiful result. When people tell you that math sucks, show them what SVD can
do.

The aim of this article is to explain how SVD can be used for rating prediction
purposes. But before we can dive into SVD in the [second part]({% post_url
2017-06-15-matrix_facto_2%}), we need to review what PCA is. PCA is only
slightly less awesome than SVD, but it is still really cool.

A little bit of PCA
===================

Let's forget the recommendation problem for 2 minutes. We'll play around with
the
[Olivetti](http://scikit-learn.org/stable/datasets/olivetti_faces.html#olivetti-faces)
dataset. It's a set of greyscale images of faces from 40 people, making up a
total of 400 images. Here are the first 10 people:

<img src="{{ site.url }}/assets/mf_post/faces/face_0/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_1/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_2/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_3/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_4/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_5/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_6/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_7/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_8/face.jpg"/>
<img src="{{ site.url }}/assets/mf_post/faces/face_9/face.jpg"/>

Friendly, right? Well, just wait a little bit...

Each image size is 64 x 64 pixels. We will flatten each of these images (we thus get
400 vectors, each with 64 x 64 = 4096 elements). We can represent our dataset
in a 400 x 4096 matrix:

$$
\newcommand{\horzbar}{\Rule{2.5ex}{0.5pt}{0.1pt}}
\newcommand{\vertbar}{\Rule{0.5pt}{1pt}{2.5ex}}
X=
\begin{pmatrix}
\horzbar & \text{Face 1} & \horzbar\\
\horzbar & \text{Face 2} & \horzbar\\
& \vdots &\\
\horzbar & \text{Face 400} & \horzbar\\
\end{pmatrix}
$$

**PCA, which stands for Principal Component Analysis, is an algorithm that will
*reveal* 400 of these guys**:

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

Creepy, right ;)?

We call these guys the **principal components** (hence the name of the
technique), and when they represent faces such as here we call them the
**eigenfaces**. Some really cool stuff can be done with eigenfaces such as
[face recognition](https://en.wikipedia.org/wiki/Eigenface), or [optimizing
your tinder
matches](http://crockpotveggies.com/2015/02/09/automating-tinder-with-eigenfaces.html)!
The reason why they're called *eigenfaces* is because they are in fact the
eigenvectors of the covariance matrix of $X$ (but we will not detail this, see
[the references]({% post_url 2017-06-17-matrix_facto_4%}/#refs) if you want to
dive further into it).

As far as we're concerned, we will call these guys the **creepy guys**. Now,
one amazing thing about them is that **they can build back all of the original
faces.** Take a look at this (these are animated gifs, about 10s long):

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

Here is what's going on. Each of the 400 original faces (i.e. each of the 400
original rows of the matrix) can be expressed as a (linear) combination of the creepy
guys. That is, we can express the first original face (i.e. its pixel values)
as a little bit of the first creepy guy,  plus a little bit of the second
creepy guy, plus a little bit of third, etc. until the last creepy guy. The
same goes for all of the other original faces: they can all be expressed as a
little bit of each creepy guy. Mathematically, we write it this way:

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

The gifs you saw above are the very translation of these math equations: the
first frame of a gif is the contribution of the first creepy guy, the second
frame is the contribution of the first two creepy guys, etc. until the last
creepy guy.

If you want to play around a bit with the creepy guys and reproduce the cool
gifs, I made a small
[notebook](http://nbviewer.jupyter.org/github/NicolasHug/nicolashug.github.io/blob/master/assets/mf_post/creepy_guys.ipynb)
for you.

<h4>Latent factors</h4>

We've actually been kind of harsh towards the creepy guys. They're not
creepy, they're **typical**. The goal of PCA is to reveal typical vectors: **each of the
creepy/typical guy represents one specific aspect underlying the data**. In an
ideal world, the first typical guy would represent (e.g.) a *typical elder person*, the
second typical guy would represent a *typical glasses wearer*, and some
other typical guys would represent concepts such as *smiley*, *sad looking*,
*big nose*, stuff like that. And with these concepts, we could define a face as
more or less *elder*, more or less *glassy*, more or less *smiling*,
etc. In practice, the concepts that PCA reveals are really not that clear:
there is no clear semantic that we could associate with any of the
creepy/typical guys that we obtained here. But the important fact remains:
**each of the typical guys captures a specific aspect of the data**. We call
these aspects the **latent factors** (latent, because they were there all the
time, we just needed PCA to reveal them). Using barbaric terms, we say that
each principal component (the creepy/typical guys) captures a specific latent
factor.

Now, this is all good and fun, but we're interested in matrix factorization for
recommendation purposes, right? So where is our matrix factorization, and what
does it have to do with recommendation? PCA is actually a plug-and-play method:
it works for any matrix. If your matrix contains images, it will reveal some
typical images that can build back all of your initial images, such as here.
**If your matrix contains potatoes, PCA will reveal some typical potatoes that
can build back all of your original potatoes**. If your matrix contains
ratings, well... Here we come.

PCA on a (dense) rating matrix
==============================

Until stated otherwise, we will consider for now that our rating matrix
$R$ is completely dense, i.e.  there are no missing entries. All the ratings
are known. This is of course not the case in real recommendation problems, but
bare with me.

<h4>PCA on the users</h4>

Here is our rating matrix, where rows are users and columns are items:

$$
R = \begin{pmatrix}
\horzbar & \text{Alice} & \horzbar\\
\horzbar & \text{Bob} & \horzbar\\
& \vdots &\\
\horzbar & \text{Zoe} & \horzbar\\
\end{pmatrix}
$$

Looks familiar? Instead of having faces in the rows represented by pixel
values, we now have users represented by their ratings. Just like PCA gave us
some typical guys before, it will now give us some **typical users**, or rather
some **typical raters**.

Here again, in an ideal world, the concepts associated with the typical users
would have a clear semantic meaning: we would obtain a typical *action movie
fan*, a typical *romance movie fan*, a typical *comedy fan*, etc. In practice,
the semantic behind the typical users is not clearly defined, but for the sake
of simplicity we will assume that they are (it doesn't change anything, this is
just for intuition/explanation purposes).

So here we are: each of our initial users (Alice, Bob...) can be expressed as a
combination of the typical users. For instance, Alice could be defined as a
little bit of an action fan, a little bit of a comedy fan, a lot of a romance
fan, etc. As for Bob, he could be more keen on action movies:

$$
\begin{align*}
\text{Alice} &= 10\% \color{#048BA8}{\text{ Action fan}} + 10\%
\color{#048BA8}{\text{ Comedy fan}} +
50\% \color{#048BA8}{\text{ Romance fan}} +\cdots\\
\text{Bob} &= 50\% \color{#048BA8}{\text{ Action fan}} + 30\%
\color{#048BA8}{\text{ Comedy fan}} + 10\%
\color{#048BA8}{\text{ Romance fan}}  +\cdots\\
\text{Zoe} &= \cdots
\end{align*}
$$

And the same goes for all of the users, you get the idea. (In practice the
coefficients are not necessarily percentages, but it's convenient for us to
think of it this way).

<h4>PCA on the items</h4>

What would happen if we transposed our rating matrix? Instead of having users
in the rows, we would now have items (movies in our case), defined as their
ratings:

$$
R^T = \begin{pmatrix}
\horzbar & \text{Titanic} & \horzbar\\
\horzbar & \text{Toy Story} & \horzbar\\
& \vdots &\\
\horzbar & \text{Fargo} & \horzbar\\
\end{pmatrix}
$$

In this case, PCA will not reveal typical faces nor typical users, but of
course **typical movies**. And here again, we will associate a semantic meaning
behind each of the typical movies, and these typical movies can build back all
of our original movies:

$$
\begin{align*}
\text{Titanic} &= 20\% \color{#048BA8}{\text{ Action}} + 0\%
\color{#048BA8}{\text{ Comedy}} +
70\% \color{#048BA8}{\text{ Romance}} +\cdots\\
\text{Toy Story} &= 30\% \color{#048BA8}{\text{ Action}} + 60\%
\color{#048BA8}{\text{ Comedy}} + 0\%
\color{#048BA8}{\text{ Romance}}  +\cdots\\
\end{align*}
$$

And the same goes for all the other movies.

We are now ready to dive into SVD in the [next part]({% post_url
2017-06-15-matrix_facto_2%}).
