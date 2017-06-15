---
layout: post
title: Insights on PCA and SVD for matrix factorization
categories: [general]
tags: [matrix factorization, PCA, SVD, recommender systems]
description:  TODO
comments: true
---

Blabla matrix is sparse our mission redict the missing entries.
We'll focus on the case of movies because we can do what the fuck we want

SVD is one of the highlights of linear algebra. It's beautiful. When people
tell you that math sucks, show them SVD. Before SVD, we need PCA which is
almost as awesome as SVD:

A little bit of PCA
===================

We'll play around with the
[Olivetti](http://scikit-learn.org/stable/datasets/olivetti_faces.html#olivetti-faces)
dataset. It's a set of greyscale images of faces from 40 people, making up a
total of 400 images. Here are the the first 10 people:

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

PCA, which stands for Principal Component Analysis, is an algorithm that will
*reveal* 400 of these guys:

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
technique), and when they represent images such as here we call them the
**eigenfaces**. Some really cool stuff can be done with eigenfaces such as
[face recognition](https://en.wikipedia.org/wiki/Eigenface) , or [optimizing
your tinder
matches](http://crockpotveggies.com/2015/02/09/automating-tinder-with-eigenfaces.html)!
The reason why they're called *eigenfaces* is because they are in fact the
eigenvectors of the covariance matrix of $X$ (but we will not detail this, see
[the references](#refs) if you want to dive further into it).

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
<img src="{{ site.url }}/assets/mf_post/faces/face_8/anim.gif">
<img src="{{ site.url }}/assets/mf_post/faces/face_9/anim.gif">
TODO: get images 8 and 9

Here is what's going on. Each of the 400 orginial faces (i.e. each of the 400
original vectors) can be expressed as a (linear) combination of the creepy
guys. That is, we can express the first original face (i.e. its pixel values)
as a little bit of the first creepy guy,  plus a little bit of the second
creepy guy, plus a little bit of third, etc. until the last creepy guy. The
same goes for all of the other original faces: they can all be exressed as a
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
~~~\cdots~~~
$$
$$
\begin{align*}
\text{Face 400}~=~&\gamma_1 \cdot \color{#048BA8}{\text{Creepy guy #1}}\\ +~ &\gamma_2 \cdot \color{#048BA8}{\text{Creepy guy #2}}\\ +~ &\cdots\\ +~ &\gamma_{400} \cdot \color{#048BA8}{\text{Creepy guy #400}}
\end{align*}
$$

The gifs you saw above are the very translation of these math equations: the
first frame of a gif is the contribution of the first creepy guy, the second
frame is the contribution of the first two creepy guys, etc. until the last
creepy guy.

<h4>Latent factors</h4>

We've actually been kind of harsh towards the creepy guys. They're not
creepy, they're **typical**. The goal of PCA is to reveal typical vectors: **each of the
creepy/typical guy represents one specific aspect underlying the data**. In an
ideal world, the first typical guy would represent (e.g.) a *typical elder person*, the
second typical guy would represent a *typical glasses wearer*, and some
other typical guys would represent concepts such as *smiley*, *sad looking*,
*big nose*, stuff like that. And with these concepts, we could define a face as
more or less *elder*, more or less *glassy*, more or less *smiling*,
etc... In practice, the concepts that PCA reveals are really not that clear:
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
typical images that can build back all of your initial images, such as here. If
your matrix contains potatoes, PCA will reveal some typical potatoes that can
build back all of your original potatoes. If your matrix contains ratings,
well... Here we come.

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

Looks familiar? Instead of having faces in the rows, we now have users
(represented as their ratings). Just like PCA gave us some typical guys before,
it will now give us some **typical users**, or rather some **typical raters**.

Here again, in an ideal world, the concepts associated with the typical users
would have a clear semantic meaning: we would obtain a typical *action movie
fan*, a typical *romance movie fan*, a typical *Comedy fan*, etc. In practice,
the semantics behind the typical users are not clearly defined, but for the
sake of simplicity we will assume that they are (it doesn't change anything,
this is just for intuition/explaination purposes).

So here we are: each of our initial users (Alice, Bob...) can be expressed a
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

We are now ready to dive into SVD.

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
