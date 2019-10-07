---
layout: post
title: Cython notes and tips
description: Cython notes and tips&#58; My workflow, memory views, the GIL and other fun stuff
tag:
- Cython
- Python
category: blog
author: nico
---

This post is a collection of notes and tips about Cython, that I learned or
discovered while developping for [scikit-learn](https://scikit-learn.org/).
I will describe my (basic) workflow, and I will describe what I learned
about avoiding Python interactions, memory views, the GIL and other fun
stuff.

This is neither a tutorial, nor an introduction to Cython. This is rather a
list things I wish I knew before writing a few thousands lines of Cython
code. I wish it will be useful to others!

# My workflow

My Cython workflow is pretty basic. Write code, compile, fix compilation
errors, then run the tests... Nothing new here. There are a few
Cython-specific steps though. Since Cython is pretty magical, a small change
in your code can induce significant drops (or gains) in performance. I
strongly suggest to:

- Always benchmark your code, extensively.
- Always track down Python interactions. Yellow is your enemy (see below).

About debugging: there is, AFAIK, no debugger for Cython. You can of course
still use a debugger for the generated C-code, but that's not really
convenient. Personally, I use the good old ``print`` statements with
extensive unit tests. It can become pretty annoying when you're debugging a
``nogil`` section where ``print()`` is forbidden though.

# Python interactions and how to avoid them

Cython generates C code that conceptually operates in 2 different modes:
either in "Python mode" or in "pure C mode".

Python mode is when the code manipulates Python objects, through the
[Python/C API](https://docs.python.org/2/c-api/index.html): for example when
you are using a dict, or a numpy array. A call to this Python/C API is
called a *Python interaction*.

Pure C mode is when the code only manipulates pure C types (things that are
``cdef``'ed) and does not make any use of the Python/C API. For example if
you want to manipulate a numpy array in pure C mode, use a memory view
instead (see below).

My simplistic proxy is that Python interaction = slow = bad, while pure C
mode = fast = good. When writing Cython, you want to avoid Python
interactions as much as possible, especially deep inside for loops. The
command ``cython -a file.pyx`` will generate a html file of the generated C
code, where each line has a different shade of yellow: more yellow means
more Python interactions.

When I write Cython code, I always use ``cython -a`` on every file. My goal
is that all the file is white (i.e. no interaction), although interactions
are unavoidable in some places (typically when arguments are passed in, or
when returning Python objects).

Some tips to avoid Python interactions:

- Use ``cdef`` as much as you can
- Use memory views to manipulate numpy arrays. If your function accepts a numpy
  array, declare a ``cdef``'ed memory view to manipulate it.
- A good way to make sure that you're not risking any interaction is to write
  code within a ``with nogil`` context manager (see below).
- For some reason, declaring a variable locally in a function may make some
  interactions disapear. Typically, I noticed that using a local variable as
  an alias to the attribute of a ``cdef``'ed class will remove Python
  interactions (``cdef int local_var = self.var``). **Don't follow these
  tips blindly though**. In some cases it will make your code faster, but
  sometimes it won't. In the end, always let your benchmarks decide.


# Using Memory views and numpy arrays

- memory aligment
- how to use mem views with custom dtype

# Multi-threading and the GIL

The GIL is this annoying thing that prevents programs run with `CPython` to
do multi-threaded parallelism. You can do multi-processing (e.g. with
[joblib](https://joblib.readthedocs.io/en/latest/)), but unless the GIL is
"released" (the GIL is a *lock*), multi-threading isn't possible.

**Cython allows you to release the GIL**. That means that you can do
multi-threading in at least 2 ways:

- Directly in Cython, using OpenMP with
  [prange](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html)).
- Using e.g. `joblib` with a multi-threading backend (the parts of your code
  that will be parallelized are the parts the release the GIL)

To release the GIL in cython, you just need to use the `with nogil:`
context manager (you can also do that directly when using ``prange``).

Your code inside of a `nogil` statement cannot have any Python interaction.
Any variable you use will have to be `cdef`ed, and you won't be allowed to
use numpy arrays: use views instead! If you call a function, it needs to be
labeled as a `nogil` function, like so:

```python
cdef void my_func(int [:] some_view) nogil:
    # ...
```

This is carefully explained in the docs, but I didn't really get it at
first: labelling a function as ``nogil`` **does not release the GIL**. It
simply tells Cython that the function *may* be called without the GIL
(Cython uses this information to do some static compilation-time checks).
As a result this function must not have any kind of Python interaction.
You are still allowed to call the function **with** the GIL, though.

Another thing that wasn't clear for me at first: it is perfectly OK to
release the GIL inside of a `def` function that has a lot of Python
interactions, as long as the Python interactions happen when the GIL is
held. That makes the interface of your functions simpler. For example, this
is perfectly possible:

```python

def f(array):  # Take Python object as input

    cdef:
        int [:] my_view = array

    # Python interactions possible here
    # ...

    with nogil:
        # No Python interactions here
        # ...
        # do stuff to my_view, maybe in parallel with prange...
        # ...

    # Python interactions possible here
    # ...

    return array  # return a Python object

# Then from Python:
out = f(np.arange(10))
```

Finally: releasing and aquiring the GIL takes time. You don't want to do that
deep inside nested for loops. It is instead better to write big chunks of
nogil code.

# the directives at the top

# Final

Cython is an amazing tool, and the Python data-science ecosystem wouldn't be
what it is without it. It can however be a bit magical sometimes, and even
with experience it is still hard to make sure that a

- some very magical stuff can happen. See issue about 2d arrays.

I found the *Cython* book by Kurt W. Smith to be very clear and pleasant to
read.
+ doc + papers

----