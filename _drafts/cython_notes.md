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
list of things I wish I knew before writing a few thousands lines of Cython
code. I hope it can be useful to others!

# Workflow

My Cython workflow is pretty basic. Write code, compile, fix compilation
errors, then run the tests... Nothing new here. There are a few
Cython-specific steps though. Since Cython is pretty magical, a small change
in your code can induce significant drops (or gains) in performance. I
strongly suggest to:

- Always benchmark your code, extensively.
- Always track down Python interactions. Yellow is your enemy (see below).

About debugging: there is, AFAIK, no debugger for Cython. You can of course
still use a debugger for the generated C-code, but that's not really
convenient. Personally, I use the good old ``print()`` statements with
extensive unit tests. It can become pretty annoying when you're debugging a
``nogil`` section where ``print()`` is forbidden though.

# Python interactions and how to avoid them

Cython generates C code that conceptually operates in 2 different modes:
either in "Python mode" or in "pure C mode".

Python mode is when the code manipulates Python objects, through the
[Python/C API](https://docs.python.org/3/c-api/intro.html): for example when
you are using a dict, or a numpy array. A call to this Python/C API is
called a *Python interaction*.

Pure C mode is when the code only manipulates pure C types (things that are
``cdef``'ed) and does not make any use of the Python/C API. For example if
you want to manipulate a numpy array in pure C mode, use a memory view
instead (see below).

My simplistic proxy is that Python interaction = slow = bad, while pure C
mode = fast = good. When writing Cython, you want to avoid Python
interactions as much as possible, especially deep inside for loops. The
command ``cython -a file.pyx`` will output a html file of the generated C
code, where each line has a different shade of yellow: more yellow means
more Python interactions.

When I write Cython code, I always use ``cython -a`` on every file. My goal
is that all the file is white (i.e. no interaction), although interactions
are unavoidable in some places:typically when arguments are passed in, or
when returning Python objects.

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

# Multi-threading and the GIL

The [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) is this
annoying thing that prevents programs run with `CPython` to do
multi-threaded parallelism. You can do multi-processing (e.g. with
[joblib](https://joblib.readthedocs.io/en/latest/)), but unless the GIL is
"released", multi-threading isn't possible.

**Cython allows you to release the GIL**. That means that you can do
multi-threading in at least 2 ways:

- Directly in Cython, using OpenMP with
  [prange](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html).
- Using e.g. `joblib` with a multi-threading backend (the parts of your code
  that will be parallelized are the parts the release the GIL)

We use both in scikit-learn.

To release the GIL in Cython, you just need to use the `with nogil:`
context manager (you can also do that directly when using ``prange``).

Your code inside of a `nogil` statement cannot have any Python interaction.
Any variable you use will have to be `cdef`'ed, and you won't be allowed to
use numpy arrays since these are objects: use views instead! If you call a
function, it needs to be labeled as a `nogil` function, like so:

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
release the GIL inside of a `def` function that has some Python
interactions, as long as the Python interactions happen outside of the
`nogil` block. In other words, no need to overthink your functions
interfaces. For example, this is perfectly possible:

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

# Then from Python, in a .py file:
out = f(np.arange(10))
```

Finally: releasing and aquiring the GIL takes time. You don't want to do that
deep inside nested for loops. It is instead better to write big chunks of
nogil code.

# Using numpy arrays and memory views

In scikit-learn we rely on numpy arrays for almost everything. Cython supports
numpy arrays but since these are Python objects, we can't manipulate them
without the GIL.

In the past, the workaround was to use pointers on the data, but that can
get ugly very quickly, especially when you need to care about the memory
alignment of 2D arrays ([C vs
Fortran](https://stackoverflow.com/questions/26998223/what-is-the-difference-between-contiguous-and-non-contiguous-arrays)).

Cython now supports memory views, which can be used without the GIL. A view
is a light struct that basically contains a pointer to the raw data, and
info about the type, memory alignment, etc.

The interaction between numpy arrays and views is pretty flexible. In
particular, the following patterns are perfectly acceptable:

```python
# Declare an argument as a view, and pass in an array
def f(int [:] my_view):
    # ...

f(np.arange(10))
```

```python
# Take an array as input, and locally map it to a view
def f(array):

    cdef int [:] my_view = array  # bind array to a view

    # You can now manipulate the view, possibly without the GIL
```

```python
# Allocate an array inside of a function, and manipulate it with a view.
def f():

    cdef int [:] my_view = np.arange(10)

    # You can now manipulate the view, possibly without the GIL

    return np.asarray(my_view)
```

I find myself not using pointers at all, unless I really need to `malloc()`
and then `free()` something in a `nogil` section.

### Other tips about memory views:

You can manipulate views over numpy structured array (i.e. arrays with a
complex dtype): see [this PR](https://github.com/cython/cython/pull/2813),
which isn't merged at the time of writing.

Since you're writing C-like code, in general you'll want to disable the
bounds checks and the wraparound to make your code faster (see the docs).

If you know whether your array is C-aligned or Fortran-aligned, definitely
let Cython know. Else, Cython will generate general code that can work with
arbitrary alignment, which is less optimized:

```python
cdef int [:, ::1] my_view  # C aligned (contiguous on the last dim)
cdef int [::1, :] my_view  # Fortran aligned (contiguous on the 1st dim)
```

# Last remarks

Cython is an amazing tool, and the whole Python data-science ecosystem owes
Cython a lot. Scikit-learn, Scipy and pandas heavily rely on it.

Much like [Numba](https://numba.pydata.org/), it can however be a bit (too)
magical sometimes, and even the smallest change can have a huge impact on
the performance of your code, sometimes for obscure reasons. I can't stress
this enough: **always benchmark your code.**

The [Cython documentation](https://cython.readthedocs.io/en/latest/) is full
of great tips, though with time the organization is becoming a bit confusing
(with e.g. some redundancy between the User Guide and the tutorials). This
[Scipy paper](http://conference.scipy.org/proceedings/SciPy2009/paper_1/)
also has some useful info that are not all covered in the docs.

I also found the *Cython* book by Kurt W. Smith to be very clear and useful.
It covers most of the topics from the docs in a very accessible way.

----
