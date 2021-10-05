"""Common Type Definitions."""

from typing import Tuple, Callable, Union, List, Any, Optional, Sequence, TypeVar, Dict
import jax.numpy as np


# Missing JAX Types.
PyTree = Any


"""A type alias for PRNGKeys.

  See https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.PRNGKey
  for details.
"""
PRNGKey = np.ndarray


"""A type alias for axes specification.

  Axes can be specified as integers (`axis=-1`) or sequences (`axis=(1, 3)`).
"""
Axes = Union[int, Sequence[int]]


"""NTK Trees.

Trees of kernels and arrays naturally emerge in certain neural
network computations computations (for example, when neural networks have nested
parallel layers).

Mimicking JAX, we use a lightweight tree structure called an NTTree. NTTrees
have internal nodes that are either Lists or Tuples and leaves which are either
array or kernel objects.
"""
T = TypeVar('T')
NTTree = Union[List[T], Tuple[T, ...], T]


Shapes = NTTree[Tuple[int, ...]]


# Layer Definition.
"""A type alias for initialization functions.

Initialization functions construct parameters for neural networks given a
random key and an input shape. Specifically, they produce a tuple giving the
output shape and a PyTree of parameters.
"""
InitFn = Callable[[PRNGKey, Shapes], Tuple[Shapes, PyTree]]


"""A type alias for apply functions.

Apply functions do computations with finite-width neural networks. They are
functions that take a PyTree of parameters and an array of inputs and produce
an array of outputs.
"""
ApplyFn = Callable[[PyTree, NTTree[np.ndarray]], NTTree[np.ndarray]]


"""Specifies `(input, output, kwargs)` axes for `vmap` in empirical NTK.
"""
_VMapAxis = Optional[NTTree[int]]
VMapAxes = Tuple[_VMapAxis, _VMapAxis, Dict[str, _VMapAxis]]
