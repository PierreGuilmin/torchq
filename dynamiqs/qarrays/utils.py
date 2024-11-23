from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, DTypeLike
from qutip import Qobj

from .._checks import check_shape
from .dense_qarray import DenseQArray, _array_to_qobj_list, _dense_to_qobj
from .layout import Layout, dense
from .qarray import (
    QArray,
    QArrayLike,
    _dims_from_qutip,
    _dims_to_qutip,
    _to_jax,
    _to_numpy,
)
from .sparsedia_qarray import (
    SparseDIAQArray,
    _array_to_sparsedia,
    _sparsedia_to_dense,
    _sparsedia_to_qobj,
)

__all__ = ['asqarray', 'stack', 'to_jax', 'to_numpy', 'to_qutip', 'sparsedia_from_dict']


def asqarray(
    x: QArrayLike, dims: tuple[int, ...] | None = None, layout: Layout | None = None
) -> QArray:
    """Convert a qarray-like object into a qarray.

    Args:
        x: Object to convert.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the Hilbert
            space tensor product. Defaults to `None` (a single system with the same
            dimension as `x`).
        layout _(dq.dense, dq.dia or None)_: Matrix layout. If `None`, the default
            layout is `dq.dense`, except for qarrays that are directly returned.

    Returns:
        Qarray representation of the input.

    Examples:
        >>> dq.asqarray([[1, 0], [0, -1]])
        QArray: shape=(2, 2), dims=(2,), dtype=int32, layout=dense
        [[ 1  0]
         [ 0 -1]]
        >>> dq.asqarray([[1, 0], [0, -1]], layout=dq.dia)
        QArray: shape=(2, 2), dims=(2,), dtype=int32, layout=dia, ndiags=1
        [[ 1  ⋅]
         [ ⋅ -1]]
        >>> dq.asqarray([qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        QArray: shape=(3, 2, 2), dims=(2,), dtype=complex64, layout=dense
        [[[ 0.+0.j  1.+0.j]
          [ 1.+0.j  0.+0.j]]
        <BLANKLINE>
         [[ 0.+0.j  0.-1.j]
          [ 0.+1.j  0.+0.j]]
        <BLANKLINE>
         [[ 1.+0.j  0.+0.j]
          [ 0.+0.j -1.+0.j]]]
    """
    if layout is None and isinstance(x, QArray):
        return x

    layout = dense if layout is None else layout
    if layout is dense:
        return _asdense(x, dims=dims)
    else:
        return _assparsedia(x, dims=dims)


def _asdense(x: QArrayLike, dims: tuple[int, ...] | None = None) -> DenseQArray:
    _warn_qarray_dims(x, dims)

    if isinstance(x, DenseQArray):
        return x
    elif isinstance(x, SparseDIAQArray):
        return _sparsedia_to_dense(x)
    elif isinstance(x, Qobj):
        dims = _dims_from_qutip(x.dims)
        x = x.full()
    elif isinstance(x, Sequence):
        x = jax.tree.map(_asjnparray, x, is_leaf=_is_leaf)

    x = jnp.asarray(x)
    dims = _init_dims(x, dims)
    return DenseQArray(dims, x)


def _assparsedia(x: QArrayLike, dims: tuple[int, ...] | None = None) -> SparseDIAQArray:
    _warn_qarray_dims(x, dims)

    if isinstance(x, SparseDIAQArray):
        return x
    elif isinstance(x, DenseQArray):
        dims = x.dims
        x = x.to_jax()
    elif isinstance(x, Qobj):
        # TODO: improve this by directly extracting the diags and offsets in case
        # the Qobj is already in sparse DIA format (only for QuTiP 5)
        dims = _dims_from_qutip(x.dims)
        x = x.full()
    elif isinstance(x, Sequence):
        x = jax.tree.map(_asjnparray, x, is_leaf=_is_leaf)

    x = jnp.asarray(x)
    dims = _init_dims(x, dims)
    return _array_to_sparsedia(x, dims=dims)


def _asjnparray(x: QArrayLike) -> Array:
    if isinstance(x, QArray):
        return x.to_jax()
    elif isinstance(x, Qobj):
        return jnp.asarray(x.full())
    else:
        return jnp.asarray(x)


def _is_leaf(x: Any) -> bool:
    if isinstance(x, (QArray, Qobj)) or is_arraylike(x):
        return True

    try:
        return jnp.asarray(x).ndim >= 2
    except (TypeError, ValueError):
        return False


def is_arraylike(x: Any) -> bool:
    # see https://github.com/jax-ml/jax/issues/8701#issuecomment-979223360
    return hasattr(x, '__array__') or hasattr(x, '__array_interface__')


def stack(qarrays: Sequence[QArray], axis: int = 0) -> QArray:
    """Join a sequence of qarrays along a new axis.

    Warning:
        All elements of the sequence `qarrays` must have identical types, shapes and
        `dims` attributes. Additionally, when stacking qarrays of type
        `SparseDIAQArray`, all elements must have identical `offsets` attributes.

    Args:
        qarrays: Qarrays to stack.
        axis: Axis in the result along which the input qarrays are stacked.

    Returns:
        Stacked qarray.

    Examples:
        >>> dq.stack([dq.fock(3, 0), dq.fock(3, 1)])
        QArray: shape=(2, 3, 1), dims=(3,), dtype=complex64, layout=dense
        [[[1.+0.j]
          [0.+0.j]
          [0.+0.j]]
        <BLANKLINE>
         [[0.+0.j]
          [1.+0.j]
          [0.+0.j]]]
    """
    # check validity of input
    if len(qarrays) == 0:
        raise ValueError('Argument `qarrays` must contain at least one element.')
    if not all(isinstance(q, QArray) for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain only elements of type `QArray`.'
        )
    dims = qarrays[0].dims
    if not all(q.dims == dims for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain elements with identical `dims` attribute.'
        )

    # stack inputs depending on type
    if all(isinstance(q, DenseQArray) for q in qarrays):
        data = jnp.stack([q.data for q in qarrays], axis=axis)
        return DenseQArray(dims, data)
    elif all(isinstance(q, SparseDIAQArray) for q in qarrays):
        unique_offsets = set()
        for qarray in qarrays:
            unique_offsets.update(qarray.offsets)
        unique_offsets = tuple(sorted(unique_offsets))

        offset_to_index = {offset: idx for idx, offset in enumerate(unique_offsets)}
        diag_list = []
        for qarray in qarrays:
            add_diags_shape = qarray.diags.shape[:-2] + (
                len(unique_offsets),
                qarray.diags.shape[-1],
            )
            updated_diags = jnp.zeros(add_diags_shape, dtype=qarray.diags.dtype)
            for i, offset in enumerate(qarray.offsets):
                idx = offset_to_index[offset]
                updated_diags = updated_diags.at[..., idx, :].set(
                    qarray.diags[..., i, :]
                )
            diag_list.append(updated_diags)
        return SparseDIAQArray(dims, unique_offsets, jnp.stack(diag_list))
    else:
        raise NotImplementedError(
            'Stacking qarrays with different types is not implemented.'
        )


def to_jax(x: QArrayLike) -> Array:
    """Convert a qarray-like object into a JAX array.

    Args:
        x: Object to convert.

    Returns:
        JAX array.

    Examples:
        >>> dq.to_jax(dq.fock(3, 1))
        Array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.to_jax([qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        Array([[[ 0.+0.j,  1.+0.j],
                [ 1.+0.j,  0.+0.j]],
        <BLANKLINE>
               [[ 0.+0.j,  0.-1.j],
                [ 0.+1.j,  0.+0.j]],
        <BLANKLINE>
               [[ 1.+0.j,  0.+0.j],
                [ 0.+0.j, -1.+0.j]]], dtype=complex64)
    """
    return _to_jax(x)


def to_numpy(x: QArrayLike) -> np.ndarray:
    """Convert a qarray-like object into a NumPy array.

    Args:
        x: Object to convert.

    Returns:
        NumPy array.

    Examples:
        >>> dq.to_numpy(dq.fock(3, 1))
        array([[0.+0.j],
               [1.+0.j],
               [0.+0.j]], dtype=complex64)
        >>> dq.to_numpy([qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        array([[[ 0.+0.j,  1.+0.j],
                [ 1.+0.j,  0.+0.j]],
        <BLANKLINE>
               [[ 0.+0.j,  0.-1.j],
                [ 0.+1.j,  0.+0.j]],
        <BLANKLINE>
               [[ 1.+0.j,  0.+0.j],
                [ 0.+0.j, -1.+0.j]]])
    """
    return _to_numpy(x)


def to_qutip(x: QArrayLike, dims: tuple[int, ...] | None = None) -> Qobj | list[Qobj]:
    r"""Convert a qarray-like object into a QuTiP Qobj or list of Qobjs.

    Args:
        x _(qarray_like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra,
            density matrix or operator.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the Hilbert
            space tensor product. Defaults to `None` (a single system with the same
            dimension as `x`).

    Returns:
        QuTiP Qobj or list of QuTiP Qobj.

    Examples:
        >>> dq.fock(3, 1)
        QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
        [[0.+0.j]
         [1.+0.j]
         [0.+0.j]]
        >>> dq.to_qutip(dq.fock(3, 1))
        Quantum object: dims=[[3], [1]], shape=(3, 1), type='ket', dtype=Dense
        Qobj data =
        [[0.]
         [1.]
         [0.]]

        # For a batched array:
        # >>> rhos = dq.stack([dq.coherent_dm(16, i) for i in range(5)])
        # >>> rhos.shape
        # (5, 16, 16)
        # todo: temporary fix
        # >>> len(dq.to_qutip(rhos))
        # 5

        Note that the tensor product structure is inferred automatically for qarrays. It
        can be specified with the `dims` argument for other types.
        >>> dq.to_qutip(dq.eye(3, 2))
        Quantum object: dims=[[3, 2], [3, 2]], shape=(6, 6), type='oper', dtype=Dense, isherm=True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    _warn_qarray_dims(x, dims)

    if isinstance(x, Qobj):
        return x
    elif isinstance(x, DenseQArray):
        return _dense_to_qobj(x)
    elif isinstance(x, SparseDIAQArray):
        return _sparsedia_to_qobj(x)
    elif isinstance(x, Sequence):
        x = jax.tree.map(_asjnparray, x, is_leaf=_is_leaf)

    x = jnp.asarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')
    dims = _init_dims(x, dims)
    dims = _dims_to_qutip(dims, x.shape)
    return _array_to_qobj_list(x, dims)


def _warn_qarray_dims(x: QArrayLike, dims: tuple[int, ...] | None = None):
    if dims is not None:
        if isinstance(x, QArray) and x.dims != dims:
            warnings.warn(
                f'Argument `x` is already a QArray with dims={x.dims}, but dims '
                f'were also provided as input with dims={dims}. Ignoring the '
                'provided `dims` and proceeding with `x.dims`.',
                stacklevel=2,
            )
        elif isinstance(x, Qobj) and _dims_from_qutip(x.dims) != dims:
            warnings.warn(
                f'Argument `x` is already a Qobj with dims={x.dims}, but dims '
                f'were also provided as input with dims={dims}. Ignoring the '
                'provided `dims` and proceeding with `x.dims`.',
                stacklevel=2,
            )


def sparsedia_from_dict(
    offsets_diags: dict[int, ArrayLike],
    dims: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
) -> SparseDIAQArray:
    """Initialize a `SparseDIAQArray` from a dictionary of offsets and diagonals.

    Args:
        offsets_diags: Dictionary where keys are offsets and values are diagonals of
            shapes _(..., n-|offset|)_ with a common batch shape between all diagonals.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the Hilbert
            space tensor product. Defaults to `None` (a single system with the same
            dimension as `x`).
        dtype: Data type of the array. If `None`, the data type is inferred from the
            diagonals.

    Returns:
        A `SparseDIAQArray` with non-zero diagonals at the specified offsets.

    Examples:
        >>> dq.sparsedia_from_dict({0: [1, 2, 3], 1: [4, 5], -1: [6, 7]})
        QArray: shape=(3, 3), dims=(3,), dtype=int32, layout=dia, ndiags=3
        [[1 4 ⋅]
         [6 2 5]
         [⋅ 7 3]]
        >>> dq.sparsedia_from_dict({0: jnp.ones((3, 2))})
        QArray: shape=(3, 2, 2), dims=(2,), dtype=float32, layout=dia, ndiags=1
        [[[1. ⋅ ]
          [ ⋅ 1.]]
        <BLANKLINE>
         [[1. ⋅ ]
          [ ⋅ 1.]]
        <BLANKLINE>
         [[1. ⋅ ]
          [ ⋅ 1.]]]
    """
    # === offsets
    offsets = tuple(offsets_diags.keys())

    # === diags
    # stack arrays in a square matrix by padding each according to its offset
    pads_width = [(abs(k), 0) if k >= 0 else (0, abs(k)) for k in offsets]
    diags = [jnp.asarray(diag) for diag in offsets_diags.values()]
    diags = [jnp.pad(diag, pad_width) for pad_width, diag in zip(pads_width, diags)]
    diags = jnp.stack(diags, dtype=dtype)
    diags = jnp.moveaxis(diags, 0, -2)

    # === dims
    n = diags.shape[-1]
    shape = (*diags.shape[:-2], n, n)
    dims = (n,) if dims is None else dims
    _check_dims_match_shape(shape, dims)

    return SparseDIAQArray(diags=diags, offsets=offsets, dims=dims)


def _init_dims(x: Array, dims: tuple[int, ...] | None = None) -> tuple[int, ...]:
    if dims is None:
        dims = (x.shape[-2],) if x.shape[-2] != 1 else (x.shape[-1],)

    _check_dims_match_shape(x.shape, dims)

    # TODO: check if is bra, ket, dm or op
    # if not (isbra(data) or isket(data) or isdm(data) or isop(data)):
    # raise ValueError(
    #     f'DenseQArray data must be a bra, a ket, a density matrix '
    #     f'or and operator. Got array with size {data.shape}'
    # )

    return dims


def _check_dims_match_shape(shape: tuple[int, ...], dims: tuple[int, ...]):
    if np.prod(dims) != np.max(shape[-2:]):
        raise ValueError(
            'The provided `dims` are incompatible with the input array. '
            f'Got dims={dims} and shape={shape}.'
        )
