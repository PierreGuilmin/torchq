from __future__ import annotations

import re
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from qutip import Qobj

from .._utils import _is_batched_scalar
from .dense_qarray import DenseQArray
from .layout import Layout, dia
from .qarray import (
    QArray,
    QArrayLike,
    _in_last_two_dims,
    _include_last_two_dims,
    _to_jax,
    isqarraylike,
)
from .sparsedia_primitives import (
    add_sparsedia_sparsedia,
    and_sparsedia_sparsedia,
    broadcast_sparsedia,
    matmul_array_sparsedia,
    matmul_sparsedia_array,
    matmul_sparsedia_sparsedia,
    mul_sparsedia_array,
    mul_sparsedia_sparsedia,
    powm_sparsedia,
    reshape_sparsedia,
    shape_sparsedia,
    sparsedia_to_array,
    trace_sparsedia,
    transpose_sparsedia,
)

__all__ = ['SparseDIAQArray']


class SparseDIAQArray(QArray):
    offsets: tuple[int, ...] = eqx.field(static=True)
    diags: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        # check diags and offsets have the right type and shape before compressing them
        if not isinstance(self.offsets, tuple):
            raise TypeError(
                'Argument `offsets` of `SparseDIAQArray` must be a tuple but got '
                f'{type(self.offsets)}'
            )

        if self.diags.ndim < 2 or self.diags.shape[-2] != len(self.offsets):
            raise ValueError(
                'Argument `diags` of `SparseDIAQArray` must be of shape '
                f'(..., len(offsets), prod(dims)), but got {self.diags.shape}'
            )

        # check that diagonals contain zeros outside the bounds of the matrix using
        # equinox runtime checks
        error = (
            'Diagonals of a `SparseDIAQArray` must contain zeros outside the '
            'matrix bounds.'
        )
        for i, offset in enumerate(self.offsets):
            zero_slice = slice(None, offset) if offset >= 0 else slice(offset, None)
            check = self.diags[..., i, zero_slice] != 0
            eqx.error_if(self.diags, check, error)

    @property
    def dtype(self) -> jnp.dtype:
        return self.diags.dtype

    @property
    def layout(self) -> Layout:
        return dia

    @property
    def shape(self) -> tuple[int, ...]:
        return shape_sparsedia(self.diags)

    @property
    def mT(self) -> QArray:
        offsets, diags = transpose_sparsedia(self.offsets, self.diags)
        return SparseDIAQArray(self.dims, offsets, diags)

    @property
    def ndiags(self) -> int:
        return len(self.offsets)

    def conj(self) -> QArray:
        return SparseDIAQArray(self.dims, self.offsets, self.diags.conj())

    def reshape(self, *shape: int) -> QArray:
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot reshape to shape {shape} because'
                f' the last two dimensions do not match current '
                f'shape dimensions ({self.shape})'
            )
        offsets, diags = reshape_sparsedia(self.offsets, self.diags, shape)
        return SparseDIAQArray(self.dims, offsets, diags)

    def broadcast_to(self, *shape: int) -> QArray:
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot broadcast to shape {shape} because'
                f' the last two dimensions do not match current '
                f'shape dimensions ({self.shape})'
            )

        offsets, diags = broadcast_sparsedia(self.offsets, self.diags, shape)
        return SparseDIAQArray(self.dims, offsets, diags)

    def ptrace(self, *keep: int) -> QArray:
        raise NotImplementedError

    def powm(self, n: int) -> QArray:
        offsets, diags = powm_sparsedia(self.offsets, self.diags, n)
        return SparseDIAQArray(self.dims, offsets, diags)

    def expm(self, *, max_squarings: int = 16) -> QArray:
        # todo: implement dia specific method or raise warning for dense conversion
        x = sparsedia_to_array(self.offsets, self.diags)
        expm_x = jax.linalg.expm(x, max_squarings=max_squarings)
        return DenseQArray(self.dims, expm_x)

    def norm(self) -> Array:
        return self.trace()

    def trace(self) -> Array:
        return trace_sparsedia(self.offsets, self.diags)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> Array:
        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            if _include_last_two_dims(axis, self.ndim):
                return self.diags.sum(axis)
            else:
                return self.to_jax().sum(axis)
        else:
            return SparseDIAQArray(self.dims, self.offsets, self.diags.sum(axis))

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            if _include_last_two_dims(axis, self.ndim):
                return self.diags.squeeze(axis)
            else:
                return self.to_jax().squeeze(axis)
        else:
            return SparseDIAQArray(self.dims, self.offsets, self.diags.squeeze(axis))

    def _eigh(self) -> tuple[Array, Array]:
        raise NotImplementedError

    def _eigvals(self) -> Array:
        raise NotImplementedError

    def _eigvalsh(self) -> Array:
        raise NotImplementedError

    def devices(self) -> set[jax.Device]:
        raise NotImplementedError

    def asdense(self) -> DenseQArray:
        data = sparsedia_to_array(self.offsets, self.diags)
        return DenseQArray(self.dims, data)

    def assparsedia(self) -> SparseDIAQArray:
        return self

    def isherm(self) -> bool:
        raise NotImplementedError

    def to_qutip(self) -> Qobj | list[Qobj]:
        return self.asdense().to_qutip()

    def to_jax(self) -> Array:
        return self.asdense().to_jax()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return self.asdense().__array__(dtype=dtype, copy=copy)

    def block_until_ready(self) -> QArray:
        _ = self.diags.block_until_ready()
        return self

    def __repr__(self) -> str:
        # === array representation with dots instead of zeros
        if jnp.issubdtype(self.dtype, jnp.complexfloating):
            # match '0. +0.j' with any number of spaces
            pattern = r'(?<!\d)0\.\s*(\+|\-)0\.j'
        elif jnp.issubdtype(self.dtype, jnp.floating):
            # match '0.' with any number of spaces
            pattern = r'(?<!\d)0\.\s*'
        elif jnp.issubdtype(self.dtype, jnp.integer):
            # match '0' with any number of spaces
            pattern = r'(?<!\d)0\s*'
        else:
            raise ValueError(
                'Unsupported dtype for SparseDIAQArray representation, got '
                f'{self.dtype}.'
            )

        # replace with a centered dot of the same length as the matched string
        replace_with_dot = lambda match: f"{'⋅':^{len(match.group(0))}}"
        data_str = re.sub(pattern, replace_with_dot, str(self.to_jax()))
        return super().__repr__() + f', ndiags={self.ndiags}\n{data_str}'

    def __mul__(self, other: QArrayLike) -> QArray:
        super().__mul__(other)

        if _is_batched_scalar(other):
            return SparseDIAQArray(self.dims, self.offsets, other * self.diags)
        elif isinstance(other, SparseDIAQArray):
            offsets, diags = mul_sparsedia_sparsedia(
                self.offsets, self.diags, other.offsets, other.diags
            )
            return SparseDIAQArray(self.dims, offsets, diags)
        elif isqarraylike(other):
            other = _to_jax(other)
            offsets, diags = mul_sparsedia_array(self.offsets, self.diags, other)
            return SparseDIAQArray(self.dims, offsets, diags)

        return NotImplemented

    def __truediv__(self, other: QArrayLike) -> QArray:
        raise NotImplementedError

    def __add__(self, other: QArrayLike) -> QArray:
        super().__add__(other)

        warning_dense_addition = (
            'A sparse array has been converted to dense format due to '
            'addition with a scalar or dense array.'
        )

        if isinstance(other, SparseDIAQArray):
            offsets, diags = add_sparsedia_sparsedia(
                self.offsets, self.diags, other.offsets, other.diags
            )
            return SparseDIAQArray(self.dims, offsets, diags)
        elif _is_batched_scalar(other) or isqarraylike(other):
            warnings.warn(warning_dense_addition, stacklevel=2)
            return self.asdense() + other

        return NotImplemented

    def __matmul__(self, other: QArrayLike) -> QArray:
        if _is_batched_scalar(other):
            raise TypeError('Attempted matrix product between a scalar and a QArray.')

        if isinstance(other, SparseDIAQArray):
            offsets, diags = matmul_sparsedia_sparsedia(
                self.offsets, self.diags, other.offsets, other.diags
            )
            return SparseDIAQArray(self.dims, offsets, diags)
        elif isqarraylike(other):
            other = _to_jax(other)
            data = matmul_sparsedia_array(self.offsets, self.diags, other)
            return DenseQArray(self.dims, data)

        return NotImplemented

    def __rmatmul__(self, other: QArrayLike) -> QArray:
        if _is_batched_scalar(other):
            raise TypeError('Attempted matrix product between a scalar and a QArray.')

        if isqarraylike(other):
            other = _to_jax(other)
            data = matmul_array_sparsedia(other, self.offsets, self.diags)
            return DenseQArray(self.dims, data)

        return NotImplemented

    def __and__(self, other: QArray) -> QArray:
        if isinstance(other, SparseDIAQArray):
            offsets, diags = and_sparsedia_sparsedia(
                self.offsets, self.diags, other.offsets, other.diags
            )
            dims = self.dims + other.dims
            return SparseDIAQArray(dims, offsets, diags)
        elif isinstance(other, DenseQArray):
            return self.asdense() & other

        return NotImplemented

    def __rand__(self, other: QArray) -> QArray:
        if isinstance(other, DenseQArray):
            return other & self.asdense()

        return NotImplemented

    def _pow(self, power: int) -> QArray:
        return SparseDIAQArray(self.dims, self.offsets, self.diags**power)

    def __getitem__(self, key: int | slice | tuple) -> QArray:
        if key in (slice(None, None, None), Ellipsis):
            return self

        _check_key_in_batch_dims(key, self.ndim)
        return SparseDIAQArray(self.dims, self.offsets, self.diags[key])


def _check_key_in_batch_dims(key: int | slice | tuple, ndim: int):
    full_slice = slice(None, None, None)
    valid_key = False
    if isinstance(key, (int, slice)):
        valid_key = ndim > 2
    elif isinstance(key, tuple):
        if Ellipsis in key:
            ellipsis_key = key.index(Ellipsis)
            key = (
                key[:ellipsis_key]
                + (full_slice,) * (ndim - len(key) + 1)
                + key[ellipsis_key + 1 :]
            )

        valid_key = (
            len(key) <= ndim - 2
            or (len(key) == ndim - 1 and key[-1] == full_slice)
            or (len(key) == ndim and key[-2] == full_slice and key[-1] == full_slice)
        )

    if not valid_key:
        raise NotImplementedError(
            'Getting items from non batching dimensions of a SparseDIAQArray is not '
            'supported.'
        )
