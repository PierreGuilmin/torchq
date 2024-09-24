from __future__ import annotations

from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .._utils import _is_batched_scalar
from .layout import Layout, dense
from .qarray import QArray, QArrayLike, _in_last_two_dims, isqarraylike

__all__ = ['DenseQArray']

# batched Kronecker product of two arrays
_bkron = jnp.vectorize(jnp.kron, signature='(a,b),(c,d)->(ac,bd)')


def _dense_to_qobj(x: DenseQArray) -> Qobj | list[Qobj]:
    if x.ndim > 2:
        return [_dense_to_qobj(sub_x, dims=x.dims) for sub_x in x]
    else:
        dims = list(x.dims)
        if x.isket():  # [[3], [1]] or for composite systems [[3, 4], [1, 1]]
            dims = [dims, [1] * len(dims)]
        elif x.isbra():  # [[1], [3]] or for composite systems [[1, 1], [3, 4]]
            dims = [[1] * len(dims), dims]
        elif x.isop():  # [[3], [3]] or for composite systems [[3, 4], [3, 4]]
            dims = [dims, dims]
        return Qobj(x, dims=dims)


def _getjaxarray(x: QArrayLike) -> Array:
    if isinstance(x, DenseQArray):
        return x.data
    elif isinstance(x, QArray):
        return x.to_jax()
    elif isinstance(x, list):
        return jnp.asarray(_getjaxarray(sub_x) for sub_x in x)
    else:
        return jnp.asarray(x)


class DenseQArray(QArray):
    r"""DenseQArray is QArray that uses JAX arrays as data storage."""

    data: Array

    @property
    def dtype(self) -> jnp.dtype:
        return self.data.dtype

    @property
    def layout(self) -> Layout:
        return dense

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def mT(self) -> QArray:
        data = self.data.mT
        return DenseQArray(self.dims, data)

    def conj(self) -> QArray:
        data = self.data.conj()
        return DenseQArray(self.dims, data)

    def reshape(self, *shape: int) -> QArray:
        data = jnp.reshape(self.data, shape)
        return DenseQArray(self.dims, data)

    def broadcast_to(self, *shape: int) -> QArray:
        data = jnp.broadcast_to(self.data, shape)
        return DenseQArray(self.dims, data)

    def ptrace(self, keep: tuple[int, ...]) -> QArray:
        from ..utils.quantum_utils.general import ptrace

        dims = tuple(self.dims[dim] for dim in keep)
        data = ptrace(self.data, keep, self.dims)
        return DenseQArray(dims, data)

    def powm(self, n: int) -> QArray:
        data = jnp.linalg.matrix_power(self.data, n)
        return DenseQArray(self.dims, data)

    def expm(self, *, max_squarings: int = 16) -> QArray:
        data = jax.scipy.linalg.expm(self.data, max_squarings=max_squarings)
        return DenseQArray(self.dims, data)

    def norm(self) -> Array:
        from ..utils.quantum_utils.general import norm

        return norm(self.data)

    def trace(self) -> Array:
        return self.data.trace(axis1=-1, axis2=-2)

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        data = self.data.sum(axis=axis)

        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            return data
        else:
            return DenseQArray(self.dims, data)

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        data = self.data.squeeze(axis=axis)

        # return array if last two dimensions are modified, qarray otherwise
        if _in_last_two_dims(axis, self.ndim):
            return data
        else:
            return DenseQArray(self.dims, data)

    def _eigh(self) -> tuple[Array, Array]:
        return jnp.linalg.eigh(self.data)

    def _eigvals(self) -> Array:
        return jnp.linalg.eigvals(self.data)

    def _eigvalsh(self) -> Array:
        return jnp.linalg.eigvalsh(self.data)

    def devices(self) -> set[Device]:
        return self.data.devices()

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        return jnp.allclose(self.data, self.data.mT.conj(), rtol=rtol, atol=atol)

    def to_qutip(self) -> Qobj | list[Qobj]:
        return _dense_to_qobj(self)

    def to_jax(self) -> Array:
        return self.data

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return np.asarray(self.data, dtype=dtype)

    def __repr__(self) -> str:
        return super().__repr__() + f'\n{self.data}'

    def __mul__(self, y: QArrayLike) -> QArray:
        super().__mul__(y)

        if _is_batched_scalar(y):
            data = self.data * y
        elif isinstance(y, DenseQArray):
            data = self.data * y.data
        elif isqarraylike(y):
            data = self.data * _getjaxarray(y)
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __truediv__(self, y: QArrayLike) -> QArray:
        super().__truediv__(y)

        if _is_batched_scalar(y):
            data = self.data / y
        elif isinstance(y, DenseQArray):
            data = self.data / y.data
        elif isqarraylike(y):
            data = self.data / _getjaxarray(y)
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __add__(self, y: QArray) -> QArray:
        super().__add__(y)

        if _is_batched_scalar(y):
            data = self.data + y
        elif isinstance(y, DenseQArray):
            data = self.data + y.data
        elif isinstance(y, get_args(ArrayLike)):
            data = self.data + _getjaxarray(y)
        else:
            return NotImplemented

        return DenseQArray(self.dims, data)

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        super().__matmul__(y)

        if isinstance(y, DenseQArray):
            dims = self.dims if len(self.dims) < len(y.dims) else y.dims
            data = self.data @ y.data
        elif isqarraylike(y):
            dims = self.dims
            data = self.data @ _getjaxarray(y)
        else:
            return NotImplemented

        if self.isbra() and y.isket():
            return data

        return DenseQArray(dims, data)

    def __rmatmul__(self, y: QArrayLike) -> QArray:
        super().__rmatmul__(y)

        if isinstance(y, DenseQArray):
            dims = self.dims if len(self.dims) < len(y.dims) else y.dims
            data = y.data @ self.data
        elif isqarraylike(y):
            dims = self.dims
            data = _getjaxarray(y) @ self.data
        else:
            return NotImplemented

        return DenseQArray(dims, data)

    def __and__(self, y: QArrayLike) -> QArray:
        super().__and__(y)

        if isinstance(y, DenseQArray):
            dims = self.dims + y.dims
            data = _bkron(self.data, y.data)
        elif isinstance(y, QArray):
            return NotImplemented
        else:
            y = _getjaxarray(y)
            y_dims = (y.shape[-2],) if y.shape[-2] != 1 else (y.shape[-1],)
            dims = self.dims + y_dims
            data = _bkron(self.data, y)

        return DenseQArray(dims, data)

    def _pow(self, power: int) -> QArray:
        data = self.data**power
        return DenseQArray(self.dims, data)

    def __getitem__(self, key: int | slice) -> QArray:
        data = self.data[key]
        return DenseQArray(self.dims, data)
