from __future__ import annotations

from math import prod

import torch
from torch import Tensor

from .operators import displace, rnd_unitary
from .tensor_types import get_cdtype
from .utils import ket_to_dm

__all__ = [
    'fock',
    'fock_dm',
    'coherent',
    'coherent_dm',
    'rnd_ket',           
]


def fock(
    dims: int | tuple[int, ...],
    states: int | tuple[int, ...],
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the ket of a Fock state or the ket of a tensor product of Fock states.

    Args:
        dims _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        states _(int or tuple of ints)_: Fock state of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, 1)_ Ket of the Fock state or tensor product of Fock states.

    Examples:
        >>> dq.fock(3, 1)
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]])
        >>> dq.fock((3, 2), (1, 0))
        tensor([[0.+0.j],
                [0.+0.j],
                [1.+0.j],
                [0.+0.j],
                [0.+0.j],
                [0.+0.j]])
    """
    # convert integer inputs to tuples by default, and check dimensions match
    dims = (dims,) if isinstance(dims, int) else dims
    states = (states,) if isinstance(states, int) else states
    if len(dims) != len(states):
        raise ValueError(
            'Arguments `states` must have the same length as `dims` of length'
            f' {len(dims)}, but has length {len(states)}.'
        )

    # compute the required basis state
    n = 0
    for dim, state in zip(dims, states):
        n = dim * n + state
    ket = torch.zeros(prod(dims), 1, dtype=get_cdtype(dtype), device=device)
    ket[n] = 1.0
    return ket


def fock_dm(
    dims: int | tuple[int, ...],
    states: int | tuple[int, ...],
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the density matrix of a Fock state or the density matrix of a tensor
    product of Fock states.

    Args:
        dims _(int or tuple of ints)_: Dimension of the Hilbert space of each mode.
        states _(int or tuple of ints)_: Fock state of each mode.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, n)_ Density matrix of the Fock state or tensor product of Fock states.

    Examples:
        >>> dq.fock_dm(3, 1)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 1.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j]])
        >>> dq.fock_dm((3, 2), (1, 0))
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """
    return ket_to_dm(fock(dims, states, dtype=get_cdtype(dtype), device=device))


def coherent(
    dim: int,
    alpha: complex | Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns the ket of a coherent state.

    Args:
        dim: Dimension of the Hilbert space.
        alpha: Coherent state amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, 1)_ Ket of the coherent state.

    Examples:
        >>> dq.coherent(4, 0.5)
        tensor([[0.882+0.j],
                [0.441+0.j],
                [0.156+0.j],
                [0.047+0.j]])
    """
    cdtype = get_cdtype(dtype)
    return displace(dim, alpha, dtype=cdtype, device=device) @ fock(
        dim, 0, dtype=cdtype, device=device
    )


def coherent_dm(
    dim: int,
    alpha: complex | Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Density matrix of a coherent state.

    Args:
        dim: Dimension of the Hilbert space.
        alpha: Coherent state amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(n, n)_ Density matrix of the coherent state.

    Examples:
        >>> dq.coherent_dm(4, 0.5)
        tensor([[0.779+0.j, 0.389+0.j, 0.137+0.j, 0.042+0.j],
                [0.389+0.j, 0.195+0.j, 0.069+0.j, 0.021+0.j],
                [0.137+0.j, 0.069+0.j, 0.024+0.j, 0.007+0.j],
                [0.042+0.j, 0.021+0.j, 0.007+0.j, 0.002+0.j]])
    """
    return ket_to_dm(coherent(dim, alpha, dtype=get_cdtype(dtype), device=device))

def rnd_ket(dim: int, *, dtype: torch.complex64 | torch.complex128 | None = None, device: str | torch.device | None = None):
    r"""
    Returns a random ket of dimension 'n, 1' sampled from the Haar measure.

    Args:
        dim: Dimension of the Hilbert space.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(..., n, 1)_ Ket.

    Examples:
        >>> dq.rnd_ket(3)
        tensor([[-0.5952-0.2949j],
                [-0.4213+0.0871j],
                [ 0.1651-0.5885j]])
    """
    psi = rnd_unitary(dim, dtype=get_cdtype(dtype), device=device) @ \
        fock(dim, 0, dtype=get_cdtype(dtype), device=device)
    return psi
