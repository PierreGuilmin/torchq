from __future__ import annotations

import equinox as eqx
from jax import Array
from jaxtyping import Scalar
from optimistix import AbstractRootFinder

from ...options import Options
from ...qarrays.qarray import QArray
from ...time_array import TimeArray


class OptionsInterface(eqx.Module):
    options: Options


class SEInterface(eqx.Module):
    """Interface for the Schrödinger equation."""

    H: TimeArray


class MEInterface(eqx.Module):
    """Interface for the Lindblad master equation."""

    H: TimeArray
    Ls: list[TimeArray]

    def L(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Ls]  # (nLs, n, n)


class MCInterface(MEInterface):
    """Interface for the Monte-Carlo jump unraveling of the master equation."""

    keys: Array
    root_finder: AbstractRootFinder | None


class SolveInterface(eqx.Module):
    Es: QArray
