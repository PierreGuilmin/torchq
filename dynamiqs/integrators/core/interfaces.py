from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import Scalar
from optimistix import AbstractRootFinder

from ..._utils import _concatenate_sort
from ...options import Options
from ...qarrays.qarray import QArray
from ...time_qarray import TimeQArray


class OptionsInterface(eqx.Module):
    options: Options


class AbstractTimeInterface(eqx.Module):
    @property
    @abstractmethod
    def discontinuity_ts(self) -> Array | None:
        pass


class SEInterface(AbstractTimeInterface):
    """Interface for the Schrödinger equation."""

    H: TimeQArray

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.H.discontinuity_ts


class MEInterface(AbstractTimeInterface):
    """Interface for the Lindblad master equation."""

    H: TimeQArray
    Ls: list[TimeQArray]

    def L(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Ls]  # (nLs, n, n)

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return _concatenate_sort(*ts)


class MCInterface(MEInterface):
    """Interface for the Monte-Carlo jump unraveling of the master equation."""

    keys: Array
    root_finder: AbstractRootFinder | None


class SolveInterface(eqx.Module):
    Es: list[QArray] | None
