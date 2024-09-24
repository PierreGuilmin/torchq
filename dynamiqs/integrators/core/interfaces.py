import equinox as eqx
from jax import Array
from jax.random import PRNGKey

from ...options import Options
from ...time_array import TimeArray
from optimistix import AbstractRootFinder


class OptionsInterface(eqx.Module):
    options: Options


class SEInterface(eqx.Module):
    """Interface for the Schrödinger equation."""

    H: TimeArray


class MEInterface(eqx.Module):
    """Interface for the Lindblad master equation."""

    H: TimeArray
    Ls: list[TimeArray]


class MCInterface(eqx.Module):
    """Interface for the Monte-Carlo jump unraveling of the master equation."""

    H: TimeArray
    Ls: list[TimeArray]
    key: PRNGKey = PRNGKey(42)
    rand: float = 0.0
    root_finder: AbstractRootFinder | None = None


class SolveInterface(eqx.Module):
    Es: Array
