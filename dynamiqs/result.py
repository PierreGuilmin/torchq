from __future__ import annotations

import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from .gradient import Gradient
from .options import Options
from .qarrays.qarray import QArray
from .qarrays.utils import to_jax
from .solver import Solver

__all__ = [
    'FloquetResult',
    'MCSolveResult',
    'MEPropagatorResult',
    'MESolveResult',
    'SEPropagatorResult',
    'SESolveResult',
]


def memory_bytes(x: Array) -> int:
    return x.itemsize * x.size


def memory_str(x: Array) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.1f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.1f} Mb'
    else:
        return f'{mem / 1024**3:.1f} Gb'


def array_str(x: Array | QArray | None) -> str | None:
    # TODO: implement memory_str for `QArray` rather than converting to JAX array
    if x is None:
        return None
    type_name = 'Array' if isinstance(x, Array) else 'QArray'
    x = to_jax(x)
    return f'{type_name} {x.dtype} {tuple(x.shape)} | {memory_str(x)}'


# the Saved object holds quantities saved during the equation integration
class Saved(eqx.Module):
    ysave: QArray
    extra: PyTree | None


class SolveSaved(Saved):
    Esave: Array | None


class PropagatorSaved(Saved):
    pass


class FloquetSaved(Saved):
    quasienergies: Array


class Result(eqx.Module):
    tsave: Array
    solver: Solver
    gradient: Gradient | None
    options: Options
    _saved: Saved
    infos: PyTree | None

    @property
    def extra(self) -> PyTree | None:
        return self._saved.extra

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError

    def block_until_ready(self) -> Result:
        _ = self._saved.ysave.block_until_ready()
        return self

    def _str_parts(self) -> dict[str, str | None]:
        return {
            'Solver': type(self.solver).__name__,
            'Gradient': (
                type(self.gradient).__name__ if self.gradient is not None else None
            ),
            'Infos': self.infos if self.infos is not None else None,
            'Extra': (eqx.tree_pformat(self.extra) if self.extra is not None else None),
        }

    def __str__(self) -> str:
        parts = self._str_parts()

        # remove None values
        parts = {k: v for k, v in parts.items() if v is not None}

        # pad to align colons
        padding = max(len(k) for k in parts) + 1
        parts_str = '\n'.join(f'{k:<{padding}}: {v}' for k, v in parts.items())
        return f'==== {self.__class__.__name__} ====\n' + parts_str

    @classmethod
    def out_axes(cls) -> SolveResult:
        return cls(None, None, None, None, 0, 0)


class SolveResult(Result):
    @property
    def states(self) -> QArray:
        return self._saved.ysave

    @property
    def final_state(self) -> QArray:
        return self.states[..., -1, :, :]

    @property
    def expects(self) -> Array | None:
        return self._saved.Esave

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {
            'States': array_str(self.states),
            'Expects': array_str(self.expects),
        }


class PropagatorResult(Result):
    @property
    def propagators(self) -> QArray:
        return self._saved.ysave

    @property
    def final_propagator(self) -> QArray:
        return self.propagators[..., -1, :, :]

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {'Propagators': array_str(self.propagators)}


class FloquetResult(Result):
    """Result of the Floquet integration.

    Attributes:
        modes _(qarray of shape (..., ntsave, n, n, 1))_: Saved Floquet modes.
        quasienergies _(array of shape (..., n))_: Saved quasienergies
        T _(float)_: Drive period
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.

    Note-: Result of running multiple simulations concurrently
        The resulting Floquet modes and quasienergies are batched according to the
        leading dimensions of the Hamiltonian `H`. For example if `H` has shape
        _(2, 3, n, n)_, then `modes` has shape _(2, 3, ntsave, n, n, 1)_.

        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.
    """

    T: float

    @property
    def modes(self) -> QArray:
        return self._saved.ysave

    @property
    def quasienergies(self) -> Array:
        return self._saved.quasienergies

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {
            'Modes': array_str(self.modes),
            'Quasienergies': array_str(self.quasienergies),
        }

    @classmethod
    def out_axes(cls) -> SolveResult:
        return cls(None, None, None, None, 0, 0, None)


class SESolveResult(SolveResult):
    r"""Result of the Schrödinger equation integration.


    Attributes:
        states _(qarray of shape (..., nsave, n, 1))_: Saved states with
            `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to `False`.
        final_state _(qarray of shape (..., n, 1))_: Saved final state.
        expects _(array of shape (..., len(exp_ops), ntsave) or None)_: Saved
            expectation values, if specified by `exp_ops`.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.

    Note-: Result of running multiple simulations concurrently
        The resulting states and expectation values are batched according to the
        leading dimensions of the Hamiltonian `H` and initial state `psi0`. The
        behaviour depends on the value of the `cartesian_batching` option

        === "If `cartesian_batching = True` (default value)"
            The results leading dimensions are
            ```
            ... = ...H, ...psi0
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `psi0` has shape _(4, n, 1)_,

            then `states` has shape _(2, 3, 4, ntsave, n, 1)_.

        === "If `cartesian_batching = False`"
            The results leading dimensions are
            ```
            ... = ...H = ...psi0  # (once broadcasted)
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `psi0` has shape _(3, n, 1)_,

            then `states` has shape _(2, 3, ntsave, n, 1)_.

        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.
    """


class MESolveResult(SolveResult):
    """Result of the Lindblad master equation integration.

    Attributes:
        states _(qarray of shape (..., nsave, n, n))_: Saved states with
            `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to `False`.
        final_state _(qarray of shape (..., n, n))_: Saved final state.
        expects _(array of shape (..., len(exp_ops), ntsave) or None)_: Saved
            expectation values, if specified by `exp_ops`.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.

    Note-: Result of running multiple simulations concurrently
        The resulting states and expectation values are batched according to the
        leading dimensions of the Hamiltonian `H`, jump operators `jump_ops` and initial
        state `rho0`. The behaviour depends on the value of the `cartesian_batching`
        option

        === "If `cartesian_batching = True` (default value)"
            The results leading dimensions are
            ```
            ... = ...H, ...L0, ...L1, (...), ...rho0
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
            - `rho0` has shape _(7, n, n)_,

            then `states` has shape _(2, 3, 4, 5, 6, 7, ntsave, n, n)_.
        === "If `cartesian_batching = False`"
            The results leading dimensions are
            ```
            ... = ...H = ...L0 = ...L1 = (...) = ...rho0  # (once broadcasted)
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
            - `rho0` has shape _(3, n, n)_,

            then `states` has shape _(2, 3, ntsave, n, n)_.

        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.
    """


class MCJumpResult(SolveResult):
    """Result of Monte Carlo jump trajectories.


    Attributes:
        states _(qarray of shape (..., ntraj, nsave, n, 1))_: Saved jump states with
            `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to `False`.
        final_state _(qarray of shape (..., ntraj, n, 1))_: Saved final jump state.
        jump_times _(Array)_: Times at which each trajectory experienced a jump. This
            quantity has shape ..., options.max_jumps where the array is filled with
            nans for the final options.max_jumps - num_jumps values.
        num_jumps _(Array)_: Number of jumps each jump trajectory experienced. The times
            at which each jump occurred is saved in jump_times.
        expects _(array of shape (..., len(exp_ops), ntsave) or None)_: Saved
            expectation values, if specified by `exp_ops`.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.
    """

    jump_times: Array
    num_jumps: Array


class MCNoJumpResult(SolveResult):
    """Result of Monte Carlo no-jump trajectories.


    Attributes:
        states _(qarray of shape (..., nsave, n, 1))_: Saved no-jump states with
            `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to `False`.
        final_state _(qarray of shape (..., n, 1))_: Saved no-jump final state.
        no_jump_prob _(Array)_: No jump probability.
        expects _(array of shape (..., len(exp_ops), ntsave) or None)_: Saved
            expectation values, if specified by `exp_ops`.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.
    """

    no_jump_prob: Array


class MCSolveResult(SolveResult):
    """Result of Monte Carlo integration.

    Attributes:
        states _(qarray of shape (..., nsave, n, n))_: Saved states with
            `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to `False`.
        final_state _(qarray of shape (..., n, n))_: Saved final state.
        no_jump_states _(qarray of shape (..., nsave, n, 1))_: Saved no-jump states with
            `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to `False`.
        no_jump_final_state _(qarray of shape (..., n, 1))_: Saved no-jump final state.
        jump_states _(qarray of shape (..., ntraj, nsave, n, 1))_: Saved jump states
            with `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to
            `False`.
        jump_final_states _(qarray of shape (..., ntraj, n, 1))_: Saved final jump
            states.
        no_jump_prob _(Array)_: No jump probability.
        jump_times _(Array)_: Times at which each trajectory experienced a jump. This
            quantity has shape ..., options.max_jumps where the array is filled with
            nans for the final options.max_jumps - num_jumps values.
        num_jumps _(Array)_: Number of jumps each jump trajectory experienced. The times
            at which each jump occurred is saved in jump_times.
        expects _(Array, optional)_: Saved expectation values.
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.


    Note-: Result of running multiple simulations concurrently
        The resulting states and expectation values are batched according to the number
        of trajectories (specified by the number of `keys` passed to `mcsolve`), the
        leading dimensions of the Hamiltonian `H`, jump operators `jump_ops` and initial
        state `psi0`. The behaviour depends on the value of the `cartesian_batching`
        option

        === "If `cartesian_batching = True` (default value)"
            The results leading dimensions are
            ```
            ... = ...H, ...L0, ...L1, (...), ...psi0
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
            - `psi0` has shape _(7, n, 1)_,
            - `keys` has len(keys) == 8

            then `states` (which is an average over the jump and no-jump states) has
            shape _(2, 3, 4, 5, 6, 7, ntsave, n, n)_.
            `no_jump_states` has shape _(2, 3, 4, 5, 6, 7, ntsave, n, 1)_ and
            `jump_states` has shape _(2, 3, 4, 5, 6, 7, 8, ntsave, n, 1)_
        === "If `cartesian_batching = False`"
            The results leading dimensions are
            ```
            ... = ...H = ...L0 = ...L1 = (...) = ...rho0 = ...keys  # (once broadcasted)
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
            - `rho0` has shape _(3, n, n)_,
            - `keys` has shape (2, 3, 4)

            then `states` has shape _(2, 3, ntsave, n, n)_,
            `no_jump_states` has shape _(2, 3, ntsave, n, 1)_ and
            `jump_states` has shape _(2, 3, 4, ntsave, n, 1)_ and

        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.
    """

    _no_jump_result: MCNoJumpResult
    _jump_result: MCJumpResult

    @property
    def no_jump_states(self) -> QArray:
        return self._no_jump_result.states

    @property
    def jump_states(self) -> QArray:
        return self._jump_result.states

    @property
    def final_no_jump_state(self) -> QArray:
        return self._no_jump_result.final_state

    @property
    def final_jump_states(self) -> QArray:
        return self._jump_result.final_state

    @property
    def no_jump_prob(self) -> Array:
        return self._no_jump_result.no_jump_prob

    @property
    def jump_times(self) -> Array:
        return self._jump_result.jump_times

    @property
    def num_jumps(self) -> Array:
        return self._jump_result.jump_times

    def __str__(self) -> str:
        parts = {
            'No-jump states': array_str(self.no_jump_states),
            'Jump states': array_str(self.jump_states),
            'No-jump probability  ': array_str(self.no_jump_prob),
            'Jump times  ': array_str(self.jump_times),
            'Number of jumps in each trajectory  ': array_str(self.num_jumps),
            'Infos': self.infos if self.infos is not None else None,
            'Expects ': array_str(self.expects) if self.expects is not None else None,
        }
        parts = {k: v for k, v in parts.items() if v is not None}
        parts_str = '\n'.join(f'{k}: {v}' for k, v in parts.items())
        return '==== MCResult ====\n' + parts_str

    @classmethod
    def out_axes(cls) -> SolveResult:
        return cls(None, None, None, None, 0, 0, 0, 0)


class SEPropagatorResult(PropagatorResult):
    r"""Result of the Schrödinger equation integration to obtain the propagator.

    Attributes:
        propagators _(qarray of shape (..., nsave, n, n))_: Saved propagators with
            `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to `False`.
        final_propagator _(qarray of shape (..., n, n))_: Saved final propagator.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.

    Note-: Result of running multiple simulations concurrently
        The resulting propagators are batched according to the leading
        dimensions of the Hamiltonian `H`. For example if `H` has shape
        _(2, 3, n, n)_, then `propagators` has shape _(2, 3, ntsave, n, n)_.

        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.
    """


class MEPropagatorResult(PropagatorResult):
    r"""Result of the Lindblad master equation integration to obtain the propagator.

    Attributes:
        propagators _(qarray of shape (..., nsave, n^2, n^2))_: Saved propagators with
            `nsave = ntsave`, or `nsave = 1` if `options.save_states` is set to `False`.
        final_propagator _(qarray of shape (..., n^2, n^2))_: Saved final propagator.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options` (see [`dq.Options`][dynamiqs.Options]).
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.

    Note-: Result of running multiple simulations concurrently
        The resulting propagators are batched according to the
        leading dimensions of the Hamiltonian `H` and jump operators `jump_ops`.
        The behaviour depends on the value of the `cartesian_batching` option

        === "If `cartesian_batching = True` (default value)"
            The results leading dimensions are
            ```
            ... = ...H, ...L0, ...L1, (...)
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,

            then `propagators` has shape _(2, 3, 4, 5, 6, ntsave, n, n)_.
        === "If `cartesian_batching = False`"
            The results leading dimensions are
            ```
            ... = ...H = ...L0 = ...L1 = (...)  # (once broadcasted)
            ```
            For example if:

            - `H` has shape _(2, 3, n, n)_,
            - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,

            then `propagators` has shape _(2, 3, ntsave, n, n)_.

        See the
        [Batching simulations](../../documentation/basics/batching-simulations.md)
        tutorial for more details.
    """
