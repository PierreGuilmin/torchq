from __future__ import annotations

import warnings
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray, Scalar

from ...qarrays.qarray import QArray
from ...qarrays.utils import stack
from ...result import Result
from ...utils.general import dag, expect
from ...utils.operators import eye_like
from .abstract_solver import StochasticBaseSolver
from .interfaces import DSMEInterface, DSSEInterface, SolveInterface
from .rouchon_solver import _cholesky_normalize
from .save_mixin import DiffusiveSolveSaveMixin


class DiffusiveState(eqx.Module):
    """State for the diffusive SSE/SME fixed step solvers."""

    state: QArray  # state (integrated from initial to current time)
    Y: Array  # measurement (integrated from initial to current time)

    def __add__(self, other: DiffusiveState) -> DiffusiveState:
        return DiffusiveState(self.state + other.state, self.Y + other.Y)


def _is_multiple_of(
    x: ArrayLike, dt: float, *, rtol: float = 1e-5, atol: float = 1e-5
) -> bool:
    x_rounded = np.round(np.array(x) / dt) * dt
    return np.allclose(x, x_rounded, rtol=rtol, atol=atol)


def _is_linearly_spaced(
    x: ArrayLike, *, rtol: float = 1e-5, atol: float = 1e-5
) -> bool:
    diffs = np.diff(x)
    return np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)


class DiffusiveSolveSolver(
    StochasticBaseSolver, DiffusiveSolveSaveMixin, SolveInterface
):
    """Solver solving the diffusive SSE/SME with a fixed step size solver."""

    def __check_init__(self):
        # check that all tsave values are exact multiples of dt
        if not _is_multiple_of(self.ts, self.dt):
            raise ValueError(
                'Argument `tsave` should only contain exact multiples of the method '
                'fixed step size `dt`.'
            )

        # check that tsave is linearly spaced
        if not _is_linearly_spaced(self.ts):
            raise ValueError('Argument `tsave` should be linearly spaced.')

        # check that options.t0 is not used
        if self.options.t0 is not None:
            raise ValueError(
                'Option `t0` is invalid for fixed step SSE or SME methods.'
            )

        if self.discontinuity_ts is not None:
            warnings.warns(
                'The Hamiltonian or jump operators are time-dependent with '
                'discontinuities, which will be ignored by the method.',
                stack_level=1,
            )

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step methods always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    @property
    def dt(self) -> float:
        return self.method.dt

    def integrate(
        self, t0: float, y0: DiffusiveState, key: PRNGKeyArray, nsteps: int
    ) -> tuple[float, DiffusiveState]:
        # integrate the SDE for nsteps of length dt
        # sample wiener
        dWs = jnp.sqrt(self.dt) * jax.random.normal(key, (nsteps, self.nmeas))

        # iterate over the fixed step size dt
        def step(carry, dW):  # noqa: ANN001, ANN202
            t, y = carry
            y = self.forward(t, y, dW)
            t = t + self.dt
            return (t, y), None

        (t, y), _ = jax.lax.scan(step, (t0, y0), dWs)

        return t, y

    def integrate_by_chunks(
        self, t0: float, y0: DiffusiveState, key: PRNGKeyArray, nsteps: int
    ) -> tuple[float, DiffusiveState]:
        # integrate the SSE/SME for nsteps of length dt, splitting the integration in
        # chunks of 1000 dts to ensure a fixed memory usage

        nsteps_per_chunk = 1000
        nchunks = int(nsteps // nsteps_per_chunk)

        # iterate over each chunk
        def step(carry, key):  # noqa: ANN001, ANN202
            t, y = carry
            t, y = self.integrate(t, y, key, nsteps_per_chunk)
            return (t, y), None

        # split the key for each chunk
        key, lastkey = jax.random.split(key)
        keys = jax.random.split(key, (nchunks,))
        (t, y), _ = jax.lax.scan(step, (t0, y0), keys)

        # integrate for the remaining number of steps (< nsubsteps)
        nremaining = nsteps % nsteps_per_chunk
        t, y = self.integrate(t, y, lastkey, nremaining)

        return t, y

    def run(self) -> Result:
        # this function assumes that ts is linearly spaced with each value an exact
        # multiple of dt

        # === define variables
        # number of save interval
        nsave = len(self.ts) - 1
        # total number of steps (of length dt)
        nsteps = int((self.t1 - self.t0) / self.dt)
        # number of steps per save interval
        nsteps_per_save = int(nsteps // nsave)
        # save time
        delta_t = self.ts[1] - self.ts[0]

        # === initial state
        # define initial state (state, Y) = (state0, 0)
        y0 = DiffusiveState(self.y0, jnp.zeros(self.nmeas))
        # save initial state at time 0
        saved0 = self.save(y0)

        # === run the simulation
        # integrate the SSE/SME for each save interval
        def outer_step(carry, key):  # noqa: ANN001, ANN202
            t, y = carry
            t, y = self.integrate_by_chunks(t, y, key, nsteps_per_save)
            return (t, y), self.save(y)

        # split the key for each save interval
        keys = jax.random.split(self.key, nsave)
        ylast, saved = jax.lax.scan(outer_step, (self.t0, y0), keys)
        ylast = ylast[1]

        # === collect and format results
        # insert the initial saved result
        saved = jax.tree.map(lambda x, y: jnp.insert(x, 0, y, axis=0), saved, saved0)
        # The averaged measurement I^{(ta, tb)} is recovered by diffing the measurement
        # Y which is integrated between ts[0] and ts[-1].
        Isave = jnp.diff(saved.Isave, axis=0) / delta_t
        saved = eqx.tree_at(lambda x: x.Isave, saved, Isave)
        saved = self.postprocess_saved(saved, ylast)

        return self.result(saved, infos=self.Infos(nsteps))

    @property
    @abstractmethod
    def nmeas(self) -> int:
        pass

    @abstractmethod
    def forward(self, t: Scalar, y: DiffusiveState, dW: Array) -> DiffusiveState:
        # return (state_{t+dt}, dY_{t+dt})
        pass


class DSSEFixedStepSolver(DiffusiveSolveSolver, DSSEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.Ls)


class DSSESolveEulerMayuramaSolver(DSSEFixedStepSolver):
    """Solver solving the diffusive SSE with the Euler-Mayurama method."""

    def forward(self, t: Scalar, y: DiffusiveState, dW: Array) -> DiffusiveState:
        psi = y.state
        L, H = self.L(t), self.H(t)

        # === measurement Y
        # dY = <L+Ld> dt + dW
        Lpsi = [_L @ psi for _L in L]
        exp = jnp.stack([expect(_L + _L.dag(), psi).real for _L in L])  # (nL)
        # todo: can be computed with 2 * (psi.dag() @ Lpsi).real
        dY = exp * self.dt + dW

        # === state psi
        dpsi = (
            -1j * self.dt * H @ psi
            - 0.5
            * self.dt
            * sum(
                [
                    _L.dag() @ _Lpsi - _exp * _Lpsi + 0.25 * _exp**2 * psi
                    for _L, _Lpsi, _exp in zip(L, Lpsi, exp, strict=True)
                ]
            )
            + sum(
                [
                    (_Lpsi - 0.5 * _exp * psi) * _dW
                    for _L, _Lpsi, _exp, _dW in zip(L, Lpsi, exp, dW, strict=True)
                ]
            )
        )

        return DiffusiveState(psi + dpsi, y.Y + dY)


dssesolve_euler_maruyama_solver_constructor = DSSESolveEulerMayuramaSolver


class DSMEFixedStepSolver(DiffusiveSolveSolver, DSMEInterface):
    @property
    def nmeas(self) -> int:
        return len(self.etas)


class DSMESolveEulerMayuramaSolver(DSMEFixedStepSolver):
    """Solver solving the diffusive SME with the Euler-Mayurama method."""

    def forward(self, t: Scalar, y: DiffusiveState, dW: Array) -> DiffusiveState:
        # The diffusive SME for a single detector is:
        #   drho = Lcal(rho)     dt + (Ccal(rho) - Tr[Ccal(rho)] rho) dW
        #   dY   = Tr[Ccal(rho)] dt + dW
        # with
        # - Lcal the Liouvillian
        # - Ccal the superoperator defined by Ccal(rho) = sqrt(eta) (L @ rho + rho @ Ld)

        rho = y.state
        L, Lm, H = self.L(t), self.Lm(t), self.H(t)
        LdL = sum([_L.dag() @ _L for _L in L])

        # === Lcal(rho)
        # (see MEDiffraxSolver in `solvers/core/diffrax_solver.py`)
        Hnh = -1j * H - 0.5 * LdL
        tmp = Hnh @ rho + sum([0.5 * _L @ rho @ _L.dag() for _L in L])
        Lcal_rho = tmp + tmp.dag()

        # === Ccal(rho)
        Lm_rho = stack([_Lm @ rho for _Lm in Lm])
        etas = self.etas[:, None, None]  # (nLm, 1, 1)
        Ccal_rho = jnp.sqrt(etas) * (Lm_rho + Lm_rho.dag())  # (nLm, n, n)
        tr_Ccal_rho = Ccal_rho.trace().real  # (nLm,)

        # === state rho
        drho_det = Lcal_rho
        drho_sto = Ccal_rho - tr_Ccal_rho[:, None, None] * rho  # (nLm, n, n)
        drho = drho_det * self.dt + (drho_sto * dW[:, None, None]).sum(0)  # (n, n)

        # === measurement Y
        dY = tr_Ccal_rho * self.dt + dW  # (nLm,)

        return DiffusiveState(rho + drho, y.Y + dY)


class DSMESolveRouchon1Solver(DSMEFixedStepSolver, SolveInterface):
    """Solver solving the diffusive SME with the Rouchon1 method."""

    def forward(self, t: Scalar, y: DiffusiveState, dW: Array) -> DiffusiveState:
        # The Rouchon update for a single loss channel is:
        #   rho_{k+1} = M_dY @ rho_k @ M_dY^\dag + M1 @ rho_k @ M1d / Tr[...]
        #   dY_{k+1} = sqrt(eta) Tr[(L+Ld) @ rho_k)] dt + dW
        # with
        #   MdY = I - (iH + 0.5 Ld @ L) dt + sqrt(self.eta) * dY * Lm
        #   M1 = sqrt(1 - eta) * L sqrt(dt)
        #
        # See comment of `_cholesky_normalize()` for the normalisation (computed for the
        # "average" Kraus operators M0 = I - (iH + 0.5 Ld @ L) dt and M1 = L sqrt(dt)).

        rho = y.state
        L, Lc, Lm, H = self.L(t), self.Lc(t), self.Lm(t), self.H(t)
        I = eye_like(H)
        LdL = sum([_L.dag() @ _L for _L in L])

        # === measurement Y
        # dY_{k+1} = sqrt(eta) Tr[(L+Ld) @ rho_k)] dt + dW
        trace = jnp.stack([expect(_Lm + _Lm.dag(), rho).real for _Lm in Lm])  # (nLm)
        dY = jnp.sqrt(self.etas) * trace * self.dt + dW  # (nLm,)

        # === state rho
        M0 = I - (1j * H + 0.5 * LdL) * self.dt
        M_dY = M0 + sum(
            [
                jnp.sqrt(eta) * _dY * _Lm
                for eta, _dY, _Lm in zip(self.etas, dY, Lm, strict=True)
            ]
        )
        Ms = [
            jnp.sqrt((1 - eta) * self.dt) * _Lm
            for eta, _Lm in zip(self.etas, Lm, strict=True)
        ] + [self.dt * _Lc for _Lc in Lc]

        if self.method.normalize:
            rho = _cholesky_normalize(M0, LdL, self.dt, rho)

        rho = M_dY @ rho @ dag(M_dY) + sum([_M @ rho @ dag(_M) for _M in Ms])
        rho = rho / rho.trace()  # normalise by signal probability

        return DiffusiveState(rho, y.Y + dY)


dsmesolve_euler_maruyama_solver_constructor = DSMESolveEulerMayuramaSolver
dsmesolve_rouchon1_solver_constructor = DSMESolveRouchon1Solver
