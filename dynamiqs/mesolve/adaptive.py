from __future__ import annotations

from torch import Tensor

from ..solvers.ode.adaptive_solver import AdaptiveSolver, DormandPrince5
from .me_solver import MESolver


class MEAdaptive(MESolver, AdaptiveSolver):
    def odefun(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        return self.lindbladian(t, rho)

    def odefun_backward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        return -self.lindbladian(t, rho)

    def odefun_adjoint(self, t: float, phi: Tensor) -> Tensor:
        # phi: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        return self.adjoint_lindbladian(t, phi)

    def odefun_augmented(
        self, t: float, rho: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        # phi: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        return self.odefun_backward(t, rho), self.odefun_adjoint(t, phi)


class MEDormandPrince5(MEAdaptive, DormandPrince5):
    pass
