from torch import Tensor

from ..ode_forward_qsolver import ODEForwardQSolver
from ..solver_utils import lindbladian


class MEEuler(ODEForwardQSolver):
    def __init__(self, *args, jump_ops: Tensor):
        super().__init__(*args)

        self.H = self.H[:, None, ...]  # (b_H, 1, n, n)
        self.jump_ops = jump_ops  # (len(jump_ops), n, n)

    def forward(self, t: float, rho: Tensor) -> Tensor:
        # Args:
        #     rho: (b_H, b_rho, n, n)
        #
        # Returns:
        #     (b_H, b_rho, n, n)

        return rho + self.options.dt * lindbladian(rho, self.H, self.jump_ops)
