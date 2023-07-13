from __future__ import annotations

from abc import ABC, abstractmethod
from math import pi, sqrt

import torch
from torch import Tensor

import dynamiqs as dq
from dynamiqs.options import Options
from dynamiqs.utils.tensor_types import TensorLike


class OpenSystem(ABC):
    def __init__(self):
        self.n = None
        self.H = None
        self.H_batched = None
        self.jump_ops = None
        self.rho0 = None
        self.rho0_batched = None
        self.exp_ops = None

    def reset(self):
        """Reset all attributes."""
        self.__init__()

    @abstractmethod
    def t_save(self, n: int) -> Tensor:
        """Compute the save time tensor."""
        pass

    def rho(self, t: float) -> Tensor:
        """Compute the exact density matrix at a given time."""
        raise NotImplementedError

    def rhos(self, t: Tensor) -> Tensor:
        return torch.stack([self.rho(t_.item()) for t_ in t])

    def expect(self, t: float) -> Tensor:
        """Compute the exact (complex) expectation values at a given time."""
        raise NotImplementedError

    def expects(self, t: Tensor) -> Tensor:
        return torch.stack([self.expect(t_.item()) for t_ in t]).swapaxes(0, 1)

    def loss(self, rho: Tensor) -> Tensor:
        """Compute an example loss function from a given density matrix."""
        return dq.expect(self.loss_op, rho).real

    def mesolve(
        self, t_save: TensorLike, options: Options
    ) -> tuple[Tensor, Tensor | None]:
        return dq.mesolve(
            self.H,
            self.jump_ops,
            self.rho0,
            t_save=t_save,
            exp_ops=self.exp_ops,
            options=options,
        )


class LeakyCavity(OpenSystem):
    # `H_batched: (3, n, n)
    # `jump_ops`: (2, n, n)
    # `rho0_batched`: (4, n, n)
    # `exp_ops`: (2, n, n)

    def __init__(
        self,
        *,
        n: int,
        kappa: float,
        delta: float,
        alpha0: complex,
        requires_grad: bool = False,
    ):
        # store parameters
        self.n = n
        self.kappa = torch.as_tensor(kappa).requires_grad_(requires_grad)
        self.delta = torch.as_tensor(delta).requires_grad_(requires_grad)
        self.alpha0 = torch.as_tensor(alpha0).requires_grad_(requires_grad)
        self.requires_grad = requires_grad

        # define gradient parameters
        self.parameters = tuple([self.delta, self.kappa, self.alpha0])

        # bosonic operators
        a = dq.destroy(self.n)
        adag = a.mH

        # loss operator
        # self.loss_op = adag @ a
        self.loss_op = (a + adag) / sqrt(2) - 1j * (adag - a) / sqrt(2)

        # prepare quantum operators
        self.H = self.delta * adag @ a
        self.H_batched = [0.5 * self.H, self.H, 2 * self.H]
        self.jump_ops = [torch.sqrt(self.kappa) * a, dq.eye(self.n)]
        self.exp_ops = [(a + adag) / sqrt(2), 1j * (adag - a) / sqrt(2)]

        # prepare initial states
        self.rho0 = dq.coherent_dm(self.n, self.alpha0)
        self.rho0_batched = [
            dq.coherent_dm(self.n, self.alpha0),
            dq.coherent_dm(self.n, 1j * self.alpha0),
            dq.coherent_dm(self.n, -self.alpha0),
            dq.coherent_dm(self.n, -1j * self.alpha0),
        ]

    def reset(self):
        self.__init__(
            n=self.n,
            kappa=self.kappa,
            delta=self.delta,
            alpha0=self.alpha0,
            requires_grad=self.requires_grad,
        )

    def t_save(self, num_t_save: int) -> Tensor:
        t_end = 2 * pi / self.delta  # a full rotation
        return torch.linspace(0.0, t_end.item(), num_t_save)

    def _alpha(self, t: float) -> Tensor:
        return (
            self.alpha0
            * torch.exp(-1j * self.delta * t)
            * torch.exp(-0.5 * self.kappa * t)
        )

    def rho(self, t: float) -> Tensor:
        return dq.coherent_dm(self.n, self._alpha(t))

    def expect(self, t: float) -> Tensor:
        alpha_t = self._alpha(t)
        exp_x = sqrt(2) * alpha_t.real
        exp_p = sqrt(2) * alpha_t.imag
        return torch.tensor([exp_x, exp_p], dtype=alpha_t.dtype)
