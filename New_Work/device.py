import torch
import torch.nn as nn
import torch.special as special
from typing import Dict, Any
from scipy.special import lambertw

__all__ = ['TwoTerminalDevice']


class TwoTerminalDevice(nn.Module):
    """
    Two-terminal nonlinear device implementing the solved diode-like current law.

    i = (c/b) * W( (b*Is/c) * exp(v/c + b*Is/c) ) - Is

    where
        v = src - des   (voltage across device)
        Is = saturation current
        a  = N * Vt     (thermal voltage * ideality factor)
        theta1, theta2 are shape parameters
        b = 1 - theta2
        c = theta1 * a
    """

    def __init__(self, num_device: int, device_cfg: Dict[str, Any]):
        super().__init__()

        self.num_device = num_device

        # Physics parameters (can be trainable or fixed)
        self.Is = nn.Parameter(torch.tensor(1e-3))     # A
        self.a = nn.Parameter(torch.tensor(0.025))     # V (~25 mV * N)
        self.theta1 = nn.Parameter(torch.tensor(1.0))  # dimensionless
        self.theta2 = nn.Parameter(torch.tensor(0.0))  # dimensionless

    def _lambertw_torch(self, z, max_iter=20, tol=1e-6):
        """
        Approximate Lambert W0(z) for real z >= 0 using Newton iteration.
        Autograd-friendly since all ops are in torch.
        """
        # Initial guess
        w = torch.log(z + 1.0)
        for _ in range(max_iter):
            ew = torch.exp(w)
            f = w * ew - z
            wp = w + 1.0
            dw = f / (ew * wp - (wp + 1.0) * f / (2.0 * wp))
            w = w - dw
            if torch.max(torch.abs(dw)) < tol:
                break
        return w


    def forward(self, t: float, src: torch.Tensor, des: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (float): time (unused placeholder).
            src (torch.Tensor): source node voltage, shape (..., num_device)
            des (torch.Tensor): destination node voltage, shape (..., num_device)

        Returns:
            torch.Tensor: device current, shape (..., num_device)
        """
        # voltage across device
        v = src - des

        # coefficients
        b = 1.0 - self.theta2
        c = self.theta1 * self.a

        # Clamp the exponent input to avoid overflow
        exp_input = torch.clamp(v / c + (b * self.Is / c), max=50.0)
        arg = (b * self.Is / c) * torch.exp(exp_input)

        # Clamp LambertW argument to valid real domain
        min_valid = -1.0 / torch.exp(torch.tensor(1.0, device=arg.device, dtype=arg.dtype))
        arg = torch.clamp(arg, min=min_valid + 1e-6)



        # Solve for current using custom LambertW
        w_val = self._lambertw_torch(arg)
        i = (c / b) * w_val - self.Is

        return i

