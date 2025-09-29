import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union, Dict, Any
from device import TwoTerminalDevice

__all__ = ['Circuit']


def _preprocess(device_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preprocess the device configuration (kept for compatibility).
    """
    if 'use_diff' not in device_cfg:
        device_cfg['use_diff'] = False
    if 'activation' not in device_cfg:
        device_cfg['activation'] = 'none'
    return device_cfg


class Circuit(nn.Module):
    """
    Circuit container that wires up multiple TwoTerminalDevice elements
    according to a net topology.
    """

    def __init__(self, net_topo: Union[torch.Tensor, Tuple, List],
                 device_cfg: Dict[str, Any],
                 noise_std: Optional[float] = 0.0):
        super().__init__()

        assert len(net_topo) == 2 and len(net_topo[0]) == len(net_topo[1]), "Invalid topology"

        self.register_buffer('src_node', net_topo[0].to(torch.int64))
        self.register_buffer('des_node', net_topo[1].to(torch.int64))

        self.num_device = len(net_topo[0])
        self.device_cfg = _preprocess(device_cfg)
        self.model = TwoTerminalDevice(self.num_device, self.device_cfg)

        self.noise_std = noise_std

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (float): time
            x (torch.Tensor): node voltages (excluding ground), shape (..., n_nodes-1)

        Returns:
            torch.Tensor: currents into each node, shape (..., n_nodes-1)
        """
        # add ground node at v=0
        aux_v = torch.cat((torch.zeros_like(x[..., :1]), x), dim=-1)

        # compute device currents
        state_i = self.model(t, aux_v[..., self.src_node], aux_v[..., self.des_node])

        # initialize result (with dummy ground node)
        result = torch.cat((torch.zeros_like(x[..., :1]), torch.zeros_like(x)), dim=-1)

        # KCL: subtract current from source node, add to destination
        result.scatter_add_(-1, self.src_node.expand_as(state_i), -state_i)
        result.scatter_add_(-1, self.des_node.expand_as(state_i), state_i)

        # remove ground node contribution
        result = result[..., 1:]
        return result
