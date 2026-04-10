"""
src/model/ewc.py — Elastic Weight Consolidation loss for continual learning.
"""

import torch
import torch.nn as nn
from typing import Dict
from src.utils.logger import get_logger

log = get_logger("ewc")


class EWCLoss(nn.Module):
    """
    EWC penalty term that prevents catastrophic forgetting.

    L_total = L_CE + lambda * SUM_i F_i * (theta_i - theta*_i)^2

    Args:
        model       : Current model.
        fisher      : Dict of parameter name → Fisher diagonal tensor.
        ref_params  : Dict of parameter name → reference (old) parameter tensor.
        lamda       : EWC penalty strength (from config: adaptive.tier2_ewc_lambda).
    """

    def __init__(self, model: nn.Module, fisher: Dict, ref_params: Dict, lamda: float = 400.0):
        super().__init__()
        self.model      = model
        self.fisher     = fisher
        self.ref_params = ref_params
        self.lamda      = lamda

    def forward(self) -> torch.Tensor:
        loss = torch.tensor(0.0, device=next(self.model.parameters()).device)
        for name, param in self.model.named_parameters():
            if name in self.fisher and name in self.ref_params:
                f   = self.fisher[name].to(param.device)
                ref = self.ref_params[name].to(param.device)
                loss += (f * (param - ref).pow(2)).sum()
        return self.lamda * loss


def load_ewc_state(production_dir) -> tuple:
    """Load Fisher matrix and reference parameters from production checkpoint."""
    import pathlib
    production_dir = pathlib.Path(production_dir)
    fisher_path    = production_dir / "fisher.pt"
    model_path     = production_dir / "best_model.pt"

    if not fisher_path.exists():
        raise FileNotFoundError(
            f"Fisher matrix not found: {fisher_path}\n"
            f"Run `python main.py train` first to generate it."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Production model not found: {model_path}\n"
            f"Run `python main.py train` first."
        )

    fisher    = torch.load(fisher_path, weights_only=True)
    state     = torch.load(model_path,  weights_only=False)
    ref_params = {k: v.clone().detach() for k, v in state["model"].items()}
    return fisher, ref_params
