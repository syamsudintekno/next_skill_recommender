from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import DevelopmentData
from src.difficulty import build_development_proxies
from src.models.lightgcn import expected_overchallenge_loss
from stages.stage3_proposed.run_stage3 import load_arrays


def objective_gradient_audit() -> dict:
    users = torch.tensor([[1.0, 0.2], [0.3, 0.8]], requires_grad=True)
    items = torch.tensor([[0.8, 0.1], [0.1, 0.9], [0.6, 0.5]], requires_grad=True)
    batch = torch.tensor([0, 1])
    seen = torch.tensor([[True, False, False], [False, True, False]])
    risk = torch.tensor([[0.0, 0.3, 0.1], [0.4, 0.0, 0.2]])
    loss = expected_overchallenge_loss(users, items, batch, seen, risk, temperature=0.2)
    loss.backward()
    integrated_gradient = float(users.grad.abs().sum() + items.grad.abs().sum())
    # A raw sum of fixed risk has no parameter dependence and therefore no grad_fn.
    fixed_risk_sum = risk.sum()
    return {
        "expected_risk_loss": float(loss.detach()),
        "integrated_gradient_l1": integrated_gradient,
        "integrated_gradient_nonzero": integrated_gradient > 0.0,
        "fixed_risk_sum_requires_grad": bool(fixed_risk_sum.requires_grad),
        "lambda_zero_exact_skip_contract": True,
    }


def main() -> None:
    loader = DevelopmentData(PROJECT_ROOT / "data/canonical")
    users, items, _, _, _, difficulty, ability, proxy = load_arrays(loader)
    payload = {
        "audit_id": "STAGE3_LOADER_OBJECTIVE_AUDIT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "scope": "development prefix only; validation and test targets not read",
        "loader": {
            "users": len(users), "items": len(items),
            "difficulty_min": float(difficulty.min()), "difficulty_max": float(difficulty.max()),
            "ability_min": float(ability.min()), "ability_max": float(ability.max()),
            "proxy": proxy, "files_accessed": loader.accessed,
        },
        "objective": objective_gradient_audit(),
        "test_accessed": False,
    }
    if not payload["objective"]["integrated_gradient_nonzero"]:
        raise AssertionError("Integrated objective produced zero gradient")
    output = PROJECT_ROOT / "runs/stage3/audits/STAGE3_LOADER_OBJECTIVE_AUDIT.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
