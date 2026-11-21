#!/usr/bin/env python3
"""Pure scheduling/objective helpers for the operator-GroupDRO pilot.

The pilot has one deliberately strict fairness contract: the mean and
GroupDRO arms must consume the exact same ordered examples and the exact same
``ops_per_example`` valid interventions.  This module builds that frozen
schedule, computes conditional aggregation weights, and supplies a low-memory
backward primitive for sharing the clean forward across operator losses.

This is *operator* GroupDRO, not the complete query-conditioned objective.
Rows may carry audited ``valid_ops``/``query_group`` fields, but in their
absence the synthetic operators are treated as valid.  Natural
counterfactual controls and affected/control query pairs remain separate work.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import torch

from qcdro_objective_draft import GroupDRO, valid_ops_for_row


SUPPORTED_SYNTHETIC_OPS = ("swap", "mask", "blur", "tshift")


def parse_ops(value: str | Sequence[str]) -> list[str]:
    """Normalize and validate an operator list while preserving its order."""

    raw = value.split(",") if isinstance(value, str) else value
    ops = list(dict.fromkeys(str(x).strip() for x in raw if str(x).strip()))
    unknown = [op for op in ops if op not in SUPPORTED_SYNTHETIC_OPS]
    if unknown:
        raise ValueError(
            f"unsupported training operators {unknown}; supported={list(SUPPORTED_SYNTHETIC_OPS)}"
        )
    if not ops:
        raise ValueError("at least one operator is required")
    return ops


def _normalized_validity_row(row: Mapping[str, object]) -> dict[str, object]:
    """Accept either a JSON list or a comma-separated manifest field."""

    out = dict(row)
    for key in ("valid_ops", "preserving_ops"):
        value = out.get(key)
        if isinstance(value, str):
            out[key] = [x.strip() for x in value.split(",") if x.strip()]
    return out


def audited_valid_ops(
    row: Mapping[str, object],
    requested: Sequence[str],
    *,
    query_group_key: str = "query_group",
    require_explicit: bool = False,
) -> list[str]:
    """Return valid synthetic operators under the pilot manifest policy."""

    if require_explicit and "valid_ops" not in row:
        return []
    return valid_ops_for_row(
        _normalized_validity_row(row), requested, query_group_key=query_group_key
    )


def _stable_tiebreak(seed: int, epoch: int, row_index: int, op: str) -> str:
    payload = f"{seed}|{epoch}|{row_index}|{op}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class ScheduleItem:
    epoch: int
    row_index: int
    ops: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"epoch": self.epoch, "row_index": self.row_index, "ops": list(self.ops)}


@dataclass(frozen=True)
class FairSchedule:
    items: tuple[ScheduleItem, ...]
    fingerprint: str
    exposure: dict[str, int]
    skipped_too_few_valid: int
    rows_with_explicit_validity: int

    def as_summary(self) -> dict[str, object]:
        return {
            "examples": len(self.items),
            "fingerprint": self.fingerprint,
            "operator_exposure": dict(self.exposure),
            "skipped_too_few_valid": self.skipped_too_few_valid,
            "rows_with_explicit_validity": self.rows_with_explicit_validity,
        }


def build_fair_schedule(
    rows: Sequence[Mapping[str, object]],
    *,
    epochs: int,
    requested_ops: Sequence[str],
    ops_per_example: int,
    seed: int,
    query_group_key: str = "query_group",
    require_explicit_validity: bool = False,
) -> FairSchedule:
    """Build a deterministic, exposure-balanced paired-operator schedule.

    At each occurrence, the least-exposed valid operators are selected.  A
    stable hash breaks ties.  Consequently two objective arms with identical
    data and seed have byte-identical fingerprints, while all four operators
    receive nearly equal exposure when every row permits them.
    """

    requested = parse_ops(requested_ops)
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not 1 <= ops_per_example <= len(requested):
        raise ValueError("ops_per_example must be in [1, len(requested_ops)]")

    exposure: Counter[str] = Counter({op: 0 for op in requested})
    items: list[ScheduleItem] = []
    skipped = 0
    explicit = sum(1 for row in rows if "valid_ops" in row)
    for epoch in range(epochs):
        order = list(range(len(rows)))
        random.Random(1234 + epoch + 1000 * seed).shuffle(order)
        for row_index in order:
            valid = audited_valid_ops(
                rows[row_index],
                requested,
                query_group_key=query_group_key,
                require_explicit=require_explicit_validity,
            )
            if len(valid) < ops_per_example:
                skipped += 1
                continue
            ranked = sorted(
                valid,
                key=lambda op: (
                    exposure[op],
                    _stable_tiebreak(seed, epoch, row_index, op),
                ),
            )
            chosen = tuple(ranked[:ops_per_example])
            exposure.update(chosen)
            items.append(ScheduleItem(epoch, row_index, chosen))

    digest = hashlib.sha256()
    for item in items:
        digest.update(
            (json.dumps(item.as_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
        )
    return FairSchedule(
        items=tuple(items),
        fingerprint=digest.hexdigest(),
        exposure={op: int(exposure[op]) for op in requested},
        skipped_too_few_valid=skipped,
        rows_with_explicit_validity=explicit,
    )


def conditional_aggregation_weights(
    sampled_groups: Sequence[str],
    *,
    objective: str,
    dro: GroupDRO | None = None,
) -> dict[str, float]:
    """Return weights normalized over the valid groups sampled this example."""

    groups = list(dict.fromkeys(sampled_groups))
    if not groups:
        raise ValueError("sampled_groups cannot be empty")
    if len(groups) != len(sampled_groups):
        raise ValueError("sampled_groups must be unique")
    if objective == "mean":
        return {g: 1.0 / len(groups) for g in groups}
    if objective != "groupdro":
        raise ValueError(f"unknown objective: {objective}")
    if dro is None:
        raise ValueError("groupdro objective requires a GroupDRO state")
    q = dro.weights()
    missing = [g for g in groups if g not in q]
    if missing:
        raise ValueError(f"unregistered GroupDRO groups: {missing}")
    denom = sum(q[g] for g in groups)
    return {g: q[g] / denom for g in groups}


def backward_one_with_clean_proxy(
    clean_score: torch.Tensor,
    loss_builder: Callable[[torch.Tensor], torch.Tensor],
    *,
    weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Backprop one operator branch without retaining the shared clean graph.

    ``loss_builder`` receives a leaf proxy containing ``clean_score``'s value.
    Its backward pass immediately frees the operator-specific visual graph.
    The returned derivative with respect to the proxy must be accumulated and
    backpropagated through the real ``clean_score`` once after all operators.
    This is algebraically identical to a monolithic weighted sum for a scalar
    clean score, but has a substantially lower activation-memory peak.
    """

    if clean_score.numel() != 1:
        raise ValueError("clean_score must be scalar")
    proxy = clean_score.detach().clone().requires_grad_(True)
    loss = loss_builder(proxy)
    if loss.numel() != 1:
        raise ValueError("operator loss must be scalar")
    (float(weight) * loss).backward()
    if proxy.grad is None:
        raise RuntimeError("loss_builder did not depend on the clean proxy")
    return loss.detach(), proxy.grad.detach().clone()


def _self_test() -> None:
    # Validity semantics and deterministic exposure balancing.
    rows = [{"video": f"v{i}.mp4"} for i in range(12)]
    rows.append({"video": "limited.mp4", "valid_ops": "mask,swap"})
    sched = build_fair_schedule(
        rows,
        epochs=2,
        requested_ops=SUPPORTED_SYNTHETIC_OPS,
        ops_per_example=2,
        seed=7,
    )
    sched2 = build_fair_schedule(
        rows,
        epochs=2,
        requested_ops=SUPPORTED_SYNTHETIC_OPS,
        ops_per_example=2,
        seed=7,
    )
    assert sched.fingerprint == sched2.fingerprint
    assert all(len(item.ops) == 2 for item in sched.items)
    assert max(sched.exposure.values()) - min(sched.exposure.values()) <= 1

    # Conditional GroupDRO must emphasize the harder observed operator.
    dro = GroupDRO(SUPPORTED_SYNTHETIC_OPS, eta=0.5, ema_decay=0.0)
    before = dro.weights()
    dro.update({"mask": torch.tensor(3.0), "swap": torch.tensor(1.0)})
    after = dro.weights()
    assert after["mask"] > before["mask"]
    pair_q = conditional_aggregation_weights(
        ["mask", "swap"], objective="groupdro", dro=dro
    )
    assert abs(sum(pair_q.values()) - 1.0) < 1e-8
    assert pair_q["mask"] > pair_q["swap"]
    assert conditional_aggregation_weights(["mask", "swap"], objective="mean") == {
        "mask": 0.5,
        "swap": 0.5,
    }

    # The streamed proxy backward must exactly match a monolithic graph.
    beta, alpha = 0.3, 0.2
    coeffs = (1.3, -0.4)
    weights = (0.7, 0.3)

    theta_a = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    clean_a = 2.1 * theta_a
    losses_a = []
    for coeff in coeffs:
        corrupt = coeff * theta_a
        margin = beta * (clean_a - corrupt)
        losses_a.append(torch.nn.functional.softplus(-margin))
    total_a = sum(w * loss for w, loss in zip(weights, losses_a)) - alpha * clean_a / 5.0
    total_a.backward()

    theta_b = torch.tensor(0.8, dtype=torch.float64, requires_grad=True)
    clean_b = 2.1 * theta_b
    clean_grad = torch.zeros_like(clean_b)
    for coeff, weight in zip(coeffs, weights):
        corrupt = coeff * theta_b

        def build(proxy: torch.Tensor, corrupt_score: torch.Tensor = corrupt) -> torch.Tensor:
            return torch.nn.functional.softplus(-beta * (proxy - corrupt_score))

        _, grad = backward_one_with_clean_proxy(clean_b, build, weight=weight)
        clean_grad = clean_grad + grad
    clean_grad = clean_grad - alpha / 5.0
    clean_b.backward(clean_grad)
    torch.testing.assert_close(theta_a.grad, theta_b.grad, rtol=1e-10, atol=1e-10)

    print(
        json.dumps(
            {
                "status": "ok",
                "schedule": sched.as_summary(),
                "dro_weights": after,
                "monolithic_grad": float(theta_a.grad),
                "streamed_grad": float(theta_b.grad),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _self_test()
