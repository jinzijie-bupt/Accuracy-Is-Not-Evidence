#!/usr/bin/env python3
"""Loss/state helpers for the proposed query-conditioned family-DRO trainer.

This is an inert local draft: nothing imports it from the live remote queues.
The model-specific trainer should supply adapter/reference log-prob advantages.

Notation used by the helpers:
  advantage = log p_theta(completion | context) - log p_ref(completion | context)
  affected_drop = advantage(clean, q_a, w_a) - advantage(I(E_a), q_a, w_a)
  control_drop  = advantage(clean, q_b, w_b) - advantage(I(E_a), q_b, w_b)

The same visual edit I(E_a) is therefore required to affect q_a more than q_b.
This paired selectivity is what prevents a generic corruption detector from
satisfying the objective by lowering every completion under an edited video.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F


TEMPORAL_QUERY_GROUPS = frozenset({"temporal_order", "state_change"})


def valid_ops_for_row(
    row: Mapping[str, object],
    requested: Sequence[str],
    *,
    query_group_key: str = "query_group",
) -> list[str]:
    """Return the operators that have a valid causal label for this query.

    A manifest may provide ``valid_ops`` explicitly.  Independently of that
    list, shuffle/reverse are rejected unless the query belongs to a temporal
    group (or the manifest explicitly sets ``shuffle_valid=true``).  Unknown
    validity is excluded rather than silently treated as a destructive edit.
    """

    explicit = row.get("valid_ops")
    allowed = set(str(x) for x in explicit) if explicit is not None else set(requested)
    group = str(row.get(query_group_key, "generic_action"))
    temporal_valid = bool(row.get("shuffle_valid", False)) or group in TEMPORAL_QUERY_GROUPS
    out: list[str] = []
    for op in requested:
        if op not in allowed:
            continue
        if op in {"shuffle", "reverse"} and not temporal_valid:
            continue
        out.append(op)
    return out


def preserving_ops_for_row(
    row: Mapping[str, object],
    requested: Sequence[str],
    *,
    query_group_key: str = "query_group",
) -> list[str]:
    """Return edits expected to preserve the answer for this query.

    An audited manifest can specify ``preserving_ops``.  As a conservative
    default, shuffle/reverse are preserving controls for non-temporal queries,
    rather than being mislabeled as evidence-destroying interventions.  The
    trainer should apply an invariance/SFT loss to these groups, not PAPO.
    """

    explicit = row.get("preserving_ops")
    if explicit is not None:
        return [op for op in requested if op in {str(x) for x in explicit}]
    group = str(row.get(query_group_key, "generic_action"))
    temporal_valid = bool(row.get("shuffle_valid", False)) or group in TEMPORAL_QUERY_GROUPS
    return [op for op in requested if op in {"shuffle", "reverse"} and not temporal_valid]


@dataclass(frozen=True)
class QCLossConfig:
    beta: float = 0.1
    min_affected_drop: float = 0.0
    min_selectivity_gap: float = 0.0
    control_tolerance: float = 0.0
    dependency_weight: float = 1.0
    selectivity_weight: float = 1.0
    invariance_weight: float = 0.5
    preserved_sft_weight: float = 0.1


def qc_intervention_loss(
    *,
    clean_affected_adv: torch.Tensor,
    corrupt_affected_adv: torch.Tensor,
    clean_control_adv: torch.Tensor,
    corrupt_control_adv: torch.Tensor,
    corrupt_control_logp_per_token: torch.Tensor | None = None,
    cfg: QCLossConfig = QCLossConfig(),
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Signed query-conditioned intervention loss for one operator/direction.

    The affected query should lose support after its own evidence is edited;
    the disjoint control query should retain support under the exact same edit.
    Parseability/accuracy is guarded with optional SFT on the preserved query.
    """

    affected_drop = clean_affected_adv - corrupt_affected_adv
    control_drop = clean_control_adv - corrupt_control_adv
    selectivity = affected_drop - control_drop

    dep = F.softplus(-cfg.beta * (affected_drop - cfg.min_affected_drop))
    sel = F.softplus(-cfg.beta * (selectivity - cfg.min_selectivity_gap))
    # Only penalize degradation of the control beyond a small tolerance.
    inv = F.softplus(cfg.beta * (control_drop - cfg.control_tolerance))
    total = (
        cfg.dependency_weight * dep
        + cfg.selectivity_weight * sel
        + cfg.invariance_weight * inv
    )
    if corrupt_control_logp_per_token is not None and cfg.preserved_sft_weight > 0:
        total = total + cfg.preserved_sft_weight * (-corrupt_control_logp_per_token)
    stats = {
        "affected_drop": affected_drop.detach(),
        "control_drop": control_drop.detach(),
        "selectivity": selectivity.detach(),
        "dependency_loss": dep.detach(),
        "selectivity_loss": sel.detach(),
        "invariance_loss": inv.detach(),
    }
    return total, stats


def natural_query_swap_loss(
    *,
    aa_adv: torch.Tensor,
    ab_adv: torch.Tensor,
    bb_adv: torch.Tensor,
    ba_adv: torch.Tensor,
    aa_logp_per_token: torch.Tensor | None = None,
    bb_logp_per_token: torch.Tensor | None = None,
    beta: float = 0.1,
    clean_sft_weight: float = 0.1,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Natural same-video query counterfactual objective.

    ``xy_adv`` scores completion/window y under query x.  Thus q_a must prefer
    w_a over w_b, while q_b must prefer w_b over w_a, with the video unchanged.
    """

    margin_a = aa_adv - ab_adv
    margin_b = bb_adv - ba_adv
    loss = F.softplus(-beta * margin_a) + F.softplus(-beta * margin_b)
    if clean_sft_weight > 0:
        if aa_logp_per_token is not None:
            loss = loss + clean_sft_weight * (-aa_logp_per_token)
        if bb_logp_per_token is not None:
            loss = loss + clean_sft_weight * (-bb_logp_per_token)
    return loss, {"query_margin_a": margin_a.detach(), "query_margin_b": margin_b.detach()}


class GroupDRO:
    """Small JSON-serializable exponentiated-gradient GroupDRO state.

    Groups should initially be operator names.  Once validity manifests are
    audited, use ``f"{query_group}/{op}"`` to optimize the worst valid semantic
    group.  The update is detached; only the weighted loss carries gradients.
    """

    def __init__(
        self,
        groups: Iterable[str],
        *,
        eta: float = 0.05,
        ema_decay: float = 0.9,
        update_clip: float = 5.0,
    ) -> None:
        groups = tuple(dict.fromkeys(str(g) for g in groups))
        if not groups:
            raise ValueError("GroupDRO requires at least one group")
        self.groups = groups
        self.eta = float(eta)
        self.ema_decay = float(ema_decay)
        self.update_clip = float(update_clip)
        self.log_q: Dict[str, float] = {g: -math.log(len(groups)) for g in groups}
        self.loss_ema: Dict[str, float | None] = {g: None for g in groups}

    def weights(self) -> Dict[str, float]:
        m = max(self.log_q.values())
        z = sum(math.exp(v - m) for v in self.log_q.values())
        return {g: math.exp(v - m) / z for g, v in self.log_q.items()}

    def update(self, losses: Mapping[str, torch.Tensor]) -> None:
        seen = [g for g in losses if g in self.log_q]
        if not seen:
            return
        for g in seen:
            value = float(losses[g].detach().float().mean().cpu())
            old = self.loss_ema[g]
            self.loss_ema[g] = value if old is None else self.ema_decay * old + (1 - self.ema_decay) * value
        available = [float(v) for v in self.loss_ema.values() if v is not None]
        center = sum(available) / len(available)
        for g in seen:
            signal = max(-self.update_clip, min(self.update_clip, float(self.loss_ema[g]) - center))
            self.log_q[g] += self.eta * signal
        # Recenter for numerical stability; this leaves softmax(q) unchanged.
        shift = max(self.log_q.values())
        for g in self.log_q:
            self.log_q[g] -= shift

    def aggregate(self, losses: Mapping[str, torch.Tensor], *, update: bool = True) -> torch.Tensor:
        seen = [g for g in losses if g in self.log_q]
        if not seen:
            raise ValueError("No registered GroupDRO group was present in this step")
        if update:
            self.update(losses)
        q = self.weights()
        # Renormalize over sampled/valid groups so skipped invalid groups do not
        # shrink this example's gradient magnitude.
        denom = sum(q[g] for g in seen)
        return sum((q[g] / denom) * losses[g] for g in seen)

    def state_dict(self) -> Dict[str, object]:
        return {
            "groups": list(self.groups),
            "eta": self.eta,
            "ema_decay": self.ema_decay,
            "update_clip": self.update_clip,
            "log_q": dict(self.log_q),
            "loss_ema": dict(self.loss_ema),
        }

    def load_state_dict(self, state: MutableMapping[str, object]) -> None:
        if tuple(state["groups"]) != self.groups:
            raise ValueError("GroupDRO group list mismatch")
        self.log_q = {g: float(v) for g, v in dict(state["log_q"]).items()}
        self.loss_ema = {
            g: (None if v is None else float(v))
            for g, v in dict(state["loss_ema"]).items()
        }


if __name__ == "__main__":
    # CPU smoke test only; it does not touch a model, dataset, or queue.
    dro = GroupDRO(["mask", "swap", "blur", "tshift"])
    x = torch.tensor(1.0, requires_grad=True)
    total = dro.aggregate({"mask": x, "swap": 2 * x})
    total.backward()
    print({"loss": float(total.detach()), "grad": float(x.grad), "q": dro.weights()})
