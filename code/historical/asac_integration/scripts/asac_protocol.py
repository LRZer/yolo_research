"""Frozen, standard-library-only ASAC Phase-1 protocol utilities.

This module deliberately contains no model, dataloader, metric, or trainer code.
It implements the scalar A2 controller and freezes A1/A3/A4 schedules solely
from A2 schedule state. Validation/test metrics, AP, fitness, and losses are
rejected and are never value-accessed by the freezer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


PANEL_SHA256 = "8636cc2983ff65e77de6351629817fb43d75ebb36a440df44e4026c13747a0fa"
OBJECT_POPULATION_SHA256 = "f57cffdeb8bc29ef66a21c384ac0c3cd7ceff9552c48245263cb6c7c51b14926"
SEEDS = (0, 1, 42)
A1_SOURCE_SEED = {0: 42, 1: 0, 42: 1}


class ProtocolError(ValueError):
    """Raised when input violates a frozen ASAC Phase-1 invariant."""


@dataclass(frozen=True)
class ASACProtocolConfig:
    """Immutable parameters from ``ASAC_PHASE1_PREREG.md``."""

    epochs: int = 50
    seeds: tuple[int, int, int] = SEEDS
    warmup_start: int = 1
    warmup_end: int = 5
    reference_epochs: tuple[int, int, int] = (3, 4, 5)
    feedback_start: int = 6
    feedback_update_end: int = 49
    feedback_end: int = 50
    q_min: float = 0.25
    q_init: float = 0.50
    q_max: float = 1.00
    q_probe: float = 0.50
    beta: float = 0.80
    stability_scale: float = 0.02
    log_rate_limit: float = 0.05
    base_scale: float = 0.50
    base_translate: float = 0.10
    a0_q: float = 1.00
    a3_left_shift: int = 10
    panel_sha256: str = PANEL_SHA256
    object_population_sha256: str = OBJECT_POPULATION_SHA256

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None = None) -> "ASACProtocolConfig":
        """Validate a serialized config without permitting preregistration drift."""

        if values is None:
            config = cls()
            config.validate()
            return config
        if not isinstance(values, Mapping):
            raise ProtocolError("protocol config must be a mapping")
        defaults = asdict(cls())
        unknown = set(values) - set(defaults)
        if unknown:
            raise ProtocolError(f"unknown protocol config fields: {sorted(unknown)}")
        merged = dict(defaults)
        merged.update(values)
        for name in ("seeds", "reference_epochs"):
            if isinstance(merged[name], list):
                merged[name] = tuple(merged[name])
        try:
            config = cls(**merged)
        except TypeError as exc:
            raise ProtocolError(f"invalid protocol config: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        """Validate types, chronology, and exact frozen values."""

        expected = asdict(type(self)())
        actual = asdict(self)
        for name, frozen in expected.items():
            value = actual[name]
            if isinstance(frozen, int) and not isinstance(frozen, bool):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ProtocolError(f"{name} must be an integer")
            if isinstance(frozen, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ProtocolError(f"{name} must be a finite number")
            if value != frozen:
                raise ProtocolError(f"{name} is frozen at {frozen!r}, got {value!r}")
        if not (
            self.warmup_start == 1
            <= self.warmup_end
            < self.feedback_start
            <= self.feedback_update_end
            < self.feedback_end == self.epochs
        ):
            raise ProtocolError("invalid ASAC phase chronology")
        if not self.q_min <= self.q_init <= self.q_max:
            raise ProtocolError("q_init must lie in [q_min, q_max]")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["seeds"] = list(self.seeds)
        result["reference_epochs"] = list(self.reference_epochs)
        return result


def _finite_number(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ProtocolError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ProtocolError(f"{name} must be a finite number")
    return value


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError(f"{name} must be an integer")
    return value


def phase_for_epoch(epoch: int, config: ASACProtocolConfig | None = None) -> str:
    config = config or ASACProtocolConfig.from_mapping()
    epoch = _integer(epoch, name="epoch")
    if not 1 <= epoch <= config.epochs:
        raise ProtocolError(f"epoch must be in [1, {config.epochs}]")
    return "warmup" if epoch <= config.warmup_end else "feedback"


def reference_stability(stability_by_epoch: Mapping[int, float]) -> float:
    """Compute the frozen ``median(S_3, S_4, S_5)`` reference."""

    missing = [epoch for epoch in (3, 4, 5) if epoch not in stability_by_epoch]
    if missing:
        raise ProtocolError(f"reference stability is missing epochs {missing}")
    values = [_finite_number(stability_by_epoch[epoch], name=f"S_{epoch}") for epoch in (3, 4, 5)]
    for epoch, value in zip((3, 4, 5), values):
        if not 0.0 <= value <= 1.0:
            raise ProtocolError(f"S_{epoch} must be in [0, 1]")
    return float(median(values))


@dataclass(frozen=True)
class FeedbackStep:
    """One scalar A2 update after a completed feedback epoch."""

    state_ema: float
    log_q_target: float
    q_target: float
    delta_log_q: float
    q_next: float


def feedback_step(
    *,
    q_epoch: float,
    state_ema_previous: float,
    stability: float,
    stability_reference: float,
    config: ASACProtocolConfig | None = None,
) -> FeedbackStep:
    """Apply the preregistered bounded state-to-action update."""

    config = config or ASACProtocolConfig.from_mapping()
    q_epoch = _finite_number(q_epoch, name="q_epoch")
    state_ema_previous = _finite_number(state_ema_previous, name="state_ema_previous")
    stability = _finite_number(stability, name="stability")
    stability_reference = _finite_number(stability_reference, name="stability_reference")
    if not config.q_min <= q_epoch <= config.q_max:
        raise ProtocolError("q_epoch is outside the frozen controller bounds")
    if not 0.0 <= state_ema_previous <= 1.0:
        raise ProtocolError("state_ema_previous must be in [0, 1]")
    if not 0.0 <= stability <= 1.0 or not 0.0 <= stability_reference <= 1.0:
        raise ProtocolError("stability and stability_reference must be in [0, 1]")

    state_ema = config.beta * state_ema_previous + (1.0 - config.beta) * stability
    log_q_target = math.log(config.q_init) + math.log(2.0) * (state_ema - stability_reference) / config.stability_scale
    q_target = math.exp(log_q_target)
    unconstrained_delta = log_q_target - math.log(q_epoch)
    delta_log_q = min(config.log_rate_limit, max(-config.log_rate_limit, unconstrained_delta))
    q_next = math.exp(math.log(q_epoch) + delta_log_q)
    q_next = min(config.q_max, max(config.q_min, q_next))
    return FeedbackStep(
        state_ema=state_ema,
        log_q_target=log_q_target,
        q_target=q_target,
        delta_log_q=delta_log_q,
        q_next=q_next,
    )


def effective_affine(q: float, config: ASACProtocolConfig | None = None) -> tuple[float, float]:
    config = config or ASACProtocolConfig.from_mapping()
    q = _finite_number(q, name="q")
    if q < 0.0:
        raise ProtocolError("q must be non-negative")
    return config.base_scale * q, config.base_translate * q


@dataclass(frozen=True)
class ControllerObservation:
    epoch: int
    train_q: float
    stability: float
    stability_reference: float | None
    state_ema: float | None
    log_q_target: float | None
    q_target: float | None
    delta_log_q: float | None
    next_q: float | None
    phase: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class A2Controller:
    """Sequential, strictly causal implementation of the A2 controller."""

    def __init__(self, config: ASACProtocolConfig | None = None) -> None:
        self.config = config or ASACProtocolConfig.from_mapping()
        self.config.validate()
        self._next_epoch = 1
        self._q_by_epoch: dict[int, float] = {1: self.config.q_init}
        self._stability_by_epoch: dict[int, float] = {}
        self._stability_reference: float | None = None
        self._state_ema: float | None = None
        self._observations: list[ControllerObservation] = []

    @property
    def next_epoch(self) -> int:
        return self._next_epoch

    @property
    def stability_reference(self) -> float | None:
        return self._stability_reference

    @property
    def observations(self) -> tuple[ControllerObservation, ...]:
        return tuple(self._observations)

    def q_for_epoch(self, epoch: int) -> float:
        epoch = _integer(epoch, name="epoch")
        if epoch not in self._q_by_epoch:
            raise ProtocolError(f"q for epoch {epoch} is not causally available yet")
        return self._q_by_epoch[epoch]

    def observe(self, epoch: int, stability: float) -> ControllerObservation:
        """Observe only joint fixed-probe stability for a completed epoch."""

        epoch = _integer(epoch, name="epoch")
        if epoch != self._next_epoch:
            raise ProtocolError(f"expected epoch {self._next_epoch}, got {epoch}")
        if epoch > self.config.epochs:
            raise ProtocolError("controller is already complete")
        stability = _finite_number(stability, name="stability")
        if not 0.0 <= stability <= 1.0:
            raise ProtocolError("stability must be in [0, 1]")

        q_epoch = self._q_by_epoch[epoch]
        self._stability_by_epoch[epoch] = stability
        update: FeedbackStep | None = None
        next_q: float | None = None

        if epoch < self.config.warmup_end:
            next_q = self.config.q_init
        elif epoch == self.config.warmup_end:
            self._stability_reference = reference_stability(self._stability_by_epoch)
            self._state_ema = self._stability_reference
            next_q = self.config.q_init
        elif self.config.feedback_start <= epoch <= self.config.feedback_update_end:
            assert self._stability_reference is not None and self._state_ema is not None
            update = feedback_step(
                q_epoch=q_epoch,
                state_ema_previous=self._state_ema,
                stability=stability,
                stability_reference=self._stability_reference,
                config=self.config,
            )
            self._state_ema = update.state_ema
            next_q = update.q_next
        # At epoch 50 the fixed probe is recorded, but no q_51 is produced.

        if next_q is not None:
            self._q_by_epoch[epoch + 1] = next_q
        observation = ControllerObservation(
            epoch=epoch,
            train_q=q_epoch,
            stability=stability,
            stability_reference=self._stability_reference,
            state_ema=self._state_ema,
            log_q_target=None if update is None else update.log_q_target,
            q_target=None if update is None else update.q_target,
            delta_log_q=None if update is None else update.delta_log_q,
            next_q=next_q,
            phase=phase_for_epoch(epoch, self.config),
        )
        self._observations.append(observation)
        self._next_epoch += 1
        return observation

    def schedule(self) -> tuple[float, ...]:
        last = min(self.config.epochs, max(self._q_by_epoch))
        return tuple(self._q_by_epoch[epoch] for epoch in range(1, last + 1))


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freezer_source_sha256() -> str:
    return sha256_file(__file__)


_KEY_TOKENS = re.compile(r"[a-z0-9]+")
_FORBIDDEN_PERFORMANCE_TOKENS = frozenset(
    {
        "accuracy", "ap", "ap50", "ap5095", "auc", "eval", "evaluation", "f1", "fitness", "loss",
        "losses", "map", "map50", "map5095", "metric", "metrics", "performance", "precision", "recall", "result",
        "results", "score", "scores", "test", "val", "validation",
    }
)


def _is_performance_key(key: str) -> bool:
    tokens = _KEY_TOKENS.findall(key.lower())
    if any(token in _FORBIDDEN_PERFORMANCE_TOKENS for token in tokens):
        return True
    compact = "".join(tokens)
    return compact.startswith(("map50", "valmap", "testmap", "valloss", "testap"))


def _project_a2_schedule_row(row: Mapping[str, Any], row_index: int, expected_seed: int) -> tuple[int, float]:
    """Project only epoch/train_q; reject performance keys before value access."""

    if not isinstance(row, Mapping):
        raise ProtocolError(f"A2 row {row_index} for seed {expected_seed} must be a mapping")
    keys = tuple(row.keys())
    if any(not isinstance(key, str) for key in keys):
        raise ProtocolError(f"A2 row {row_index} for seed {expected_seed} has non-string keys")
    forbidden = sorted(key for key in keys if _is_performance_key(key))
    if forbidden:
        raise ProtocolError(
            f"A2 row {row_index} for seed {expected_seed} contains forbidden performance fields: {forbidden}"
        )
    required = {"epoch", "variant", "train_q"}
    missing = required - set(keys)
    if missing:
        raise ProtocolError(f"A2 row {row_index} is missing schedule fields: {sorted(missing)}")
    epoch = _integer(row["epoch"], name=f"row {row_index} epoch")
    if row["variant"] != "A2":
        raise ProtocolError(f"row {row_index} variant must be 'A2'")
    if "seed" in keys and _integer(row["seed"], name=f"row {row_index} seed") != expected_seed:
        raise ProtocolError(f"row {row_index} seed does not match input seed {expected_seed}")
    q = _finite_number(row["train_q"], name=f"row {row_index} train_q")
    return epoch, q


def project_a2_schedule(
    a2_state_rows: Iterable[Mapping[str, Any]], *, seed: int, config: ASACProtocolConfig | None = None
) -> tuple[tuple[int, float], ...]:
    """Validate one complete A2 run and return ordered ``(epoch, train_q)``."""

    config = config or ASACProtocolConfig.from_mapping()
    seed = _integer(seed, name="seed")
    if seed not in config.seeds:
        raise ProtocolError(f"seed must be one of {config.seeds}")
    projected = [_project_a2_schedule_row(row, index, seed) for index, row in enumerate(a2_state_rows, start=1)]
    if len(projected) != config.epochs:
        raise ProtocolError(f"A2 seed {seed} must contain exactly {config.epochs} rows, got {len(projected)}")
    epochs = [epoch for epoch, _ in projected]
    if len(set(epochs)) != len(epochs):
        raise ProtocolError(f"A2 seed {seed} contains duplicate epochs")
    if set(epochs) != set(range(1, config.epochs + 1)):
        raise ProtocolError(f"A2 seed {seed} epochs must be exactly 1..50")
    projected.sort(key=lambda item: item[0])
    for epoch, q in projected:
        if not config.q_min <= q <= config.q_max:
            raise ProtocolError(f"A2 seed {seed} train_q at epoch {epoch} is outside [q_min, q_max]")
    q_by_epoch = dict(projected)
    for epoch in range(config.warmup_start, config.feedback_start + 1):
        if q_by_epoch[epoch] != config.q_init:
            raise ProtocolError(f"A2 seed {seed} train_q at epoch {epoch} must equal q_init={config.q_init}")
    for epoch in range(config.feedback_start + 1, config.feedback_end + 1):
        log_step = math.log(q_by_epoch[epoch]) - math.log(q_by_epoch[epoch - 1])
        if abs(log_step) > config.log_rate_limit + 1e-12:
            raise ProtocolError(f"A2 seed {seed} exceeds the frozen log-q rate limit before epoch {epoch}")
    return tuple(projected)


def _q_fingerprint(q_values: Sequence[float]) -> Counter[str]:
    return Counter(float(value).hex() for value in q_values)


def a3_source_epoch_map(config: ASACProtocolConfig | None = None) -> dict[int, int]:
    """Frozen left rotation: target epoch 6 reads source epoch 16."""

    config = config or ASACProtocolConfig.from_mapping()
    feedback_epochs = list(range(config.feedback_start, config.feedback_end + 1))
    shift = config.a3_left_shift % len(feedback_epochs)
    shifted_sources = feedback_epochs[shift:] + feedback_epochs[:shift]
    result = {epoch: epoch for epoch in range(config.warmup_start, config.warmup_end + 1)}
    result.update(dict(zip(feedback_epochs, shifted_sources)))
    return result


def _make_schedule_rows(
    *, variant: str, target_seed: int, source_seed: int, source_epoch_by_target: Mapping[int, int],
    source_q_by_epoch: Mapping[int, float], config: ASACProtocolConfig,
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for target_epoch in range(1, config.epochs + 1):
        source_epoch = source_epoch_by_target[target_epoch]
        q = source_q_by_epoch[source_epoch]
        scale, translate = effective_affine(q, config)
        rows.append(
            {
                "epoch": target_epoch, "variant": variant, "seed": target_seed,
                "phase": phase_for_epoch(target_epoch, config), "train_q": q,
                "effective_scale": scale, "effective_translate": translate,
                "source_variant": "A2", "source_seed": source_seed, "source_epoch": source_epoch,
            }
        )
    return tuple(rows)


def validate_frozen_controls(
    *, projected_by_seed: Mapping[int, Sequence[tuple[int, float]]],
    schedules: Mapping[str, Mapping[int, Sequence[Mapping[str, Any]]]],
    config: ASACProtocolConfig | None = None,
) -> None:
    """Hard-check A1/A3/A4 provenance and exact multiset invariants."""

    config = config or ASACProtocolConfig.from_mapping()
    if set(projected_by_seed) != set(config.seeds):
        raise ProtocolError("projected A2 seeds must be exactly {0, 1, 42}")
    if set(schedules) != {"A1", "A3", "A4"}:
        raise ProtocolError("control schedules must contain exactly A1, A3, and A4")
    q_by_seed = {seed: dict(rows) for seed, rows in projected_by_seed.items()}
    a3_map = a3_source_epoch_map(config)
    identity_map = {epoch: epoch for epoch in range(1, config.epochs + 1)}

    for variant in ("A1", "A3", "A4"):
        if set(schedules[variant]) != set(config.seeds):
            raise ProtocolError(f"{variant} schedules must cover exactly seeds {config.seeds}")
        for target_seed in config.seeds:
            rows = schedules[variant][target_seed]
            if len(rows) != config.epochs or [row.get("epoch") for row in rows] != list(range(1, 51)):
                raise ProtocolError(f"{variant} seed {target_seed} epochs must be exactly 1..50")
            if any(row.get("variant") != variant or row.get("seed") != target_seed for row in rows):
                raise ProtocolError(f"{variant} seed {target_seed} identity fields are inconsistent")
            source_seed = A1_SOURCE_SEED[target_seed] if variant == "A1" else target_seed
            source_map = a3_map if variant == "A3" else identity_map
            for row in rows:
                target_epoch = row["epoch"]
                source_epoch = source_map[target_epoch]
                if row.get("source_seed") != source_seed or row.get("source_epoch") != source_epoch:
                    raise ProtocolError(f"{variant} seed {target_seed} provenance mapping is incorrect")
                if float(row["train_q"]).hex() != float(q_by_seed[source_seed][source_epoch]).hex():
                    raise ProtocolError(f"{variant} seed {target_seed} altered a source A2 q value")
            if variant in {"A1", "A4"}:
                target_qs = [row["train_q"] for row in rows]
                source_qs = [q_by_seed[source_seed][epoch] for epoch in range(1, 51)]
            else:
                target_qs = [row["train_q"] for row in rows[5:]]
                source_qs = [q_by_seed[source_seed][epoch] for epoch in range(6, 51)]
                if any(float(rows[epoch - 1]["train_q"]).hex() != float(config.q_init).hex() for epoch in range(1, 6)):
                    raise ProtocolError(f"A3 seed {target_seed} changed warmup epochs")
            if _q_fingerprint(target_qs) != _q_fingerprint(source_qs):
                raise ProtocolError(f"{variant} seed {target_seed} failed exact q-multiset preservation")

    if any(A1_SOURCE_SEED[seed] == seed for seed in config.seeds):
        raise ProtocolError("A1 source-seed mapping has a fixed point")
    a2_collection = [q for seed in config.seeds for _, q in projected_by_seed[seed]]
    a1_collection = [row["train_q"] for seed in config.seeds for row in schedules["A1"][seed]]
    if _q_fingerprint(a1_collection) != _q_fingerprint(a2_collection):
        raise ProtocolError("A1 does not preserve the three-seed A2 q collection")


@dataclass(frozen=True)
class FrozenSchedules:
    schedules: dict[str, dict[int, tuple[dict[str, Any], ...]]]
    manifest: dict[str, Any]


def freeze_control_schedules(
    a2_state_rows_by_seed: Mapping[int, Iterable[Mapping[str, Any]]], *,
    input_state_sha256_by_seed: Mapping[int, str] | None = None,
    source_sha256: str | None = None, config: ASACProtocolConfig | None = None,
) -> FrozenSchedules:
    """Freeze all A1/A3/A4 controls after all three live A2 runs exist."""

    config = config or ASACProtocolConfig.from_mapping()
    if set(a2_state_rows_by_seed) != set(config.seeds):
        raise ProtocolError(f"A2 state inputs must cover exactly seeds {config.seeds}")
    if input_state_sha256_by_seed is not None:
        if set(input_state_sha256_by_seed) != set(config.seeds):
            raise ProtocolError(f"input state hashes must cover exactly seeds {config.seeds}")
        if not all(_valid_sha256(input_state_sha256_by_seed[seed]) for seed in config.seeds):
            raise ProtocolError("every input state hash must be a lowercase SHA256 hex digest")
    if source_sha256 is not None and not _valid_sha256(source_sha256):
        raise ProtocolError("freezer source hash must be a lowercase SHA256 hex digest")
    projected = {seed: project_a2_schedule(a2_state_rows_by_seed[seed], seed=seed, config=config) for seed in config.seeds}
    q_by_seed = {seed: dict(rows) for seed, rows in projected.items()}
    identity_map = {epoch: epoch for epoch in range(1, config.epochs + 1)}
    a3_map = a3_source_epoch_map(config)
    schedules: dict[str, dict[int, tuple[dict[str, Any], ...]]] = {"A1": {}, "A3": {}, "A4": {}}
    for target_seed in config.seeds:
        a1_source = A1_SOURCE_SEED[target_seed]
        schedules["A1"][target_seed] = _make_schedule_rows(
            variant="A1", target_seed=target_seed, source_seed=a1_source,
            source_epoch_by_target=identity_map, source_q_by_epoch=q_by_seed[a1_source], config=config,
        )
        schedules["A3"][target_seed] = _make_schedule_rows(
            variant="A3", target_seed=target_seed, source_seed=target_seed,
            source_epoch_by_target=a3_map, source_q_by_epoch=q_by_seed[target_seed], config=config,
        )
        schedules["A4"][target_seed] = _make_schedule_rows(
            variant="A4", target_seed=target_seed, source_seed=target_seed,
            source_epoch_by_target=identity_map, source_q_by_epoch=q_by_seed[target_seed], config=config,
        )
    validate_frozen_controls(projected_by_seed=projected, schedules=schedules, config=config)

    schedule_hashes = {
        variant: {str(seed): canonical_sha256(list(schedules[variant][seed])) for seed in config.seeds}
        for variant in ("A1", "A3", "A4")
    }
    a3_mapping_rows = [{"target_epoch": target, "source_epoch": source} for target, source in a3_map.items()]
    manifest = {
        "schema": "asac-phase1-freezer-v2",
        "protocol_sha256": canonical_sha256(config.to_dict()),
        "freezer_source_sha256": source_sha256,
        "input_state_sha256_by_seed": None if input_state_sha256_by_seed is None else {
            str(seed): input_state_sha256_by_seed[seed] for seed in config.seeds
        },
        "a2_schedule_projection_sha256_by_seed": {
            str(seed): canonical_sha256([
                {"epoch": epoch, "variant": "A2", "seed": seed, "train_q": q} for epoch, q in projected[seed]
            ]) for seed in config.seeds
        },
        "output_schedule_sha256": schedule_hashes,
        "a1_source_seed_by_target_seed": {str(seed): A1_SOURCE_SEED[seed] for seed in config.seeds},
        "a1_source_seed_mapping_sha256": canonical_sha256([
            {"target_seed": seed, "source_seed": A1_SOURCE_SEED[seed]} for seed in config.seeds
        ]),
        "a3_shift": {"direction": "left", "epochs": config.a3_left_shift, "phase": [6, 50]},
        "a3_source_epoch_by_target_epoch": {str(target): source for target, source in a3_map.items()},
        "a3_source_epoch_mapping_sha256": canonical_sha256(a3_mapping_rows),
        "performance_fields_consumed": [],
        "schedule_fields_consumed": ["epoch", "variant", "train_q"],
        "optional_integrity_fields_consumed": ["seed"],
    }
    return FrozenSchedules(schedules=schedules, manifest=manifest)


def _reject_json_constant(token: str) -> None:
    raise ProtocolError(f"non-finite JSON number is forbidden: {token}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def parse_a2_state_jsonl(raw: bytes) -> tuple[Mapping[str, Any], ...]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProtocolError("A2 state JSONL must be UTF-8") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ProtocolError(f"blank JSONL line at {line_number}")
        try:
            row = json.loads(line, object_pairs_hook=_object_without_duplicate_keys, parse_constant=_reject_json_constant)
        except (json.JSONDecodeError, ProtocolError) as exc:
            raise ProtocolError(f"invalid A2 state JSONL line {line_number}: {exc}") from exc
        if not isinstance(row, Mapping):
            raise ProtocolError(f"A2 state JSONL line {line_number} must be an object")
        rows.append(row)
    if not rows:
        raise ProtocolError("A2 state JSONL is empty")
    return tuple(rows)


def freeze_jsonl_bytes_by_seed(
    raw_by_seed: Mapping[int, bytes], config: ASACProtocolConfig | None = None
) -> FrozenSchedules:
    """Freeze from three raw JSONL files while retaining exact input hashes."""

    config = config or ASACProtocolConfig.from_mapping()
    if set(raw_by_seed) != set(config.seeds):
        raise ProtocolError(f"raw A2 inputs must cover exactly seeds {config.seeds}")
    return freeze_control_schedules(
        {seed: parse_a2_state_jsonl(raw_by_seed[seed]) for seed in config.seeds},
        input_state_sha256_by_seed={seed: sha256_bytes(raw_by_seed[seed]) for seed in config.seeds},
        source_sha256=freezer_source_sha256(), config=config,
    )


def _valid_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_frozen_schedules(bundle: FrozenSchedules, output_directory: str | os.PathLike[str]) -> dict[str, Path]:
    """Atomically write pinned artifacts; refuse manifests lacking raw/source hashes."""

    if not _valid_sha256(bundle.manifest.get("freezer_source_sha256")):
        raise ProtocolError("cannot write unpinned schedules: freezer source SHA256 is missing")
    input_hashes = bundle.manifest.get("input_state_sha256_by_seed")
    if not isinstance(input_hashes, Mapping) or set(input_hashes) != {"0", "1", "42"}:
        raise ProtocolError("cannot write unpinned schedules: three input state hashes are required")
    if not all(_valid_sha256(value) for value in input_hashes.values()):
        raise ProtocolError("cannot write unpinned schedules: invalid input state SHA256")
    output_directory = Path(output_directory)
    paths: dict[str, Path] = {}
    for variant in ("A1", "A3", "A4"):
        for seed in SEEDS:
            key = f"{variant.lower()}_seed{seed}"
            path = output_directory / f"{key}_schedule.json"
            _atomic_write(path, canonical_json_bytes(list(bundle.schedules[variant][seed])))
            expected_hash = bundle.manifest["output_schedule_sha256"][variant][str(seed)]
            if sha256_file(path) != expected_hash:
                raise ProtocolError(f"written schedule hash mismatch for {variant} seed {seed}")
            paths[key] = path
    manifest_path = output_directory / "freeze_manifest.json"
    _atomic_write(manifest_path, canonical_json_bytes(bundle.manifest))
    paths["manifest"] = manifest_path
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze A1/A3/A4 from three completed live A2 state JSONL files.")
    parser.add_argument("--a2-seed0", type=Path, required=True)
    parser.add_argument("--a2-seed1", type=Path, required=True)
    parser.add_argument("--a2-seed42", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    bundle = freeze_jsonl_bytes_by_seed({
        0: arguments.a2_seed0.read_bytes(), 1: arguments.a2_seed1.read_bytes(),
        42: arguments.a2_seed42.read_bytes(),
    })
    paths = write_frozen_schedules(bundle, arguments.output_dir)
    print(json.dumps({name: str(path) for name, path in paths.items()}, sort_keys=True))
    print(json.dumps(bundle.manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
