"""Portable automations — triggers + actions as plain dicts (§17, §36).

An automation is a data record that travels between nodes, Brain and
dashboards unchanged::

    {
        "id": "abc123",
        "name": "lights off when empty",
        "enabled": true,
        "trigger": {
            "type": "condition",               # time | condition | event
            "when": "snr_mean > 4",            # rule-language expression
            "labels": ["empty"],               # optional prediction gate
            "for_s": 30.0                      # must hold this long
        },
        "action": {
            "type": "home_assistant",          # executor Action.type
            "config": {"entity_id": "light.hall", "service": "turn_off"},
            "cooldown_seconds": 60.0
        }
    }

Trigger kinds
-------------
- ``time``      — ``interval_s`` periodic, or ``at: ["07:30", "18:45"]``
                  daily wall-clock times (node-local).
- ``condition`` — numeric ``when`` expression over window features
                  (same parser as :class:`RuleProcessor`), optionally
                  gated by ``labels``/``min_confidence`` on the latest
                  prediction. ``for_s`` debounces: the condition must
                  hold continuously before firing.
- ``event``     — ``{"on": "label", "label": "occupied"}`` fires once
                  when the prediction label *becomes* ``label`` (edge).
                  ``{"on": "prediction"}`` fires on every prediction
                  that passes ``labels``/``min_confidence``.

Actions are executor action configs — the same dicts ``Action`` already
accepts — plus ``type: "lan"`` for actuators on other nodes reachable
via ``whispy.lan``::

    {"type": "lan",
     "config": {"host": "10.0.0.22", "port": 5001, "token": "...",
                "actuator": "light-c483",
                "operation": "set_color",
                "params": {"rgb": [0, 255, 0]}}}

String fields in action ``config``/``params`` may reference
``{label}``, ``{confidence}``, ``{device_id}``, ``{model_id}`` and
``{timestamp}`` — rendered at fire time.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from .contracts import Prediction

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------

@dataclass
class Automation:
    """A stored automation record (trigger + action + runtime state)."""

    id: str = ""
    name: str = ""
    enabled: bool = True
    trigger: Dict[str, Any] = field(default_factory=dict)
    action: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    # Non-spec runtime state (persisted alongside, harmless to travel).
    state: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Automation":
        return cls(
            id=str(data.get("id") or uuid.uuid4().hex[:12]),
            name=str(data.get("name") or "automation"),
            enabled=bool(data.get("enabled", True)),
            trigger=dict(data.get("trigger") or {}),
            action=dict(data.get("action") or {}),
            created_at=float(data.get("created_at") or time.time()),
            state=dict(data.get("state") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id, "name": self.name, "enabled": self.enabled,
            "trigger": dict(self.trigger), "action": dict(self.action),
            "created_at": self.created_at, "state": dict(self.state),
        }


# ---------------------------------------------------------------------------
# Trigger evaluation
# ---------------------------------------------------------------------------

def _feature_resolver(ctx: Mapping[str, Any],
                      features: Any = None):
    """Resolve a rule-language identifier from context then features.

    ``features`` is a ``WindowFeatures``-like object (has ``feature(name)``)
    consulted lazily so only identifiers the expression actually uses are
    computed. Returns NaN for unknown identifiers — every comparison
    against NaN is False, mirroring RuleProcessor's "unknown feature ⇒
    no match".
    """
    def resolve(name: str) -> float:
        try:
            val = ctx.get(name)
            if val is not None:
                if isinstance(val, bool):
                    return 1.0 if val else 0.0
                return float(val)
        except (TypeError, ValueError):
            return float("nan")
        if features is not None:
            try:
                return float(features.feature(name))
            except Exception:
                pass
        return float("nan")
    return resolve


def eval_expression(expr: str, ctx: Mapping[str, Any],
                    features: Any = None) -> bool:
    """Evaluate a rule-language boolean expression against ctx + features."""
    from .processors.rules import _Parser, _tokenize
    try:
        return bool(_Parser(_tokenize(expr),
                            _feature_resolver(ctx, features)).parse())
    except Exception as exc:
        logger.debug("automation expression %r failed: %s", expr, exc)
        return False


def _labels_match(trigger: Mapping[str, Any], ctx: Mapping[str, Any]) -> bool:
    labels = trigger.get("labels") or trigger.get("label")
    if isinstance(labels, str):
        labels = [labels]
    if labels:
        return str(ctx.get("label") or "") in [str(x) for x in labels]
    return True


def _confidence_ok(trigger: Mapping[str, Any], ctx: Mapping[str, Any]) -> bool:
    try:
        return float(ctx.get("confidence") or 0.0) >= \
            float(trigger.get("min_confidence") or 0.0)
    except (TypeError, ValueError):
        return False


def _clock_seconds() -> int:
    lt = time.localtime()
    return lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec


def should_fire(auto: Automation, ctx: Mapping[str, Any],
                now: Optional[float] = None,
                features: Any = None) -> bool:
    """Decide whether ``auto``'s trigger fires for this tick/context.

    Edge semantics live in ``auto.state`` (mutated in place) — callers
    persist it however they like. ``ctx`` carries prediction scalars
    (``label``, ``confidence``, ``model_id``); ``features`` is an
    optional ``WindowFeatures`` for ``when`` expressions.
    """
    now = time.time() if now is None else now
    trig = auto.trigger or {}
    ttype = str(trig.get("type") or "")
    st = auto.state

    if ttype == "time":
        interval = trig.get("interval_s")
        if interval is not None:
            last = float(st.get("last_fired") or 0.0)
            if now - last >= float(interval):
                st["last_fired"] = now
                return True
            return False
        # Daily wall-clock times, e.g. {"at": ["07:30", "18:45"]}.
        times = trig.get("at") or []
        if isinstance(times, str):
            times = [times]
        cur = _clock_seconds()
        hit = False
        for t in times:
            try:
                hh, mm = [int(x) for x in str(t).split(":")[:2]]
            except ValueError:
                continue
            target = hh * 3600 + mm * 60
            if abs(cur - target) <= 30:          # within the same minute
                hit = True
                break
        marker = f"{time.strftime('%Y-%m-%d')}:{cur // 60}"
        if hit and st.get("last_clock_minute") != marker:
            st["last_clock_minute"] = marker
            return True
        return False

    if ttype == "condition":
        ok = _labels_match(trig, ctx) and _confidence_ok(trig, ctx)
        expr = trig.get("when")
        if ok and expr:
            ok = eval_expression(str(expr), ctx, features)
        for_s = float(trig.get("for_s") or 0.0)
        if ok:
            armed = st.get("armed_since")
            if for_s <= 0:
                st["armed_since"] = None
                if not st.get("latched"):      # fire-once until reset
                    st["latched"] = True
                    return True
                return False
            if armed is None:
                st["armed_since"] = now
                return False
            if now - float(armed) >= for_s:
                if not st.get("latched"):
                    st["latched"] = True
                    return True
            return False
        # Condition cleared → reset debounce + latch.
        st["armed_since"] = None
        st["latched"] = False
        return False

    if ttype == "event":
        on = str(trig.get("on") or "label")
        if on == "prediction":
            return (_labels_match(trig, ctx) and _confidence_ok(trig, ctx))
        # on="label" — edge: label became (or changed to) target.
        target = str(trig.get("label") or (trig.get("labels") or [""])[0])
        prev = st.get("prev_label")
        cur = str(ctx.get("label") or "")
        st["prev_label"] = cur
        if _confidence_ok(trig, ctx) and cur == target and prev != target:
            return True
        return False

    logger.warning("unknown automation trigger type %r", ttype)
    return False


# ---------------------------------------------------------------------------
# Action rendering + lan executor
# ---------------------------------------------------------------------------

def render(template: Any, fmt: Mapping[str, Any]) -> Any:
    """Recursively ``{placeholder}``-render strings in a config tree."""
    if isinstance(template, str):
        try:
            return template.format(**fmt)
        except (KeyError, IndexError, ValueError):
            return template
    if isinstance(template, dict):
        return {k: render(v, fmt) for k, v in template.items()}
    if isinstance(template, list):
        return [render(v, fmt) for v in template]
    return template


def format_context(prediction: Optional[Prediction],
                   ctx: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Build the ``{...}`` substitution map for an action config."""
    out: Dict[str, Any] = {
        "label": getattr(prediction, "label", "") or "",
        "confidence": getattr(prediction, "confidence", "") or "",
        "device_id": getattr(prediction, "device_id", "") or "",
        "model_id": getattr(prediction, "runtime_model_id", "") or "",
        "timestamp": time.time(),
    }
    if ctx:
        out.update({k: v for k, v in ctx.items()
                    if isinstance(k, str)})
    return out


__all__ = [
    "Automation",
    "eval_expression",
    "format_context",
    "render",
    "should_fire",
]
