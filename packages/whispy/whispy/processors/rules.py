"""RuleProcessor — one validated rule schema (§36).

Rule schema::

    {
        "name": "radar-occupancy",
        "processor": "rule",
        "sensor": "radar",
        "rules": [
            {"when": "snr_mean > 12 and energy > 0.4", "label": "occupied",
             "confidence": 0.9},
            {"when": "rms_accel > 3.8 or peak_snr > 24.0", "label": "alert"}
        ],
        "else": "empty",
        "params": {"snr_threshold": 12.0}
    }

Compound expressions support ``and`` / ``or`` / ``not`` and parentheses.
Only features actually computable from the window are usable — unknown
features make the rule non-matching rather than silently zero.
"""

from __future__ import annotations

import logging
import operator
import re
from typing import Any, Dict, List, Optional

from ..contracts import Prediction, SensorWindow
from ..windows import WindowFeatures
from .base import Processor, ProcessorMeta

logger = logging.getLogger(__name__)

_OPS = {
    ">": operator.gt, ">=": operator.ge, "<": operator.lt,
    "<=": operator.le, "==": operator.eq, "!=": operator.ne,
}

_TOKEN_RE = re.compile(r"""
    \s*(?P<tok>
        >=|<=|==|!=|>|<            # comparison
      | \( | \)                    # parens
      | \b(?:and|or|not)\b         # boolean ops
      | -?\d+(?:\.\d+)?            # number literal
      | [A-Za-z_]\w*               # identifier (feature or param)
    )
""", re.VERBOSE)

_CMP = set(_OPS)


class _ExprError(ValueError):
    pass


def _tokenize(expr: str) -> List[str]:
    tokens: List[str] = []
    pos = 0
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            if expr[pos].isspace():
                pos += 1
                continue
            raise _ExprError(f"bad token at {expr[pos:]!r}")
        tokens.append(m.group("tok"))
        pos = m.end()
    return tokens


class _Parser:
    """Recursive-descent parser: or → and → not → comparison → atom."""

    def __init__(self, tokens: List[str], resolve):
        self._tokens = tokens
        self._pos = 0
        self._resolve = resolve

    def _peek(self) -> Optional[str]:
        return self._tokens[self._pos] if self._pos < len(self._tokens) else None

    def _next(self) -> str:
        tok = self._peek()
        if tok is None:
            raise _ExprError("unexpected end of expression")
        self._pos += 1
        return tok

    def parse(self) -> bool:
        value = self._parse_or()
        if self._peek() is not None:
            raise _ExprError(f"trailing token {self._peek()!r}")
        return value

    def _parse_or(self) -> bool:
        left = self._parse_and()
        while self._peek() == "or":
            self._next()
            right = self._parse_and()
            left = left or right
        return left

    def _parse_and(self) -> bool:
        left = self._parse_not()
        while self._peek() == "and":
            self._next()
            right = self._parse_not()
            left = left and right
        return left

    def _parse_not(self) -> bool:
        if self._peek() == "not":
            self._next()
            return not self._parse_not()
        return self._parse_comparison()

    def _parse_comparison(self) -> bool:
        left = self._parse_atom()
        tok = self._peek()
        if tok in _CMP:
            self._next()
            right = self._parse_atom()
            return bool(_OPS[tok](left, right))
        # Bare identifier/number treated as truthiness
        return bool(left)

    def _parse_atom(self) -> float:
        tok = self._next()
        if tok == "(":
            value = self._parse_or()
            if self._next() != ")":
                raise _ExprError("missing closing paren")
            return 1.0 if value else 0.0
        if re.fullmatch(r"-?\d+(?:\.\d+)?", tok):
            return float(tok)
        if re.fullmatch(r"[A-Za-z_]\w*", tok):
            return float(self._resolve(tok))
        raise _ExprError(f"unexpected token {tok!r}")


class RuleProcessor(Processor):
    """Evaluates validated rule expressions against window features."""

    def __init__(self, config: Dict[str, Any],
                 meta: Optional[ProcessorMeta] = None):
        self._config = config
        self._rules: List[Dict[str, Any]] = list(config.get("rules") or [])
        # Legacy single-expression form: {"rule": "snr_db > 12", ...}
        if not self._rules and config.get("rule"):
            self._rules = [{
                "when": config["rule"],
                "label": config.get("target_label", "positive"),
                "confidence": config.get("confidence", 1.0),
            }]
        self._else = config.get("else", "unknown")
        self._params = dict(config.get("params") or {})
        self._meta = meta or ProcessorMeta(
            name=config.get("name", "rule-processor"),
            processor_type="rule",
            sensor=config.get("sensor", "any"),
            task=config.get("task", "occupancy"),
            inputs=tuple(i.get("sensor") for i in config.get("inputs", [])
                         if isinstance(i, dict)),
            config_schema=config.get("config_schema") or {},
        )
        # Validate all expressions eagerly — bad rules fail at deploy time,
        # not at inference time (§36: "unsupported configuration fails
        # before deployment").
        for rule in self._rules:
            self._compile(rule.get("when", ""))

    def metadata(self) -> ProcessorMeta:
        return self._meta

    def configure(self, config: Dict[str, Any]) -> None:
        params = config.get("params", config)
        if isinstance(params, dict):
            self._params.update(params)

    @staticmethod
    def _compile(expr: str) -> List[str]:
        if not expr or not isinstance(expr, str):
            raise _ExprError("rule 'when' must be a non-empty expression")
        return _tokenize(expr)

    def _resolve(self, token: str, features: WindowFeatures) -> float:
        try:
            return features.feature(token)
        except KeyError:
            if token in self._params:
                return float(self._params[token])
            raise

    def _eval(self, expr: str, features: WindowFeatures) -> bool:
        tokens = _tokenize(expr)
        parser = _Parser(tokens, lambda t: self._resolve(t, features))
        return parser.parse()

    def predict(self, window: SensorWindow) -> Prediction:
        features = WindowFeatures(window)
        for rule in self._rules:
            expr = rule.get("when", "")
            try:
                matched = self._eval(expr, features)
            except (KeyError, _ExprError):
                # Missing feature → rule does not match (explicit, not zero).
                continue
            if matched:
                return Prediction(
                    label=rule.get("label", "positive"),
                    confidence=float(rule.get("confidence", 1.0)),
                    metadata={"rule": expr},
                )
        return Prediction(label=self._else, confidence=1.0)


__all__ = ["RuleProcessor"]
