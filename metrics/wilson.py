"""Wilson score confidence intervals for binomial proportions.

Required by the evaluation plan to report rigorous confidence intervals
on small benchmark sets (e.g. 40-60 split-context cases, InjecAgent subsets).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class WilsonInterval:
    """Represents a Wilson score confidence interval for a proportion.

    Attributes:
        successes: Number of positive occurrences (e.g., successful attacks or completed tasks).
        total: Total number of evaluated cases.
        rate: Estimated empirical proportion (successes / total).
        lower: Lower bound of the confidence interval (clamped to [0.0, 1.0]).
        upper: Upper bound of the confidence interval (clamped to [0.0, 1.0]).
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        z: Normal distribution critical value corresponding to confidence.
    """
    successes: int
    total: int
    rate: float
    lower: float
    upper: float
    confidence: float = 0.95
    z: float = 1.95996

    @property
    def margin_of_error(self) -> float:
        """Approximate symmetric half-width margin of error."""
        return (self.upper - self.lower) / 2.0

    def as_pct(self, precision: int = 1) -> str:
        """Format as human-readable string: e.g. '24.0% [CI: 16.5% - 33.4%]'."""
        return (
            f"{self.rate * 100:.{precision}f}% "
            f"[CI: {self.lower * 100:.{precision}f}% - {self.upper * 100:.{precision}f}%]"
        )

    def as_ci_only(self, precision: int = 1) -> str:
        """Format only the interval: e.g. '[16.5% - 33.4%]'."""
        return f"[{self.lower * 100:.{precision}f}% - {self.upper * 100:.{precision}f}%]"

    def as_markdown(self, precision: int = 1) -> str:
        """Format for inclusion in Markdown tables."""
        return f"**{self.rate * 100:.{precision}f}%** <br><sub>[{self.lower * 100:.{precision}f}%, {self.upper * 100:.{precision}f}%]</sub>"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "successes": self.successes,
            "total": self.total,
            "rate": round(self.rate, 4),
            "lower": round(self.lower, 4),
            "upper": round(self.upper, 4),
            "confidence": self.confidence,
            "formatted": self.as_pct(),
        }


def _get_z_score(confidence: float) -> float:
    """Return standard two-tailed z-score for given confidence."""
    # Common confidence values lookup for precision
    z_lookup = {
        0.90: 1.6448536269514722,
        0.95: 1.959963984540054,
        0.98: 2.3263478740408408,
        0.99: 2.5758293035489004,
    }
    if confidence in z_lookup:
        return z_lookup[confidence]
    # Fallback to inverse error function approximation
    p = 1.0 - (1.0 - confidence) / 2.0
    # Winitzki approximation for erf inverse
    a = 0.147
    term = 2.0 / (math.pi * a) + math.log(1.0 - (2.0 * p - 1.0) ** 2) / 2.0
    val = (
        term**2 - math.log(1.0 - (2.0 * p - 1.0) ** 2) / a
    )
    inv_erf = math.copysign(math.sqrt(math.sqrt(max(0.0, val)) - term), 2.0 * p - 1.0)
    return math.sqrt(2.0) * inv_erf


def wilson_interval(
    successes: int,
    total: int,
    confidence: float = 0.95,
) -> WilsonInterval:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Args:
        successes: Number of successes / positive cases (0 <= successes <= total).
        total: Total number of trials / cases.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        WilsonInterval with empirical rate and exact [lower, upper] bounds.
    """
    if total <= 0:
        return WilsonInterval(
            successes=0,
            total=0,
            rate=0.0,
            lower=0.0,
            upper=0.0,
            confidence=confidence,
            z=_get_z_score(confidence),
        )

    successes = max(0, min(successes, total))
    p_hat = successes / total
    z = _get_z_score(confidence)
    z2 = z * z

    # Wilson score interval formulation
    denom = 1.0 + z2 / total
    center = (p_hat + z2 / (2.0 * total)) / denom
    spread = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z2 / (4.0 * total * total)))

    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    # Guard against floating-point drift so the point estimate always lies inside
    # the interval (e.g. p=1.0 previously produced upper=0.9999999999999999).
    lower = min(lower, p_hat)
    upper = max(upper, p_hat)

    return WilsonInterval(
        successes=successes,
        total=total,
        rate=p_hat,
        lower=lower,
        upper=upper,
        confidence=confidence,
        z=z,
    )


def format_ci(successes: int, total: int, confidence: float = 0.95, precision: int = 1) -> str:
    """Convenience helper to compute and format a Wilson interval in one call."""
    return wilson_interval(successes, total, confidence).as_pct(precision)
