"""The bounds a plan runs inside (G-360), read from the host's ``plans`` config.

A value outside its range is REFUSED at construction, never clamped: an
operator who asked for 500 steps should learn that 40 is the most an update can
carry, not find out from a plan that was silently cut.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["EVENT_BYTES_CEILING", "STEPS_CEILING", "PlanLimits"]

#: Fancy contract g282.3: a plan carries at most 40 steps.
STEPS_CEILING = 40

#: Fancy contract g282.3: the first update is under 8 KB.
EVENT_BYTES_CEILING = 8192


@dataclass(frozen=True)
class PlanLimits:
    max_steps: int = 20
    max_attempts: int = 3
    max_pauses_per_step: int = 1
    max_changes: int = 5
    max_advances: int = 60
    context_max_chars: int = 2000
    context_window: int = 8
    event_max_bytes: int = EVENT_BYTES_CEILING
    require_approval: bool = True
    run_steps: bool = False
    stale_after: int = 300

    def __post_init__(self) -> None:
        ranges: dict[str, tuple[object, int, int]] = {
            "max_steps": (self.max_steps, 1, STEPS_CEILING),
            "max_attempts": (self.max_attempts, 1, 20),
            "max_pauses_per_step": (self.max_pauses_per_step, 0, 10),
            "max_changes": (self.max_changes, 0, 50),
            "max_advances": (self.max_advances, 1, 10000),
            "context.max_chars": (self.context_max_chars, 500, 20000),
            "context.window": (self.context_window, 0, STEPS_CEILING),
            "events.max_bytes": (self.event_max_bytes, 1024, EVENT_BYTES_CEILING),
            "reaper.stale_after": (self.stale_after, 1, 86400),
        }
        for name, (value, low, high) in ranges.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"plans.{name} must be a whole number.")
            if value < low or value > high:
                raise ValueError(f"plans.{name} must be between {low} and {high}.")
        for name, flag in (
            ("require_approval", self.require_approval),
            ("events.run_steps", self.run_steps),
        ):
            if not isinstance(flag, bool):
                raise TypeError(f"plans.{name} must be true or false.")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> PlanLimits:
        def whole(value: object, default: int, name: str) -> int:
            if value is None:
                return default
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"plans.{name} must be a whole number.")
            return value

        def flag(value: object, default: bool, name: str) -> bool:
            if value is None:
                return default
            if not isinstance(value, bool):
                raise TypeError(f"plans.{name} must be true or false.")
            return value

        context = config.get("context")
        events = config.get("events")
        reaper = config.get("reaper")
        context = context if isinstance(context, Mapping) else {}
        events = events if isinstance(events, Mapping) else {}
        reaper = reaper if isinstance(reaper, Mapping) else {}

        return cls(
            max_steps=whole(config.get("max_steps"), 20, "max_steps"),
            max_attempts=whole(config.get("max_attempts"), 3, "max_attempts"),
            max_pauses_per_step=whole(config.get("max_pauses_per_step"), 1, "max_pauses_per_step"),
            max_changes=whole(config.get("max_changes"), 5, "max_changes"),
            max_advances=whole(config.get("max_advances"), 60, "max_advances"),
            context_max_chars=whole(context.get("max_chars"), 2000, "context.max_chars"),
            context_window=whole(context.get("window"), 8, "context.window"),
            event_max_bytes=whole(events.get("max_bytes"), EVENT_BYTES_CEILING, "events.max_bytes"),
            require_approval=flag(config.get("require_approval"), True, "require_approval"),
            run_steps=flag(events.get("run_steps"), False, "events.run_steps"),
            stale_after=whole(reaper.get("stale_after"), 300, "reaper.stale_after"),
        )

    def to_dict(self) -> dict[str, int | bool]:
        return {
            "max_steps": self.max_steps,
            "max_attempts": self.max_attempts,
            "max_pauses_per_step": self.max_pauses_per_step,
            "max_changes": self.max_changes,
            "max_advances": self.max_advances,
            "context_max_chars": self.context_max_chars,
            "context_window": self.context_window,
            "event_max_bytes": self.event_max_bytes,
            "require_approval": self.require_approval,
            "run_steps": self.run_steps,
            "stale_after": self.stale_after,
        }
