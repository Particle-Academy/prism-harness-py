"""Check the harness configuration before a run does it for you."""

from __future__ import annotations

from dataclasses import dataclass, field

from prism_harness.modes import ModeRegistry
from prism_harness.stores.manager import SessionStoreManager
from prism_harness.tools import ToolAuthorizer, ToolRegistry

__all__ = ["DoctorFinding", "DoctorReport", "diagnose"]


@dataclass(frozen=True)
class DoctorFinding:
    check: str
    ok: bool
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    findings: list[DoctorFinding] = field(default_factory=list)

    @property
    def problems(self) -> int:
        return sum(1 for finding in self.findings if not finding.ok)

    @property
    def ok(self) -> bool:
        return self.problems == 0

    def summary(self) -> str:
        lines = [
            "Harness configuration is consistent."
            if self.ok
            else f"{self.problems} problem(s) found in the harness configuration."
        ]
        lines.extend(
            f"  {'ok  ' if finding.ok else 'FAIL'} {finding.check}: {finding.detail}"
            for finding in self.findings
        )
        return "\n".join(lines)


def diagnose(
    modes: ModeRegistry | None = None,
    tools: ToolRegistry | None = None,
    stores: SessionStoreManager | None = None,
    authorizer: ToolAuthorizer | None = None,
) -> DoctorReport:
    """Every check here corresponds to a failure this package already refuses at
    runtime. Those refusals are correct and they are also LATE: a mode nobody
    has entered yet keeps its broken subagent reference until someone switches
    to it, and the first person to find out is a user mid-conversation.

    So this resolves EVERY mode rather than the default one, and reports what a
    run would have raised. A CLI is left to the consumer -- this returns the
    report as data, which is also what makes it testable and what lets a health
    endpoint serve it.
    """
    findings: list[DoctorFinding] = []

    if stores is not None:
        findings.append(_check_stores(stores))

    if authorizer is not None:
        findings.append(
            DoctorFinding(
                check="authorizer",
                ok=True,
                detail="enabled -- tools are filtered per run and per call"
                if authorizer.enabled
                else "DISABLED -- every registered tool is offered to every run",
            )
        )

    if modes is not None:
        findings.extend(_check_modes(modes, tools))

    return DoctorReport(findings=findings)


def _check_stores(stores: SessionStoreManager) -> DoctorFinding:
    try:
        durable = stores.durable()
        stores.ephemeral()
    except Exception as error:  # noqa: BLE001 - reporting, not handling
        return DoctorFinding(check="stores", ok=False, detail=str(error))

    return DoctorFinding(
        check="stores", ok=True, detail=f"durable slot is {durable.durability().value}"
    )


def _check_modes(modes: ModeRegistry, tools: ToolRegistry | None) -> list[DoctorFinding]:
    """Resolve EVERY mode, and check the tools each one names.

    A mode listing a tool the registry cannot produce fails only when a run
    reaches for it -- which is to say, in front of whoever is talking to the
    agent. Checked against the registry's static names; a tool only a provider
    can supply needs a session, so a mode using ``*`` is not flagged.
    """
    names = modes.names()

    if not names:
        return [DoctorFinding(check="modes", ok=False, detail="no modes are configured")]

    findings: list[DoctorFinding] = []
    known = set(tools.names()) if tools is not None else set()

    for name in names:
        try:
            mode = modes.resolve(name)
        except Exception as error:  # noqa: BLE001 - reporting, not handling
            findings.append(DoctorFinding(check=f"mode:{name}", ok=False, detail=str(error)))
            continue

        missing = (
            []
            if tools is None or "*" in mode.tools
            else [tool for tool in mode.tools if tool not in known]
        )

        if missing:
            findings.append(
                DoctorFinding(
                    check=f"mode:{name}",
                    ok=False,
                    detail=f"names {len(missing)} tool(s) the registry cannot produce: "
                    + ", ".join(missing),
                )
            )
            continue

        detail = [
            f"{len(mode.tools)} tool(s)",
            f"{mode.max_steps} step(s)",
            f"{len(mode.subagents)} subagent(s)",
        ]
        if mode.requires_approval:
            detail.append("approval: " + ", ".join(mode.requires_approval))

        findings.append(DoctorFinding(check=f"mode:{name}", ok=True, detail=", ".join(detail)))

    # The default has to resolve, or every session that does not name a mode
    # fails -- and that is the ordinary path, not an edge.
    try:
        modes.resolve(None)
        findings.append(DoctorFinding(check="default mode", ok=True, detail=modes.default()))
    except Exception as error:  # noqa: BLE001 - reporting, not handling
        findings.append(DoctorFinding(check="default mode", ok=False, detail=str(error)))

    return findings
