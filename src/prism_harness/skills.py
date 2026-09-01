"""Skills a mode can load into its system prompt, read from disk."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from prism_harness.errors import HarnessError

__all__ = ["SkillRegistry"]

_SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_TRAVERSAL = re.compile(r"(^|/)\.\.(/|$)")


class SkillRegistry:
    """The read is GUARDED, and that is the whole of it.

    A skill name and a path both arrive from configuration or, worse, from a
    model that was told it may fetch a referenced file. Reading either without
    checking is a path traversal with extra steps, and the file it would reach
    is on the machine running the agent.

    Three checks, in this order, each closing something the others do not:

    1. the NAME matches a strict pattern -- this is what stops ``../`` in the
       name itself, before it is ever joined to anything;
    2. the relative PATH is rejected lexically if it is absolute or contains a
       ``..`` segment;
    3. the resolved REAL path must still sit inside the skill's own real root,
       which is what catches a symlink pointing out of it -- the one thing the
       first two checks cannot see.

    The third is not redundant. A lexically innocent ``notes/link.md`` that is a
    symlink to ``/etc/passwd`` passes both earlier checks.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def augment_prompt(self, system_prompt: str, names: Sequence[str]) -> str:
        """The system prompt with each named skill appended as a tagged section.

        Returns the prompt UNCHANGED when no skills are named, rather than
        appending an empty preamble that would tell the model skills are
        available when none are.
        """
        sections = [
            f'<skill name="{name}">\n{self.read(name, "SKILL.md")}\n</skill>' for name in names
        ]

        if not sections:
            return system_prompt

        joined = "\n\n".join(sections)

        return (
            f"{system_prompt}\n\nThe following Harness-owned skills are available. Follow their "
            "routing instructions and use skill_read for referenced files. Do not copy skill "
            f"files into the project workspace.\n\n{joined}"
        ).strip()

    def read(self, name: str, path: str) -> str:
        """Read one file from inside a skill. See the class docstring for the guard."""
        if not _SKILL_NAME.match(name):
            raise HarnessError.skill_path_refused(f"the skill name [{name}] is not a valid name")

        relative = path.replace("\\", "/").strip()

        if not relative or relative.startswith("/") or _TRAVERSAL.search(relative):
            raise HarnessError.skill_path_refused(f"[{path}] must stay inside the skill")

        try:
            skill_root = (self.root / name).resolve(strict=True)
            file = (self.root / name).joinpath(*relative.split("/")).resolve(strict=True)
        except OSError as error:
            raise HarnessError.skill_path_refused(f"[{name}/{relative}] was not found") from error

        # The check the lexical ones cannot make: a symlink inside the skill
        # that points out of it is lexically innocent and resolves elsewhere.
        if not file.is_relative_to(skill_root) or file == skill_root or not file.is_file():
            raise HarnessError.skill_path_refused(f"[{name}/{relative}] resolves outside the skill")

        return file.read_text(encoding="utf-8")
