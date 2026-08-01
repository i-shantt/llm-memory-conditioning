"""Read-time context conditioners."""

from .base import (
    ComposeConditioner,
    Conditioner,
    Conditioning,
    IdentityConditioner,
    RenderedUnit,
    apply,
    condition,
    render_context,
)
from .supersede import LATEST, OUTDATED, SupersedeConditioner
from .temporal import TemporalConditioner

__all__ = [
    "ComposeConditioner", "Conditioner", "Conditioning", "IdentityConditioner",
    "RenderedUnit", "SupersedeConditioner", "TemporalConditioner",
    "LATEST", "OUTDATED", "apply", "build", "condition", "render_context",
]

_BUILDERS = {
    "identity": lambda: IdentityConditioner(),
    "supersede": lambda: SupersedeConditioner("mark"),
    "supersede:mark": lambda: SupersedeConditioner("mark"),
    "supersede:drop": lambda: SupersedeConditioner("drop"),
    "supersede:order": lambda: SupersedeConditioner("order"),
    "supersede:mark:naive": lambda: SupersedeConditioner("mark", require_conflict=False),
    "temporal": lambda: TemporalConditioner(sort=True),
    "temporal:norank": lambda: TemporalConditioner(sort=False),
    "all": lambda: ComposeConditioner(
        SupersedeConditioner("mark"), TemporalConditioner(sort=True), name="all"),
    "safe": lambda: ComposeConditioner(
        SupersedeConditioner("order"), TemporalConditioner(sort=True), name="safe"),
}


def build(name: str) -> Conditioner:
    """Name -> conditioner, so an arm is reproducible from its tag alone.

    memllm leaves the equivalent factory stranded in a script, which forced a
    cross-script import. Keeping it in the package avoids repeating that.
    """
    try:
        return _BUILDERS[name]()
    except KeyError:
        raise ValueError(
            f"unknown conditioner {name!r}; choose from {sorted(_BUILDERS)}"
        ) from None
