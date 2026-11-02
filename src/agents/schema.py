"""Pydantic models for agent action JSON."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator

ActionType = Literal["forge_alliance", "break_alliance", "hold", "broadcast_state"]


class Action(BaseModel):
    """Strict action schema returned by every agent's decide().

    Attributes:
        action: One of the four legal action types.
        target: Faction id the action refers to, or None.
        vote: Faction id of the agent's preferred unifying leader.
        dialogue: Short first-person string.
    """

    model_config = ConfigDict(extra="forbid")

    action: ActionType
    target: Optional[str] = None
    vote: str
    dialogue: str = ""

    @field_validator("target", mode="before")
    @classmethod
    def _normalize_null_target(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, str) and v.strip().lower() in ("", "null", "none", "nil"):
            return None
        return v
