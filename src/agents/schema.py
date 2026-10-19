"""Pydantic models for agent action JSON."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict

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
