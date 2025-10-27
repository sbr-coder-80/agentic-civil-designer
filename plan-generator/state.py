from typing import TypedDict, Literal

class State(TypedDict):
    committed_history: list[dict]   # only validated messages
    staged_user_msg: dict | None
    staged_model_msg: dict | None
    input_kind: Literal["valid","follow-up","invalid"]
    output_kind: Literal["valid","follow-up","invalid",]
    violations: list[str]