# validators.py
import json
from typing import Literal
from openai import OpenAI
from state import State
from config import config

# Initialize OpenAI once per process
CLIENT = OpenAI(api_key=config.openai_api_key)
MODEL = "gpt-4o-mini"

ALLOWED_INPUT_KINDS: tuple[Literal["valid","follow-up","invalid"], ...] = ("valid","follow-up","invalid")
ALLOWED_OUTPUT_KINDS: tuple[Literal["valid","follow-up","invalid"], ...] = ("valid","follow-up","invalid")

def _commit_if_valid(state: State, ok: bool) -> None:
    if ok and state.get("staged_user_msg"):
        state["committed_history"].append(state["staged_user_msg"])
    # always clear the staged user msg after decision
    state["staged_user_msg"] = None

def _commit_model_if_valid(state: State, ok: bool) -> None:
    """Commit staged_model_msg to committed_history if valid."""
    if ok and state.get("staged_model_msg"):
        state["committed_history"].append(state["staged_model_msg"])
    # always clear the staged model msg after decision
    state["staged_model_msg"] = None

def validate_user_fn(state: State) -> State:
    """
    Calls LLM to validate if staged_user_msg is about layouts/floor plans.
    Sets state['input_kind'] ∈ {'valid','follow-up','invalid'}
    Commits staged_user_msg to committed_history only if valid or follow-up.
    """
    msg = state.get("staged_user_msg")
    if not msg or not isinstance(msg, dict) or not msg.get("content"):
        state["input_kind"] = "invalid"
        state["violations"].append("Empty or missing user message.")
        _commit_if_valid(state, ok=False)
        return state

    # Prepare compact history (last N messages to keep prompt small)
    history = state.get("committed_history", [])[-8:]

    system_prompt = (
        "You are an input validator for a civil-planning chatbot. "
        "Decide if the USER message is about site layouts, floor plans, room programs, "
        "plot dimensions, setbacks, orientation, circulation, adjacency, elevations, or related clarifications. "
        "Return strict JSON with fields: {\"input_kind\":\"valid|follow-up|invalid\",\"reasons\":[...]}.\n"
        "- valid: clearly provides requirements or content to design/iterate a floor/site plan.\n"
        "- follow-up: asks for clarifications or provides missing details relevant to the plan.\n"
        "- invalid: off-topic (e.g., jokes, unrelated coding, finance, etc.)."
    )

    user_payload = {
        "history": history,           # prior validated turns only
        "candidate_user_msg": msg,    # untrusted message to judge
    }

    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        print("validator response:", data)
        input_kind = data.get("input_kind", "invalid")
        reasons = data.get("reasons", [])

        if input_kind not in ALLOWED_INPUT_KINDS:
            input_kind = "invalid"

        state["input_kind"] = input_kind
        if input_kind == "invalid":
            state["violations"].append(
                f"User input rejected by validator: {reasons if reasons else 'No reasons provided.'}"
            )

        _commit_if_valid(state, ok=(input_kind in ("valid", "follow-up")))
        return state

    except Exception as e:
        state["input_kind"] = "invalid"
        state["violations"].append(f"Validator error: {e}")
        _commit_if_valid(state, ok=False)
        return state

def validate_model_fn(state: State) -> State:
    """
    Calls LLM to validate if staged_model_msg is appropriate and helpful.
    Sets state['output_kind'] ∈ {'valid','follow-up','invalid'}
    Commits staged_model_msg to committed_history only if valid or follow-up.
    """
    msg = state.get("staged_model_msg")
    if not msg or not isinstance(msg, dict) or not msg.get("content"):
        state["output_kind"] = "invalid"
        state["violations"].append("Empty or missing model message.")
        _commit_model_if_valid(state, ok=False)
        return state

    # Prepare compact history (last N messages to keep prompt small)
    history = state.get("committed_history", [])[-8:]

    system_prompt = (
        "You are an output validator for a civil-planning chatbot. "
        "Decide if the ASSISTANT message is appropriate, helpful, and relevant. "
        "Return strict JSON with fields: {\"output_kind\":\"valid|follow-up|invalid\",\"reasons\":[...]}.\n"
        "- valid: message is helpful, relevant, and appropriate for a civil planning assistant.\n"
        "- follow-up: message asks clarifying questions to better understand requirements.\n"
        "- invalid: message is off-topic, inappropriate, or fails to address civil planning needs."
    )

    user_payload = {
        "history": history,           # prior conversation
        "candidate_model_msg": msg,   # untrusted model message to judge
    }

    try:
        resp = CLIENT.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": json.dumps(user_payload, ensure_ascii=False),
                },
            ],
            temperature=0,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        print("model validator response:", data)
        output_kind = data.get("output_kind", "invalid")
        reasons = data.get("reasons", [])

        if output_kind not in ALLOWED_OUTPUT_KINDS:
            output_kind = "invalid"

        state["output_kind"] = output_kind
        if output_kind == "invalid":
            state["violations"].append(
                f"Model output rejected by validator: {reasons if reasons else 'No reasons provided.'}"
            )

        _commit_model_if_valid(state, ok=(output_kind in ("valid", "follow-up")))
        return state

    except Exception as e:
        state["output_kind"] = "invalid"
        state["violations"].append(f"Model validator error: {e}")
        _commit_model_if_valid(state, ok=False)
        return state
