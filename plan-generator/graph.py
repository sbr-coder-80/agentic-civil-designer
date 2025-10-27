# graph.py
import os
from langgraph.graph import StateGraph, END
from state import State
from validators import validate_user_fn, validate_model_fn
from planner import plan_or_clarify_router_fn
from config import config


# --- Node fns (minimal stubs) ---

def intro_fn(state: State) -> State:
    intro_msg = {
        "role": "assistant",
        "content": "👷‍♂️ Hello! I’m your Agentic Civil Planner. Share your plot or layout idea, and I’ll help design a floor plan."
    }
    state["committed_history"].append(intro_msg)
    print(f"[Civil Planner Agent]: {intro_msg['content']}")
    return state

def intake_fn(state: State) -> State:
    """
    Intake node:
    - Reads user input (from console, API, or passed state).
    - Places it into staged_user_msg for validation.
    - Does not commit it yet.
    """
    # If user input already seeded in state (e.g., from main.py), just reuse it
    if state.get("staged_user_msg"):
        user_msg = state["staged_user_msg"]
    else:
        # Prompt the user from console (basic v1 example)
        raw_text = input("👤 You: ")
        user_msg = {"role": "user", "content": raw_text}
        state["staged_user_msg"] = user_msg

    # Print what we staged
    print(f"[Intake] Staged user message: {user_msg['content']}")
    return state

def reject_user_fn(state: State) -> State:
    """
    Reject invalid user input:
    - Creates a rejection message explaining why the input was invalid
    - Adds the message to committed_history
    - Clears any staged messages
    """
    violations = state.get("violations", [])
    
    # Build rejection message
    rejection_content = (
        "❌ I couldn't process that input. "
        "Please provide information about site layouts, floor plans, room programs, "
        "plot dimensions, setbacks, orientation, circulation, adjacency, or elevations.\n"
    )
    
    if violations:
        rejection_content += f"\nReason(s): {'; '.join(violations)}\n"
    
    rejection_content += (
        "\nExamples of valid inputs:\n"
        "- 'I need a 2-bedroom apartment layout for a 1000 sq ft plot'\n"
        "- 'Design a floor plan with kitchen adjacent to dining area'\n"
        "- 'Show me options for a residential building with 6 units'"
    )
    
    rejection_msg = {
        "role": "assistant",
        "content": rejection_content
    }
    
    print(f"[Civil Planner Agent]: {rejection_content}")
    
    # Clear any staged messages
    state["staged_user_msg"] = None
    state["staged_model_msg"] = None
    
    return state

def ask_clarifying_fn(state: State) -> State:
    # expects to set output_kind; default to "follow-up"
    state["output_kind"] = state.get("output_kind", "follow-up")
    return state

def generate_draft_fn(state: State) -> State:
    # expects to set output_kind; default to "valid"
    state["output_kind"] = state.get("output_kind", "valid")
    return state

def repair_or_refuse_fn(state: State) -> State:
    """
    Repair or refuse invalid model output.
    Can attempt to repair the output or refuse gracefully.
    """
    violations = state.get("violations", [])
    
    if state.get("output_kind") == "invalid":
        repair_content = (
            "⚠️ I'm having trouble generating a proper response. "
            "Please provide more specific requirements for your civil planning needs.\n"
        )
        
        if violations:
            repair_content += f"\nIssues: {'; '.join(violations[-3:])}\n"
        
        repair_msg = {
            "role": "assistant",
            "content": repair_content
        }
        
        # Add the repair message and clear staged
        state["committed_history"].append(repair_msg)
        print(f"[Civil Planner Agent]: {repair_content}")
        
        # Clear staged model msg
        state["staged_model_msg"] = None
    
    return state

# --- Graph ---

g = StateGraph(State)

g.add_node("Intro", intro_fn)
g.add_node("Intake", intake_fn)
g.add_node("ValidateUser", validate_user_fn)
g.add_node("RejectUser", reject_user_fn)
g.add_node("PlanOrClarifyRouter", plan_or_clarify_router_fn)
g.add_node("AskClarifying", ask_clarifying_fn)
g.add_node("GenerateDraft", generate_draft_fn)
g.add_node("ValidateModel", validate_model_fn)
g.add_node("RepairOrRefuse", repair_or_refuse_fn)

g.set_entry_point("Intro")
g.add_edge("Intro", "Intake")
g.add_edge("Intake", "ValidateUser")

g.add_conditional_edges(
    "ValidateUser",
    lambda s: s["input_kind"],
    {
        "valid": "PlanOrClarifyRouter",
        "follow-up": "PlanOrClarifyRouter",
        "invalid": "RejectUser",
    },
)

g.add_conditional_edges(
    "PlanOrClarifyRouter",
    lambda s: "AskClarifying" if s["input_kind"] == "follow-up" else "GenerateDraft",
    {
        "AskClarifying": "AskClarifying",
        "GenerateDraft": "GenerateDraft",
    },
)

g.add_edge("AskClarifying", "ValidateModel")
g.add_edge("GenerateDraft", "ValidateModel")

g.add_conditional_edges(
    "ValidateModel",
    lambda s: s["output_kind"],
    {
        "valid": END,
        "follow-up": END,
        "invalid": "RepairOrRefuse",
    },
)

g.add_conditional_edges(
    "RepairOrRefuse",
    lambda s: s["output_kind"],
    {
        "valid": END,
        "follow-up": END,
        "invalid": END,
    },
)
app = g.compile()