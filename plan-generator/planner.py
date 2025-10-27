
import json
from typing import Literal
from openai import OpenAI
from state import State
from config import config

# Initialize OpenAI once per process
CLIENT = OpenAI(api_key=config.openai_api_key)
MODEL = "gpt-4o-mini"

def plan_or_clarify_router_fn(state: State) -> State:
    """
    Router that generates an LLM response based on the conversation history.
    Passes committed history including the last user question to OpenAI,
    generates a response, prints it, and adds it to staged_model_msg for validation.
    """
    # Get the conversation history
    history = state.get("committed_history", [])
    
    if not history:
        print("[PlanOrClarifyRouter] No conversation history available.")
        return state
    
    # Prepare system message for the civil planner
    system_message = (
        "You are an expert civil planning assistant. Your role is to help users "
        "design floor plans, site layouts, and architectural solutions. Based on "
        "the conversation history, provide helpful guidance, ask clarifying questions, "
        "or generate design suggestions. Be concise, professional, and focused on "
        "civil planning and architecture topics."
    )
    
    try:
        # Call OpenAI with the conversation history
        messages = [{"role": "system", "content": system_message}]
        
        # Add conversation history
        for msg in history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        
        print("[PlanOrClarifyRouter] Calling OpenAI to generate response...")
        
        response = CLIENT.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.7,
        )
        
        # Extract the response content
        response_content = response.choices[0].message.content
        
        if response_content:
            # Create the assistant message
            assistant_msg = {
                "role": "assistant",
                "content": response_content
            }
            
            # Add to staged_model_msg (will be validated later)
            state["staged_model_msg"] = assistant_msg
            
            # Print the response
            print(f"[Civil Planner Agent]: {response_content}")
        else:
            print("[PlanOrClarifyRouter] Empty response from OpenAI")
            
    except Exception as e:
        print(f"[PlanOrClarifyRouter] Error generating response: {e}")
        error_msg = {
            "role": "assistant",
            "content": "Sorry, I encountered an error processing your request. Please try again."
        }
        state["staged_model_msg"] = error_msg
        state["violations"].append(f"PlanOrClarifyRouter error: {e}")
    
    return state