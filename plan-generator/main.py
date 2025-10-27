from graph import app
from state import State
from config import config

# Validate configuration at startup
config.validate()

def main():
    # Initial state (seed the conversation)
    init_state: State = {
        "committed_history": [],
        "staged_user_msg": None,
        "staged_model_msg": None,
        "input_kind": "valid",       # seed for routing (later your validator sets this)
        "output_kind": "valid",      # seed for routing (later validator sets this)
        "violations": [],
    }

    # Run the compiled graph once
    final_state = app.invoke(init_state)

    # Print results
    print("\n=== Final State ===")
    for msg in final_state["committed_history"]:
        print(f"{msg['role']}: {msg['content']}")
    print("Violations:", final_state["violations"])

if __name__ == "__main__":
    main()
