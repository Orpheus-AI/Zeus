import json
import os
import sys
from typing import Dict

import bittensor as bt
from zeus.base.validator import BaseValidatorNeuron
from zeus.utils.results_state import ResultsState, save_state

def main():
    # Inject environment variables into sys.argv if they are missing
    # This allows the script to be run via PM2 without passing args explicitly
    if "--netuid" not in sys.argv and "NETUID" in os.environ:
        sys.argv.extend(["--netuid", os.environ["NETUID"]])
    if "--wallet.name" not in sys.argv and "WALLET_NAME" in os.environ:
        sys.argv.extend(["--wallet.name", os.environ["WALLET_NAME"]])
    if "--wallet.hotkey" not in sys.argv and "WALLET_HOTKEY" in os.environ:
        sys.argv.extend(["--wallet.hotkey", os.environ["WALLET_HOTKEY"]])

    # Parse arguments just like the validator does to get the correct paths
    config = BaseValidatorNeuron.config()
    BaseValidatorNeuron.check_config(config)
    
    full_path = config.neuron.full_path
    v2_path = os.path.join(full_path, "state_v2.json")
    v3_path = os.path.join(full_path, "state_v3.json")

    print(f"Looking for v2 state at: {v2_path}")
    
    if not os.path.exists(v2_path):
        print(f"Error: File {v2_path} does not exist.")
        return

    try:
        with open(v2_path, "r") as f:
            v2_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Could not parse {v2_path} as JSON: {e}")
        return

    step = v2_data.get("step")
    v2_variables = v2_data.get("variables", {})

    v3_variables: Dict[str, ResultsState] = {}

    for var_name, old_content in v2_variables.items():
        state_key = f"{var_name}@0_48"
        v3_variables[state_key] = ResultsState(
            name=state_key,
            rank_history=old_content.get("rank_history", {}).copy(),
            best_10_miners=old_content.get("best_10_miners", []).copy()
        )

    # Save the new state using the existing utility function
    save_state(v3_path, v3_variables, step)
    print(f"Successfully migrated {v2_path} to {v3_path}")

if __name__ == "__main__":
    main()
