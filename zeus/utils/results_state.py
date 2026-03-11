from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json
import os
import numpy as np
@dataclass
class ResultsState:
    name: str
    rank_history: Dict[str, List[float]] = field(default_factory=dict)
    best_10_miners: List[str] = field(default_factory=list)

    def prune(self, current_hotkeys: List[str], max_len: int):
        """Prunes hotkeys not in metagraph and limits history length."""
        pruned_hotkeys = []
        for hotkey in list(self.rank_history.keys()):
            if hotkey not in current_hotkeys:
                self.rank_history.pop(hotkey, None)
                if hotkey in self.best_10_miners:
                    self.best_10_miners.remove(hotkey)
                pruned_hotkeys.append(hotkey)
                continue
            if len(self.rank_history[hotkey]) > max_len:
                self.rank_history[hotkey] = self.rank_history[hotkey][-max_len:]
        
        return pruned_hotkeys
    
    # update rank history
    def insert_rank_history(self, rewards: np.ndarray, hotkeys_list: List[str]):
        for rank, hotkey in zip(rewards.tolist(), hotkeys_list):
            self.rank_history.setdefault(str(hotkey), []).append(float(rank))

    def to_dict(self):
        """Allows for easy serialization and extension with new fields."""
        return {
            "rank_history": self.rank_history,
            "best_10_miners": self.best_10_miners,
        }


def save_state(
    state_path: str,
    variables: Dict[str, ResultsState],
    step: Optional[int] = None,
):
    state_data: Dict = {
        "variables": {name: var.to_dict() for name, var in variables.items()}
    }
    if step is not None:
        state_data["step"] = int(step)
    with open(state_path, "w") as f:
        json.dump(state_data, f, indent=4)


def load_state(
    state_path: str,
) -> Tuple[Dict[str, ResultsState], Optional[int]]:
    """
    Load variable state (and optional step/hotkeys) from disk.
    Returns ({}, None, None) if path does not exist or read/parse fails.
    """
    if not os.path.exists(state_path):
        return ({}, None)

    try:
        with open(state_path, "r") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        import logging
        logging.getLogger(__name__).warning(
            "Could not load state from %s: %s. Using empty state.", state_path, e
        )
        return ({}, None)

    variables = {}
    for name, content in data.get("variables", {}).items():
        variables[name] = ResultsState(
            name=name,
            rank_history=content.get("rank_history", {}),
            best_10_miners=content.get("best_10_miners", []),
        )

    step = data.get("step")
    if step is not None:
        step = int(step)
    return (variables, step)

