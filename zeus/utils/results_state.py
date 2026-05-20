from typing import Dict, List, Optional, Tuple
import json
import os
import sqlite3
import time
import pandas as pd
import numpy as np
import bittensor as bt
from zeus.validator.constants import RANK_HISTORY_DATABASE_LOCATION, RANK_HISTORY_PRUNE_DAYS, RANK_HISTORY_ALLOWED_ABSENCE

def get_db_connection():
    conn = sqlite3.connect(str(RANK_HISTORY_DATABASE_LOCATION), timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL;") # read and write can happen concurrently
    conn.execute("PRAGMA busy_timeout=30000;") # if 2 writes or writes and deletes the same time then it will wait for 30 seconds until we get an error
    conn.row_factory = sqlite3.Row
    return conn

def init_result_state_db():
    os.makedirs(RANK_HISTORY_DATABASE_LOCATION.parent, exist_ok=True)
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS challenge_ranks (
                state_key TEXT,
                miner_hotkey TEXT,
                challenge_enddate REAL,
                rank REAL,
                is_participated BOOLEAN,
                PRIMARY KEY (state_key, miner_hotkey, challenge_enddate)
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_history 
            ON challenge_ranks(state_key, miner_hotkey, challenge_enddate DESC)
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS best_10_miners (
                state_key TEXT,
                miner_hotkey TEXT,
                position INTEGER DEFAULT 0,
                updated_at REAL DEFAULT 0,
                PRIMARY KEY (state_key, miner_hotkey)
            )
        """)
        conn.commit()
    finally:
        conn.close()

class ResultsState:
    def __init__(self, name: str):
        self.name = name

    @property
    def best_10_miners(self) -> List[str]:
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT miner_hotkey
                FROM best_10_miners
                WHERE state_key = ?
                ORDER BY position ASC, miner_hotkey ASC
            """, (self.name,))
            miners = [row['miner_hotkey'] for row in cursor.fetchall()]
            bt.logging.debug(f"Found {len(miners)} best miners from ResultsState {self.name}")
            return miners
        finally:
            conn.close()

    @best_10_miners.setter
    def best_10_miners(self, hotkeys: List[str]):
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            updated_at = time.time()
            cursor.execute("DELETE FROM best_10_miners WHERE state_key = ?", (self.name,))
            if hotkeys:
                cursor.executemany(
                    """
                    INSERT INTO best_10_miners (state_key, miner_hotkey, position, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    [(self.name, h, position, updated_at) for position, h in enumerate(hotkeys)]
                )
            conn.commit()
            bt.logging.debug(f"ResultsState best_10_miners for {self.name} has been updated")
        finally:
            conn.close()

    @property
    def rank_history(self) -> Dict[str, List[float]]:
        conn = get_db_connection()
        cutoff_ts = (
            pd.Timestamp.now("UTC").floor("6h") - pd.Timedelta(hours=6)
        ).timestamp()
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT miner_hotkey, rank 
                FROM challenge_ranks 
                WHERE state_key = ? AND challenge_enddate <= ?
                ORDER BY challenge_enddate ASC
            """, (self.name, cutoff_ts))
            
            history = {}
            for row in cursor.fetchall():
                history.setdefault(row['miner_hotkey'], []).append(row['rank'])
            bt.logging.debug(f"Returning history for {len(history)} miners from ResultsState {self.name}")
            return history
        finally:
            conn.close()

    # update rank history
    def insert_rank_history(self, rewards: np.ndarray, hotkeys_list: List[str], miner_penalty_bool_list: List[bool], challenge_enddate: float):
        if len(rewards) == 0 or len(hotkeys_list) == 0:
            bt.logging.warning(f"[ResultsState] insert_rank_history called with empty rewards/hotkeys for {self.name}; skipping.")
            return

        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            
            # Get all hotkeys that have ever been scored for this state_key
            cursor.execute("SELECT DISTINCT miner_hotkey FROM challenge_ranks WHERE state_key = ?", (self.name,))
            all_hotkeys = {row['miner_hotkey'] for row in cursor.fetchall()}
            
            # Add current participating hotkeys to the set of all hotkeys
            participating_set = set(hotkeys_list)
            all_hotkeys.update(participating_set)
            
            penalty_rank = float(len(all_hotkeys))
            
            records_to_insert = []
            
            max_rank_in_rewards = max(rewards)
            # Add participating miners
            for rank, hotkey, penalty_bool in zip(rewards.tolist(), hotkeys_list, miner_penalty_bool_list):
                if rank == max_rank_in_rewards and penalty_bool:
                    rank = penalty_rank
                records_to_insert.append((self.name, str(hotkey), challenge_enddate, float(rank), True))
                
            # Add non-participating miners (the penalty)
            for hotkey in all_hotkeys:
                if hotkey not in participating_set:
                    records_to_insert.append((self.name, str(hotkey), challenge_enddate, penalty_rank, False))
                    
            cursor.executemany("""
                INSERT OR REPLACE INTO challenge_ranks (state_key, miner_hotkey, challenge_enddate, rank, is_participated)
                VALUES (?, ?, ?, ?, ?)
            """, records_to_insert)
            
            # N-Strike History Removal
            # Find hotkeys that have is_participated = False for the last RANK_HISTORY_ALLOWED_ABSENCE consecutive challenges
            for hotkey in all_hotkeys:
                cursor.execute("""
                    SELECT is_participated FROM challenge_ranks 
                    WHERE state_key = ? AND miner_hotkey = ?
                    ORDER BY challenge_enddate DESC
                    LIMIT ?
                """, (self.name, hotkey, RANK_HISTORY_ALLOWED_ABSENCE))
                recent_participations = [row['is_participated'] for row in cursor.fetchall()]
                
                if len(recent_participations) == RANK_HISTORY_ALLOWED_ABSENCE and not any(recent_participations):
                    # RANK_HISTORY_ALLOWED_ABSENCE consecutive misses, wipe history
                    cursor.execute("DELETE FROM challenge_ranks WHERE state_key = ? AND miner_hotkey = ?", (self.name, hotkey))
            
            bt.logging.debug(f"Insert rank history for ResultsState {self.name} successful!")
            conn.commit()
        finally:
            conn.close()

def prune_rank_database(current_hotkeys: List[str]) -> List[str]:
    """Prunes hotkeys not in metagraph and limits history length based on RANK_HISTORY_PRUNE_DAYS."""
    conn = get_db_connection()
    try:
        cursor = conn.cursor()

        # Collect hotkeys from BOTH tables so N-strike-deleted entries that only
        # remain in best_10_miners are also cleaned up.
        cursor.execute("""
            SELECT DISTINCT miner_hotkey FROM challenge_ranks
            UNION
            SELECT DISTINCT miner_hotkey FROM best_10_miners
        """)
        db_hotkeys = [row['miner_hotkey'] for row in cursor.fetchall()]

        current_hotkeys_set = set(current_hotkeys)
        to_delete = [hotkey for hotkey in db_hotkeys if hotkey not in current_hotkeys_set]
        if to_delete:
            sql_to_delete_params = [(hotkey,) for hotkey in to_delete]
            cursor.executemany("DELETE FROM challenge_ranks WHERE miner_hotkey = ?", sql_to_delete_params)
            cursor.executemany("DELETE FROM best_10_miners WHERE miner_hotkey = ?", sql_to_delete_params)
                
        # days * 24 hours * 60 mins * 60 secs = total seconds
        time_in_seconds = RANK_HISTORY_PRUNE_DAYS * 24 * 60 * 60
        cutoff_date = pd.Timestamp.now('UTC').timestamp() - time_in_seconds
        cursor.execute("DELETE FROM challenge_ranks WHERE challenge_enddate < ?", (cutoff_date, ))
        
        conn.commit()
        bt.logging.debug(f"Pruned {len(to_delete)} hotkeys from all ResultsStates")
        return to_delete
    finally:
        conn.close()


def load_rank_history_snapshot(state_keys: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """Load rank history for several state keys in one read-only snapshot."""
    if not state_keys:
        return {}

    cutoff_ts = (
        pd.Timestamp.now("UTC").floor("6h") - pd.Timedelta(hours=6)
    ).timestamp()
    placeholders = ",".join("?" for _ in state_keys)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            f"""
            SELECT state_key, miner_hotkey, rank
            FROM challenge_ranks
            WHERE state_key IN ({placeholders})
              AND challenge_enddate <= ?
            ORDER BY state_key ASC, miner_hotkey ASC, challenge_enddate ASC
            """,
            [*state_keys, cutoff_ts],
        )

        snapshot: Dict[str, Dict[str, List[float]]] = {}
        for row in cursor.fetchall():
            state_history = snapshot.setdefault(row["state_key"], {})
            state_history.setdefault(row["miner_hotkey"], []).append(row["rank"])
        return snapshot
    finally:
        conn.close()
    
def migrate_state_to_db(old_state_path: str):
    """Migrate all state from JSON structure to SQLite database."""
    if os.path.exists(RANK_HISTORY_DATABASE_LOCATION):
        # Check if already populated
        conn = get_db_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM challenge_ranks LIMIT 1")
            if cursor.fetchone():
                bt.logging.info(f"Database already populated at {RANK_HISTORY_DATABASE_LOCATION}. Skipping migration.")
                return
        finally:
            conn.close()  # always closes exactly once
        
    if not os.path.exists(old_state_path):
        bt.logging.info(f"No old state file found at {old_state_path}. Skipping migration.")
        return

    bt.logging.info(f"Migrating state from {old_state_path} to SQLite...")
        
    try:
        with open(old_state_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        bt.logging.error(f"Failed to read {old_state_path}: {e}")
        return
        
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
                
        for name, content in data.get("variables", {}).items():
            best_10 = content.get("best_10_miners", [])
            if best_10:
                updated_at = time.time()
                cursor.executemany(
                    """
                    INSERT OR REPLACE INTO best_10_miners
                        (state_key, miner_hotkey, position, updated_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    [(name, h, position, updated_at) for position, h in enumerate(best_10)]
                )
                
            old_rank_history = content.get("rank_history", {})
            if old_rank_history:
                # Keep the timezone-aware Timestamp so .timestamp() always returns UTC seconds.
                base_time = pd.Timestamp.now('UTC').floor('6h')
                if "@0_48" in name:
                    base_time -= pd.Timedelta(days=7)
                else:
                    base_time -= pd.Timedelta(days=20)
                    
                records_to_insert = []
                for hotkey, ranks in old_rank_history.items():
                    # Backfill recursively backwards
                    current_time = base_time
                    for rank in reversed(ranks):
                        records_to_insert.append((name, hotkey, current_time.timestamp(), float(rank), True))
                        current_time -= pd.Timedelta(hours=6)
                        
                cursor.executemany("""
                    INSERT OR REPLACE INTO challenge_ranks (state_key, miner_hotkey, challenge_enddate, rank, is_participated)
                    VALUES (?, ?, ?, ?, ?)
                """, records_to_insert)
                
        conn.commit()
        bt.logging.info("Migration to SQLite complete.")
    finally:
        conn.close()
    
def load_state() -> Dict[str, ResultsState]:
    """
    Load variable states from disk.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        
        # Get all unique state_keys
        cursor.execute("""
            SELECT DISTINCT state_key FROM challenge_ranks
            UNION
            SELECT DISTINCT state_key FROM best_10_miners
        """)
        state_keys = [row['state_key'] for row in cursor.fetchall()]
        
        variables = {key: ResultsState(name=key) for key in state_keys}
        
        return variables
    finally:
        conn.close()
   