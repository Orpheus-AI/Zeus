import pathlib
import sqlite3
import time
from typing import Callable, List, Set, Tuple

import bittensor as bt
import pandas as pd
import torch

from zeus.data.loaders.era5_cds import Era5CDSLoader
from zeus.data.sample import Era5Sample
from zeus.validator.challenge_spec import offsets_from_predict_hours
from zeus.validator.constants import METADATA_DATABASE_LOCATION, TIME_WINDOWS_PER_CHALLENGE
from zeus.validator.miner_data import MinerData
from zeus.utils.time import to_timestamp
from zeus.utils.compression import decompress_prediction
import os
import numpy as np
import xarray as xr

def save_best_miner_prediction(self, sample : Era5Sample, miner : MinerData, is_random: bool, best_10_hotkeys: list[str]):
    # this is for the proxy to save the best miner prediction after the hash phase is done!

    bt.logging.info(f"[save_best_miner_prediction] Begining the process for miner uid {miner.uid}, which is random = {is_random}")
    start_time = sample.start_timestamp
    start_time_timestamp = to_timestamp(start_time)
    start_time_str = start_time_timestamp.strftime("%Y%m%d%H")
    end_time = sample.end_timestamp
    end_time_timespamp =  to_timestamp(end_time)
    end_time_str = end_time_timespamp.strftime("%Y%m%d%H")

    prediction = decompress_prediction(miner.prediction)
    
    # Create subdirectory for the variable if it doesn't exist
    if not os.path.exists(self.best_predictions_path):
        os.makedirs(self.best_predictions_path, exist_ok=True)
    
    variable_name = sample.variable.split("@")[0]
    variable_folder = os.path.join(self.best_predictions_path, variable_name)
    os.makedirs(variable_folder, exist_ok=True)

    if prediction is None:
        bt.logging.warning(f"[save_best_miner_prediction] Prediction is None for miner {miner.uid} {miner.hotkey}")
        return
    
    # Convert torch tensor to numpy array and save
    prediction_numpy = prediction.detach().cpu().to(torch.float32).numpy()
    time_coords = pd.date_range(start_time_timestamp, end_time_timespamp, freq=f"{sample.step_size}h")

    xr_dataarray = xr.DataArray(
        data = prediction_numpy, 
        dims = ["time", "latitude", "longitude"],
        coords = dict(
            time = time_coords,
            latitude = np.arange(sample.lat_start, sample.lat_end+0.25, 0.25), 
            longitude = np.arange(sample.lon_start, sample.lon_end+0.25, 0.25)
        ))
    
    rank = f"{best_10_hotkeys.index(miner.hotkey)+1}" if miner.hotkey in best_10_hotkeys else "unknown"
    randomness_tag = "random" if is_random else f"rank_{rank}" 
    filename = f"{start_time_str}-{end_time_str}-S{sample.step_size}_miner_{miner.hotkey}_{randomness_tag}.nc"
    filepath = os.path.join(variable_folder, filename)
    # Convert DataArray to Dataset with the variable name set to the sample.variable
    xr_dataset = xr.Dataset({sample.variable: xr_dataarray})
    xr_dataset.to_netcdf(filepath)
    bt.logging.success(f"[save_best_miner_prediction] Saved best miner prediction to {filepath}")

class OptimizedWeatherStorage:
    def __init__(
        self,
        cds_loader: Era5CDSLoader,
        db_path: pathlib.Path = METADATA_DATABASE_LOCATION,
    ):
        self.cds_loader = cds_loader
        self.db_path = db_path
        self.last_synced_block = 0

        bt.logging.warning(f"Initializing OptimizedWeatherStorage: db_path={db_path}")

        # Initialize SQLite for Metadata
        self._create_tables()
        bt.logging.debug(f"SQLite metadata tables created/verified at {db_path}")

    def should_score(self) -> bool:
        """
        Check if the database should score its stored miner predictions.
        """
        if not self.cds_loader.is_ready():
            bt.logging.debug("CDS loader not ready")
            return False
        now = pd.Timestamp.now('UTC')
        if now.hour%6 != 0 and now.hour%6 != 5 and now.hour%6 != 4:
            bt.logging.info("It is okay time to try to score")
            return True
        return False

    def _create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS challenges (
                    uid INTEGER PRIMARY KEY AUTOINCREMENT,
                    lat_start REAL, 
                    lat_end REAL, 
                    lon_start REAL, 
                    lon_end REAL,
                    start_timestamp REAL, 
                    end_timestamp REAL,
                    hours_to_predict INTEGER, 
                    inserted_at REAL,
                    variable TEXT DEFAULT '2m_temperature'
                );
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS challenge_hotkey_map (
                    miner_hotkey TEXT,
                    miner_uid TEXT,
                    challenge_uid INTEGER,
                    good_miner BOOLEAN DEFAULT 1,
                    PRIMARY KEY (challenge_uid, miner_hotkey),
                    FOREIGN KEY (challenge_uid) REFERENCES challenges (uid)
                );
            """)
            # miner responses, the hashes they returned
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS hash_responses (
                    miner_hotkey TEXT,
                    miner_uid TEXT,
                    challenge_uid INTEGER,
                    hash TEXT,
                    FOREIGN KEY (challenge_uid) REFERENCES challenges (uid)
                );
                """
            )
            conn.commit()
            bt.logging.debug("Created/verified SQLite tables: challenges, challenge_hotkey_map, hash_responses")

    def insert(self,sample: Era5Sample, miners_data: List[MinerData], good_miners = True) -> bool:
        """
        Insert a challenge and responses into the database.
        If a challenge already exists find the uid and insert the responses
        If a challenge doesn't exist, insert the challenge and the responses
        
        Return :
            boolean : whether the insertion was successful
        """

        challenge_uid = self._find_challenge_id(sample)
        if challenge_uid is not None:
            bt.logging.debug(f"insert: found existing challenge_uid={challenge_uid}")
        else:
            bt.logging.debug("insert: no existing challenge, inserting new challenge")
            challenge_uid = self._insert_challenge(sample)

        if challenge_uid is None:
            bt.logging.error("insert: failed to obtain challenge_uid (insert_challenge returned None)")
            return False
        success = self._insert_hash_responses(challenge_uid, miners_data, good_miners)

        if success:
            bt.logging.success(f"insert: success for challenge_uid={challenge_uid}, {len(miners_data)} miner responses")
        else:
            bt.logging.warning(f"insert: _insert_hash_responses failed for challenge_uid={challenge_uid}")
        return success
     
    def get_hashing_data_for_sample(self, sample: Era5Sample) -> Tuple[List[str], List[str]]:
        """
        Given a sample, find the corresponding challenge and return 
        the hashes, hotkeys, and uids of miners who submitted a hash.
        """
        challenge_uid = self._find_challenge_id(sample)
        
        if challenge_uid is None:
            bt.logging.warning(f"get_hashing_data_for_sample: No challenge found for sample at {sample.start_timestamp}")
            return [], []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # Standard JOIN ensures we only get miners who actually have a hash record
            cursor.execute("""
                SELECT 
                    hash, 
                    miner_hotkey
                FROM hash_responses 
                WHERE challenge_uid = ?;
            """, (challenge_uid,))

            results = cursor.fetchall()
            
            if not results:
                return [], []

            hashes = [r[0] for r in results]
            hotkeys = [r[1] for r in results]

            return hashes, hotkeys

        except Exception as e:
            bt.logging.error(f"get_hashing_data_for_sample: Failed to retrieve data — {e}")
            return [], []
        finally:
            conn.close()

    def _find_challenge_id(self, sample: Era5Sample):
        """
        Check if a challenge matching the sample already exists in the database.
        Returns the challenge_uid if found, None otherwise.
        """
        params = (
            *sample.get_bbox(),
            sample.start_timestamp,
            sample.end_timestamp,
            sample.predict_hours,
            sample.query_timestamp,
            sample.variable,
        )
        bt.logging.debug(
            f"_find_challenge_id: querying challenges with bbox={sample.get_bbox()}, start={to_timestamp(sample.start_timestamp)}, end={to_timestamp(sample.end_timestamp)}, hours={sample.predict_hours}, variable={sample.variable}"
        )
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT uid FROM challenges 
                WHERE lat_start = ? AND lat_end = ? AND lon_start = ? AND lon_end = ?
                AND start_timestamp = ? AND end_timestamp = ?
                AND hours_to_predict = ? AND inserted_at = ? AND variable = ?;
                """,
                params,
            )
            result = cursor.fetchone()
            if result:
                bt.logging.debug(f"_find_challenge_id: found challenge_uid={result[0]}")
                return result[0]
            bt.logging.debug("_find_challenge_id: no matching challenge found")
            return None
        finally:
            conn.close()
    
    def _insert_challenge(self, sample: Era5Sample) -> int:
        """
        Insert a sample into the database and return the challenge UID.
        """
        bt.logging.debug(
            f"_insert_challenge: inserting challenge bbox={sample.get_bbox()}, start={to_timestamp(sample.start_timestamp)}, end={to_timestamp(sample.end_timestamp)}, hours={sample.predict_hours}"
        )
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO challenges (lat_start, lat_end, lon_start, lon_end, 
                start_timestamp, end_timestamp, hours_to_predict, inserted_at, variable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
                (
                    *sample.get_bbox(),
                    sample.start_timestamp,
                    sample.end_timestamp,
                    sample.predict_hours,
                    sample.query_timestamp,
                    sample.variable,
                ),
            )
            challenge_uid = cursor.lastrowid
            conn.commit()
            bt.logging.warning(f"_insert_challenge: created challenge_uid={challenge_uid}")
            return challenge_uid

        except Exception as e:
            conn.rollback()
            bt.logging.exception(f"_insert_challenge: failed — {e}")
            raise e
        finally:
            conn.close()

    def _insert_hash_responses(self, challenge_uid: int, miners_data: List[MinerData], good_miners = True):
        """
        Insert the responses from the miners into the database.
        If a challenge_uid is already in the responses table,
        append the row with the new hotkeys

        Returns boolean whether the insertion was successful
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            data_to_insert = []
            hotkeys = []
            miner_uids = []
            # prepare data for insertion
            for miner in miners_data:
                hotkeys.append(miner.hotkey)
                miner_uids.append(miner.uid)
                if good_miners:
                    data_to_insert.append((miner.hotkey, miner.uid, challenge_uid, miner.prediction_hash))

            if good_miners:
                cursor.executemany(
                    """
                    INSERT INTO hash_responses (miner_hotkey, miner_uid, challenge_uid, hash)
                    VALUES (?, ?, ?, ?);
                    """,
                    data_to_insert,
                )
                conn.commit()

            return self._insert_hotkeys(challenge_uid, hotkeys, miner_uids, good_miners)
        except Exception as e:
            conn.rollback()
            bt.logging.exception(f"_insert_challenge: failed — {e}")
            raise e
        finally:
            conn.close()
                
    def _insert_hotkeys(self, challenge_uid: int, hotkeys : List[str], miner_uids: List[int], good_miners = True) -> bool:
        """
        If a challenge_uid is already in the responses table,
        append the row with the new hotkeys

        Returns boolean whether the insertion was successful
        """
        conn = sqlite3.connect(self.db_path)
        try:
            bt.logging.debug(f"[_insert_hotkeys] For challenge {challenge_uid} inserting infor for {miner_uids}")
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO challenge_hotkey_map (miner_hotkey, miner_uid, challenge_uid, good_miner)
                VALUES (?, ?, ?, ?);
                """,
                zip(hotkeys, miner_uids, [challenge_uid]*len(hotkeys), [good_miners]*len(hotkeys)),
            )
            conn.commit()
            return True
        except Exception as e:
            conn.rollback()
            bt.logging.exception(f"_insert_hotkeys: failed for challenge_uid={challenge_uid} — {e}")
            return False
        finally:
            conn.close()

    def mark_miners_as_bad(self, sample: Era5Sample, hotkeys : List[str]) -> bool:
        """
        Update the miiner status from good to bad for hotkeys and challenge as given in the parameters
        Return bool whether the insertion was successful
        """

        challenge_uid = self._find_challenge_id(sample)
        if not challenge_uid:
            bt.logging.error("_punish_miner: no challenge_uid was found for this sample")
            return False
        
        if not hotkeys:
            bt.logging.debug("_punish_miner: no hotkeys provided, nothing to update")
            return True
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            # In the table challenge_hotkey_map update all the miners with hotkey in hotkeys for challenge_uid good_miner = 0
            placeholders = ','.join('?' for _ in hotkeys)
            cursor.execute(
                f"""
                UPDATE challenge_hotkey_map 
                SET good_miner = 0 
                WHERE challenge_uid = ? AND miner_hotkey IN ({placeholders});
                """,
                [challenge_uid] + hotkeys
            )

            # Remove rows from hash_responses for the punished miners
            cursor.execute(
                f"""
                DELETE FROM hash_responses 
                WHERE challenge_uid = ? AND miner_hotkey IN ({placeholders});
                """,
                [challenge_uid] + hotkeys
            )
            
            rows_updated = cursor.rowcount
            conn.commit()
            bt.logging.warning(f"_punish_miner: updated {rows_updated} miners to good_miner=0 for challenge_uid={challenge_uid}")
            return True
        except Exception as e:
            conn.rollback()
            bt.logging.exception("_punish_miner: There was a failure updating the miners from good to bad")
            return False
        finally:
            conn.close()

    def _get_hotkeys_and_uids_for_challenge(self, challenge_uid: int, good_miners = True) -> Tuple[List[str], List[str]]:
        """
        Return all the hotkeys for a given challenge for good/bad miners
        """
        bt.logging.debug(f"_get_hotkeys_and_uids_for_challenge: challenge_uid={challenge_uid}, good_miners={good_miners}")
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT miner_hotkey, miner_uid FROM challenge_hotkey_map 
                WHERE challenge_uid = ? AND good_miner = ?;
            """, (challenge_uid, good_miners))

            result = cursor.fetchall()

            if result:
                hotkeys = [r[0] for r in result]
                uids = [int(r[1]) for r in result]
                bt.logging.debug(f"_get_hotkeys_and_uids_for_challenge: challenge_uid={challenge_uid} returned {len(hotkeys)} hotkeys")
                return hotkeys, uids
            bt.logging.debug(f"_get_hotkeys_and_uids_for_challenge: challenge_uid={challenge_uid} has no hotkeys")
            return [], []

        except Exception as e:
            conn.rollback()
            bt.logging.exception(f"_get_hotkeys_and_uids_for_challenge: failed for challenge_uid={challenge_uid} — {e}")
            return [], []
    
    def get_responding_miners_hotkeys(self) -> set[str]:
        """
        Get the hotkeys of all the miners that have responded to at least one challenge
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            
            # Get all hotkeys from rows where good_miners is True
            cursor.execute("""
                SELECT miner_hotkey FROM challenge_hotkey_map 
                WHERE good_miner = ?;
            """, (True,))

            results = cursor.fetchall()
            hotkeys_set = {r[0] for r in results}         
            return hotkeys_set
        
        except Exception as e:
            conn.rollback()
            return set()
        finally:
            conn.close()

    async def score_and_prune(self, score_func: Callable):
        latest_available = self.cds_loader.last_stored_timestamp.timestamp()
        bt.logging.warning(
            f"score_and_prune: starting, latest_available={to_timestamp(latest_available)}"
        )

        # Fetch and CLOSE the connection immediately
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM challenges WHERE end_timestamp <= ?;",
                (latest_available,),
            )
            challenges = cursor.fetchall()

        bt.logging.warning(f"score_and_prune: found {len(challenges)} challenges with end_timestamp <= {to_timestamp(latest_available)}")

        for chal in challenges:
            bt.logging.warning(f"score_and_prune: chal: {chal}")
            c_uid, lat_s, lat_e, lon_s, lon_e, start_t, end_t, hours, ins_at, var = chal

            try:
                start_offset, end_offset = offsets_from_predict_hours(hours, TIME_WINDOWS_PER_CHALLENGE)
            except ValueError:
                # TODO check currently in the competition if same amount of hours
                bt.logging.warning(
                    f"score_and_prune: challenge {c_uid} has hours_to_predict={hours} "
                    f"that doesn't match any current window, deleting"
                )
                self._delete_challenge(c_uid)
                continue

            sample = Era5Sample(start_t, end_t, lat_s, lat_e, lon_s, lon_e, var, ins_at, None, hours,
                                start_offset=start_offset, end_offset=end_offset)
            output = self.cds_loader.get_output(sample)
            sample.output_data = output
            #bt.logging.warning(f'score_and_prune: output: {output}')
            if (
                output is None
                or output.shape[0] != hours
                or not torch.isfinite(output).all()
            ):
                if output is not None:
                    bt.logging.warning(f'{output.shape[0]} != {hours}')
                if end_t < (latest_available - pd.Timedelta(days=3).total_seconds()):
                    bt.logging.warning(f"score_and_prune: challenge {c_uid} unscoreable (end_t={end_t}, latest={to_timestamp(latest_available)}), deleting")
                    if output is not None:
                        bt.logging.warning(f"score_and_prune: output shape {output.shape} vs desired hours {hours}")
                    else:
                        bt.logging.warning(f"score_and_prune: output is None for challenge {c_uid}")
                    self._delete_challenge(c_uid)
                else:
                    bt.logging.warning(f"score_and_prune: skipping challenge {c_uid} (unscoreable but end_t within 3 days) end_t: {to_timestamp(end_t)}, latest: {to_timestamp(latest_available)}")
                continue

            # First get the good miners from the SQL database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM hash_responses WHERE challenge_uid = ?;
                    """,
                    (c_uid,),
                )
                responses = cursor.fetchall()

                miner_hotkeys = [r[0] for r in responses]
                miner_uids = [int(r[1]) for r in responses]
                hashes = [r[3] for r in responses]


            # then get the bad miners with empty hashes
            current_challenge_bad_miner_hotkeys, current_challenge_bad_miner_uids = self._get_hotkeys_and_uids_for_challenge(c_uid, good_miners=False)

            current_challenge_all_miner_hotkeys = miner_hotkeys + current_challenge_bad_miner_hotkeys
            miner_uids += current_challenge_bad_miner_uids
            hashes += [None]*len(current_challenge_bad_miner_hotkeys)
            
            # the boolean with whether the miner is good or not (Not really needed because if the hash is empty then the miner is bad, but in case you still want it):
            is_good = [1]*len(miner_hotkeys) + [0]*len(current_challenge_bad_miner_hotkeys)
                
            await score_func(sample, current_challenge_all_miner_hotkeys, miner_uids, hashes, is_good)
            self._delete_challenge(c_uid)

            # don't score miners too quickly in succession and always wait after last scoring
            time.sleep(1)

        bt.logging.warning(f"score_and_prune: completed {len(challenges)} challenges")
        return len(challenges) > 0

    def _delete_challenge(self, challenge_uid: int):
        """Cleanup the SQLite"""
        bt.logging.warning(f"_delete_challenge: starting for challenge_uid={challenge_uid}")
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            cursor.execute("DELETE FROM challenges WHERE uid = ?;", (challenge_uid,))
            cursor.execute("DELETE FROM challenge_hotkey_map WHERE challenge_uid = ?;", (challenge_uid,))
            cursor.execute("DELETE FROM hash_responses WHERE challenge_uid = ?;", (challenge_uid,))
            
            conn.commit()
        except Exception as e:
            conn.rollback()
            bt.logging.exception(f"_delete_challenge: failed for challenge_uid={challenge_uid} — {e}")
        finally:
            conn.close()
 
    def prune_hotkeys(self, hotkeys: List[str]):
        """Remove hotkeys from challenge_hotkey_map and hash_responses for hotkeys that are no longer in the metagraph."""
        if not hotkeys:
            bt.logging.debug("prune_hotkeys: no hotkeys to prune, skipping")
            return
        bt.logging.warning(f"prune_hotkeys: pruning {len(hotkeys)} hotkeys no longer in metagraph")
        # hotkeys_set = set(hotkeys)
        challenge_uid_for_deletion = []

        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()

            
            cursor.execute(
                """
                DELETE FROM challenge_hotkey_map WHERE miner_hotkey IN ({});
                """.format(','.join('?' for _ in hotkeys)),
                hotkeys
            )

            cursor.execute(
                """
                DELETE FROM hash_responses WHERE miner_hotkey IN ({});
                """.format(','.join('?' for _ in hotkeys)),
                hotkeys
            )
            conn.commit()
           
        except Exception as e:
            conn.rollback()
            bt.logging.exception(f"prune_hotkeys: failed — {e}")
        finally:
            conn.close()
            for c in challenge_uid_for_deletion:
                self._delete_challenge(c) 
