# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List, Callable, Union, Optional
import sqlite3
import time
import torch
import pandas as pd
import numpy as np
import json
from collections import defaultdict

from zeus.data.base.loader import BaseDataLoader
from zeus.validator.constants import DATABASE_LOCATION
from zeus.data.base.sample import BaseSample
from zeus.validator.miner_data import MinerData
from zeus.validator.constants import MechanismType


class ResponseDatabase:

    MECHANISM_TO_DB_NAME = {
        MechanismType.ERA5: "challenges",
        MechanismType.WEATHER_XM: "challenges_weatherxm"
    }

    def __init__(
        self,
        db_path: str = DATABASE_LOCATION,
    ):
        self.db_path = db_path
        self.create_tables()
        # start at 0 so it always syncs at startup
        self.last_synced_block = defaultdict(int)

    def should_score(self, block: int, dataloader: BaseDataLoader) -> bool:
        """
        Check if the database should score its stored miner predictions.
        This is done roughly hourly, so with one block every 12 seconds this means
        if the current block is more than 300 blocks ahead of the last synced block, we should score.
        """
        if not dataloader.is_ready():
            return False
        if block - self.last_synced_block[dataloader.mechanism] > 300:
            self.last_synced_block[dataloader.mechanism] = block
            return True
        return False

    def create_tables(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS challenges (
                    uid INTEGER PRIMARY KEY AUTOINCREMENT,
                    lat_start REAL,
                    lat_end REAL,
                    lon_start REAL,
                    lon_end REAL,
                    start_timestamp REAL,
                    end_timestamp REAL,
                    hours_to_predict INTEGER,
                    baseline TEXT,
                    inserted_at REAL,
                    variable TEXT DEFAULT '2m_temperature',
                    ifs_hres_baseline TEXT
                );
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS challenges_weatherxm (
                    uid INTEGER PRIMARY KEY AUTOINCREMENT,
                    lat REAL,
                    lon REAL,
                    elevation REAL,
                    station_id TEXT,
                    start_timestamp REAL,
                    end_timestamp REAL,
                    hours_to_predict INTEGER,
                    baseline TEXT,
                    inserted_at REAL,
                    variable TEXT
                );
                """
            )

            # miner responses, we will use JSON for the tensor.
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    miner_hotkey TEXT,
                    challenge_uid INTEGER,
                    prediction TEXT,
                    response_time REAL DEFAULT 5.0,
                    mechanism INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY (challenge_uid) REFERENCES challenges (uid)
                );
                """
            )
            # migrate from v1.5.3 -> v2.0.0
            if not "mechanism" in get_column_names(cursor, "responses"):
                cursor.execute("ALTER TABLE responses ADD COLUMN mechanism INTEGER DEFAULT 0;")

            conn.commit()

    def insert(
        self,
        sample: BaseSample,
        miners_data: List[MinerData],
    ):
        """
        Insert a challenge and responses into the database.
        """
        challenge_uid = self._insert_challenge(sample=sample)
        self._insert_responses(challenge_uid, sample.mechanism, miners_data=miners_data)

    def _insert_challenge(self, sample: BaseSample) -> int:
        """
        Insert a challenge into the database.

        Returns:
            The challenge uid.
        """
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            db_name = self.MECHANISM_TO_DB_NAME[sample.mechanism]

            column_names = get_column_names(cursor, table_name=db_name)
            column_names.remove("uid") # don't set that one
            values = []
            # get values from sample based on column names matching sample's properties
            for column in column_names:
                value = getattr(sample, column)
                if type(value) in (torch.Tensor, np.ndarray):
                    value = serialize(value)
                values.append(value)
            
            placeholders = ", ".join(["?"] * len(values))
            cursor.execute(
                f"INSERT INTO {db_name} ({', '.join(column_names)}) VALUES ({placeholders});",
                tuple(values)
            )
            challenge_uid = cursor.lastrowid
            conn.commit()
            return challenge_uid

    def _insert_responses(
        self,
        challenge_uid: int,
        mechanism: MechanismType,
        miners_data: List[MinerData],
    ):
        """
        Insert the responses from the miners into the database.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            data_to_insert = []
            # prepare data for insertion
            for miner in miners_data:
                data_to_insert.append((
                    miner.hotkey, 
                    challenge_uid, 
                    serialize(miner.prediction), 
                    miner.response_time, 
                    mechanism.value
                ))

            cursor.executemany(
                """
                INSERT INTO responses (miner_hotkey, challenge_uid, prediction, response_time, mechanism)
                VALUES (?, ?, ?, ?, ?);
                """,
                data_to_insert,
            )
            conn.commit()

    def score_and_prune(
        self, 
        dataloader: BaseDataLoader, 
        score_func: Callable[[BaseSample, List[str], List[torch.Tensor], List[float]], None]
    ):
        """
        Check the database for challenges and responses, and prune them if they are not needed anymore.

        If a challenge is found that should be finished, the correct output is fetched.
        Next, all miner predictions are loaded and the score_func is called with the sample, miner hotkeys and predictions.
        """
        latest_available = dataloader.get_last_available().timestamp()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # get all challenges that we can now score
            cursor.execute(
                f"""
                SELECT * FROM {self.MECHANISM_TO_DB_NAME[dataloader.mechanism]} WHERE end_timestamp <= ?;
                """,
                (latest_available,),
            )
            challenges = cursor.fetchall()

        columns = [description[0] for description in cursor.description]
        columns.remove("uid")

        for challenge in challenges:
            (challenge_uid, *values) = challenge

            # convert json strings to tensors
            for vi, value in enumerate(values):
                if isinstance(value, str):
                    try:
                        values[vi] = deserialize(value)
                    except: # i.e. the 'variable' string
                        pass

            # instantiate based on column names matching class init arguments
            sample: BaseSample = dataloader.sample_cls(**dict(zip(columns, values)))

            # load the correct output and set it if it is available
            output = dataloader.get_output(sample)
            sample.output_data = output

            if output is None or output.shape[0] != sample.hours_to_predict or not torch.isfinite(output).all():
                if sample.end_timestamp < (latest_available - pd.Timedelta(days=3).total_seconds()):
                    # challenge is unscore-able, delete it
                    self._delete_challenge(challenge_uid, dataloader.mechanism)
                continue
        
            # load the miner predictions
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM responses WHERE challenge_uid = ? AND mechanism = ?;
                    """,
                    (challenge_uid, dataloader.mechanism.value),
                )
                responses = cursor.fetchall()

                miner_hotkeys = [r[0] for r in responses]
                predictions = [deserialize(r[2]) for r in responses]
                response_times = [r[3] for r in responses]
            
            # don't score while database is open in case there is a metagraph delay.
            score_func(sample, miner_hotkeys, predictions, response_times)
            self._delete_challenge(challenge_uid, dataloader.mechanism)

            # don't score miners too quickly in succession and always wait after last scoring
            time.sleep(1)

    def _delete_challenge(self, challenge_uid: int, mechanism: MechanismType):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # prune the challenge and the responses
            cursor.execute(
                f"""
                DELETE FROM {self.MECHANISM_TO_DB_NAME[mechanism]} WHERE uid = ?;
                """,
                (challenge_uid,),
            )
            cursor.execute(
                """
                DELETE FROM responses WHERE challenge_uid = ? AND mechanism = ?;
                """,
                (challenge_uid, mechanism.value),
            )
            conn.commit()

    def prune_hotkeys(self, hotkeys: List[str]):
        """
        Prune the database of hotkeys that are no longer participating.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM responses WHERE miner_hotkey IN ({});
                """.format(','.join('?' for _ in hotkeys)),
                hotkeys
            )
            conn.commit()


def get_column_names(cursor: sqlite3.Cursor, table_name: str) -> List[str]:	
    """
    Get the column names of a table.

    Returns:
        List[str]: The column names.
    """
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return columns

def serialize(tensor: Optional[Union[np.ndarray, torch.Tensor]]) -> str:
    """
    Serialize a tensor to a JSON string.

    Returns:
        str: The serialized tensor.
    """
    if tensor is None:
        return '[]'
    return json.dumps(tensor.tolist())

def deserialize(str_tensor: Optional[str]) -> Optional[torch.Tensor]:
    """
    Deserialize a JSON string to a tensor.

    Returns:
        Optional[torch.Tensor]: The deserialized tensor.
    """
    if str_tensor is None:
        return None
    return torch.tensor(json.loads(str_tensor))