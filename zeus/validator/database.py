from typing import List, Callable
import sqlite3
import time
import torch
import pandas as pd
import json

from zeus.data.loaders.era5_cds import Era5CDSLoader
from zeus.validator.constants import DATABASE_LOCATION
from zeus.data.sample import Era5Sample


class ResponseDatabase:

    def __init__(
        self,
        cds_loader: Era5CDSLoader,
        db_path: str = DATABASE_LOCATION,
    ):
        self.cds_loader = cds_loader
        self.db_path = db_path
        self.create_tables()
        # start at 0 so it always syncs at startup
        self.last_synced_block = 0

    def should_score(self, block: int) -> bool:
        """
        Check if the database should score its stored miner predictions.
        This is done roughly hourly, so with one block every 12 seconds this means
        if the current block is more than 300 blocks ahead of the last synced block, we should score.
        """
        if not self.cds_loader.is_ready():
            return False
        if block - self.last_synced_block > 300:
            self.last_synced_block = block
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
                    variable TEXT DEFAULT '2m_temperature'
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
                    FOREIGN KEY (challenge_uid) REFERENCES challenges (uid)
                );
                """
            )
            conn.commit()

    def insert(
        self,
        sample: Era5Sample,
        miner_hotkeys: List[str],
        predictions: List[torch.Tensor],
    ):
        """
        Insert a challenge and responses into the database.
        """
        challenge_uid = self._insert_challenge(sample)
        self._insert_responses(challenge_uid, miner_hotkeys, predictions)

    def _insert_challenge(self, sample: Era5Sample) -> int:
        """
        Insert a sample into the database and return the challenge UID.
        Assumes the sample's output data is the baseline.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO challenges (lat_start, lat_end, lon_start, lon_end, start_timestamp, end_timestamp, hours_to_predict, baseline, inserted_at, variable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    *sample.get_bbox(),
                    sample.start_timestamp,
                    sample.end_timestamp,
                    sample.predict_hours,
                    json.dumps(sample.output_data.tolist()),
                    sample.query_timestamp,
                    sample.variable
                ),
            )
            challenge_uid = cursor.lastrowid
            conn.commit()
            return challenge_uid

    def _insert_responses(
        self,
        challenge_uid: int,
        miner_hotkeys: List[str],
        predictions: List[torch.Tensor],
    ):
        """
        Insert the responses from the miners into the database.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            data_to_insert = []
            # prepare data for insertion
            for miner_hotkey, prediction in zip(miner_hotkeys, predictions):
                prediction_json = json.dumps(prediction.tolist())

                data_to_insert.append((miner_hotkey, challenge_uid, prediction_json))

            cursor.executemany(
                """
                INSERT INTO responses (miner_hotkey, challenge_uid, prediction)
                VALUES (?, ?, ?);
            """,
                data_to_insert,
            )
            conn.commit()

    def score_and_prune(
        self, score_func: Callable[[Era5Sample, torch.Tensor, List[str], List[torch.Tensor]], None]
    ):
        """
        Check the database for challenges and responses, and prune them if they are not needed anymore.

        If a challenge is found that should be finished, the correct output is fetched.
        Next, all miner predictions are loaded and the score_func is called with the sample, miner hotkeys and predictions.
        """
        latest_available = self.cds_loader.last_stored_timestamp.timestamp()

        with sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            cursor = conn.cursor()
            # get all challenges that we can now score
            cursor.execute(
                """
                SELECT * FROM challenges WHERE end_timestamp <= ?;
            """,
                (latest_available,),
            )
            challenges = cursor.fetchall()

        for i, challenge in enumerate(challenges):
            # load the sample
            (
                challenge_uid,
                lat_start,
                lat_end,
                lon_start,
                lon_end,
                start_timestamp,
                end_timestamp,
                hours_to_predict,
                baseline,
                inserted_at,
                variable,
            ) = challenge

            sample = Era5Sample(
                variable=variable,
                query_timestamp=inserted_at,
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
                lat_start=lat_start,
                lat_end=lat_end,
                lon_start=lon_start,
                lon_end=lon_end,
                predict_hours=hours_to_predict,
            )
            # load the correct output and set it if it is available
            output = self.cds_loader.get_output(sample)
            sample.output_data = output

            if output is None or output.shape[0] != hours_to_predict:
                if end_timestamp < (latest_available - pd.Timedelta(days=3).total_seconds()):
                    # challenge is unscore-able, delete it
                    self._delete_challenge(challenge_uid)
                continue
        
            baseline = torch.tensor(json.loads(baseline))
            # load the miner predictions
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    SELECT * FROM responses WHERE challenge_uid = ?;
                """,
                    (challenge_uid,),
                )
                responses = cursor.fetchall()

                miner_hotkeys = [response[0] for response in responses]
                predictions = [
                    torch.tensor(json.loads(response[2])) for response in responses
                ]
            
            # don't score while database is open in case there is a metagraph delay.
            score_func(sample, baseline, miner_hotkeys, predictions)
            self._delete_challenge(challenge_uid)

            # don't score miners too quickly in succession and always wait after last scoring
            if i > 0:
                time.sleep(4)

    def _delete_challenge(self, challenge_uid: int):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # prune the challenge and the responses
            cursor.execute(
                """
                DELETE FROM challenges WHERE uid = ?;
            """,
                (challenge_uid,),
            )
            cursor.execute(
                """
                DELETE FROM responses WHERE challenge_uid = ?;
            """,
                (challenge_uid,),
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


def column_exists(cursor: sqlite3.Cursor, table_name: str, column_name: str):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]
    return column_name in columns