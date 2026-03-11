import sqlite3
from typing import List, Optional

from zeus.validator.constants import PERFORMANCE_DATABASE_LOCATION
from zeus.validator.miner_data import MinerData
from zeus.data.sample import Era5Sample

class PerformanceDatabase:

    def __init__(
        self,
        v_hotkey: str,
        db_path: str = PERFORMANCE_DATABASE_LOCATION,
    ):
        self.db_path = db_path
        self.create_tables()
        self.v_hotkey = v_hotkey


    def create_tables(self):
        """
        Create the tables for the performance database.
        # TODO : this is only temporary until we actually have a database where all the validators write to and we make a dashboard from that
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS challenges (
                    uid INTEGER PRIMARY KEY AUTOINCREMENT,
                    v_hotkey TEXT,
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
                """,
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS responses (
                    v_hotkey TEXT,
                    miner_hotkey TEXT,
                    miner_uid INTEGER,
                    challenge_uid INTEGER,
                    rmse REAL,
                    score REAL,
                    shape_penalty INTEGER,
                    FOREIGN KEY (challenge_uid) REFERENCES challenges (uid)
                );
                """,        
            )

            conn.commit()

    def log_performance(self, sample: Era5Sample, miners_data : List[MinerData]):
        """
        Inserts/logs the sample/challenge and the miners' data into the database

        Parameters:
            sample: Era5Sample - The sample to be added to the database
            miners_data: List[MinerData] - The miners' data to be inserted

        Step 1 ) check if the sample already exists in the past_challenges table, 
        Step 2.1) if it does, get the challenge_id and jump to step 3)
        Step 2.2) otherwise, add the challenge (i.e. sample) to the past_challenges table
        Step 3) add the miners'data to the miner_performance table
        """
        
        # Step 1)
        challenge_id = self.find_challenge_id(sample)

        if challenge_id is None: # Step 2.2)
            challenge_id = self.insert_challenge(sample)
        
        # Step 3) 
        self.insert_miner_perfromance(challenge_id, miners_data)

    def find_challenge_id(self, sample: Era5Sample) -> Optional[int]:
        """
        Looks if a given challenge/sample is already saved in the challenges table.
        If the sample has been saved, the challenge_id (uid) is returned, otherwise returns None.
        
        Parameters:
            sample: FtFSample - The sample to search for
            
        Returns:
            Optional[int] - The challenge uid if found, None otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT uid FROM challenges 
                WHERE lat_start = ? 
                AND lat_end = ? 
                AND lon_start = ? 
                AND lon_end = ? 
                AND start_timestamp = ? 
                AND end_timestamp = ? 
                AND hours_to_predict = ? 
                AND inserted_at = ?
                AND variable = ?
                AND v_hotkey = ?
                """,
                (
                    sample.lat_start,
                    sample.lat_end,
                    sample.lon_start,
                    sample.lon_end,
                    sample.start_timestamp,
                    sample.end_timestamp,
                    sample.predict_hours,
                    sample.query_timestamp,
                    sample.variable,
                    self.v_hotkey,
                ),
            )
            result = cursor.fetchone()
            return result[0] if result else None

    def insert_challenge(self, sample: Era5Sample) -> int:
        """
        Insert a challenge into the database.

        Parameters:
            sample: FtFSample - The sample to be added to the database

        Returns:
            The challenge uid.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO challenges (v_hotkey, lat_start, lat_end, lon_start, lon_end, start_timestamp, end_timestamp, hours_to_predict, inserted_at, variable)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    self.v_hotkey,
                    sample.lat_start,
                    sample.lat_end,
                    sample.lon_start,
                    sample.lon_end,
                    sample.start_timestamp,
                    sample.end_timestamp,
                    sample.predict_hours,        
                    sample.query_timestamp,
                    sample.variable,
                ),
            )
            challenge_uid = cursor.lastrowid
            conn.commit()
            return challenge_uid

    def insert_miner_perfromance(self, challenge_uid: int, miners_data: List[MinerData]):
        """
        Insert the responses from the miners into the database.

        Parameters:
            challenge_uid: int - The challenge uid to insert the miners' data into
            miners_data: List[MinerData] - The miners' data to be inserted
        Returns:
            None
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            data_to_insert = []
            # prepare data for insertion
            for miner in miners_data:
                data_to_insert.append((
                    self.v_hotkey,
                    miner.hotkey, 
                    miner.uid,
                    challenge_uid,
                    miner.rmse, 
                    miner.score, 
                    int(miner.shape_penalty)
                ))

            cursor.executemany(
                """
                INSERT INTO responses (v_hotkey, miner_hotkey, miner_uid, challenge_uid, rmse, score, shape_penalty)
                VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                data_to_insert,
            )
            conn.commit()  