import os

import pandas as pd
from zeus.validator.constants import MIN_HOURS_BETWEEN_REQUESTS, CHALLENGE_HASHING_MAX_MINUTE

# Set ZEUS_TEST_MODE=1 to force immediate hash/prediction phases on every forward() call.
_TEST_MODE = os.environ.get("ZEUS_TEST_MODE", "0") == "1"

class Scheduler:

    def __init__(self):

        self.last_hash_request_at: pd.Timestamp = pd.Timestamp(0,  tz='UTC')
        self.have_hashes_should_query_best: bool = False

    def is_hash_commit_time(self) -> bool:
        """
        The requests are once every 6 hours, at "00:CHALLENGE_HASHING_MAX_MINUTE", "06:CHALLENGE_HASHING_MAX_MINUTE", "12:CHALLENGE_HASHING_MAX_MINUTE", "18:CHALLENGE_HASHING_MAX_MINUTE"
        """
        if _TEST_MODE:
            if (pd.Timestamp.now('UTC') - self.last_hash_request_at) > pd.Timedelta(minutes=2):
                self.last_hash_request_at = pd.Timestamp.now('UTC')
                self.have_hashes_should_query_best = True
                return True
            return False

        now = pd.Timestamp.now('UTC')

        if (now.hour%6 == 0 and now.minute >= CHALLENGE_HASHING_MAX_MINUTE) and (now - self.last_hash_request_at) > pd.Timedelta(hours=MIN_HOURS_BETWEEN_REQUESTS):
            self.last_hash_request_at = now
            self.have_hashes_should_query_best = True
            return True

        return False

    def is_query_best_time(self) -> bool:
        """
        The query best happens at least 30 minutes after the hash stage, only once per 6 hours.
        """
        if _TEST_MODE and self.have_hashes_should_query_best:
            self.have_hashes_should_query_best = False
            return True

        now = pd.Timestamp.now('UTC')

        if not (now.hour%6 == 0 and now.minute >= CHALLENGE_HASHING_MAX_MINUTE) and self.have_hashes_should_query_best:
            self.have_hashes_should_query_best = False
            return True

        return False