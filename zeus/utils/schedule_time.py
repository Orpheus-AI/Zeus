import pandas as pd
from zeus.validator.constants import MIN_HOURS_BETWEEN_REQUESTS

class Scheduler:
      
    def __init__(self):

        self.last_hash_request_at: pd.Timestamp = pd.Timestamp(0,  tz='UTC')
        self.last_query_best_at: pd.Timestamp = pd.Timestamp(0,  tz='UTC')

    def is_hash_commit_time(self) -> bool:
        """
        The requests are once every 6 hours, at "00", "06", "12", "18"
        """    
        now = pd.Timestamp.now('UTC')

        if now.hour%6 == 0 and (now - self.last_hash_request_at) > pd.Timedelta(hours=MIN_HOURS_BETWEEN_REQUESTS):
            self.last_hash_request_at = now
            return True
        
        return False
    
    def is_query_best_time(self) -> bool:
        """
        The query best happens at least one hour after the hash stage
        """    
        now = pd.Timestamp.now('UTC')

        if now.hour%6 != 0 and (now - self.last_query_best_at) > pd.Timedelta(hours=MIN_HOURS_BETWEEN_REQUESTS):
            self.last_query_best_at = now
            return True
        
        return False