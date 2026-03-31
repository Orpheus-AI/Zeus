import pandas as pd
from zeus.validator.constants import MIN_HOURS_BETWEEN_REQUESTS

class Scheduler:
      
    def __init__(self):

        self.last_hash_request_at: pd.Timestamp = pd.Timestamp(0,  tz='UTC')
        self.have_hashes_should_query_best: bool = False

    def is_hash_commit_time(self) -> bool:
        """
        The requests are once every 6 hours, at "00:30", "06:30", "12:30", "18:30"
        """    
        now = pd.Timestamp.now('UTC')

        if (now.hour%6 == 0 and now.minute >= 30) and (now - self.last_hash_request_at) > pd.Timedelta(hours=MIN_HOURS_BETWEEN_REQUESTS):
            self.last_hash_request_at = now
            self.have_hashes_should_query_best = True
            return True
        
        return False
    
    def is_query_best_time(self) -> bool:
        """
        The query best happens at least 30 minutes after the hash stage, only once per 6 hours. 
        """    
        now = pd.Timestamp.now('UTC')

        if not (now.hour%6 == 0 and now.minute >= 30) and self.have_hashes_should_query_best:
            self.have_hashes_should_query_best = False
            return True
        
        return False