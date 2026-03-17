from dataclasses import dataclass
from typing import Optional

@dataclass
class MinerData:
    hotkey: str
    prediction: Optional[bytes] = None
    uid: Optional[int] = None  # all below are not set initially
    score: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    shape_penalty: Optional[bool] = None
    prediction_hash: Optional[str] = None

    @property
    def metrics(self):
        return {
             "RMSE": self.rmse,
             "MAE": self.mae, 
             "score": self.score,
             "shape_penalty": self.shape_penalty,
        }