from typing import List, Optional
import asyncio
import threading
import aiohttp
import bittensor as bt
import traceback
import math
from zeus.validator.miner_data import MinerData
from zeus.data.sample import Era5Sample
from zeus.protocol import HashedTimePredictionSynapse

LOG_COLOR = "\033[1;35m"
LOG_COLOR_RESET = "\033[0m"


class PerformanceDatabaseConnection:
    """
    Connection class for sending challenge and miner performance data to the Performance Database API.
    
    This class uses Bittensor's dendrite to send cryptographically signed requests to the API,
    this is done to ensure that only authenticated validators can insert data.
    """

    def __init__(
        self,
        wallet: bt.wallet,
        api_url: str,
    ):
        """
        Initialize the Performance Database Connection.
        
        Args:
            wallet: Bittensor wallet instance (required for signing requests)
            api_url: API server URL
        """
        if wallet is None:
            raise ValueError(f"{LOG_COLOR}[PerformanceDatabaseConnection] Wallet is required for signing requests{LOG_COLOR_RESET}")
        if not api_url:
            raise ValueError(f"{LOG_COLOR}[PerformanceDatabaseConnection] API URL is required{LOG_COLOR_RESET}")
        
        self.wallet = wallet
        self.api_url = api_url
        self.session = None
        
        # Create a dedicated background event loop for thread-safe async tasks
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._start_background_loop,
            args=(self._loop,),
            daemon=True,
            name="PerfDBThread"
        )
        self._thread.start()
        
        # Create dendrite for sending signed requests
        self.dendrite = bt.dendrite(wallet=self.wallet)
        
        bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Initialized API endpoint: {self.api_url}{LOG_COLOR_RESET}")

# --------------------------  Helper functions --------------------------
    def _start_background_loop(self, loop: asyncio.AbstractEventLoop):
        """Runs forever in the background thread."""
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _fire_and_forget(self, coro):
        """Safely schedules a coroutine on the dedicated background loop from ANY thread."""
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(coro, self._loop)
        else:
            bt.logging.warning(f"{LOG_COLOR}[PerformanceDatabaseConnection] Background loop is not running. Cannot schedule task.{LOG_COLOR_RESET}")

    async def _get_session(self):
        """Get or create the aiohttp ClientSession."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    def close(self):
        """Close the aiohttp ClientSession and stop the background thread."""
        if self.session and not self.session.closed:
            try:
                # Schedule the close on the background loop and wait for it
                future = asyncio.run_coroutine_threadsafe(self.session.close(), self._loop)
                future.result(timeout=5.0)
            except Exception as e:
                bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Failed to close session: {e}{LOG_COLOR_RESET}")
        
        # Stop the background event loop
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _get_challenge_data(self, sample: Era5Sample) -> dict:
        """Helper to convert an Era5Sample to the expected challenge dictionary."""
        return {
            "v_hotkey": self.wallet.hotkey.ss58_address,
            "lat_start": sample.lat_start,
            "lat_end": sample.lat_end,
            "lon_start": sample.lon_start,
            "lon_end": sample.lon_end,
            "start_timestamp": sample.start_timestamp,
            "end_timestamp": sample.end_timestamp,
            "hours_to_predict": sample.predict_hours,
            "variable": sample.variable
        }

    async def _send_signed_request(self, endpoint: str, request_data: dict, success_msg: str):
        """Helper to sign the request and send it to the API."""
        # Create API endpoint info for signing
        api_axon = bt.AxonInfo(
            ip="this should be okay",
            port=55555,
            hotkey="api_hotkey",
            version=1,
            ip_type=4,
            coldkey="api_coldkey",
        )
        
        # Create synapse and sign it
        synapse = bt.Synapse()
        signed_synapse = self.dendrite.preprocess_synapse_for_request(
            target_axon_info=api_axon,
            synapse=synapse,
            timeout=30.0
        )
        signed_headers = signed_synapse.to_headers()
        
        api_url_insert = f"{self.api_url}/{endpoint}"
        
        # Send HTTP request with signed headers
        session = await self._get_session()
        async with session.post(
            url=api_url_insert,
            headers=signed_headers,
            json=request_data,
            timeout=aiohttp.ClientTimeout(total=30.0)
        ) as response:
            if response.status == 200:
                response_data = await response.json()
                bt.logging.success(
                    f"{LOG_COLOR}[PerformanceDatabaseConnection] {success_msg} "
                    f"Challenge ID: {response_data.get('challenge_id', 'N/A')} "
                    f"| Info: {response_data}{LOG_COLOR_RESET}"
                )
                return response_data
            else:
                error_detail = await response.text()
                bt.logging.warning(f"{LOG_COLOR}[PerformanceDatabaseConnection] API returned status {response.status}: {error_detail}{LOG_COLOR_RESET}")
                return None

# --------------------------  Log functions --------------------------
    def log_scores(self, sample: Era5Sample, miners_data: List[MinerData]):
        """
        Log challenge and miner performance data to the API.
        
        This method:
        1. Converts Era5Sample to ChallengeData format
        2. Converts List[MinerData] to List[MinerPerformanceData] format
        3. Sends a cryptographically signed request to the API using Bittensor dendrite
        4. Handles the response and logs any errors
        
        Parameters:
            sample: Era5Sample - The challenge/sample data
            miners_data: List[MinerData] - The miners' performance data
            
        Returns:
            None (logs errors but doesn't raise exceptions to avoid disrupting validator flow)
        """
        try:
            bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Trying to log information to the performance database.{LOG_COLOR_RESET}")
            self._fire_and_forget(self._log_scores_async(sample, miners_data))
        except Exception as e:
            bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Failed to schedule performance data logging: {e}. {LOG_COLOR_RESET}")

    async def _log_scores_async(self, sample: Era5Sample, miners_data: List[MinerData]):
        """
        Async implementation of log_scores method.
        
        Parameters:
            sample: Era5Sample - The challenge/sample data
            miners_data: List[MinerData] - The miners' performance data
        """
        try:
            challenge_data = self._get_challenge_data(sample)
            
            # Convert List[MinerData] to List[MinerPerformanceData] format
            miners_performance = []
            for miner in miners_data:
                rmse_value = float(miner.rmse) if miner.rmse is not None and miner.rmse != float('inf') else -1.0
                mae_value = float(miner.mae) if miner.mae is not None and miner.mae != float('inf') else -1.0
                
                miners_performance.append({
                    "miner_hotkey": miner.hotkey,
                    "miner_uid": miner.uid,
                    "rmse": rmse_value,
                    "mae": mae_value,
                    "rank": miner.score,
                    "penalty": int(miner.shape_penalty)
                })
            
            # Skip if no valid miners to log
            if not miners_performance:
                bt.logging.warning(f"{LOG_COLOR}[PerformanceDatabaseConnection] No valid miner performance data to log{LOG_COLOR_RESET}")
                return
            
            request_data = {
                "challenge": challenge_data,
                "miners_performance": miners_performance
            }
            
            bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Sending performance data to API: challenge for {sample.variable}, {len(miners_performance)} miners{LOG_COLOR_RESET}")
            
            await self._send_signed_request(
                endpoint="insert_data", 
                request_data=request_data, 
                success_msg="Successfully logged performance data."
            )
                
        except Exception as e:
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Error in _log_scores_async ({error_type}): {e}\nTraceback:\n{error_traceback}{LOG_COLOR_RESET}")

    def log_rank_aggregates(self, miners_metadata: List[dict], variable: str):
        """
        Log per-variable rolling average rank rows to the performance API (signed POST).
        miners_metadata entries must include hotkey, rank, miner_window (from compute_min_rank_weights).
        """
        try:
            bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Trying to log rank aggregates to the performance database.{LOG_COLOR_RESET}")
            self._fire_and_forget(self._log_rank_aggregates_async(miners_metadata, variable))
        except Exception as e:
            bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Failed to schedule rank aggregates logging: {e}. {LOG_COLOR_RESET}")

    async def _log_rank_aggregates_async(self, miners_metadata: List[dict], variable: str):
        """Async implementation of log_rank_aggregates."""
        try:
            if not miners_metadata:
                bt.logging.warning(f"{LOG_COLOR}[PerformanceDatabaseConnection] No rank aggregates to log{LOG_COLOR_RESET}")
                return

            api_rows = [
                {
                    "variable": variable,
                    "miner_hotkey": m["hotkey"],
                    "rank": m["avg_rank"],
                    "miner_window": m["miner_window"],
                }
                for m in miners_metadata
            ]

            bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Sending rank aggregates to API: {len(api_rows)} rows ({variable}){LOG_COLOR_RESET}")

            await self._send_signed_request(
                endpoint="insert_hotkey_rank_aggregates",
                request_data=api_rows,
                success_msg="Successfully logged rank aggregates."
            )

        except Exception as e:
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Error in _log_rank_aggregates_async ({error_type}): {e}\nTraceback:\n{error_traceback}{LOG_COLOR_RESET}")

    def log_hash_responses_info(self, sample: Era5Sample, requested_at: float, responses: List[HashedTimePredictionSynapse], all_queried_uid: List[int], queried_miners_corresponding_attempts: List[int], successful_uids: List[int]):
        """
        Log hash responses info to the API.
        
        Parameters:
            sample: Era5Sample - The challenge/sample data
            requested_at: float - Timestamp when the request was sent
            responses: List[HashedTimePredictionSynapse] - The responses from miners
            all_queried_uid: List[int] - UIDs of all queried miners
            queried_miners_corresponding_attempts: List[int] - Attempt numbers for each queried miner
            successful_uids: List[int] - UIDs of miners that successfully responded
        """
        try:
            bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Trying to log hash responses info to the performance database.{LOG_COLOR_RESET}")
            self._fire_and_forget(self._log_hash_responses_info_async(sample, requested_at, responses, all_queried_uid, queried_miners_corresponding_attempts, successful_uids))
        except Exception as e:
            bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Failed to schedule hash responses info logging: {e}. {LOG_COLOR_RESET}")

    async def _log_hash_responses_info_async(self, sample: Era5Sample, requested_at: float, responses: List[HashedTimePredictionSynapse], all_queried_uid: List[int], queried_miners_corresponding_attempts: List[int], successful_uids: List[int]):
        """
        Async implementation of log_hash_responses_info method.
        """
        try:
            challenge_data = self._get_challenge_data(sample)
            
            # Prepare hash responses info
            hash_responses_info = []
            for i, uid in enumerate(all_queried_uid):
                # Safely get response time if available
                response_time = float(responses[i].dendrite.process_time) if i < len(responses) and responses[i].dendrite.process_time else 0.0
                attempt_number = queried_miners_corresponding_attempts[i] if i < len(queried_miners_corresponding_attempts) else 0
                successful_attempt = uid in successful_uids
                
                hash_responses_info.append({
                    "miner_uid": uid,
                    "queried_at": requested_at,
                    "response_time": response_time,
                    "attempt_number": attempt_number,
                    "successful_attempt": successful_attempt
                })
            
            if not hash_responses_info:
                bt.logging.warning(f"{LOG_COLOR}[PerformanceDatabaseConnection] No valid hash response info to log{LOG_COLOR_RESET}")
                return
                
            request_data = {
                "challenge": challenge_data,
                "hash_responses_info": hash_responses_info
            }
            
            bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Sending hash response info to API: challenge for {sample.variable}, {len(hash_responses_info)} miners{LOG_COLOR_RESET}")
            
            await self._send_signed_request(
                endpoint="insert_hash_response_info", 
                request_data=request_data, 
                success_msg="Successfully logged hash response info."
            )
                        
        except Exception as e:
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Error in _log_hash_responses_info_async ({error_type}): {e}\nTraceback:\n{error_traceback}{LOG_COLOR_RESET}")

    def insert_top_k_info(self, sample: Era5Sample, miner_hotkeys: List[str], miner_uids: List[int], unsuccessful_hotkeys: List[str]):
        """
        Insert top k info to the API.
        """
        try:
            bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Trying to log top k info to the performance database.{LOG_COLOR_RESET}")
            self._fire_and_forget(self._insert_top_k_info_async(sample, miner_hotkeys, miner_uids, unsuccessful_hotkeys))
        except Exception as e:
            bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Failed to schedule top k info logging: {e}. {LOG_COLOR_RESET}")

    async def _insert_top_k_info_async(self, sample: Era5Sample, miner_hotkeys: List[str], miner_uids: List[int], unsuccessful_hotkeys: List[str]):
        """
        Async implementation of insert_top_k_info method.
        """
        try:
            challenge_data = self._get_challenge_data(sample)
            
            # Prepare top miners predictions info
            top_miners_predictions = []
            for i, hotkey in enumerate(miner_hotkeys):
                uid = miner_uids[i] if i < len(miner_uids) else -1
                received_penalty = hotkey in unsuccessful_hotkeys
                
                top_miners_predictions.append({
                    "miner_hotkey": hotkey,
                    "miner_uid": uid,
                    "received_penalty": received_penalty
                })
            
            if not top_miners_predictions:
                bt.logging.warning(f"{LOG_COLOR}[PerformanceDatabaseConnection] No valid top miners predictions to log{LOG_COLOR_RESET}")
                return
                
            request_data = {
                "challenge": challenge_data,
                "top_miners_predictions": top_miners_predictions
            }
            
            bt.logging.info(f"{LOG_COLOR}[PerformanceDatabaseConnection] Sending top miners predictions to API: challenge for {sample.variable}, {len(top_miners_predictions)} miners{LOG_COLOR_RESET}")
            
            await self._send_signed_request(
                endpoint="insert_top_miners_predictions", 
                request_data=request_data, 
                success_msg="Successfully logged top miners predictions."
            )
                        
        except Exception as e:
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            bt.logging.error(f"{LOG_COLOR}[PerformanceDatabaseConnection] Error in _insert_top_k_info_async ({error_type}): {e}\nTraceback:\n{error_traceback}{LOG_COLOR_RESET}")