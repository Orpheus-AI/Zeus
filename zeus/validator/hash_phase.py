import time
from typing import List

import bittensor as bt
from copy import deepcopy
from zeus.base.validator import BaseValidatorNeuron
from zeus.data.sample import Era5Sample
from zeus.protocol import HashedTimePredictionSynapse
from zeus.utils.coordinates import bbox_to_str
from zeus.utils.time import timestamp_to_str
from zeus.utils.uids import get_available_uids
from zeus.validator.constants import HASH_DENDRITE_SETTINGS, MAINNET_UID
from zeus.validator.miner_data import MinerData
from zeus.validator.responses_processing import (
    _build_bad_miners_data,
)
import math

async def run_all_hash_phases(self, challenges):
    """Run hash phase for all challenges, collecting miner hash commitments.
    
    Args:
        self: BaseValidatorNeuron instance
        challenges: List of Era5Sample challenges to process
        
    Returns:
        Dictionary mapping challenge strings to lists of good MinerData objects
    """
    all_uids_to_query = get_available_uids(
        self.metagraph,
        self.config.neuron.vpermit_tao_limit,
        MAINNET_UID,
        exclude=set(),
    )

    for i, sample in enumerate(challenges):
        bt.logging.info(f"[run_all_hash_phases] Start of the commit phase for challenge {i}/{len(challenges)}")
        bt.logging.info(f"[run_all_hash_phases] Data sampled with bounding box {bbox_to_str(sample.get_bbox())} for variable {sample.variable}")
        bt.logging.info(f"[run_all_hash_phases] Data sampled starts from {timestamp_to_str(sample.start_timestamp)} | Predict {sample.predict_hours} steps (step={sample.step_size}h).")
        
        str_sample = str(sample)
        

        good_miners_data, bad_miners_uids = await _run_single_hash_phase(self, all_uids_to_query, sample)
        bt.logging.warning(f"hash phase:[{str_sample}] {len(bad_miners_uids)} miners do not have a valid hash or did not respond: {bad_miners_uids}")
        if len(bad_miners_uids) > 0:
            good_miners_uids = [miner.uid for miner in good_miners_data]
            
            bt.logging.warning(f'good miners uids: {good_miners_uids}')
            bad_miners_data = _build_bad_miners_data(self, bad_miners_uids)



            successful_insertion = self.database.insert(sample, bad_miners_data, good_miners=False)
            bt.logging.warning(f"successful_insertion of bad miners? {successful_insertion}")
            if successful_insertion:
                #self.uid_tracker.mark_finished(bad_miners_uids, good=False)
                bt.logging.success(f"Storing bad miner responses in SQLite database: {bad_miners_uids}")
        
      
    


async def _run_single_hash_phase(self: BaseValidatorNeuron, all_uids_to_query: List[int], sample: Era5Sample):
    """Run hash phase for a single challenge, querying miners in batches.
    
    Args:
        self: BaseValidatorNeuron instance
        all_uids_to_query: List of miner UIDs to query
        sample: Era5Sample challenge to process
        
    Returns:
        Tuple of (good_miners_data, bad_miners_uids) where good_miners_data is a list
        of MinerData objects with valid hashes, and bad_miners_uids is a set of UIDs
        that failed to provide valid hashes
    """
    bt.logging.debug(f'[_run_single_hash_phase] Number of all miners uids {len(all_uids_to_query)}')

    # in the last few trials, if there are any miners that were not
    steps_to_see_everyone_once = math.ceil(len(all_uids_to_query) / HASH_DENDRITE_SETTINGS.response_batch_k)
    good_miners_data: List[MinerData] = []
    good_miners_uids: set[int] = set()
    bad_miners_uids: set[int] = set()
    
    forward_max_uid_attempts = steps_to_see_everyone_once * HASH_DENDRITE_SETTINGS.attempts_per_miner
    self.uid_tracker.init_count_map()
    for attempt in range(forward_max_uid_attempts):
        # Exclude only good_miners so we never re-query them; bad miners can be sampled again
        miner_uids = self.uid_tracker.get_next_batch(
            k=HASH_DENDRITE_SETTINGS.response_batch_k,
            good_miners_uids=good_miners_uids,
            allowed_uids= None,
            allowed_attempts=HASH_DENDRITE_SETTINGS.attempts_per_miner
        )
        if not miner_uids:
            continue
        
        bt.logging.debug(f"[_run_single_hash_phase] len miner_uids: {len(set(miner_uids))} queried uids: {miner_uids}")

        axons_to_query = [self.metagraph.axons[uid] for uid in miner_uids]

        start_time = time.time()
        responses = await self.dendrite_hash(
            axons=axons_to_query,
            synapse=sample.build_synapse(HashedTimePredictionSynapse),
            deserialize=False,
            timeout=self.config.neuron.hash_timeout,
        )
        end_time = time.time()
        bt.logging.warning(f"Time taken to query hashes: {end_time - start_time} seconds")
       
        batch_good_miners_data = _parse_hashes(miner_uids, axons_to_query, responses)
        batch_successful_uids = [miner_data.uid for miner_data in batch_good_miners_data]
        good_miners_data.extend(batch_good_miners_data)
       
        if len(batch_good_miners_data) > 0:
            successful_insertion = self.database.insert(sample, batch_good_miners_data, good_miners=True)
            bt.logging.debug(f"[_run_single_hash_phase] successful_insertion? {successful_insertion}")
            if successful_insertion:
                bt.logging.success(f"Storing challenge and sensible miner responses in SQLite database: {batch_successful_uids}")
        
        bt.logging.info(f"Commit batch: {len(batch_successful_uids)}/{len(miner_uids)} miners succeeded.")

        good_miners_uids.update(batch_successful_uids)

        

        bad_miners_uids = set(all_uids_to_query) - good_miners_uids
        # Introduce a delay to prevent spamming requests

        if len(good_miners_uids) >= len(all_uids_to_query):
            bt.logging.info(f"All {len(all_uids_to_query)} available miners are good; stopping batch loop.")
            break
        
        time.sleep(10) #max(0, FORWARD_DELAY_SECONDS - (time.time() - start_forward)))
        

        
    return good_miners_data, bad_miners_uids


def _parse_hashes(miner_uids, axons_to_query, responses) -> List[MinerData]:
    """Parse hash responses from miners and create MinerData objects.
    
    Args:
        miner_uids: List of miner UIDs that were queried
        axons_to_query: List of axon objects corresponding to the UIDs
        responses: List of HashedTimePredictionSynapse responses from miners
        
    Returns:
        List of MinerData objects for miners that provided valid hashes
    """
    miners_data = []
    for uid, axon, response in zip(miner_uids, axons_to_query, responses):
        if response.hash:
            miners_data.append(MinerData(uid=uid, hotkey=axon.hotkey, prediction_hash=response.hash))
            bt.logging.debug(f'[_parse_hashes] stored hash {response.hash} for uid {uid}')
        else:
            # Log why this miner was not counted (so stress-test / timeout cases are visible)
            reason = "no hash"
            if getattr(response, "is_timeout", None):
                reason = "timeout"
            elif getattr(response, "is_failure", None):
                reason = "failure"
            bt.logging.debug(
                f"Commit: uid {uid} hotkey {axon.hotkey} not stored: {reason} "
                f"(is_timeout={getattr(response, 'is_timeout', None)}, is_failure={getattr(response, 'is_failure', None)})"
            )
    return miners_data




    


