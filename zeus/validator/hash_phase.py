import time
from typing import Dict, List

import bittensor as bt

from zeus.base.validator import BaseValidatorNeuron
from zeus.commitment import ChallengeCommitment, read_all_miner_commitments
from zeus.data.sample import Era5Sample
from zeus.utils.coordinates import bbox_to_str
from zeus.utils.time import timestamp_to_str, _block_parameters_for_sample
from zeus.utils.uids import get_available_uids
from zeus.validator.constants import MAINNET_UID
from zeus.validator.miner_data import MinerData
from zeus.validator.responses_processing import (
    _build_bad_miners_data,
)


def _insert_hash_data_to_db(
    self: BaseValidatorNeuron, 
    sample: Era5Sample, 
    good_miners_data: List[MinerData], 
    bad_miners_uids: set, 
    metagraph
):
    """Helper to insert good/bad miner data into DB and handle logging. Returns True if any insertion succeeded."""

    if good_miners_data:
        successful_insertion = self.database.insert(sample, good_miners_data, good_miners=True)
        bt.logging.debug(f"[run_all_hash_phases] successful good insertion? {successful_insertion}")
        if successful_insertion:
            good_miners_uids = [miner.uid for miner in good_miners_data]
            bt.logging.success(f"Storing challenge and chain hashes in SQLite database: {good_miners_uids}")
         

    if bad_miners_uids:
        bt.logging.warning(f"hash phase:[{str(sample)}] {len(bad_miners_uids)} miners do not have a valid hash on chain: {bad_miners_uids}")
        bad_miners_data = _build_bad_miners_data(self, bad_miners_uids, metagraph=metagraph)
        successful_insertion = self.database.insert(sample, bad_miners_data, good_miners=False)
        bt.logging.warning(f"successful_insertion of bad miners? {successful_insertion}")
        if successful_insertion:
            bt.logging.success(f"Storing bad miner responses in SQLite database: {bad_miners_uids}")
    


async def run_all_hash_phases(self, challenges):
    """Run hash phase for all challenges, sourcing miner hash commitments from chain.

    Args:
        self: BaseValidatorNeuron instance
        challenges: List of Era5Sample challenges to process
    """
    all_uids_to_query = get_available_uids(
        self.metagraph,
        self.config.neuron.vpermit_tao_limit,
        MAINNET_UID,
        exclude=set(),
    )
    
    # avoid re-reading commitments for the same block
    commitments_by_block: Dict[int | None, Dict[int, ChallengeCommitment]] = {}

    for i, sample in enumerate(challenges):
        bt.logging.info(f"[run_all_hash_phases] Start of the commit phase for challenge {i}/{len(challenges)}")
        bt.logging.info(f"[run_all_hash_phases] Data sampled with bounding box {bbox_to_str(sample.get_bbox())} for variable {sample.variable}")
        bt.logging.info(f"[run_all_hash_phases] Data sampled starts from {timestamp_to_str(sample.start_timestamp)} | Predict {sample.predict_hours} steps (step={sample.step_size}h).")
        if self.database.find_challenge_id(sample) is not None: # avoid duplicate challenges
            bt.logging.info(f"[run_all_hash_phases] Challenge {sample.state_key} already exists in database, skipping hash phase")
            continue
        
        chain_head_block = self.subtensor.get_current_block()
        chain_head_time = self.subtensor.get_timestamp(chain_head_block)

        commit_block, read_block, reference_block, max_blocks_older = _block_parameters_for_sample(
            sample, chain_head_block, chain_head_time
        )
        bt.logging.info(
            f"[run_all_hash_phases] {sample.state_key} commit_block={commit_block} "
            f"read_block={read_block or 'latest'} reference_block={reference_block}"
        )

        chain_commitments = commitments_by_block.get(read_block)
        if chain_commitments is None:
            start_time = time.time()
            try:
                chain_commitments = read_all_miner_commitments(
                    self.subtensor, # use last subtensor and metagraph
                    self.metagraph,
                    self.config.netuid,
                    block=read_block,
                    current_block=chain_head_block,
                    vpermit_tao_limit=self.config.neuron.vpermit_tao_limit
                )
            except Exception as e:
                bt.logging.error(
                    f"[run_all_hash_phases] Failed to read commitments from chain "
                    f"at block {read_block or 'latest'}: {e}. Treating all miners as bad for this cycle."
                )
                chain_commitments = {}
            commitments_by_block[read_block] = chain_commitments
            bt.logging.info(
                f"[run_all_hash_phases] Read {len(chain_commitments)} miner commitments "
                f"from chain at block {read_block or 'latest'} in {time.time() - start_time:.2f}s"
            )
            if len(chain_commitments) == 0:
                bt.logging.warning(
                    f"[run_all_hash_phases] No miner commitments found on chain at block {read_block or 'latest'}. "
                    f"All {len(all_uids_to_query)} miners will be marked as bad."
                )

        good_miners_data, bad_miners_uids = await _run_single_hash_phase(
            self.metagraph,
            all_uids_to_query,
            sample,
            chain_commitments,
            reference_block,
            max_blocks_older,
        )
        successful_uids = [miner_data.uid for miner_data in good_miners_data]
        committed_at_by_uid = {
            miner_data.uid: miner_data.block_hash_committed_at
            for miner_data in good_miners_data
            if miner_data.block_hash_committed_at is not None
        }
        self.performance_database_api.log_hash_responses_info(
            sample,
            time.time(),
            all_uids_to_query,
            successful_uids,
            committed_at_by_uid,
        )

        _insert_hash_data_to_db(self, sample, good_miners_data, bad_miners_uids, self.metagraph)


async def _run_single_hash_phase(
    metagraph,
    all_uids_to_query: List[int],
    sample: Era5Sample,
    chain_commitments: Dict[int, ChallengeCommitment],
    reference_block: int,
    max_blocks_older: int,
):
    """Run hash phase for a single challenge by reading hashes from chain.

    Args:
        metagraph: Metagraph snapshot (live ``self.metagraph`` or historical at challenge block).
        all_uids_to_query: List of miner UIDs to include in this cycle
        sample: Era5Sample challenge to process
        chain_commitments: Cached on-chain commitments for the relevant block
        reference_block: Block number used to assess commitment freshness.
        max_blocks_older: Maximum accepted age in blocks before a commitment is
            treated as stale.

    Returns:
        Tuple of (good_miners_data, bad_miners_uids) where good_miners_data is a list
        of MinerData objects with valid hashes, and bad_miners_uids is a set of UIDs
        that do not have a valid hash on chain for this state_key
    """
    bt.logging.debug(f"[_run_single_hash_phase] Number of all miners uids {len(all_uids_to_query)}")
    good_miners_data: List[MinerData] = []
    good_miners_uids: set[int] = set()
    ZERO_HASH = "0" * 64  # 32 bytes of zeros as hex
    
    for uid in all_uids_to_query:
        axon = metagraph.axons[uid]
        commitment = chain_commitments.get(uid)
        if commitment is None:
            bt.logging.debug(
                f"[_run_single_hash_phase] uid {uid} hotkey {axon.hotkey} has no commitment on chain"
            )
            continue

        prediction_hash = commitment.hashes.get(sample.state_key)
        if not prediction_hash or prediction_hash == ZERO_HASH:
            bt.logging.debug(
                f"[_run_single_hash_phase] uid {uid} hotkey {axon.hotkey} has no chain hash for {sample.state_key}"
            )
            continue

        committed_at = commitment.committed_at_block
        if committed_at is None:
            bt.logging.debug(
                f"[_run_single_hash_phase] uid {uid} commit lacks block metadata — treating as stale"
            )
            continue

        is_old_commitment = (reference_block - committed_at) > max_blocks_older
        if is_old_commitment:
            bt.logging.debug(
                f"[_run_single_hash_phase] uid {uid} commit stale — "
                f"committed_at={committed_at} reference_block={reference_block} age={reference_block - committed_at}"
            )
            continue

        good_miners_data.append(
            MinerData(
                uid=uid,
                hotkey=axon.hotkey,
                prediction_hash=prediction_hash,
                block_hash_committed_at=committed_at,
            )
        )
        good_miners_uids.add(uid)
        bt.logging.debug(f"[_run_single_hash_phase] stored chain hash for uid {uid}")

    bad_miners_uids = set(all_uids_to_query) - good_miners_uids
    return good_miners_data, bad_miners_uids

