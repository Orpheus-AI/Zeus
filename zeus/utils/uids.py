import random
import bittensor as bt
import numpy as np
from typing import Set


def check_uid_availability(
    metagraph: "bt.metagraph.Metagraph",
    uid: int,
    vpermit_tao_limit: int,
    mainnet_uid: int,
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has
    less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
        mainnet_uid (int): The UID of the mainnet
    Returns:
        bool: True if uid is available, False otherwise
    """
    if not metagraph.axons[uid].is_serving:
        return False

    if (
        metagraph.netuid == mainnet_uid
        and metagraph.validator_permit[uid]
        and metagraph.S[uid] > vpermit_tao_limit
    ):
        return False
    return True


def get_available_uids(
    metagraph: "bt.metagraph.Metagraph",

    vpermit_tao_limit: int,
    mainnet_uid: int,
    exclude: Set[int] = None,
) -> np.ndarray:
    """Returns all available uids from the metagraph.
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        vpermit_tao_limit (int): Validator permit tao limit
        mainnet_uid (int): The UID of the mainnet
        exclude (List[int], optional): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): All available uids.
    Notes:
        - If there are no available non-excluded `uids`, returns an empty array.
    """
    if exclude is None:
        exclude = set()

    avail_uids = []
    for uid in range(metagraph.n.item()):
        available = check_uid_availability(
            metagraph, uid, vpermit_tao_limit, mainnet_uid
        )
        if available:
            avail_uids.append(uid)

    candidate_uids = [uid for uid in avail_uids if uid not in exclude]
    return candidate_uids

def get_random_uids(
    metagraph: "bt.metagraph.Metagraph",
    k: int,
    vpermit_tao_limit: int,
    mainnet_uid: int,
    exclude: Set[int] = None,
) -> np.ndarray:
    """Returns k available random uids from the metagraph.
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        k (int): Number of uids to return. Must be non-negative.
        vpermit_tao_limit (int): Validator permit tao limit
        mainnet_uid (int): The UID of the mainnet
        exclude (List[int], optional): List of uids to exclude from the random sampling.
    Returns:
        uids (np.ndarray): Randomly sampled available uids.
    Notes:
        - If `k` is larger than the number of available non-excluded `uids`,
          the function will return all available non-excluded `uids` in random order.
        - If there are no available non-excluded `uids`, returns an empty array.
    """
    candidate_uids = get_available_uids(metagraph, vpermit_tao_limit, mainnet_uid, exclude)
    sample_size = min(k, len(candidate_uids))
    if sample_size == 0:
        return np.array([], dtype=int)

    return np.array(random.sample(candidate_uids, sample_size))
