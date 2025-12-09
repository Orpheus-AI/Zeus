# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

from typing import List
import bittensor as bt

from zeus.validator.constants import MAINNET_UID, VPERMIT_TAO_LIMIT

def is_miner_uid(
    metagraph: bt.Metagraph,
    uid: int,
    vpermit_tao_limit: int = VPERMIT_TAO_LIMIT,
) -> bool:
    """Check if uid is available. The UID should be available if it is serving and has
    less than vpermit_tao_limit stake
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        uid (int): uid to be checked
        vpermit_tao_limit (int): Validator permit tao limit
    Returns:
        bool: True if uid is available, False otherwise
    """
    if not metagraph.axons[uid].is_serving:
        return False

    if (
        metagraph.validator_permit[uid]
        and metagraph.S[uid] > vpermit_tao_limit
        and metagraph.netuid == MAINNET_UID
    ):
        return False
    return True

def get_uids(
    metagraph: bt.Metagraph,
    vpermit_tao_limit: int,
    miners: bool = True,
) -> List[int]:
    """Get all available uids.
    Args:
        metagraph (:obj: bt.metagraph.Metagraph): Metagraph object
        vpermit_tao_limit (int): Validator permit tao limit
        available (bool): Whether to get available uids only
    Returns:
        List[int]: List of available uids
    """
    return [
        uid for uid in range(metagraph.n) if
        (
            is_miner_uid(metagraph, uid, vpermit_tao_limit) if miners 
            else not is_miner_uid(metagraph, uid, vpermit_tao_limit)
        )
    ]