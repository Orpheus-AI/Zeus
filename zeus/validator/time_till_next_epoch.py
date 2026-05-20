def current_tempo_bounds(current_block: int, netuid: int, tempo: int) -> tuple[int, int]:
    current_end = current_block + time_till_next_epoch(current_block, netuid, tempo) - 1 # -1 because we want to get the tempo and not the epoch
    current_start = current_end - tempo
    
    return current_start, current_end


def time_till_next_epoch(current_block: int, netuid: int, tempo: int) -> int:
    """
    Calculates the number of blocks until the next epoch.
    Check file calculate_next_epoch_time.md for more details.
    Args:
        current_block: The current block.
        netuid: The subnet ID.
        tempo: The tempo of the subnet.
    Returns:
        The number of blocks until the next epoch.

    """
    interval = tempo + 1

    blocks_until = (
        interval - ((current_block + netuid + 1) % interval)
    ) % interval

    return blocks_until
