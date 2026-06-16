from dataclasses import dataclass, field
from typing import Dict, List, Optional

import bittensor as bt

from zeus.validator.constants import (
    ERA5_DATA_VARS,
    LONG_CHALLENGE,
    SHORT_CHALLENGE,
    MAINNET_UID,
)
from zeus.utils.uids import check_uid_availability, is_registered_after_release_zeus_v2

# Sorted variable names — deterministic order for binary packing.
VARIABLES_ORDER: List[str] = sorted(ERA5_DATA_VARS.keys())

RAW_MAX_BYTES = 128


@dataclass
class ChallengeCommitment:
    """Holds binary sha256 hashes per (variable × time_window) challenge.
    On-chain commitment of per-challenge prediction hashes.

    Chain constraints (Commitments pallet):
    - max 3 fields per set_commitment call
    - each field is Data::Raw, max Raw128 (128 bytes)

    Layout:
    Field 0 — LONG challenge hashes (0_360): concat of binary sha256 per variable, sorted
    Field 1 — SHORT challenge hashes (0_48): same layout
    Field 2 — reserved for future use

    Variables are always sorted alphabetically so both miner and validator
    agree on position → variable mapping.
    hashes: state_key -> sha256 hex string  (e.g. "2m_temperature@0_48" -> "ab12...")
    """

    hashes: Dict[str, str] = field(default_factory=dict)
    # Block number at which this commitment was set on chain (Registration.block).
    # Populated by read_commitment; None for locally-constructed commitments
    # that haven't been read back from chain.
    committed_at_block: Optional[int] = None

    def _pack_field(self, window: tuple[int, int]) -> bytes:
        """Pack hashes for one time window into binary (concat of 32-byte digests)."""
        parts = []
        for var in VARIABLES_ORDER:
            key = f"{var}@{window[0]}_{window[1]}"
            hex_hash = self.hashes.get(key)
            if hex_hash is None:
                raise ValueError(f"Missing hash for {key}")
            parts.append(bytes.fromhex(hex_hash)) # converts string hash to bytes
        data = b"".join(parts)
        if len(data) > RAW_MAX_BYTES:
            raise ValueError(
                f"Field for window {window} is {len(data)} bytes, max {RAW_MAX_BYTES}. "
                f"Too many variables ({len(VARIABLES_ORDER)})."
            )
        return data

    def to_fields(self) -> list[bytes]:
        """Return list of binary fields ready for set_commitment."""
        return [
            self._pack_field(LONG_CHALLENGE),
            self._pack_field(SHORT_CHALLENGE),
        ]

    @classmethod
    def from_fields(
        cls,
        fields: list[bytes],
        committed_at_block: Optional[int] = None,
    ) -> "ChallengeCommitment":
        """Reconstruct from on-chain binary fields."""
        hashes: Dict[str, str] = {}
        for field_data, window in zip(fields, [LONG_CHALLENGE, SHORT_CHALLENGE]):
            for i, var in enumerate(VARIABLES_ORDER):
                key = f"{var}@{window[0]}_{window[1]}"
                digest = field_data[i * 32 : (i + 1) * 32]
                hashes[key] = digest.hex()
        return cls(hashes=hashes, committed_at_block=committed_at_block)


def commit_to_chain(
    subtensor: "bt.subtensor",
    wallet,
    netuid: int,
    commitment: ChallengeCommitment,
) -> bool:
    """Submit commitment hashes to chain via set_commitment (2 × Raw128 fields)."""
    binary_fields = commitment.to_fields()

    fields = [{f"Raw{len(f)}": f} for f in binary_fields]

    call = subtensor.substrate.compose_call(
        call_module="Commitments",
        call_function="set_commitment",
        call_params={
            "netuid": netuid,
            "info": {"fields": [fields]},
        },
    )

    success, message = subtensor.sign_and_send_extrinsic(
        call=call,
        wallet=wallet,
        sign_with="hotkey",
        wait_for_inclusion=False,
        wait_for_finalization=True,
    )
    if not success:
        raise RuntimeError(f"set_commitment failed: {message}")
    return True


def read_all_miner_commitments(
    subtensor: "bt.subtensor",
    metagraph,
    netuid: int,
    block: int | None = None,
    current_block: int | None = None,
    vpermit_tao_limit: int | None = None,
) -> Dict[int, ChallengeCommitment]:
    """Read prediction hash commitments from all serving miners.

    Returns dict of miner_uid -> ChallengeCommitment.
    Skips validators (validator_permit) and miners without a valid commitment.
    """
    results: Dict[int, ChallengeCommitment] = {}
    for uid in range(metagraph.n.item()):
        is_miner = check_uid_availability(metagraph, uid = uid, vpermit_tao_limit=vpermit_tao_limit, mainnet_uid=netuid) if netuid == MAINNET_UID else True
        registered_after_v2 = is_registered_after_release_zeus_v2(metagraph.block_at_registration[uid], current_block)
        if not is_miner or not registered_after_v2:
            continue
        hotkey = metagraph.hotkeys[uid]
        try:
            commitment = read_commitment(subtensor, netuid, hotkey, block=block)
            results[uid] = commitment
        except Exception as e:
            bt.logging.error(f"Failed to read commitment for {uid} on block {block}: {e}")
            continue
    return results


def read_commitment(
    subtensor: "bt.subtensor",
    netuid: int,
    hotkey: str,
    block: int | None = None,
) -> ChallengeCommitment:
    """Read commitment from chain and reconstruct ChallengeCommitment."""
    result = subtensor.substrate.query(
        module="Commitments",
        storage_function="CommitmentOf",
        params=[netuid, hotkey],
        block_hash=subtensor.determine_block_hash(block),
    )
    if result is None:
        raise ValueError(f"No commitment found for {hotkey} on netuid {netuid}")

    # result can be ScaleType (has .value) or plain dict depending on substrate version
    data = result.value if hasattr(result, "value") else result
    info = data["info"]
    # Registration.block — block at which the commitment was set on chain.
    # Used downstream to detect stale commitments from previous cycles.
    #  CommitmentOfResponse has block what about CommitmentOf https://docs.learnbittensor.org/python-api/html/autoapi/bittensor/core/types/#bittensor.core.types.CommitmentOfResponse 
    committed_at_block = data.get("block") if isinstance(data, dict) else None

    binary_fields = []
    for field_item in info["fields"][0]: 
        for key, val in field_item.items():
            if not key.startswith("Raw"):
                continue
            # val can be: tuple of ints, bytes, hex string, or tuple-wrapped
            if isinstance(val, tuple) and len(val) == 1 and isinstance(val[0], tuple):
                val = val[0]
            if isinstance(val, (tuple, list)):
                binary_fields.append(bytes(val))
            elif isinstance(val, bytes):
                binary_fields.append(val)
            elif isinstance(val, str) and val.startswith("0x"):
                binary_fields.append(bytes.fromhex(val[2:]))
            else:
                binary_fields.append(val.encode())

    return ChallengeCommitment.from_fields(binary_fields, committed_at_block=committed_at_block)
