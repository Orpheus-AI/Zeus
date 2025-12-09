#!/bin/bash

set -a
source miner.env
set +a

MINER_PROCESS_NAME="zeus_miner"


if pm2 list | grep -q "$MINER_PROCESS_NAME"; then
  echo "Process '$MINER_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $MINER_PROCESS_NAME
fi

pm2 start neurons/miner.py --name $MINER_PROCESS_NAME -- \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $AXON_PORT \
  --blacklist.force_validator_permit $BLACKLIST_FORCE_VALIDATOR_PERMIT \
  --weatherxm.api_key $WEATHERXM_API_KEY \
  --logging.info

# synchronise the process list with the pm2 ecosystem file
pm2 save