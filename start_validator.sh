#!/bin/bash

# Load environment variables from .env file & set defaults
set -a
source validator.env
set +a

# Login to Weights & Biases
if ! wandb login $WANDB_API_KEY; then
  echo "Failed to login to Weights & Biases with the provided API key."
fi

if [ -z "$CDS_API_KEY" ]; then
  echo "Please specify a CDS API KEY to login to CDS! You will not be able to download live ERA5 data."
  exit 1
fi

VALIDATOR_PROCESS_NAME="zeus_validator"

if pm2 list | grep -q "$VALIDATOR_PROCESS_NAME"; then
  echo "Process '$VALIDATOR_PROCESS_NAME' is already running. Deleting it..."
  pm2 delete $VALIDATOR_PROCESS_NAME
fi

echo "Starting validator process"
pm2 start neurons/validator.py --name $VALIDATOR_PROCESS_NAME -- \
  --netuid $NETUID \
  --subtensor.network $SUBTENSOR_NETWORK \
  --subtensor.chain_endpoint $SUBTENSOR_CHAIN_ENDPOINT \
  --wallet.name $WALLET_NAME \
  --wallet.hotkey $WALLET_HOTKEY \
  --axon.port $AXON_PORT \
  --proxy.port $PROXY_PORT \
  --logging.info

# synchronise the process list with the pm2 ecosystem file
pm2 save