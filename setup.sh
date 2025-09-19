#!/bin/bash

# check if sudo exists as it doesn't on RunPod
if command -v sudo 2>&1 >/dev/null; then
    PREFIX="sudo"
else
    PREFIX=""
fi

$PREFIX apt update -y
$PREFIX apt install -y \
    python3-pip \
    nano \
    libgl1 \
    npm

$PREFIX npm install -g pm2@6.0.5

# Make sure TCP processes have sufficient memory
# Doesn't work on Docker, so check if modifyable first
if [ ! -f /.dockerenv ]; then
    TCP_SETTINGS="
    # Custom Zeus TCP buffer settings for high-concurrency workloads
    net.ipv4.tcp_rmem=4096 131072 13107200
    net.ipv4.tcp_wmem=4096 16384 13107200
    "
    echo "$TCP_SETTINGS" | $PREFIX tee /etc/sysctl.d/99-zeus-subnet-tcp.conf > /dev/null
    $PREFIX sysctl --system
    echo "Increased TCP memory settings for optimal concurrency!"
fi

# install repository itself and CPU only torch
pip install --extra-index-url https://download.pytorch.org/whl/cpu -e . --use-pep517

# Create miner.env if it doesn't exist
if [ -f "miner.env" ]; then
    echo "File 'miner.env' already exists. Skipping creation."
else
    cat > miner.env << 'EOL'
# Subtensor Network Configuration:
NETUID=                                        # Network UID options: 18, 301
SUBTENSOR_NETWORK=                             # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=
                                               # Endpoints:
                                               # - wss://entrypoint-finney.opentensor.ai:443
                                               # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Miner Settings:
AXON_PORT=
BLACKLIST_FORCE_VALIDATOR_PERMIT=True          # Default setting to force validator permit for blacklisting
EOL
    echo "File 'miner.env' created."
fi

# Create validator.env if it doesn't exist
if [ -f "validator.env" ]; then
    echo "File 'validator.env' already exists. Skipping creation."
else
    cat > validator.env << 'EOL'
NETUID=                                         # Netuids: 18 (for finney), 301 (for testnet)
SUBTENSOR_NETWORK=                              # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=
                                                # Endpoints:
                                                # - wss://entrypoint-finney.opentensor.ai:443
                                                # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Validator Port Setting:
AXON_PORT=
PROXY_PORT=

# API Keys:
WANDB_API_KEY=                  # https://wandb.ai/authorize
CDS_API_KEY=                    # https://github.com/Orpheus-AI/Zeus/blob/main/docs/Validating.md#ecmwf
OPEN_METEO_API_KEY=             # https://open-meteo.com/en/pricing#plans (Cheapest one suffices)
PROXY_API_KEY=                  # Your Proxy API Key, you can generate it yourself

# Optional integrations
DISCORD_WEBHOOK=                # https://www.svix.com/resources/guides/how-to-make-webhook-discord/
EOL
    echo "File 'validator.env' created."
fi

echo "Environment setup completed successfully."