# Validator Guide

## Table of Contents

1. [Installation 🔧](#installation)
   - [Data 📊](#data)
   - [Registration ✍️](#registration)
2. [Validating ✅](#validating)
3. [Requirements 💻](#requirements)

## Before you proceed ⚠️

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml). 

## Installation

Download the repository and navigate to the folder.
```bash
git clone https://github.com/Orpheus-AI/ClimateAI.git && cd ClimateAI
```

We recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install), and create a virtual environment with this command:

```bash
conda create -y -n climate python=3.11
```

To activate your virtual environment, run `conda activate climate`. To deactivate, `conda deactivate`.

Install the remaining necessary requirements with the following chained command.

```bash
conda activate climate
chmod +x setup.sh 
./setup.sh
```

## Registration

To validate on our subnet, you must have a registered hotkey.

#### Mainnet

```bash
btcli s register --netuid [net_uid] --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid [testnet_uid] --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```


## Validating
Before launching your validator, make sure to create a file called `validator.env`. This file will not be tracked by git. 
You can use the sample below as a starting point, but make sure to replace **wallet_name**, **wallet_hotkey**, **axon_port**, **wandb_api_key** and **cds_api_key**.

```bash
NETUID=34                                      # Network User ID options: 34, 168
SUBTENSOR_NETWORK=finney                       # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                                # Endpoints:
                                                # - wss://entrypoint-finney.opentensor.ai:443
                                                # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Note: If you're using RunPod, you must select a port >= 70000 for symmetric mapping
# Validator Port Setting:
AXON_PORT=8092
PROXY_PORT=10913

# API Keys:
WANDB_API_KEY=your_wandb_api_key_here
CDS_API_KEY=your_cds_api_key_here
```
> [!IMPORTANT]
> In order to send miners challenges involving the latest ERA5 data, you need to provide a Copernicus CDS API key. These can be obtained from the [following website](https://cds.climate.copernicus.eu/how-to-api). Please first create an account or login, and then scroll down until you see the code-box with 'key' in it on the 'How to API'-page. 

If you don't have a W&B API key, please reach out to Ørpheus A.I. via Discord and we can provide one. Without W&B, miners will not be able to see their live scores, 
so we highly recommend enabling this.

Now you're ready to run your validator!

```bash
conda activate climate
./start_validator.sh
```

## Requirements
We strive to make validation as simple as possible on our subnet, aiming to minimise storage and hardware requirements for our validators.
All data send to miners is streamed live from Google's ERA5 storage, meaning you need **no local data storage** to validate on our subnet! As long as you have enough storage to install our standard Python dependencies (i.e. PyTorch), you can run our entire codebase. 

Data processing is done locally, but since this has been highly optimised, you will also **not need any GPU** or CUDA support. You will only need a decent CPU machine, where we recommend having at least 8GB of RAM. Since data is loaded over the internet, it is useful to have at least a moderately decent (>3MB/s) internet connection.

You are not required to provide API keys/funds for any external services. We would kindly ask you to link you validator to Weights and Biases, since this helps both miners and outside parties to obtain visualisation of the current state of the subnet. This can be done by specifying your API key in the ``validator.env` file. 

> [!TIP]
> Should you need any assistance with setting up the validator, W&B or anything else, please don't hesitate to reach out to the team at Ørpheus A.I. via Discord!