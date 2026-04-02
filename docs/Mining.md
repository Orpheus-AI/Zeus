# Miner Guide

## Table of Contents

1. [Installation 🔧](#installation)
2. [Registration ✍️](#registration)
3. [Setup ⚙️](#setup)
4. [Mining ⛏️](#mining)
5. [What to return in each phase](#what-to-return-in-each-phase)

## Before you proceed ⚠️

**Be aware of the minimum compute requirements** for our subnet, detailed in [Minimum compute YAML configuration](../min_compute.yml). A GPU is required for training (unless you want to wait weeks for training to complete), but is not required for inference while running a miner.

## Installation
_RunPod has not been updates yet : TODO_
> [!TIP]
> If you are using RunPod, you can use our [dedicated template](https://runpod.io/console/deploy?template=x2lktx2xex&ref=97t9kcqz) which comes pre-installed with all required dependencies! Even without RunPod the [Docker image](https://hub.docker.com/repository/docker/ericorpheus/zeus/) behind this template might still work for your usecase. If you are using this template/image, you can skip all steps below except for cloning.

Download the repository and navigate to the folder.
```bash
git clone https://github.com/Orpheus-AI/Zeus.git && cd Zeus
```

We recommend using a Conda virtual environment to install the necessary Python packages.<br>
You can set up Conda with this [quick command-line install](https://docs.anaconda.com/free/miniconda/#quick-command-line-install). Note that after you run the last commands in the miniconda setup process, you'll be prompted to start a new shell session to complete the initialization. 

With miniconda installed, you can create a virtual environment with this command:

```bash
conda create -y -n zeus python=3.11
```

To activate your virtual environment, run `conda activate zeus`. To deactivate, `conda deactivate`.

Install the remaining necessary requirements with the following chained command. This may take a few minutes to complete.

```bash
conda activate zeus
chmod +x setup.sh 
./setup.sh
```


## Registration

To mine on our subnet, you must have a registered hotkey.

*Note: For testnet tao, you can make requests in the [Bittensor Discord's "Requests for Testnet Tao" channel](https://discord.com/channels/799672011265015819/1190048018184011867)*

To reduce the risk of deregistration due to technical issues or a poor performing model, we recommend the following:
1. Test your miner on testnet before you start mining on mainnet.
2. Before registering your hotkey on mainnet, make sure your port is open by running `curl your_ip:your_port`
3. If you've trained a custom model, test it's performance by deploying to testnet. 


#### Mainnet

```bash
btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 301 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```

## Setup
Before launching your miner, make sure to create a file called `miner.env`. This file will not be tracked by git. 
You can use the sample below as a starting point, but make sure to replace **wallet_name**, **wallet_hotkey**, and **axon_port**.


```bash
# Subtensor Network Configuration:
NETUID=18                                      # Network User ID options: 18,301
SUBTENSOR_NETWORK=finney                       # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
                                               # Endpoints:
                                               # - wss://entrypoint-finney.opentensor.ai:443
                                               # - wss://test.finney.opentensor.ai:443/

# Wallet Configuration:
WALLET_NAME=default
WALLET_HOTKEY=default

# Miner Settings:
AXON_PORT=
BLACKLIST_FORCE_VALIDATOR_PERMIT=True          # Default setting to force validator permit for blacklisting
```

## Mining
Now you're ready to run your miner!

```bash
conda activate zeus
./start_miner.sh
```


### Input and desired output data
The datasource for this subnet consists of ERA5 reanalysis data from the Climate Data Store (CDS) of the European Union's Earth observation programme (Copernicus). This comprises the largest global environmental dataset to date, containing hourly measurements across a multitude of variables. 

**Request Schedule:**
There are 4 forecast rounds per day, anchored at 00:30, 06:30, 12:30, and 18:30 UTC. In each round the validator may issue **multiple challenges**: one per combination of **ERA5 variable** and **forecast horizon** (see [constants](../zeus/validator/constants.py)). Geography and step size are the same for all challenges:

- **Geographical coverage**: Always the entire Earth (full latitude and longitude range)
- **Step size**: Always 1 hour between prediction time steps
- **Forecast horizons** (identified by `requested_hours`, the length of the time dimension):
  - **Short-term:** 49 hourly steps from the current hour (t=0) through **+48 hours** (`requested_hours == 49`).
  - **Long-term:** 361 hourly steps from the current hour through **+360 hours** (15 days, `requested_hours == 361`).

Each challenge focuses on a single variable. Supported variables and weights (each split across both horizons) are defined in [constants](../zeus/validator/constants.py) (e.g. 2 m temperature, 100 m wind components, surface solar radiation downwards).

**Input Format:**
The validator sends you a request containing:
- **Bounding box coordinates**: `latitude_start`, `latitude_end`, `longitude_start`, `longitude_end` (always -90 to 90 for latitude, -180 to 179.75 for longitude)
- **Time range**: `start_time` and `end_time` (float timestamps in UTC, aligned to the horizon)
- **Number of time steps**: `requested_hours` — **49** for short-term or **361** for long-term
- **Step size**: `step_size` (always 1 hour)
- **Variable**: `variable` (string identifying the ERA5 variable to predict)

The geographical grid is generated from the bounding box with a resolution of 0.25 degrees, resulting in a fixed grid size of 721 × 1440 points (latitude × longitude) for the entire Earth.

**Scoring:**
You will be scored based on both the Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) between your predictions and the actual ground truth at those locations for the requested timepoints. The final score is the average of these two metrics: `(RMSE + MAE) / 2`. Ground truth is not available at request time; scoring runs once ERA5 covers **every** timestep in that challenge’s window. Short horizons therefore tend to be scored sooner than the 15-day long horizon.

Your goal is to minimize both RMSE and MAE, which will improve your ranking and subnet incentive. Scoring uses:
- **Competition ranking**: Miners are ranked based on their scores, with lower scores (better predictions) receiving better ranks
- **Latitude weighting**: Additional latitude-based weighting is applied to ensure fair evaluation across different regions

Miners with incorrect output shapes, non-finite values, or missing responses receive shape penalties.

> [!IMPORTANT]
> There are 4 scheduling anchors per day at 00:30, 06:30, 12:30, and 18:30 UTC. Each challenge is always for the entire Earth (721 × 1440 grid points) and either **49** or **361** hourly steps. Compress the float16 array with the same layout: `(requested_hours, 721, 1440)`.

### What to return in each phase

The validator sends two kinds of requests. Your miner must detect the synapse type and return the correct field in each case.

| Phase | Request type | What you return | Do not send |
|-------|----------------|-----------------|-------------|
| **Commit (hash)** | `HashedTimePredictionSynapse` | Set **`synapse.hash`** to the commitment string: `sha256(compressed_bytes + hotkey_ss58.encode("utf-8")).hexdigest()`, where `compressed_bytes` is the **blosc2-compressed** bytes of your float16 prediction tensor **`(synapse.requested_hours, 721, 1440)`** (49 or 361). Use your wallet hotkey SS58 address as `hotkey_ss58`. See [zeus.utils.hash.prediction_hash](../zeus/utils/hash.py). | Do **not** set `predictions`; the validator only expects the hash in this phase. |
| **Reveal / Scoring (full prediction)** | `TimePredictionSynapse`  | Set **`synapse.predictions`** to the **base64-encoded** string of the **same** blosc2-compressed prediction (the one you hashed). So: same tensor → `compress_prediction(tensor)` → `base64.b64encode(compressed).decode("ascii")`. The validator verifies that `sha256(compressed + hotkey)` equals the hash you committed earlier. If this verification fails you get a penalty. | Do not change or substitute a different prediction; it must match the hash or you are marked bad. Honor `requested_hours` and time fields from the synapse.|


**Summary:**

1. **Hash phase:** Compute your prediction tensor, compress it with blosc2, then return only the hash: `hash = sha256(compressed_bytes + hotkey_bytes).hexdigest()` in the `hash` field. 
2. **Reveal / scoring phase:** Return the **same** compressed prediction as base64 in the `predictions` field. Use the same compression (and blosc2 version) as in the requirements so the validator can decode and verify.

The [default miner](../neurons/miner.py) implements `_forward_hashed` (commit) and `_forward_unhashed_predictions` (reveal/scoring); 
The [protocol](../zeus/protocol.py) defines `HashedTimePredictionSynapse` and `TimePredictionSynapse`.



