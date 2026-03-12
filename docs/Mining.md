# Miner Guide

## Table of Contents

1. [Installation 🔧](#installation)
2. [Registration ✍️](#registration)
3. [Setup ⚙️](#setup)
4. [Mining ⛏️](#mining)
   - [Input and desired output data](#input-and-desired-output-data)
   - [What to return in each phase](#what-to-return-in-each-phase)

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

_Note: For testnet tao, you can make requests in the [Bittensor Discord's "Requests for Testnet Tao" channel](https://discord.com/channels/799672011265015819/1190048018184011867)_

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
There are 4 forecast requests per day, sent at 00:00, 06:00, 12:00, and 18:00 UTC. Each request follows a fixed format:

- **Geographical coverage**: Always the entire Earth (full latitude and longitude range)
- **Time steps**: Always 49 time steps, covering from the current hour (t=0) to +48 hours (t=48) in 1-hour intervals
- **Step size**: Always 1 hour between prediction time steps

Each challenge focuses on a single variable. Currently supported variables include temperature two meters above the earth's surface and wind 100 meters u and v components. For an up-to-date list and their scoring weights, see the [constants](../zeus/validator/constants.py) file.

**Input Format:**
The validator sends you a request containing:

- **Bounding box coordinates**: `latitude_start`, `latitude_end`, `longitude_start`, `longitude_end` (always -90 to 90 for latitude, -180 to 179.75 for longitude)
- **Time range**: `start_time` and `end_time` (float timestamps in UTC, always rounded to the hour)
- **Number of time steps**: `requested_hours` (always 49)
- **Step size**: `step_size` (always 1 hour)
- **Variable**: `variable` (string identifying the ERA5 variable to predict, e.g., `"2m_temperature"`, `"100m_u_component_of_wind"`, `"100m_v_component_of_wind"`)

The geographical grid is generated from the bounding box with a resolution of 0.25 degrees, resulting in a fixed grid size of 721 × 1440 points (latitude × longitude) for the entire Earth.

**Scoring:**
You will be scored based on both the Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) between your predictions and the actual ground truth at those locations for the requested timepoints. The final score is the average of these two metrics: `(RMSE + MAE) / 2`. The actual ground truth are not yet known at the time you receive the challenge, so you will be scored in the future when these data become available (typically 7 days later).

Your goal is to minimize both RMSE and MAE, which will improve your ranking and subnet incentive. Scoring uses:

- **Competition ranking**: Miners are ranked based on their scores, with lower scores (better predictions) receiving better ranks
- **Latitude weighting**: Additional latitude-based weighting is applied to ensure fair evaluation across different regions

Miners with incorrect output shapes, non-finite values, or missing responses receive shape penalties.

> [!IMPORTANT]
> There are 4 requests per day at 00:00, 06:00, 12:00, and 18:00 UTC. Each request is always for the entire Earth (721 × 1440 grid points) and always for 49 time steps (from now to +48 hours in 1-hour intervals). Because the request format is fixed and predictable, miners can precompute forecasts ahead of time. Make sure to reformat your final output to the correct `(49, 721, 1440)` structure before compression.

### What to return in each phase

The validator sends two kinds of requests. Your miner must detect the synapse type and return the correct field in each case.

| Phase                                  | Request type                  | What you return                                                                                                                                                                                                                                                                                                                                                                    | Do not send                                                                                                                                                                        |
| -------------------------------------- | ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Commit (hash)**                      | `HashedTimePredictionSynapse` | Set **`synapse.hash`** to the commitment string: `sha256(compressed_bytes + hotkey_ss58.encode("utf-8")).hexdigest()`, where `compressed_bytes` is the **blosc2-compressed** bytes of your float16 prediction tensor `(49, 721, 1440)`. Use your wallet hotkey SS58 address as `hotkey_ss58`. See [zeus.utils.hash.prediction_hash](../zeus/utils/hash.py).                        | Do **not** set `predictions`; the validator only expects the hash in this phase.                                                                                                   |
| **Reveal / Scoring (full prediction)** | `TimePredictionSynapse`       | Set **`synapse.predictions`** to the **base64-encoded** string of the **same** blosc2-compressed prediction (the one you hashed). So: same tensor → `compress_prediction(tensor)` → `base64.b64encode(compressed).decode("ascii")`. The validator verifies that `sha256(compressed + hotkey)` equals the hash you committed earlier. If this verification fails you get a penalty. | Do not change or substitute a different prediction; it must match the hash or you are marked bad. Return the predictions only in the time intervals specified in the miner's code. |

**Summary:**

1. **Hash phase:** Compute your prediction tensor, compress it with blosc2, then return only the hash: `hash = sha256(compressed_bytes + hotkey_bytes).hexdigest()` in the `hash` field.
2. **Reveal / scoring phase:** Return the **same** compressed prediction as base64 in the `predictions` field. Use the same compression (and blosc2 version) as in the requirements so the validator can decode and verify.

The [default miner](../neurons/miner.py) implements `_forward_hashed` (commit) and `_forward_unhashed_predictions` (reveal/scoring);
The [protocol](../zeus/protocol.py) defines `HashedTimePredictionSynapse` and `TimePredictionSynapse`.
