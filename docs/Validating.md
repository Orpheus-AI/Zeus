# Validator Guide

## Table of Contents

1. [Installation 🔧](#installation)
   - [Registration ✍️](#registration)
2. [Validating ✅](#validating)
   - [ECMWF 🌎](#ecmwf)
   - [Validator phases](#validator-phases-forward-pass)
   - [Validator state and ranking storage](#validator-state-and-ranking-storage)
   - [Weight setting and burn](#weight-setting-and-burn)
3. [Hardware requirements 💻](#hardware-requirements)

## Before you proceed ⚠️

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

## Installation
> [!TIP]

The instructions below are specifically tailored for AWS.
1. Deploy an EC2 instance.
It should have at least **4 vCPUs** and **16GB RAM** and **25GBit network performance**. The recommended choice is the `m5n.xlarge` (or better).
To simplify setup, choose the 'Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Amazon Linux 2023)' AMI. We will not use the GPU, but do use a lot of its other pre-installed libraries.
Set its storage to at least **80GB**.

2. Install a non-ancient version of Python: 
```bash
 sudo dnf install python3.11 -y
```
3. Create a virtual environment and activate it
```bash
 python3.11 -m venv zeus-venv && source zeus-venv/bin/activate
```

4. Download the repository and navigate to the folder.
```bash
git clone https://github.com/Orpheus-AI/Zeus.git && cd Zeus
```

5. Install the necessary requirements with the following script (make sure zeus-venv is active!)
```bash
./setup.sh
```

## Registration

To validate on our subnet, you must have a registered hotkey.

#### Mainnet

```bash
btcli s register --netuid 18 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network finney
```

#### Testnet

```bash
btcli s register --netuid 301 --wallet.name [wallet_name] --wallet.hotkey [wallet.hotkey] --subtensor.network test
```


## Validating
Before launching your validator, make sure to create a file called `validator.env`. This file will not be tracked by git. 
You can use the sample below as a starting point, but make sure to replace **wallet_name**, **wallet_hotkey**, **axon_port**, **proxy_port**, and **cds_api_key**.

```bash
NETUID=18                                       # Network User ID options: 18,301
SUBTENSOR_NETWORK=finney                        # Networks: finney, test, local
SUBTENSOR_CHAIN_ENDPOINT=wss://entrypoint-finney.opentensor.ai:443
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
CDS_API_KEY=                    # https://github.com/Orpheus-AI/Zeus/blob/main/docs/Validating.md#ecmwf
PROXY_API_KEY=                  # Your Proxy API Key, you can generate it yourself

# Optional integrations
DISCORD_WEBHOOK=                # https://www.svix.com/resources/guides/how-to-make-webhook-discord/
```

### ECMWF
> [!IMPORTANT]
> In order to send miners challenges involving the latest ERA5 data, you need to provide a Copernicus CDS API key. The steps below explain how to obtain this key. If you encounter any difficulty in the process, please let us know and we will create an account for you.

1. Go to the official [CDS website](https://cds.climate.copernicus.eu/how-to-api).
2. Click on the "Login - Register" button in the top right of the page.
3. Click the "I understand" button on the screen that pops up to be redirected to the next page.
4. Unless you already have an account, click the blue "Register" button in the gray box below the login page.
5. Fill in your details and complete the Captcha. Keep in mind that you need to be able to access the email address used. Then click the blue register button.
6. Go to your email and click the link in the email from `servicedesk@ecmwf.int`. You should be taken to a page to enter more information. If not, go the link from step 1 and try to login instead of registering. 
7. Fill in the extra details (they are not checked at all and don't have to be accurate) and accept the statements. Click the "activate your profile" button.
8. You should be redirected back to the [CDS website](https://cds.climate.copernicus.eu/how-to-api). Scroll down to the section labeled '1. Setup the CDS API personal access token.' You will find a code block containing your API key. **Crucially, copy only the value of the 'key' portion of this code block into your `validator.env` file.**
    For example, the code block will resemble the following:

    ```
    url: https://cds.climate.copernicus.eu/api
    key: YOUR_API_KEY_THAT_SHOULD_BE_COPIED
    ```

    **Only copy the string following 'key:' (i.e., `YOUR_API_KEY_THAT_SHOULD_BE_COPIED`) into your environment file.**
9. Please ensure you accept the [terms for downloading ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences), as this is the dataset used for validator queries.

Now you're ready to run your validator!

```bash
source zeus-venv/bin/activate # from root folder, not GitHub repo
cd Zeus/
./start_validator.sh
```

The startup script loads `validator.env` and starts the validator under pm2. If you run the validator manually, use `neurons/validator.py` with the same wallet, subtensor, and netuid values from `validator.env`.

> [!NOTE]
> If you experience problems with v-trust and your validator can't connect to the performance endpoint [constants -> PERFORMANCE_DATABASE_URL](../zeus/validator/constants.py), then please reach out to the team at Ørpheus A.I. via Discord.

### Validator phases (forward pass) 



The validator uses an **on-chain commit-reveal** flow so miners cannot copy others' answers. Miners publish prediction hashes through the Subtensor `Commitments` pallet, and validators read those chain commitments before requesting full predictions. The logic is in [forward.py](../zeus/validator/forward.py) and [hash_phase.py](../zeus/validator/hash_phase.py). Each forward pass can run one or more of these phases, depending on the current time and chain state.

**Challenges:** Each forward samples one challenge per **ERA5 variable × time window**. There are two windows: **short-term** (49 hourly steps, 0–48 h from the challenge start) and **long-term** (361 steps, 0–360 h / 15 days). Registry and per-window dendrite limits are in [constants.py](../zeus/validator/constants.py) (`TIME_WINDOWS_PER_CHALLENGE`, `TOPK_PREDICTION_SETTINGS_PER_WINDOW`, `SCORING_PREDICTION_SETTINGS_PER_WINDOW`, `CHALLENGE_REGISTRY`). Weight aggregation uses separate rolling windows for short vs long challenges (`--neuron.score_time_window_short`, `--neuron.score_time_window_long` in [config](../zeus/utils/config.py)).

| Phase | When it runs | What the validator does |
|-------|----------------|--------------------------|
| **1. Hash commit read** | At 00:45, 06:45, 12:45, 18:45 UTC (every 6 hours, subject to a minimum interval between requests). Miners should have committed before the hash window closes, currently minute `:45`. | Reads **hash commitments** from the Subtensor `Commitments` pallet for all eligible miners instead of requesting hashes over axon. Each commitment is unpacked into per-challenge hashes, checked for freshness against the expected challenge block, and stored in SQLite with the block it was committed at. Missing, empty, or stale commitments are recorded as bad. Implemented in `run_all_hash_phases()`. |
| **2. Query best (long-horizon reveal)** | At least one hour after the hash phase, in the same 6-hour window (when `hour % 6 != 0`). | Queries only a **subset** of miners for 15-day challenges using `TimePredictionSynapse`: miners in the historical top set that also have an accepted on-chain hash, with a random fallback when no top miner is available. Wind-component challenges use the union of top miners from the u and v components so wind speed can come from consistent miners. Revealed predictions are verified against committed hashes; failed reveals are marked bad. Implemented in `run_initial_prediction_top_k_phases()`. |
| **3. Scoring** | For challenges whose ground truth is ready, runs the **final prediction phase**. | Requests full predictions from miners who committed, verifies the hash reveal, computes RMSE/MAE vs ground truth, applies shape/collusion penalties, writes rank history, refreshes top miners, and logs performance data. Implemented via `database.score_and_prune()` and `run_final_prediction_phase()`. |

Timing is controlled by [schedule_time.py](../zeus/utils/schedule_time.py) (`is_hash_commit_time`, `is_query_best_time`).

### Validator state and ranking storage

Validator state is now stored in SQLite instead of only in JSON:

| Data | Default location | Purpose |
|------|------------------|---------|
| Challenge metadata | `~/.cache/zeus/challenges_v2.db` | Stores pending challenges, on-chain prediction hashes, committed block numbers, bad miners, reveal/scoring status, and cleanup state. |
| Rank history and top miners | `~/.cache/zeus/rank_history.db` | Stores per-challenge miner ranks and ordered top-miner lists used for top-k querying and weight calculation. |
| ERA5 cache | `~/.cache/zeus/era5` | Stores downloaded ERA5 data needed for scoring. |
| Burn amounts | `~/.cache/zeus/burn_amounts.json` | Caches per-epoch burn amounts fetched from the performance API. |

On startup, the validator initializes the rank-history database and migrates older `state_v3.json` data when available. Rank history is kept for up to `RANK_HISTORY_PRUNE_DAYS`, and miners that are absent for `RANK_HISTORY_ALLOWED_ABSENCE` consecutive challenge results can have their per-challenge rank history removed. The top-miner list is maintained per challenge and is used by the query-best reveal phase.

Only eligible miners are considered for querying and weighting. Eligible miners must be serving, must not exceed the validator-permit stake limit on mainnet, and must be registered after the Zeus v2 cutoff used by `zeus.utils.uids`.

### Weight setting and burn

Weight setting now runs in a dedicated background `WeightSetter` thread instead of being performed inline with every forward pass. The thread:

- Refreshes burn data near the end of each epoch.
- Waits until the configured block window before epoch end before attempting `set_weights`.
- Respects the subnet weight rate limit from chain hyperparameters.
- Loads a single SQLite rank snapshot for all challenge keys when calculating weights.

Weights are calculated from rolling average ranks. The default windows are `--neuron.score_time_window_short 4` for short-term challenges and `--neuron.score_time_window_long 8` for long-term challenges. For each available challenge, the best ranked eligible miner receives most of that challenge's weight (`PERCENTAGE_GOING_TO_WINNER`), the remainder is distributed logarithmically, and challenge weights are combined using the variable and time-window weights in [constants.py](../zeus/validator/constants.py). If there is no rank history yet, the validator temporarily rewards miners that have responded recently.

Burn is skipped on testnet. On mainnet, the validator fetches per-epoch burn entries from the performance API and applies them before emitting weights. If burn data cannot be fetched, it falls back to `--neuron.burn_percent` (default `0.5`) and can send a Discord notification if `DISCORD_WEBHOOK` is configured.

## Hardware requirements
We strive to make validation as simple as possible on our subnet, aiming to minimise storage and hardware requirements for our validators.
Only a couple days of environmental data need to be stored at a time, which take around 1GB of storage. Miner hashes of their predictions are also temporarily stored in an SQLite database for challenges where the ground-truth is not yet known, which can reach around 15GB. 
Therefore we recommend a total storage of around 80GB, allowing for ample space to install all dependencies and store the miner predictions.

Data processing is done locally, but since this has been highly optimised, you will also **not need any GPU** or CUDA support. You will only need a decent CPU machine, where we recommend having at least 16GB of RAM. Since data is loaded over the internet, it is useful to have at least a moderately decent (>5GBit/s) internet connection.

> [!TIP]
> Should you need any assistance with setting up the validator or anything else, please don't hesitate to reach out to the team at Ørpheus A.I. via Discord!
