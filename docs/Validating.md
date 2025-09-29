# Validator Guide

## Table of Contents

1. [Installation 🔧](#installation)
   - [Registration ✍️](#registration)
2. [Validating ✅](#validating)
   - [ECMWF 🌎](#ecmwf)
3. [Requirements 💻](#requirements)

## Before you proceed ⚠️

**Ensure you are running Subtensor locally** to minimize outages and improve performance. See [Run a Subtensor Node Locally](https://github.com/opentensor/subtensor/blob/main/docs/running-subtensor-locally.md#compiling-your-own-binary).

## Installation
> [!TIP]
> We highly recommend using AWS for Validators. Since response speed is part of the incentive and we value a subnet which brings value globally, we ideally want to split validators geographically.
> Please reach out to us and will help select the ideal location for you. 

The instructions below are specifically tailored for AWS.
1. Deploy an EC2 instance.
It should have at least **4 vCPUs** and **16GB RAM** and **25GBit network performance**. The recommend choice is the `m5n.xlarge` (or better)
To simplify setup, choose the 'Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Amazon Linux 2023)' AMI. We will not use the GPU, but do use a lot of its other pre-installed libraries.
Set its storage to at least **60GB** (80 is recommended).

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
You can use the sample below as a starting point, but make sure to replace **wallet_name**, **wallet_hotkey**, **axon_port**, **wandb_api_key** and **cds_api_key**.

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
WANDB_API_KEY=                  # https://wandb.ai/authorize
CDS_API_KEY=                    # https://github.com/Orpheus-AI/Zeus/blob/main/docs/Validating.md#ecmwf
OPEN_METEO_API_KEY=             # https://open-meteo.com/en/pricing#plans (Cheapest one suffices)
PROXY_API_KEY=                  # Your Proxy API Key, you can generate it yourself

# Optional integrations
DISCORD_WEBHOOK=                # https://www.svix.com/resources/guides/how-to-make-webhook-discord/
```
If you don't have a W&B API key, please reach out to Ørpheus A.I. via Discord. Without W&B, miners will not be able to see their live scores, 
so we highly recommend enabling this.

### 1. OpenMeteo
In order to validate you need an [OpenMeteo API key](https://open-meteo.com/en/pricing) for the API standard plan. After purchasing, it will be send to your email address!
So look out for an email from info@open-meteo.com. Please enter the API key in the validator.env file, under the key `OPEN_METEO_API_KEY=mykey`.

### 2. ECMWF
> [!IMPORTANT]
> In order to send miners challenges involving the latest ERA5 data, you need to provide a Copernicus CDS API key. The steps below explain how to obtain this key. If you encounter any difficulty in the process, please let us know and we will create an account for you.

1. Go the the official [CDS website](https://cds.climate.copernicus.eu/how-to-api).
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
pm2 start run_neuron.py -- --validator 
```

- Auto updates are enabled by default. To disable, run with `--no-auto-updates`.
- Self-healing restarts are disabled by default (every 3 hours). To enable, run with `--self-heal`.

## Hardware requirements
We strive to make validation as simple as possible on our subnet, aiming to minimise storage and hardware requirements for our validators.
Only a couple days of environmental data need to be stored at a time, which take around 1GB of storage. Miner predictions are also temporarily stored in an SQLite database for challenges where the ground-truth is not yet known, which can reach around 15GB. 
Therefore we recommend a total storage of around 80GB, allowing for ample space to install all dependencies and store the miner predictions.

Data processing is done locally, but since this has been highly optimised, you will also **not need any GPU** or CUDA support. You will only need a decent CPU machine, where we recommend having at least 16GB of RAM. Since data is loaded over the internet, it is useful to have at least a moderately decent (>5GBit/s) internet connection.

> [!TIP]
> Should you need any assistance with setting up the validator, W&B or anything else, please don't hesitate to reach out to the team at Ørpheus A.I. via Discord!
