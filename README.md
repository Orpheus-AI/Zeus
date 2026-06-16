<p align="center">
  <img src="static/zeus-icon.png" alt="Zeus Logo" width="150"/>
</p>
<h1 align="center">SN18: Zeus Environmental Forecasting Subnet<br><small>Ørpheus AI</small></h1>


![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Welcome to the Zeus Subnet! This repository contains all the necessary information to get started, understand our subnet architecture, and contribute.


## Quick Links
- [Mining Guide ⛏️](docs/Mining.md)
- [Incentive mechanism 🎁](docs/ScoringChallengesCalculatingWeights.ipynb)
- [Validator Guide 🔧](docs/Validating.md)

> [!IMPORTANT]
> If you are new to Bittensor, we recommend familiarizing yourself with the basics on the [Bittensor Website](https://bittensor.com/) before proceeding.

## Predicting future environmental variables within a decentralized framework

**Overview:**
The Zeus Subnet leverages advanced AI models within the Bittensor network to forecast environmental data. This platform is engineered on a decentralized, incentive-driven framework to enhance trustworthiness and stimulate continuous technological advancement. The datasource for this subnet consists of ERA5 reanalysis data from the Climate Data Store (CDS) of the European Union's Earth observation programme (Copernicus). This comprises the largest global environmental dataset to date, containing hourly measurements from 1940 until the present across hundreds of variables. Validators currently issue global ERA5 forecasting challenges for 2 m temperature, 100 m wind components, and surface solar radiation downwards.

**Purpose:**
Traditionally, environmental forecasting is achieved through physics-based numerical weather/environmental prediction (NWP). While this allows for very accurate predictions, it is also highly cost-ineffective, requiring large amounts of computing power for a single forecast. Furthermore, predictions are time expensive to obtain, since the simulation process of these NWP algorithms can take multiple hours to finish. Currently, there is a lot of ongoing research into the development of intelligent, data-driven algorithms for environmental prediction. Such algorithms can potentially be much faster, more accurate, at a fraction of the cost and carbon emissions. This subnet incentives the development of novel and groundbreaking architectures for environmental data prediction. Through the continuous evolution of this subnet, we are able to allow miners to tackle increasingly difficult problems over time.

**Features:**

- **Short- and long-horizon forecasts:** Validators issue both **short-range** challenges (hourly steps from the current hour through **+48 hours**, 49 timesteps) and **long-range** challenges (the same grid through **+360 hours**, i.e. 15 days, 361 timesteps). Each ERA5 variable is evaluated on both horizons; see [constants](zeus/validator/constants.py) for windows and weights.
- **On-chain commit-reveal validation:** Miners commit prediction hashes to the Subtensor `Commitments` pallet, then later reveal the matching prediction over axon. Validators read commitments from chain and verify reveals against those hashes before using predictions for proxy/top-miner queries or final scoring.
- **Persistent validator state:** Validators store challenge metadata, rank history, and per-challenge top miners in local SQLite databases.
- **Rolling rank weights:** Validator weights are set from per-challenge rolling rank history. Short and long forecast windows use separate averaging windows, and the best miner in each available challenge receives most of that challenge's weight before challenge weights are combined.
- **Burn-aware weight setting:** A background weight setter handles epoch timing, chain rate limits, and burn amounts fetched from the performance API, with a configurable fallback burn percentage.
- **Model Evolution:** Our platform continuously integrates the latest research and developments in AI to adapt to evolving generative techniques.

**Core Components:**

- **Miners:** Tasked with running forecasting algorithms that predict environmental variables on the global ERA5 grid for the requested variable and horizon.
  - **Research Integration:** We systematically update our detection models and methodologies in response to emerging academic research. Through the global ERA5 dataset, we are able to provide validators and miners with near infinite amounts of environmental data, which can also be used for training their models. All data is publicly available to everyone.
- **Validators:** Responsible for challenging miners with global ERA5 forecasts, verifying commit-reveal responses, maintaining rank history, and setting subnet weights from recent performance. See the [Validator Guide](docs/Validating.md#validator-phases-forward-pass) for phases and timing.
  - **Resource Expansion:** We continuously add new environmental challenges and data modalities to our subnet in order to evolve our subnet and solve a multitude of distinct problems.

## Community
For real-time discussions, community support, and regular updates, <a href="https://discord.com/invite/bittensor">join the bittensor discord</a>. Connect with developers, researchers, and users to get the most out of the Zeus Subnet.

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2024 Opentensor Foundation

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
```
