<p align="center">
  <img src="static/zeus-icon.png" alt="Zeus Logo" width="150"/>
</p>
<h1 align="center">SN18: Zeus Environmental Forecasting Subnet<br><small>Ørpheus AI</small></h1>


![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

[Website](https://www.zeussubnet.com/) · [X](https://x.com/zeussubnet) · [LinkedIn](https://www.linkedin.com/company/orpheus-ai-nl) · [Discord](https://discord.com/invite/bittensor) · [huggingface](https://huggingface.co/orpheus-zeus)



## Quick Links

- [Mining Guide ⛏️](docs/Mining.md)
- [Incentive mechanism 🎁](docs/ScoringChallengesCalculatingWeights.ipynb)
- [Validator Guide 🔧](docs/Validating.md)

> [!IMPORTANT]
> If you are new to Bittensor, we recommend familiarizing yourself with the basics on the [Bittensor Website](https://bittensor.com/) before proceeding.



## Table of Contents

- [Who are we](#who-are-we)
- [What we do](#what-we-do)
  - [Overview](#overview)
  - [Purpose](#purpose)
  - [Features](#features)
- [Feedback and Contributions](#feedback-and-contributions)
- [License](#license)
- [Contacts](#contacts)



## Who are we

Ørpheus AI builds Zeus (Bittensor subnet 18): a decentralized weather forecasting network that turns competitive machine-learning forecasts into actionable atmospheric intelligence for energy markets. Find out more on our website!

## What we do
### Overview

The Zeus Subnet leverages advanced AI models within the Bittensor network to forecast environmental data on a decentralized, incentive-driven framework. The datasource for this subnet consists of ERA5 reanalysis data from the Climate Data Store (CDS) of the European Union's Earth observation programme (Copernicus). This comprises the largest global environmental dataset to date, containing hourly measurements from 1940 until the present across hundreds of variables. Validators issue global ERA5 forecasting challenges for four surface variables used heavily in energy trading: 2 m temperature, 100 m u- and v-components of wind, and surface solar radiation downwards. Miners compete to produce the best forecasts on the full Earth grid; validators score reveals against ERA5, keep rank history, and set subnet weights from verified performance.

### Purpose

Traditionally, environmental forecasting relies on physics-based numerical weather prediction (NWP). While this allows for very accurate predictions, it is also highly cost-ineffective, requiring large amounts of computing power for a single forecast. Furthermore, predictions are time expensive to obtain, since the simulation process of these NWP algorithms can take multiple hours to finish. Currently, there is a lot of ongoing research into the development of intelligent, data-driven algorithms for environmental prediction. Such algorithms can potentially be much faster, more accurate, at a fraction of the cost and carbon emissions. This subnet incentives the development of novel and groundbreaking architectures for environmental data prediction. Through the continuous evolution of this subnet, we are able to allow miners to tackle increasingly difficult problems over time.

### Features

- **Hourly 15-day forecasts for energy trading desks.** Zeus targets hourly forecasts out to 15 days for four surface variables that matter to power and gas books: 2 m temperature, 100 m eastward and northward wind, and surface solar radiation downwards. Challenges run on a global 0.25° ERA5 grid with both shorter and longer horizons so the market prices skill where desks actually use weather—not only at synoptic publish times of traditional NWP.
- **Dynamic attention shifting by client need.** Scoring and incentives can shift focus depending on commercial requirements—horizon, resolution, variables, and coordinates—through challenge weighting and geographic scalars (for example regional boosts over Europe and Germany). That lets pilots and products steer miner effort toward the regions and lead times that improve portfolio outcomes, rather than treating every grid cell as equally valuable.
- **Open-source verification of miner predictions via Hugging Face.** A public dataset and tutorial support trustless checks of forecast quality: see the [trustless verification tutorial](https://huggingface.co/datasets/orpheus-zeus/Zeus-API-forecasts/blob/main/trustless_verification_tutorial.ipynb). A selection of predictions has been available since 17 June, with a 7-day upload delay. Businesses can verify forecasts using the blockchain and the hashes stored in prediction metadata; the notebook walks through the end-to-end process.
- **Anti-gaming through on-chain commit-reveal.** Miners commit a hash of their compressed prediction to the blockchain before revealing the full forecast. That has two significant outcomes: it effectively stops relay mining on subnet 18 (copying another miner’s answer after seeing it), and it allows businesses to verify predictions as described above by matching revealed bytes to the on-chain commitment.
- **Epoch dynamics and dynamic burn.** Weight setting runs on a dedicated epoch schedule: a background setter refreshes burn data near epoch end, waits for the configured block window, respects chain weight rate limits, then emits weights from a fresh rank snapshot. **Dynamic burn** assigns a varying share of emissions to a burn UID each epoch (fetched from the performance API, with a configurable fallback). The goal is to deter **weight-copying validators**—operators that mirror another validator’s weights without doing the scoring work—by making copied weight vectors misaligned when burn changes every epoch, so honest validators that compute ranks and apply the current burn remain the ones emissions follow.
- **Incentive and validator infrastructure tied to verified skill.** Validators store challenge metadata, rank history, and per-challenge top miners in local SQLite databases. Subnet weights come from rolling averages of recent ranks (separate windows for short- and long-horizon challenges), with most of each challenge’s weight going to the best-ranked miner before variable and horizon weights are combined. Emissions stay aligned with forecast quality that has already been scored and hashed on-chain.
- **Decentralized R&D and continuous model evolution.** Miners run independent forecasting stacks on publicly available ERA5 data, near-unlimited history for training, and compete under the same challenge rules. Validators expand modalities and difficulty over time; the subnet absorbs new research so the competitive market keeps pushing toward faster, lower-cost, higher-accuracy alternatives to traditional NWP for energy-relevant variables.



## Feedback and Contributions

We welcome issues, pull requests, and discussion from miners, validators, researchers, and anyone building on Zeus. Start with the [Mining Guide](docs/Mining.md) and [Validator Guide](docs/Validating.md), and use the [incentive notebook](docs/ScoringChallengesCalculatingWeights.ipynb) when proposing scoring or docs changes. For real-time questions and updates, join the [Bittensor Discord](https://discord.com/invite/bittensor) and reach out to the Ørpheus AI team in the Zeus channels.

## License

This repository is licensed under the MIT License. See `[LICENSE](LICENSE)` for the full text.

```text
MIT License

Copyright (c) 2023 Opentensor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```



## Contacts

|     | Link |
|:---:|------|
| <span style="font-size:18px;vertical-align:middle;">🌐</span> | [zeussubnet.com](https://www.zeussubnet.com/) |
| <img src="https://api.iconify.design/simple-icons:x.svg" alt="X" width="18" height="18" style="vertical-align:middle;" /> | [x.com/zeussubnet](https://x.com/zeussubnet) |
| <img src="https://api.iconify.design/simple-icons:linkedin.svg?color=%230077B5" alt="LinkedIn" width="18" height="18" style="vertical-align:middle;" /> | [Orpheus AI on LinkedIn](https://www.linkedin.com/company/orpheus-ai-nl) |
| <img src="https://api.iconify.design/simple-icons:discord.svg?color=%235865F2" alt="Discord" width="18" height="18" style="vertical-align:middle;" /> | [Bittensor Discord (Zeus / Ørpheus AI channels)](https://discord.com/invite/bittensor) |
| <img src="https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.svg" alt="Hugging Face" width="18" height="18" style="vertical-align:middle; background:#FFD21C; border-radius:3px;"/>  | [huggingface.co/orpheus-zeus](https://huggingface.co/orpheus-zeus) |