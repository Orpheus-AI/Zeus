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
The Zeus Subnet leverages advanced AI models within the Bittensor network to forecast environmental data. This platform is engineered on a decentralized, incentive-driven framework to enhance trustworthiness and stimulate continuous technological advancement. The datasource for this subnet consists of ERA5 reanalysis data from the Climate Data Store (CDS) of the European Union's Earth observation programme (Copernicus). This comprises the largest global environmental dataset to date, containing hourly measurements from 1940 until the present across a multitude of hundreds of variables. Validators can stream data from this data source in real-time, allowing miners to be queried on terabytes of data.

**Purpose:**
Traditionally, environmental forecasting is achieved through physics-based numerical weather/environmental prediction (NWP). While this allows for very accurate predictions, it is also highly cost-ineffective, requiring large amounts of computing power for a single forecast. Furthermore, predictions are time expensive to obtain, since the simulation process of these NWP algorithms can take multiple hours to finish. Currently, there is a lot of ongoing research into the development of intelligent, data-driven algorithms for environmental prediction. Such algorithms can potentially be much faster, more accurate, at a fraction of the cost and carbon emissions. This subnet incentives the development of novel and groundbreaking architectures for environmental data prediction. Through the continuous evolution of this subnet, we are able to allow miners to tackle increasingly difficult problems over time.

**Features:**

- **Model Evolution:** Our platform continuously integrates the latest research and developments in AI to adapt to evolving generative techniques.

**Core Components:**

- **Miners:** Tasked with running forecasting algorithms that predict environmental variables at specific locations and timestamps.
  - **Research Integration:** We systematically update our detection models and methodologies in response to emerging academic research. Through the global ERA5 dataset, we are able to provide validators and miners with near infinite amounts of environmental data, which can also be used for training their models. All data is publicly available to everyone.
- **Validators:** Responsible for challenging miners with a subsets of environmental data and evaluating miner performance on heldout data. Validation uses a commit–reveal flow; see the [Validator Guide](docs/Validating.md#validator-phases) for phases and timing.
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
