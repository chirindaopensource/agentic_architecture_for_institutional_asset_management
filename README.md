# **`README.md`**

# The Self-Driving Portfolio: Agentic Architecture for Institutional Asset Management

<!-- PROJECT SHIELDS -->
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Source](https://img.shields.io/badge/Source-ArXiv%20Preprint-B31B1B.svg)](https://arxiv.org/abs/2604.02279)
[![Year](https://img.shields.io/badge/Year-2026-purple)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Discipline](https://img.shields.io/badge/Discipline-Institutional%20Investment%20Governance-00529B)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Discipline](https://img.shields.io/badge/Discipline-Macro--Financial%20Econometrics-00529B)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Discipline](https://img.shields.io/badge/Discipline-Mathematical%20Finance-00529B)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Discipline](https://img.shields.io/badge/Discipline-Multi--Agent%20Systems%20(MAS)-00529B)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Data Sources](https://img.shields.io/badge/Data-Bloomberg%20%7C%20Apex%20Data%20%7C%20FRED-lightgrey)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Core Method](https://img.shields.io/badge/Method-LLM--as--Judge%20%7C%20Modified%20Borda%20Count-orange)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Core Method](https://img.shields.io/badge/Method-Sequential%20Convex%20Programming-orange)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Analysis](https://img.shields.io/badge/Analysis-Agentic%20Orchestration%20%7C%20ReAct-red)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Validation](https://img.shields.io/badge/Validation-Fail--Closed%20Schema%20Gating-green)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Robustness](https://img.shields.io/badge/Robustness-Ceteris%20Paribus%20Sensitivity-yellow)](https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Type Checking: mypy](https://img.shields.io/badge/type%20checking-mypy-blue)](http://mypy-lang.org/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-%23412991.svg?style=flat&logo=openai&logoColor=white)](https://openai.com/)
[![AutoGen](https://img.shields.io/badge/AutoGen-Multi--Agent%20Orchestration-blue.svg)](https://microsoft.github.io/autogen/)
[![SciPy](https://img.shields.io/badge/SciPy-Optimization-%238CAAE6.svg?style=flat&logo=scipy&logoColor=white)](https://scipy.org/)
[![JSON Schema](https://img.shields.io/badge/JSON%20Schema-Contracts-brightgreen.svg)](https://json-schema.org/)

**Repository:** `https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management`

**Owner:** 2026 Craig Chirinda (Open Source Projects)

This repository contains an **independent**, professional-grade Python implementation of the systems engineering workflow and quantitative methodologies described in the April 2026 ArXiv preprint entitled **"The Self-Driving Portfolio: Agentic Architecture for Institutional Asset Management"** by:

*   **Andrew Ang**
*   **Nazym Azimbayev**
*   **Andrey Kim**

The project provides a complete, end-to-end computational framework for operationalizing the industrialization of the fiduciary intelligence cycle. It delivers a modular, highly optimized pipeline that executes the entire Strategic Asset Allocation (SAA) workflow: from macro-regime classification and Capital Market Assumption (CMA) generation, through multi-method portfolio optimization, to structured multi-agent deliberation, and final Chief Investment Officer (CIO) ensemble synthesis.

## Table of Contents

- [Introduction](#introduction)
- [Theoretical Background](#theoretical-background)
- [Features](#features)
- [Methodology Implemented](#methodology-implemented)
- [Core Components (Notebook Structure)](#core-components-notebook-structure)
- [Key Callable: `orchestrate_fiduciary_intelligence_cycle`](#key-callable-orchestrate_fiduciary_intelligence_cycle)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Input Data Structure](#input-data-structure)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Project Structure](#project-structure)
- [Customization](#customization)
- [Contributing](#contributing)
- [Recommended Extensions](#recommended-extensions)
- [License](#license)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

## Introduction

This project provides a rigorous Python implementation of the architectural pattern outlined by Ang, Azimbayev, and Kim (2026). The core of this repository is the iPython Notebook `agentic_architecture_for_institutional_asset_management_draft.ipynb`, which contains a comprehensive suite of 37 orchestrated tasks to replicate and formalize the workflow.

The pipeline addresses a critical bottleneck in institutional asset management: the finite bandwidth of human decision-makers. By decomposing the SAA workflow into a multi-agent system (MAS), the architecture allows for the simultaneous evaluation of dozens of asset classes and portfolio construction methodologies. 

Crucially, the codebase operationalizes this paradigm by enforcing a strict **Separation of Concerns**: Large Language Models (LLMs) are utilized exclusively for judgment, interpretation, and narrative generation, while all arithmetic, linear algebra, and stochastic optimization are delegated to deterministic Python scripts. The entire system is bounded by the Investment Policy Statement (IPS), ensuring fiduciary-grade compliance at every node.

## Theoretical Background

The implemented methods formalize the workflow into a rigorous mathematical and systems architecture.

**1. IPS Constraint Computability:**
The Investment Policy Statement is translated into strict mathematical boundaries for the optimization solvers:
$$ \sigma_p = \sqrt{w^\top \Sigma w} \in [0.08, 0.12] $$
$$ TE = \sqrt{(w-w_b)^\top \Sigma (w-w_b)} \le 0.06 $$

**2. Capital Market Assumptions (CMA):**
Asset Class agents generate expected returns using multiple structural models, including the Black-Litterman equilibrium prior and the Grinold-Kroner building-block model:
$$ \pi = \delta \Sigma w_{mkt} $$
$$ \hat{\mu}_4 = y_{div} + y_{buyback} + g + \Delta v $$

**3. Adversarial Diversification:**
To prevent ensemble groupthink, an Adversarial Diversifier agent solves a non-convex Sequential Convex Programming (SCP) problem to maximize tracking variance relative to the ensemble centroid $\bar{w}$, subject to a Sharpe-ratio floor:
$$ \max_w (w-\bar{w})^\top \Sigma (w-\bar{w}) \quad \text{s.t.} \quad SR(w) \ge 0.75 \cdot SR(w_{\max SR}) $$

**4. Social Choice Aggregation:**
The Strategy Review phase aggregates decentralized agent judgments using a Modified Borda Count, blended with quantitative metrics via a regime-dependent weight $\alpha$:
$$ VoteScore_j = \sum_{a \neq j} \text{points}(a \to j) $$
$$ Composite_j = \alpha(\text{regime}) \cdot \widetilde{VoteScore}_j + (1-\alpha(\text{regime})) \cdot \widetilde{MetricScore}_j $$

**5. CIO Ensemble Synthesis:**
The apex CIO agent evaluates multiple ensemble techniques, selecting the optimal IPS-compliant portfolio based on a multi-criteria composite score:
$$ CIO\_Score_j = 0.25 SR_j + 0.15 IPS_j + 0.15 Div_j + 0.20 RF_j + 0.15 ER_j + 0.10 CU_j $$

Below is a diagram which summarizes the proposed approach:

<div align="center">
  <img src="https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management/blob/main/agentic_architecture_for_institutional_asset_management_ipo_main.png" alt="Agentic SAA Architecture" width="100%">
</div>

## Features

The provided iPython Notebook implements the full research pipeline, including:

-   **Zero-Trust Separation of Concerns:** LLMs are strictly prohibited from performing arithmetic. All mathematical operations are executed by deterministic Python tools injected via `functools.partial` closures to prevent prompt injection or hallucination.
-   **Fail-Closed Schema Gating:** Inter-agent communication is mediated entirely through the filesystem using strict JSON Schema (Draft 7) contracts. Any schema violation immediately halts the pipeline, preventing cascading state corruption.
-   **Deterministic FSM Orchestration:** The AutoGen `GroupChatManager` is overridden with a custom Finite State Machine (FSM) to guarantee that the multi-agent deliberation protocol executes in the exact sequence mandated by the architecture (Phases A through G).
-   **Robust Covariance Estimation:** Implements pairwise Ledoit-Wolf shrinkage and eigenvalue clipping to guarantee Positive Semi-Definite (PSD) covariance matrices, even when historical asset data contains missing observations or differing inception dates.
-   **Exact Euclidean Weight Projection:** Replaces heuristic weight clipping with a continuous knapsack bisection algorithm, guaranteeing that optimized portfolio weights sum exactly to 1.0 without floating-point drift.
-   **Cryptographic Provenance:** Computes SHA-256 hashes of all raw ingested dataframes and the configuration dictionary, sealing the entire execution state into an immutable, compressed archive for fiduciary auditability.

## Methodology Implemented

The core analytical steps directly implement the workflow from the source paper:

1.  **Initialization & Governance (Tasks 1-16):** Validates the `config.yaml`, enforces point-in-time data truncation, aligns temporal calendars, and generates the strict JSON output schemas.
2.  **Macro Regime Synthesis (Tasks 17, 20):** Computes expanding z-scores for economic indicators and maps them to a discrete regime (Expansion, Late-cycle, Recession, Recovery) using a softmax confidence function.
3.  **CMA Generation (Tasks 18, 21-24):** Executes 18 parallel ReAct loops. The "CMA Judge" evaluates method dispersion and applies regime-conditional logic to select final expected returns.
4.  **Portfolio Construction (Tasks 19, 25-26):** Dispatches 20 canonical optimizers (e.g., Risk Parity, Min CVaR) and the Adversarial Diversifier to generate candidate weight vectors.
5.  **Strategy Review (Tasks 27-30):** The CRO Agent generates risk reports. PC Agents engage in randomized peer review and submit structured Borda-count votes. The top-5 candidates are shortlisted via a diversity constraint and allowed to revise their proposals.
6.  **CIO Synthesis (Task 31):** The CIO Agent evaluates 7 ensemble methods, selects the highest-scoring IPS-compliant portfolio, and generates a natural-language Board Memo.
7.  **Robustness & Diagnostics (Tasks 32-37):** Computes ex-post backtest diagnostics, executes ceteris paribus data sensitivity trials, measures LLM variability (NLP drift), and packages the final reproducible archive.

## Core Components (Notebook Structure)

The notebook is structured as a logical pipeline with modular orchestrator functions for each of the 37 major tasks. All functions are self-contained, fully documented with strict type hints and comprehensive docstrings, and designed for professional-grade execution.

## Key Callable: `orchestrate_fiduciary_intelligence_cycle`

The project is designed around a single, top-level user-facing interface function:

-   **`orchestrate_fiduciary_intelligence_cycle`:** This apex orchestrator function runs the entire automated research pipeline from end-to-end. A single call to this function reproduces the entire workflow, managing static initialization, the baseline end-to-end run, the data and modeling sensitivity analysis, the LLM variability analysis, and the final cryptographic packaging of the entire execution state.

## Prerequisites

-   Python 3.10+
-   Core Python dependencies: `numpy`, `pandas`, `scipy`, `jsonschema`, `pyyaml`, `autogen`, `openai`.
-   Optional ML dependencies (for NLP drift metrics): `scikit-learn`.

## Installation

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management.git
    cd agentic_architecture_for_institutional_asset_management
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Python dependencies:**
    ```sh
    pip install numpy pandas scipy jsonschema pyyaml pyautogen openai scikit-learn
    ```

## Input Data Structure

The pipeline requires a configuration dictionary (`config.yaml`) and 10 distinct, high-fidelity pandas DataFrames:

1.  **`df_macro_raw`:** Monthly economic indicators (GDP, NFP, CPI, Brent, Fed Funds, FCI).
2.  **`df_benchmark_factors_raw`:** Daily Fama-French factors and risk-free rate.
3.  **`df_ohlcv_raw`:** Multi-indexed daily pricing data.
4.  **`df_fundamentals_raw`:** Multi-indexed valuation metrics (CAPE, P/E, Yields).
5.  **`df_signals_raw`:** Multi-indexed technical and sentiment signals (RSI, Momentum, Flows).
6.  **`df_total_return_raw`:** Multi-indexed strictly positive total return indices.
7.  **`df_benchmark_60_40_raw`:** Daily benchmark leg indices and weights.
8.  **`df_survey_cma_raw`:** Monthly analyst consensus return expectations.
9.  **`df_fixed_income_curves_spreads_raw`:** Monthly Treasury yields and credit spreads.
10. **`df_cpi_index_raw`:** Monthly CPI index levels.
11. **`universe_map`:** A dictionary linking the 18 canonical IPS asset class names to their respective tickers and categories.

## Usage

The notebook provides a complete, step-by-step guide. The primary workflow is to execute the final cell, which demonstrates how to load the configuration, generate synthetic data (or load proprietary data), and use the apex orchestrator to execute the pipeline:

```python
import pandas as pd
import numpy as np
import yaml
from typing import Dict, Any

# 1. Load the master configuration from the YAML file.
# (Assumes config.yaml is in the working directory)
def load_study_configuration(filepath: str = "config.yaml") -> Dict[str, Any]:
    if not isinstance(filepath, str):
        raise TypeError(f"filepath must be a string, got {type(filepath)}.")
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
        print(f"Successfully loaded configuration from {filepath}")
        return config
    except FileNotFoundError as e:
        print(f"CRITICAL ERROR: {filepath} not found in the working directory.")
        raise e

config = load_study_configuration("config.yaml")

# 2. Generate or load input dataframes and universe map
# (Example assumes raw_dataframes dict is populated per the Input Data Structure)
# raw_dataframes = load_proprietary_data() 

# Inject the universe_map into the configuration registries
if "REGISTRIES" not in config:
    config["REGISTRIES"] = {}
config["REGISTRIES"]["universe_map"] = universe_map

# 3. Execute the entire fiduciary intelligence cycle.
if __name__ == "__main__":
    
    class MockOpenAIClient:
        """A mock client to satisfy the type signature for the orchestrator."""
        pass
        
    mock_client = MockOpenAIClient()
    
    if config and raw_dataframes:
        print("\nInitiating The Self-Driving Portfolio Pipeline...")
        
        cycle_artifacts = orchestrate_fiduciary_intelligence_cycle(
            raw_config=config,
            raw_dataframes=raw_dataframes,
            client=mock_client,
            base_output_path="./run_outputs"
        )
        
        # 4. Access results
        print("\n" + "="*80)
        print("CYCLE EXECUTION COMPLETE")
        print("="*80)
        
        print("\n[Generated Artifact Paths]")
        for artifact_name, path in cycle_artifacts.items():
            if path:
                print(f"- {artifact_name}: {path}")
```

## Output Structure

The pipeline returns a master dictionary containing the absolute paths to the generated artifacts, serialized to disk under the unique `run_outputs/cycle_{timestamp}_{uuid}/` directory:

-   **`baseline_archive_path`**: The zipped archive of the baseline end-to-end run, containing all JSON artifacts, Markdown reports, and the `board_memo.md`.
-   **`data_robustness_report_path`**: The JSON report detailing the ceteris paribus sensitivity analysis (e.g., impact of altering the covariance window).
-   **`llm_variability_report_path`**: The JSON report detailing the protocol stability analysis (e.g., Spearman rank correlation of voting outcomes under different reasoning efforts).
-   **`master_deliverable_path`**: The cryptographically sealed `.zip` archive of the entire master cycle directory, including the `provenance.json` and `reproduction_report.md`.

## Project Structure

```
agentic_architecture_for_institutional_asset_management/
│
├── agentic_architecture_for_institutional_asset_management_draft.ipynb  # Main implementation notebook
├── agent_feature_creation.py                                            # Deterministic feature engineering and data augmentation
├── agent_llm_infrastructure.py                                          # Secure LLM interaction, configuration, and response parsing
├── agent_registry_and_skills_payload_creation.py                        # Agent operational boundaries and skills registry serialization
├── agent_tools.py                                                       # 70+ deterministic Python callables (computational substrate)
├── config.yaml                                                          # Master configuration file (IPS and methodological parameters)
├── requirements.txt                                                     # Python package dependencies
│
├── LICENSE                                                              # MIT Project License File
└── README.md                                                            # This file
```

## Customization

The pipeline is highly customizable via the `config.yaml` file. Users can modify study parameters such as:
-   **IPS Governance:** Adjust the `target_real_return_spread`, `volatility_band`, or `max_drawdown_limit` to reflect different institutional risk tolerances.
-   **LLM Invocation Policy:** Modify the `reasoning_effort` schedule (e.g., upgrading the CMA Judge to `xhigh`) or adjust `max_output_tokens`.
-   **Methodological Parameters:** Alter the `MACRO_SCORING_WEIGHTS`, the `COVARIANCE_ESTIMATION` window, or the `vote_metric_blend_alpha_schedule` for the Borda count aggregation.

## Contributing

Contributions are welcome. Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes. Adherence to PEP 8, strict type hinting, and the 1:1 inline comment-to-code-line ratio is required. All new tools must be registered with strict JSON Schema definitions.

## Recommended Extensions

Future extensions, as suggested by the architectural constraints, could include:
-   **Live API Integration:** Replacing the synthetic data generation with live feeds from Bloomberg (B-PIPE) or FactSet to enable real-time regime monitoring.
-   **Advanced Meta-Learning:** Expanding the Meta-Agent's capabilities to utilize reinforcement learning from human feedback (RLHF) based on Investment Committee overrides.
-   **Expanded Asset Universe:** Scaling the IPS universe beyond 18 assets to include granular private market proxies or individual equity selection agents.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Citation

If you use this code or the methodology in your research, please cite the original ArXiv preprint:

```bibtex
@misc{ang2026selfdriving,
  author = {Ang, Andrew and Azimbayev, Nazym and Kim, Andrey},
  title = {The Self-Driving Portfolio: Agentic Architecture for Institutional Asset Management},
  year = {2026},
  eprint = {2604.02279},
  archivePrefix = {arXiv},
  primaryClass = {q-fin.PM},
  url = {https://arxiv.org/abs/2604.02279}
}
```

For the implementation itself, you may cite this repository:
```
Chirinda, C. (2026). The Self-Driving Portfolio: A Formal Python Implementation.
GitHub repository: https://github.com/chirindaopensource/agentic_architecture_for_institutional_asset_management
```

## Acknowledgments

-   Credit to **Andrew Ang, Nazym Azimbayev, and Andrey Kim** for the foundational architectural design that forms the entire basis for this computational formalization.
-   This project is built upon the exceptional tools provided by the open-source community. Sincere thanks to the developers of the scientific Python ecosystem, particularly the **OpenAI**, **AutoGen**, **SciPy**, and **Pandas** contributors.


--

*This README was generated based on the structure and content of the `agentic_architecture_for_institutional_asset_management_draft.ipynb` notebook and follows best practices for research software documentation.*
