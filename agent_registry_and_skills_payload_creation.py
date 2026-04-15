# =============================================================================
# SELF-DRIVING PORTFOLIO: AGENT SKILLS REGISTRY GENERATOR
# =============================================================================
# Implements the foundational skill taxonomy for the agentic Strategic Asset
# Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# This module defines the strict operational boundaries for all 15 agent types
# by assigning exactly 6 to 9 discrete "Skills" to each agent. It then
# serializes these skills as markdown contracts to the working drive, ensuring
# that LLMs have explicit, auditable instructions for tool invocation.
#
# All functions are purely deterministic Python callables.
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Union

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# Initialise a named logger so callers can configure log levels independently
logger: logging.Logger = logging.getLogger(__name__)


def generate_and_persist_agent_skills_registry(
    base_dir: Union[str, Path]
) -> Dict[str, List[str]]:
    """
    Constructs the definitive registry mapping each agent in the Self-Driving
    Portfolio architecture to its 6-9 required 'Skills', and serializes each
    skill as a markdown contract file to the specified working directory.

    This function enforces the separation of concerns by generating the exact
    documentation the LLM will read to understand its boundaries and the
    Python tools it is permitted to invoke.

    Parameters
    ----------
    base_dir : Union[str, Path]
        The root directory path where the 'skills' folder hierarchy will be
        generated.

    Returns
    -------
    Dict[str, List[str]]
        The dictionary mapping agent slugs to their respective list of skill slugs.

    Raises
    ------
    TypeError
        If ``base_dir`` is not a string or pathlib.Path.
    PermissionError
        If the script lacks write access to the specified ``base_dir``.
    OSError
        If directory or file creation fails due to system-level constraints.
    """
    # ------------------------------------------------------------------
    # Input validation: type check for base_dir
    # ------------------------------------------------------------------
    if not isinstance(base_dir, (str, Path)):
        raise TypeError(
            f"base_dir must be a str or pathlib.Path, "
            f"got {type(base_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Convert base_dir to a resolved pathlib.Path object for robust routing
    # ------------------------------------------------------------------
    root_path: Path = Path(base_dir).resolve()

    # ------------------------------------------------------------------
    # Define the strict, immutable mapping of agents to their 6-9 skills.
    # This dictionary serves as the absolute truth for agent capabilities.
    # ------------------------------------------------------------------
    registry: Dict[str, List[str]] = {
        # Macro Agent (7 Skills): Synthesizes economic indicators into regimes.
        "macro_agent": [
            "macro-data-ingestion",
            "growth-momentum-scoring",
            "inflation-trajectory-scoring",
            "monetary-policy-scoring",
            "financial-conditions-scoring",
            "regime-classification-logic",
            "macro-narrative-synthesis"
        ],
        # Equity AC Agent (8 Skills): Produces CMAs for equity asset classes.
        "ac_equity_agent": [
            "historical-return-statistics",
            "equity-signal-processing",
            "historical-erp-modeling",
            "regime-adjusted-erp-modeling",
            "black-litterman-equilibrium",
            "inverse-gordon-valuation",
            "implied-erp-extraction",
            "asset-class-report-generation"
        ],
        # Fixed Income AC Agent (7 Skills): Produces CMAs for bond asset classes.
        "ac_fi_agent": [
            "historical-return-statistics",
            "fixed-income-signal-processing",
            "yield-curve-modeling",
            "credit-spread-duration-analysis",
            "sector-concentration-assessment",
            "fi-cma-building-block",
            "asset-class-report-generation"
        ],
        # Real Assets AC Agent (6 Skills): Produces CMAs for commodities/REITs/Gold.
        "ac_real_assets_agent": [
            "historical-return-statistics",
            "real-asset-signal-processing",
            "inflation-beta-estimation",
            "historical-erp-modeling",
            "regime-adjusted-erp-modeling",
            "asset-class-report-generation"
        ],
        # Cash AC Agent (6 Skills): Produces CMAs for cash equivalents.
        "ac_cash_agent": [
            "historical-return-statistics",
            "cash-yield-proxy-fetching",
            "cash-drag-estimation",
            "liquidity-premium-assessment",
            "regime-optionality-valuation",
            "asset-class-report-generation"
        ],
        # CMA Judge (7 Skills): Evaluates method dispersion and selects final CMA.
        "cma_judge": [
            "cma-dispersion-classification",
            "regime-conditional-tilting",
            "valuation-threshold-gating",
            "signal-alignment-verification",
            "range-constraint-enforcement",
            "confidence-weighted-blending",
            "cma-rationale-synthesis"
        ],
        # Covariance Agent (6 Skills): Script-only agent for matrix estimation.
        "covariance_agent": [
            "returns-matrix-construction",
            "sample-covariance-estimation",
            "ledoit-wolf-shrinkage",
            "eigenvalue-clipping-psd-repair",
            "nearest-psd-projection",
            "covariance-metadata-serialization"
        ],
        # Generic PC Agent (7 Skills): Proposes standard portfolio allocations.
        "pc_generic_agent": [
            "ips-constraint-parsing",
            "heuristic-weighting-schemes",
            "mean-variance-optimization",
            "risk-structured-optimization",
            "optimization-convergence-diagnostics",
            "pc-weight-serialization",
            "pc-rationale-reporting"
        ],
        # Adversarial PC Agent (6 Skills): Maximizes tracking variance vs centroid.
        "pc_adversarial_agent": [
            "ips-constraint-parsing",
            "ensemble-centroid-computation",
            "sharpe-floor-calibration",
            "tracking-variance-maximization",
            "pc-weight-serialization",
            "adversarial-rationale-reporting"
        ],
        # Researcher PC Agent (6 Skills): Discovers novel optimization objectives.
        "pc_researcher_agent": [
            "ips-constraint-parsing",
            "pc-registry-gap-analysis",
            "academic-literature-retrieval",
            "objective-function-formulation",
            "method-specification-validation",
            "novel-method-serialization"
        ],
        # CRO Narrative Agent (8 Skills): Computes risk metrics and writes reports.
        "cro_narrative_agent": [
            "ex-ante-volatility-computation",
            "tracking-error-computation",
            "backtest-sharpe-computation",
            "maximum-drawdown-computation",
            "fama-french-factor-regression",
            "ips-compliance-verification",
            "cro-json-serialization",
            "cro-narrative-synthesis"
        ],
        # Peer Review Agent (6 Skills): Critiques intra/inter-category peers.
        "peer_review_agent": [
            "ips-compliance-critique",
            "risk-profile-critique",
            "diversification-concentration-critique",
            "regime-fit-critique",
            "estimation-risk-critique",
            "actionable-improvement-formulation"
        ],
        # Voting Agent (6 Skills): Executes modified Borda count logic.
        "voting_agent": [
            "peer-review-parsing",
            "cro-report-parsing",
            "borda-count-ranking-logic",
            "bottom-flag-penalty-logic",
            "self-vote-exclusion-enforcement",
            "vote-schema-serialization"
        ],
        # Top-5 Revision Agent (6 Skills): Integrates feedback and re-optimizes.
        "top5_revision_agent": [
            "peer-feedback-integration",
            "cro-diagnostic-integration",
            "constraint-override-formulation",
            "re-optimization-triggering",
            "revision-memo-synthesis",
            "no-revision-confirmation"
        ],
        # CIO Agent (8 Skills): Synthesizes final ensemble and writes board memo.
        "cio_agent": [
            "cio-composite-scoring",
            "simple-average-ensembling",
            "inverse-te-ensembling",
            "regime-conditional-ensembling",
            "ips-compliance-ensemble-verification",
            "best-ensemble-selection",
            "final-weight-serialization",
            "board-memo-generation"
        ]
    }

    # ------------------------------------------------------------------
    # Define the master 'skills' directory path within the base directory
    # ------------------------------------------------------------------
    skills_root: Path = root_path / "skills"

    # ------------------------------------------------------------------
    # Create the master 'skills' directory.
    # exist_ok=True prevents errors if the directory already exists.
    # parents=True ensures any missing parent directories are also created.
    # ------------------------------------------------------------------
    try:
        skills_root.mkdir(parents=True, exist_ok=True)
        logger.info("Successfully verified/created skills root directory at '%s'.", skills_root)
    except OSError as exc:
        logger.error("Failed to create skills root directory at '%s'.", skills_root)
        raise

    # ------------------------------------------------------------------
    # Iterate over each agent and its corresponding list of skills
    # ------------------------------------------------------------------
    for agent_slug, skills_list in registry.items():

        # Define the specific directory path for the current agent's skills
        agent_dir: Path = skills_root / agent_slug

        # Create the agent-specific directory
        try:
            agent_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.error("Failed to create directory for agent '%s' at '%s'.", agent_slug, agent_dir)
            raise

        # ------------------------------------------------------------------
        # Iterate over each skill slug assigned to the current agent
        # ------------------------------------------------------------------
        for skill_slug in skills_list:

            # Define the full file path for the markdown skill contract
            skill_file_path: Path = agent_dir / f"{skill_slug}.md"

            # Construct the standardized, implementation-grade markdown content.
            # This enforces the separation of concerns: the LLM reads the methodology
            # and constraints, but is explicitly instructed to call the Python script.
            markdown_content: str = (
                f"# Skill: {skill_slug}\n"
                f"**Agent Owner:** {agent_slug}\n\n"
                f"## 1. Methodological Mandate\n"
                f"This document serves as the strict operational boundary for the `{skill_slug}` capability. "
                f"The agent must not perform arithmetic or probabilistic guessing related to this domain. "
                f"All logic must be deferred to the registered Python tool bindings.\n\n"
                f"## 2. Required Inputs\n"
                f"- [To be populated dynamically by the Orchestrator during ReAct/AutoGen initialization]\n\n"
                f"## 3. Tool Bindings\n"
                f"The agent must invoke the following Python scripts to execute this skill:\n"
                f"- `scripts/{agent_slug}/{skill_slug}_executor.py`\n\n"
                f"## 4. Output Contract\n"
                f"Execution of this skill must result in strict adherence to the predefined JSON schemas "
                f"located in `STUDY_CONFIG['OUTPUT_SCHEMAS']`.\n"
            )

            # Write the constructed markdown content to the file system using UTF-8 encoding
            try:
                skill_file_path.write_text(markdown_content, encoding="utf-8")
                logger.debug("Successfully wrote skill contract: '%s'", skill_file_path)
            except OSError as exc:
                logger.error("Failed to write skill contract at '%s'.", skill_file_path)
                raise

    # Log the successful completion of the registry generation
    logger.info(
        "generate_and_persist_agent_skills_registry: Successfully generated and "
        "persisted skills registry for %d agents.", len(registry)
    )

    # ------------------------------------------------------------------
    # Return the registry dictionary to the caller for downstream configuration
    # ------------------------------------------------------------------
    return registry
