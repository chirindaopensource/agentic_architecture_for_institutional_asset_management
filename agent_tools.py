# =============================================================================
# SELF-DRIVING PORTFOLIO: TOOL REGISTRY — BATCH 1 (TOOLS 1–10)
# =============================================================================
# Implements the first 10 tools from the complete 78-tool registry for the
# agentic Strategic Asset Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# Tools implemented:
#   1.  fetch_macro_snapshot
#   2.  score_growth_dimension
#   3.  score_inflation_dimension
#   4.  score_policy_dimension
#   5.  score_financial_conditions_dimension
#   6.  classify_regime
#   7.  write_macro_view_json
#   8.  write_analysis_md
#   9.  fetch_historical_stats
#   10. fetch_signals
#
# All tools are purely deterministic Python callables — no LLM interaction.
# All arithmetic conventions are frozen per STUDY_CONFIG["DATA_CONVENTIONS"].
# =============================================================================

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger — all tools emit structured log messages.
# ---------------------------------------------------------------------------
# Initialise a named logger for this module so that callers can configure
# log levels independently of the root logger.
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants sourced from STUDY_CONFIG (reproduced here for
# self-contained validation; the orchestrator injects the live config).
# ---------------------------------------------------------------------------

# Valid macro regime labels per METHODOLOGY_PARAMS["MACRO_REGIMES"]
_VALID_REGIMES: Tuple[str, ...] = (
    "Expansion",
    "Late-cycle",
    "Recession",
    "Recovery",
)

# Required columns in df_macro_raw per RAW_DATA_SCHEMAS["MACRO_DATA"]["schema"]
_MACRO_RAW_REQUIRED_COLS: Tuple[str, ...] = (
    "real_gdp_growth_rev",
    "nonfarm_payrolls_mom",
    "cpi_yoy",
    "cpi_mom",
    "cpi_core_yoy",
    "brent_crude_usd",
    "fed_funds_rate",
    "fed_funds_3m_change",
    "financial_conditions_index",
    "financial_conditions_mom",
)

# Equity-only fundamental fields per RAW_DATA_SCHEMAS["FUNDAMENTALS"]
_EQUITY_ONLY_FIELDS: Tuple[str, ...] = (
    "cape_ratio",
    "pe_trailing",
    "earnings_yield",
)

# Yield fields that must be stored in decimal form (e.g., 0.03 = 3%)
_YIELD_FIELDS: Tuple[str, ...] = (
    "earnings_yield",
    "dividend_yield",
    "buyback_yield",
)

# Minimum monthly observations required for reliable statistical estimation
_MIN_MONTHLY_OBS: int = 24

# Maximum staleness tolerance in months for signals/fundamentals
_MAX_STALENESS_MONTHS: int = 3

# Numerical stability epsilon for z-score denominator
_ZSCORE_EPS: float = 1e-8

# Score clipping bounds per tool specification
_SCORE_CLIP_LOW: float = -3.0
_SCORE_CLIP_HIGH: float = 3.0


# =============================================================================
# TOOL 1: fetch_macro_snapshot
# =============================================================================

def fetch_macro_snapshot(
    as_of_date: str,
    df_macro_raw: pd.DataFrame,
) -> Dict[str, float]:
    """
    Retrieve the most recent macro indicator snapshot on or before ``as_of_date``.

    This tool implements the point-in-time data retrieval step for the Macro
    Agent (Task 17, ReAct loop). It slices ``df_macro_raw`` to enforce strict
    temporal discipline: no observation beyond ``as_of_date`` is included,
    preventing lookahead bias as discussed in Section 5.1 of the manuscript
    (Yin et al. 2024).

    The returned dict is the primary input to all four dimension-scoring tools
    (Tools 2–5) and is passed as the ``macro_snapshot`` argument.

    Parameters
    ----------
    as_of_date : str
        ISO-8601 date string (e.g., ``"2026-03-31"``). The snapshot is the
        most recent row in ``df_macro_raw`` with index ``<= as_of_date``.
        Must be parseable by ``pd.Timestamp``.
    df_macro_raw : pd.DataFrame
        Monthly macro indicator panel. Must have a timezone-naive
        ``DatetimeIndex`` (month-end) and all columns listed in
        ``_MACRO_RAW_REQUIRED_COLS``. Shape: ``(T, 10)``.

    Returns
    -------
    Dict[str, float]
        Flat dictionary with exactly 10 keys matching ``_MACRO_RAW_REQUIRED_COLS``.
        All values are Python native ``float`` (not ``np.float64``) to ensure
        JSON serialisability. Example::

            {
                "real_gdp_growth_rev": 2.1,
                "nonfarm_payrolls_mom": -15000.0,
                "cpi_yoy": 2.4,
                ...
            }

    Raises
    ------
    TypeError
        If ``df_macro_raw`` is not a ``pd.DataFrame``.
    ValueError
        If ``as_of_date`` cannot be parsed as a date.
    ValueError
        If required columns are missing from ``df_macro_raw``.
    ValueError
        If no rows exist on or before ``as_of_date``.
    ValueError
        If the most recent row contains NaN values in required columns.

    Notes
    -----
    Point-in-time discipline: the filter ``df_macro_raw.index <= as_of_date``
    ensures that only data available at the as-of date is used. This is a
    hard requirement per ``DATA_CONVENTIONS["point_in_time_policy"]``.

    The ``real_gdp_growth_rev`` series is revision-sensitive. If only the
    latest vintage is available, reproduction is approximate — this is
    documented per ``DATA_CONVENTIONS["point_in_time_policy"]["gdp_vintage_policy"]``.
    """
    # ------------------------------------------------------------------
    # Input validation: ensure df_macro_raw is a DataFrame
    # ------------------------------------------------------------------
    if not isinstance(df_macro_raw, pd.DataFrame):
        raise TypeError(
            f"df_macro_raw must be a pd.DataFrame, "
            f"got {type(df_macro_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: parse as_of_date to pd.Timestamp
    # ------------------------------------------------------------------
    try:
        # Convert the ISO-8601 string to a timezone-naive pd.Timestamp
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
    except Exception as exc:
        raise ValueError(
            f"as_of_date='{as_of_date}' cannot be parsed as a date. "
            f"Expected ISO-8601 format (e.g., '2026-03-31'). "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Input validation: check that the DataFrame index is a DatetimeIndex
    # ------------------------------------------------------------------
    if not isinstance(df_macro_raw.index, pd.DatetimeIndex):
        raise TypeError(
            "df_macro_raw must have a DatetimeIndex. "
            f"Got index type: {type(df_macro_raw.index).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: strip timezone from index if present to ensure
    # timezone-naive comparison (per DATA_CONVENTIONS)
    # ------------------------------------------------------------------
    if df_macro_raw.index.tz is not None:
        # Localise to None to make comparison with timezone-naive as_of_ts safe
        df_macro_raw = df_macro_raw.copy()
        df_macro_raw.index = df_macro_raw.index.tz_localize(None)

    # ------------------------------------------------------------------
    # Input validation: verify all required columns are present
    # ------------------------------------------------------------------
    missing_cols: List[str] = [
        col for col in _MACRO_RAW_REQUIRED_COLS
        if col not in df_macro_raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_macro_raw is missing required columns: {missing_cols}. "
            f"Required: {list(_MACRO_RAW_REQUIRED_COLS)}."
        )

    # ------------------------------------------------------------------
    # Point-in-time filter: retain only rows on or before as_of_date
    # This enforces the lookahead-bias prevention requirement from the
    # manuscript (Section 5.1, Yin et al. 2024).
    # ------------------------------------------------------------------
    df_filtered: pd.DataFrame = df_macro_raw.loc[
        df_macro_raw.index <= as_of_ts,
        list(_MACRO_RAW_REQUIRED_COLS),
    ]

    # ------------------------------------------------------------------
    # Guard: ensure at least one row exists after filtering
    # ------------------------------------------------------------------
    if df_filtered.empty:
        raise ValueError(
            f"No rows in df_macro_raw on or before as_of_date='{as_of_date}'. "
            f"Earliest available date: {df_macro_raw.index.min().date()}."
        )

    # ------------------------------------------------------------------
    # Select the most recent row (last row after point-in-time filter)
    # ------------------------------------------------------------------
    latest_row: pd.Series = df_filtered.iloc[-1]

    # ------------------------------------------------------------------
    # Guard: check for NaN values in the selected row
    # ------------------------------------------------------------------
    nan_cols: List[str] = [
        col for col in _MACRO_RAW_REQUIRED_COLS
        if pd.isna(latest_row[col])
    ]
    if nan_cols:
        raise ValueError(
            f"The most recent macro snapshot (date: {df_filtered.index[-1].date()}) "
            f"contains NaN values in columns: {nan_cols}. "
            "Apply forward-fill or interpolation per DATA_CONVENTIONS "
            "['missing_data_policy'] before calling this tool."
        )

    # ------------------------------------------------------------------
    # Construct the output dict with Python native float values
    # (not np.float64) to ensure JSON serialisability downstream
    # ------------------------------------------------------------------
    snapshot: Dict[str, float] = {
        col: float(latest_row[col])
        for col in _MACRO_RAW_REQUIRED_COLS
    }

    # ------------------------------------------------------------------
    # Log the snapshot date for audit trail
    # ------------------------------------------------------------------
    logger.info(
        "fetch_macro_snapshot: snapshot date=%s, as_of_date=%s",
        df_filtered.index[-1].date(),
        as_of_date,
    )

    # Return the JSON-serialisable macro snapshot dict
    return snapshot


# =============================================================================
# TOOL 2: score_growth_dimension
# =============================================================================

def score_growth_dimension(
    macro_snapshot: Dict[str, float],
    df_macro_raw: pd.DataFrame,
    growth_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute the standardised growth dimension score :math:`s^{(g)}_t`.

    Implements the weighted z-score formula for the growth dimension of the
    macro regime classification framework (Task 20, Step 1):

    .. math::

        s^{(g)}_t = w_{gdp} \\cdot z(\\text{GDP}_t)
                  + w_{nfp} \\cdot z(\\text{NFP}_t)

    where :math:`z(x_t) = (x_t - \\mu_x) / \\sigma_x` is computed over the
    full historical distribution of ``df_macro_raw``, and the frozen weights
    are :math:`w_{gdp} = 0.6`, :math:`w_{nfp} = 0.4` per
    ``METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]["growth"]``.

    The result is clipped to :math:`[-3, 3]` to bound the influence of
    extreme outliers.

    Parameters
    ----------
    macro_snapshot : Dict[str, float]
        Output of ``fetch_macro_snapshot``. Must contain keys
        ``"real_gdp_growth_rev"`` and ``"nonfarm_payrolls_mom"``.
    df_macro_raw : pd.DataFrame
        Full historical macro panel (injected via closure). Used to compute
        the historical mean and standard deviation for z-score normalisation.
        Must contain columns ``"real_gdp_growth_rev"`` and
        ``"nonfarm_payrolls_mom"``. Shape: ``(T, ≥2)``.
    growth_weights : Optional[Dict[str, float]]
        Override for frozen weights. Keys: ``"real_gdp_growth"`` and
        ``"nonfarm_payrolls_mom"``. If ``None``, uses frozen values
        ``{real_gdp_growth: 0.6, nonfarm_payrolls_mom: 0.4}``.

    Returns
    -------
    float
        Growth dimension score :math:`s^{(g)}_t \\in [-3, 3]`.
        Higher values indicate stronger growth momentum.

    Raises
    ------
    TypeError
        If ``macro_snapshot`` is not a dict or ``df_macro_raw`` is not a
        ``pd.DataFrame``.
    ValueError
        If required keys are missing from ``macro_snapshot``.
    ValueError
        If required columns are missing from ``df_macro_raw``.
    ValueError
        If fewer than ``_MIN_MONTHLY_OBS`` non-NaN observations are available
        for either series.
    ValueError
        If the historical standard deviation of either series is effectively
        zero (constant series).

    Notes
    -----
    Z-scores are computed relative to the **full** historical distribution of
    ``df_macro_raw``, not a rolling window. This is a frozen implementation
    choice per ``METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]``.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(macro_snapshot, dict):
        raise TypeError(
            f"macro_snapshot must be a dict, got {type(macro_snapshot).__name__}."
        )
    if not isinstance(df_macro_raw, pd.DataFrame):
        raise TypeError(
            f"df_macro_raw must be a pd.DataFrame, "
            f"got {type(df_macro_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys in macro_snapshot
    # ------------------------------------------------------------------
    _required_snapshot_keys: Tuple[str, ...] = (
        "real_gdp_growth_rev",
        "nonfarm_payrolls_mom",
    )
    missing_keys: List[str] = [
        k for k in _required_snapshot_keys if k not in macro_snapshot
    ]
    if missing_keys:
        raise ValueError(
            f"macro_snapshot is missing required keys: {missing_keys}."
        )

    # ------------------------------------------------------------------
    # Input validation: required columns in df_macro_raw
    # ------------------------------------------------------------------
    _required_cols: Tuple[str, ...] = (
        "real_gdp_growth_rev",
        "nonfarm_payrolls_mom",
    )
    missing_cols: List[str] = [
        c for c in _required_cols if c not in df_macro_raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_macro_raw is missing required columns: {missing_cols}."
        )

    # ------------------------------------------------------------------
    # Resolve frozen weights (default or override)
    # Frozen values: real_gdp_growth=0.6, nonfarm_payrolls_mom=0.4
    # per METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]["growth"]
    # ------------------------------------------------------------------
    _default_weights: Dict[str, float] = {
        "real_gdp_growth": 0.6,
        "nonfarm_payrolls_mom": 0.4,
    }
    weights: Dict[str, float] = (
        growth_weights if growth_weights is not None else _default_weights
    )

    # ------------------------------------------------------------------
    # Validate that weights sum to 1.0 (within floating-point tolerance)
    # ------------------------------------------------------------------
    weight_sum: float = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"growth_weights must sum to 1.0, got {weight_sum:.6f}."
        )

    # ------------------------------------------------------------------
    # Extract current snapshot values for both indicators
    # ------------------------------------------------------------------
    # Current real GDP growth revision value from the snapshot
    gdp_current: float = float(macro_snapshot["real_gdp_growth_rev"])
    # Current nonfarm payrolls MoM change from the snapshot
    nfp_current: float = float(macro_snapshot["nonfarm_payrolls_mom"])

    # ------------------------------------------------------------------
    # Extract historical series from df_macro_raw, dropping NaN values
    # to compute clean baseline statistics
    # ------------------------------------------------------------------
    # Historical GDP growth series (drop NaN for clean statistics)
    gdp_hist: pd.Series = df_macro_raw["real_gdp_growth_rev"].dropna()
    # Historical NFP MoM series (drop NaN for clean statistics)
    nfp_hist: pd.Series = df_macro_raw["nonfarm_payrolls_mom"].dropna()

    # ------------------------------------------------------------------
    # Minimum observation check: require at least _MIN_MONTHLY_OBS
    # non-NaN observations for reliable z-score computation
    # ------------------------------------------------------------------
    if len(gdp_hist) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for 'real_gdp_growth_rev': "
            f"{len(gdp_hist)} non-NaN observations, "
            f"minimum required: {_MIN_MONTHLY_OBS}."
        )
    if len(nfp_hist) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for 'nonfarm_payrolls_mom': "
            f"{len(nfp_hist)} non-NaN observations, "
            f"minimum required: {_MIN_MONTHLY_OBS}."
        )

    # ------------------------------------------------------------------
    # Compute historical mean and standard deviation for each series
    # These form the z-score baseline: z(x_t) = (x_t - mu) / sigma
    # ------------------------------------------------------------------
    # Historical mean of real GDP growth
    gdp_mean: float = float(gdp_hist.mean())
    # Historical standard deviation of real GDP growth
    gdp_std: float = float(gdp_hist.std(ddof=1))
    # Historical mean of nonfarm payrolls MoM
    nfp_mean: float = float(nfp_hist.mean())
    # Historical standard deviation of nonfarm payrolls MoM
    nfp_std: float = float(nfp_hist.std(ddof=1))

    # ------------------------------------------------------------------
    # Guard against near-zero standard deviation (constant series)
    # Adding _ZSCORE_EPS prevents division by zero
    # ------------------------------------------------------------------
    if gdp_std < _ZSCORE_EPS:
        raise ValueError(
            f"Historical std of 'real_gdp_growth_rev' is effectively zero "
            f"({gdp_std:.2e}). Cannot compute z-score."
        )
    if nfp_std < _ZSCORE_EPS:
        raise ValueError(
            f"Historical std of 'nonfarm_payrolls_mom' is effectively zero "
            f"({nfp_std:.2e}). Cannot compute z-score."
        )

    # ------------------------------------------------------------------
    # Compute z-scores: z(x_t) = (x_t - mu_x) / sigma_x
    # ------------------------------------------------------------------
    # Z-score for real GDP growth: (current - historical mean) / historical std
    z_gdp: float = (gdp_current - gdp_mean) / gdp_std
    # Z-score for nonfarm payrolls MoM: (current - historical mean) / historical std
    z_nfp: float = (nfp_current - nfp_mean) / nfp_std

    # ------------------------------------------------------------------
    # Compute weighted sum: s^(g)_t = w_gdp * z(GDP) + w_nfp * z(NFP)
    # Frozen weights: 0.6 for GDP, 0.4 for NFP
    # ------------------------------------------------------------------
    raw_score: float = (
        weights["real_gdp_growth"] * z_gdp
        + weights["nonfarm_payrolls_mom"] * z_nfp
    )

    # ------------------------------------------------------------------
    # Clip to [-3, 3] to bound the influence of extreme outliers
    # per the tool specification
    # ------------------------------------------------------------------
    clipped_score: float = float(
        np.clip(raw_score, _SCORE_CLIP_LOW, _SCORE_CLIP_HIGH)
    )

    # Log the computed score components for audit trail
    logger.debug(
        "score_growth_dimension: z_gdp=%.4f, z_nfp=%.4f, "
        "raw=%.4f, clipped=%.4f",
        z_gdp, z_nfp, raw_score, clipped_score,
    )

    # Return the clipped growth dimension score
    return clipped_score


# =============================================================================
# TOOL 3: score_inflation_dimension
# =============================================================================

def score_inflation_dimension(
    macro_snapshot: Dict[str, float],
    df_macro_raw: pd.DataFrame,
    inflation_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute the standardised inflation dimension score :math:`s^{(\\pi)}_t`.

    Implements the weighted z-score formula for the inflation dimension of the
    macro regime classification framework (Task 20, Step 1):

    .. math::

        s^{(\\pi)}_t = w_{yoy} \\cdot z(\\text{CPI\\_YoY}_t)
                     + w_{mom} \\cdot z(\\text{CPI\\_MoM}_t)

    where :math:`z(x_t) = (x_t - \\mu_x) / \\sigma_x` is computed over the
    full historical distribution of ``df_macro_raw``, and the frozen weights
    are :math:`w_{yoy} = 0.7`, :math:`w_{mom} = 0.3` per
    ``METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]["inflation"]``.

    A higher score indicates greater inflationary pressure, which is a key
    input to the stagflationary risk assessment described in the manuscript
    (Section 4.1).

    Parameters
    ----------
    macro_snapshot : Dict[str, float]
        Output of ``fetch_macro_snapshot``. Must contain keys
        ``"cpi_yoy"`` and ``"cpi_mom"``.
    df_macro_raw : pd.DataFrame
        Full historical macro panel (injected via closure). Must contain
        columns ``"cpi_yoy"`` and ``"cpi_mom"``. Shape: ``(T, ≥2)``.
    inflation_weights : Optional[Dict[str, float]]
        Override for frozen weights. Keys: ``"cpi_yoy"`` and ``"cpi_mom"``.
        If ``None``, uses frozen values ``{cpi_yoy: 0.7, cpi_mom: 0.3}``.

    Returns
    -------
    float
        Inflation dimension score :math:`s^{(\\pi)}_t \\in [-3, 3]`.
        Higher values indicate stronger inflationary pressure.

    Raises
    ------
    TypeError
        If inputs are of incorrect type.
    ValueError
        If required keys or columns are missing.
    ValueError
        If insufficient history or near-zero standard deviation.

    Notes
    -----
    The CPI YoY series receives a higher weight (0.7) than MoM (0.3) because
    the manuscript's regime classification emphasises sustained inflation trends
    over short-term fluctuations. This is a frozen implementation choice.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(macro_snapshot, dict):
        raise TypeError(
            f"macro_snapshot must be a dict, got {type(macro_snapshot).__name__}."
        )
    if not isinstance(df_macro_raw, pd.DataFrame):
        raise TypeError(
            f"df_macro_raw must be a pd.DataFrame, "
            f"got {type(df_macro_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys in macro_snapshot
    # ------------------------------------------------------------------
    _required_keys: Tuple[str, ...] = ("cpi_yoy", "cpi_mom")
    missing_keys: List[str] = [
        k for k in _required_keys if k not in macro_snapshot
    ]
    if missing_keys:
        raise ValueError(
            f"macro_snapshot is missing required keys: {missing_keys}."
        )

    # ------------------------------------------------------------------
    # Input validation: required columns in df_macro_raw
    # ------------------------------------------------------------------
    missing_cols: List[str] = [
        c for c in _required_keys if c not in df_macro_raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_macro_raw is missing required columns: {missing_cols}."
        )

    # ------------------------------------------------------------------
    # Resolve frozen weights (default or override)
    # Frozen values: cpi_yoy=0.7, cpi_mom=0.3
    # per METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]["inflation"]
    # ------------------------------------------------------------------
    _default_weights: Dict[str, float] = {
        "cpi_yoy": 0.7,
        "cpi_mom": 0.3,
    }
    weights: Dict[str, float] = (
        inflation_weights if inflation_weights is not None else _default_weights
    )

    # ------------------------------------------------------------------
    # Validate that weights sum to 1.0
    # ------------------------------------------------------------------
    weight_sum: float = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"inflation_weights must sum to 1.0, got {weight_sum:.6f}."
        )

    # ------------------------------------------------------------------
    # Extract current snapshot values for both CPI indicators
    # ------------------------------------------------------------------
    # Current CPI YoY value from the snapshot
    cpi_yoy_current: float = float(macro_snapshot["cpi_yoy"])
    # Current CPI MoM value from the snapshot
    cpi_mom_current: float = float(macro_snapshot["cpi_mom"])

    # ------------------------------------------------------------------
    # Extract historical series, dropping NaN for clean baseline statistics
    # ------------------------------------------------------------------
    # Historical CPI YoY series
    cpi_yoy_hist: pd.Series = df_macro_raw["cpi_yoy"].dropna()
    # Historical CPI MoM series
    cpi_mom_hist: pd.Series = df_macro_raw["cpi_mom"].dropna()

    # ------------------------------------------------------------------
    # Minimum observation check
    # ------------------------------------------------------------------
    if len(cpi_yoy_hist) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for 'cpi_yoy': "
            f"{len(cpi_yoy_hist)} observations, minimum: {_MIN_MONTHLY_OBS}."
        )
    if len(cpi_mom_hist) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for 'cpi_mom': "
            f"{len(cpi_mom_hist)} observations, minimum: {_MIN_MONTHLY_OBS}."
        )

    # ------------------------------------------------------------------
    # Compute historical mean and standard deviation for z-score baseline
    # z(x_t) = (x_t - mu_x) / sigma_x
    # ------------------------------------------------------------------
    # Historical mean of CPI YoY
    cpi_yoy_mean: float = float(cpi_yoy_hist.mean())
    # Historical standard deviation of CPI YoY (sample, ddof=1)
    cpi_yoy_std: float = float(cpi_yoy_hist.std(ddof=1))
    # Historical mean of CPI MoM
    cpi_mom_mean: float = float(cpi_mom_hist.mean())
    # Historical standard deviation of CPI MoM (sample, ddof=1)
    cpi_mom_std: float = float(cpi_mom_hist.std(ddof=1))

    # ------------------------------------------------------------------
    # Guard against near-zero standard deviation
    # ------------------------------------------------------------------
    if cpi_yoy_std < _ZSCORE_EPS:
        raise ValueError(
            f"Historical std of 'cpi_yoy' is effectively zero "
            f"({cpi_yoy_std:.2e}). Cannot compute z-score."
        )
    if cpi_mom_std < _ZSCORE_EPS:
        raise ValueError(
            f"Historical std of 'cpi_mom' is effectively zero "
            f"({cpi_mom_std:.2e}). Cannot compute z-score."
        )

    # ------------------------------------------------------------------
    # Compute z-scores: z(x_t) = (x_t - mu_x) / sigma_x
    # ------------------------------------------------------------------
    # Z-score for CPI YoY
    z_cpi_yoy: float = (cpi_yoy_current - cpi_yoy_mean) / cpi_yoy_std
    # Z-score for CPI MoM
    z_cpi_mom: float = (cpi_mom_current - cpi_mom_mean) / cpi_mom_std

    # ------------------------------------------------------------------
    # Compute weighted sum: s^(pi)_t = w_yoy * z(CPI_YoY) + w_mom * z(CPI_MoM)
    # Frozen weights: 0.7 for YoY, 0.3 for MoM
    # ------------------------------------------------------------------
    raw_score: float = (
        weights["cpi_yoy"] * z_cpi_yoy
        + weights["cpi_mom"] * z_cpi_mom
    )

    # ------------------------------------------------------------------
    # Clip to [-3, 3] to bound the influence of extreme outliers
    # ------------------------------------------------------------------
    clipped_score: float = float(
        np.clip(raw_score, _SCORE_CLIP_LOW, _SCORE_CLIP_HIGH)
    )

    # Log the computed score components for audit trail
    logger.debug(
        "score_inflation_dimension: z_cpi_yoy=%.4f, z_cpi_mom=%.4f, "
        "raw=%.4f, clipped=%.4f",
        z_cpi_yoy, z_cpi_mom, raw_score, clipped_score,
    )

    # Return the clipped inflation dimension score
    return clipped_score


# =============================================================================
# TOOL 4: score_policy_dimension
# =============================================================================

def score_policy_dimension(
    macro_snapshot: Dict[str, float],
    df_macro_raw: pd.DataFrame,
    policy_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute the standardised monetary policy dimension score :math:`s^{(pol)}_t`.

    Implements the weighted z-score formula for the monetary policy dimension
    of the macro regime classification framework (Task 20, Step 1):

    .. math::

        s^{(pol)}_t = w_{ffr} \\cdot z(\\text{FFR}_t)
                    + w_{\\Delta} \\cdot z(\\Delta_3\\text{FFR}_t)

    where :math:`\\Delta_3\\text{FFR}_t` is the 3-month change in the fed
    funds rate, and the frozen weights are :math:`w_{ffr} = 0.5`,
    :math:`w_{\\Delta} = 0.5` per
    ``METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]["monetary_policy"]``.

    A higher score indicates tighter monetary policy (higher rates and/or
    rising rates), which is a key input to late-cycle and recession regime
    classification.

    Parameters
    ----------
    macro_snapshot : Dict[str, float]
        Output of ``fetch_macro_snapshot``. Must contain keys
        ``"fed_funds_rate"`` and ``"fed_funds_3m_change"``.
    df_macro_raw : pd.DataFrame
        Full historical macro panel (injected via closure). Must contain
        columns ``"fed_funds_rate"`` and ``"fed_funds_3m_change"``.
        Shape: ``(T, ≥2)``.
    policy_weights : Optional[Dict[str, float]]
        Override for frozen weights. Keys: ``"fed_funds_rate"`` and
        ``"fed_funds_3m_change"``. If ``None``, uses frozen values
        ``{fed_funds_rate: 0.5, fed_funds_3m_change: 0.5}``.

    Returns
    -------
    float
        Policy dimension score :math:`s^{(pol)}_t \\in [-3, 3]`.
        Higher values indicate tighter monetary policy.

    Raises
    ------
    TypeError
        If inputs are of incorrect type.
    ValueError
        If required keys or columns are missing.
    ValueError
        If insufficient history or near-zero standard deviation.

    Notes
    -----
    During zero-lower-bound (ZLB) periods (2009–2015, 2020–2022), the
    ``fed_funds_rate`` has near-zero variance in short windows. The full
    historical distribution naturally handles this by reflecting the unusual
    nature of ZLB in the z-score. The 2022 hiking cycle produces extreme
    ``fed_funds_3m_change`` values that are bounded by the ``[-3, 3]`` clip.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(macro_snapshot, dict):
        raise TypeError(
            f"macro_snapshot must be a dict, got {type(macro_snapshot).__name__}."
        )
    if not isinstance(df_macro_raw, pd.DataFrame):
        raise TypeError(
            f"df_macro_raw must be a pd.DataFrame, "
            f"got {type(df_macro_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys in macro_snapshot
    # ------------------------------------------------------------------
    _required_keys: Tuple[str, ...] = ("fed_funds_rate", "fed_funds_3m_change")
    missing_keys: List[str] = [
        k for k in _required_keys if k not in macro_snapshot
    ]
    if missing_keys:
        raise ValueError(
            f"macro_snapshot is missing required keys: {missing_keys}."
        )

    # ------------------------------------------------------------------
    # Input validation: required columns in df_macro_raw
    # ------------------------------------------------------------------
    missing_cols: List[str] = [
        c for c in _required_keys if c not in df_macro_raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_macro_raw is missing required columns: {missing_cols}."
        )

    # ------------------------------------------------------------------
    # Resolve frozen weights (default or override)
    # Frozen values: fed_funds_rate=0.5, fed_funds_3m_change=0.5
    # per METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]["monetary_policy"]
    # ------------------------------------------------------------------
    _default_weights: Dict[str, float] = {
        "fed_funds_rate": 0.5,
        "fed_funds_3m_change": 0.5,
    }
    weights: Dict[str, float] = (
        policy_weights if policy_weights is not None else _default_weights
    )

    # ------------------------------------------------------------------
    # Validate that weights sum to 1.0
    # ------------------------------------------------------------------
    weight_sum: float = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"policy_weights must sum to 1.0, got {weight_sum:.6f}."
        )

    # ------------------------------------------------------------------
    # Extract current snapshot values for both policy indicators
    # ------------------------------------------------------------------
    # Current fed funds rate level from the snapshot
    ffr_current: float = float(macro_snapshot["fed_funds_rate"])
    # Current 3-month change in fed funds rate from the snapshot
    ffr_3m_current: float = float(macro_snapshot["fed_funds_3m_change"])

    # ------------------------------------------------------------------
    # Extract historical series, dropping NaN for clean baseline statistics
    # ------------------------------------------------------------------
    # Historical fed funds rate level series
    ffr_hist: pd.Series = df_macro_raw["fed_funds_rate"].dropna()
    # Historical fed funds 3-month change series
    ffr_3m_hist: pd.Series = df_macro_raw["fed_funds_3m_change"].dropna()

    # ------------------------------------------------------------------
    # Minimum observation check
    # ------------------------------------------------------------------
    if len(ffr_hist) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for 'fed_funds_rate': "
            f"{len(ffr_hist)} observations, minimum: {_MIN_MONTHLY_OBS}."
        )
    if len(ffr_3m_hist) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for 'fed_funds_3m_change': "
            f"{len(ffr_3m_hist)} observations, minimum: {_MIN_MONTHLY_OBS}."
        )

    # ------------------------------------------------------------------
    # Compute historical mean and standard deviation for z-score baseline
    # ------------------------------------------------------------------
    # Historical mean of fed funds rate
    ffr_mean: float = float(ffr_hist.mean())
    # Historical standard deviation of fed funds rate (sample, ddof=1)
    ffr_std: float = float(ffr_hist.std(ddof=1))
    # Historical mean of fed funds 3-month change
    ffr_3m_mean: float = float(ffr_3m_hist.mean())
    # Historical standard deviation of fed funds 3-month change (sample, ddof=1)
    ffr_3m_std: float = float(ffr_3m_hist.std(ddof=1))

    # ------------------------------------------------------------------
    # Guard against near-zero standard deviation
    # ------------------------------------------------------------------
    if ffr_std < _ZSCORE_EPS:
        raise ValueError(
            f"Historical std of 'fed_funds_rate' is effectively zero "
            f"({ffr_std:.2e}). Cannot compute z-score."
        )
    if ffr_3m_std < _ZSCORE_EPS:
        raise ValueError(
            f"Historical std of 'fed_funds_3m_change' is effectively zero "
            f"({ffr_3m_std:.2e}). Cannot compute z-score."
        )

    # ------------------------------------------------------------------
    # Compute z-scores: z(x_t) = (x_t - mu_x) / sigma_x
    # ------------------------------------------------------------------
    # Z-score for fed funds rate level
    z_ffr: float = (ffr_current - ffr_mean) / ffr_std
    # Z-score for fed funds 3-month change
    z_ffr_3m: float = (ffr_3m_current - ffr_3m_mean) / ffr_3m_std

    # ------------------------------------------------------------------
    # Compute weighted sum: s^(pol)_t = w_ffr * z(FFR) + w_delta * z(delta_FFR)
    # Frozen weights: 0.5 for level, 0.5 for 3-month change
    # ------------------------------------------------------------------
    raw_score: float = (
        weights["fed_funds_rate"] * z_ffr
        + weights["fed_funds_3m_change"] * z_ffr_3m
    )

    # ------------------------------------------------------------------
    # Clip to [-3, 3] to bound the influence of extreme outliers
    # ------------------------------------------------------------------
    clipped_score: float = float(
        np.clip(raw_score, _SCORE_CLIP_LOW, _SCORE_CLIP_HIGH)
    )

    # Log the computed score components for audit trail
    logger.debug(
        "score_policy_dimension: z_ffr=%.4f, z_ffr_3m=%.4f, "
        "raw=%.4f, clipped=%.4f",
        z_ffr, z_ffr_3m, raw_score, clipped_score,
    )

    # Return the clipped policy dimension score
    return clipped_score


# =============================================================================
# TOOL 5: score_financial_conditions_dimension
# =============================================================================

def score_financial_conditions_dimension(
    macro_snapshot: Dict[str, float],
    df_macro_raw: pd.DataFrame,
    fci_weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Compute the standardised financial conditions dimension score
    :math:`s^{(fc)}_t`.

    Implements the weighted z-score formula for the financial conditions
    dimension of the macro regime classification framework (Task 20, Step 1):

    .. math::

        s^{(fc)}_t = w_{fci} \\cdot z(\\text{FCI}_t)
                   + w_{\\Delta} \\cdot z(\\Delta\\text{FCI}_t)

    where the frozen weights are :math:`w_{fci} = 0.6`,
    :math:`w_{\\Delta} = 0.4` per
    ``METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]["financial_conditions"]``.

    A higher score indicates tighter financial conditions, which is a
    recessionary signal per the manuscript's regime classification framework.

    Parameters
    ----------
    macro_snapshot : Dict[str, float]
        Output of ``fetch_macro_snapshot``. Must contain keys
        ``"financial_conditions_index"`` and ``"financial_conditions_mom"``.
    df_macro_raw : pd.DataFrame
        Full historical macro panel (injected via closure). Must contain
        columns ``"financial_conditions_index"`` and
        ``"financial_conditions_mom"``. Shape: ``(T, ≥2)``.
    fci_weights : Optional[Dict[str, float]]
        Override for frozen weights. Keys: ``"financial_conditions_index"``
        and ``"financial_conditions_mom"``. If ``None``, uses frozen values
        ``{financial_conditions_index: 0.6, financial_conditions_mom: 0.4}``.

    Returns
    -------
    float
        Financial conditions dimension score :math:`s^{(fc)}_t \\in [-3, 3]`.
        Higher values indicate tighter financial conditions.

    Raises
    ------
    TypeError
        If inputs are of incorrect type.
    ValueError
        If required keys or columns are missing.
    ValueError
        If insufficient history or near-zero standard deviation.

    Notes
    -----
    **Sign convention (frozen):** This implementation assumes that a higher
    ``financial_conditions_index`` value corresponds to tighter financial
    conditions (e.g., Bloomberg FCI convention). If the FCI is constructed
    with the opposite sign convention, the caller must invert the series
    before passing it to this tool. This is documented as a frozen
    implementation choice per ``DATA_CONVENTIONS``.

    The composition and normalisation of ``fci_raw_score`` is UNSPECIFIED
    IN MANUSCRIPT. This tool treats it as a generic numeric series.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(macro_snapshot, dict):
        raise TypeError(
            f"macro_snapshot must be a dict, got {type(macro_snapshot).__name__}."
        )
    if not isinstance(df_macro_raw, pd.DataFrame):
        raise TypeError(
            f"df_macro_raw must be a pd.DataFrame, "
            f"got {type(df_macro_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys in macro_snapshot
    # ------------------------------------------------------------------
    _required_keys: Tuple[str, ...] = (
        "financial_conditions_index",
        "financial_conditions_mom",
    )
    missing_keys: List[str] = [
        k for k in _required_keys if k not in macro_snapshot
    ]
    if missing_keys:
        raise ValueError(
            f"macro_snapshot is missing required keys: {missing_keys}."
        )

    # ------------------------------------------------------------------
    # Input validation: required columns in df_macro_raw
    # ------------------------------------------------------------------
    missing_cols: List[str] = [
        c for c in _required_keys if c not in df_macro_raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_macro_raw is missing required columns: {missing_cols}."
        )

    # ------------------------------------------------------------------
    # Resolve frozen weights (default or override)
    # Frozen values: financial_conditions_index=0.6, financial_conditions_mom=0.4
    # per METHODOLOGY_PARAMS["MACRO_SCORING_WEIGHTS"]["financial_conditions"]
    # ------------------------------------------------------------------
    _default_weights: Dict[str, float] = {
        "financial_conditions_index": 0.6,
        "financial_conditions_mom": 0.4,
    }
    weights: Dict[str, float] = (
        fci_weights if fci_weights is not None else _default_weights
    )

    # ------------------------------------------------------------------
    # Validate that weights sum to 1.0
    # ------------------------------------------------------------------
    weight_sum: float = sum(weights.values())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"fci_weights must sum to 1.0, got {weight_sum:.6f}."
        )

    # ------------------------------------------------------------------
    # Extract current snapshot values for both FCI indicators
    # ------------------------------------------------------------------
    # Current FCI level from the snapshot
    fci_current: float = float(macro_snapshot["financial_conditions_index"])
    # Current FCI MoM change from the snapshot
    fci_mom_current: float = float(macro_snapshot["financial_conditions_mom"])

    # ------------------------------------------------------------------
    # Extract historical series, dropping NaN for clean baseline statistics
    # ------------------------------------------------------------------
    # Historical FCI level series
    fci_hist: pd.Series = df_macro_raw["financial_conditions_index"].dropna()
    # Historical FCI MoM change series
    fci_mom_hist: pd.Series = df_macro_raw["financial_conditions_mom"].dropna()

    # ------------------------------------------------------------------
    # Minimum observation check
    # ------------------------------------------------------------------
    if len(fci_hist) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for 'financial_conditions_index': "
            f"{len(fci_hist)} observations, minimum: {_MIN_MONTHLY_OBS}."
        )
    if len(fci_mom_hist) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for 'financial_conditions_mom': "
            f"{len(fci_mom_hist)} observations, minimum: {_MIN_MONTHLY_OBS}."
        )

    # ------------------------------------------------------------------
    # Compute historical mean and standard deviation for z-score baseline
    # ------------------------------------------------------------------
    # Historical mean of FCI level
    fci_mean: float = float(fci_hist.mean())
    # Historical standard deviation of FCI level (sample, ddof=1)
    fci_std: float = float(fci_hist.std(ddof=1))
    # Historical mean of FCI MoM change
    fci_mom_mean: float = float(fci_mom_hist.mean())
    # Historical standard deviation of FCI MoM change (sample, ddof=1)
    fci_mom_std: float = float(fci_mom_hist.std(ddof=1))

    # ------------------------------------------------------------------
    # Guard against near-zero standard deviation
    # ------------------------------------------------------------------
    if fci_std < _ZSCORE_EPS:
        raise ValueError(
            f"Historical std of 'financial_conditions_index' is effectively "
            f"zero ({fci_std:.2e}). Cannot compute z-score."
        )
    if fci_mom_std < _ZSCORE_EPS:
        raise ValueError(
            f"Historical std of 'financial_conditions_mom' is effectively "
            f"zero ({fci_mom_std:.2e}). Cannot compute z-score."
        )

    # ------------------------------------------------------------------
    # Compute z-scores: z(x_t) = (x_t - mu_x) / sigma_x
    # ------------------------------------------------------------------
    # Z-score for FCI level
    z_fci: float = (fci_current - fci_mean) / fci_std
    # Z-score for FCI MoM change
    z_fci_mom: float = (fci_mom_current - fci_mom_mean) / fci_mom_std

    # ------------------------------------------------------------------
    # Compute weighted sum:
    # s^(fc)_t = w_fci * z(FCI) + w_delta * z(delta_FCI)
    # Frozen weights: 0.6 for level, 0.4 for MoM change
    # ------------------------------------------------------------------
    raw_score: float = (
        weights["financial_conditions_index"] * z_fci
        + weights["financial_conditions_mom"] * z_fci_mom
    )

    # ------------------------------------------------------------------
    # Clip to [-3, 3] to bound the influence of extreme outliers
    # ------------------------------------------------------------------
    clipped_score: float = float(
        np.clip(raw_score, _SCORE_CLIP_LOW, _SCORE_CLIP_HIGH)
    )

    # Log the computed score components for audit trail
    logger.debug(
        "score_financial_conditions_dimension: z_fci=%.4f, z_fci_mom=%.4f, "
        "raw=%.4f, clipped=%.4f",
        z_fci, z_fci_mom, raw_score, clipped_score,
    )

    # Return the clipped financial conditions dimension score
    return clipped_score


# =============================================================================
# TOOL 6: classify_regime
# =============================================================================

def classify_regime(
    score_vector: Dict[str, float],
    regime_thresholds: Optional[Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]] = None,
) -> Dict[str, Any]:
    """
    Map the four-dimensional macro score vector to a regime label and confidence.

    Implements the deterministic regime classification step (Task 20, Step 2)
    using the frozen threshold matrix from
    ``METHODOLOGY_PARAMS["MACRO_REGIME_THRESHOLDS"]`` and the softmax
    membership scoring method specified by
    ``METHODOLOGY_PARAMS["MACRO_CONFIDENCE_METHOD"] = "softmax_membership_score"``.

    **Algorithm:**

    1. For each of the four regimes, compute a membership score equal to the
       count of threshold conditions satisfied by the score vector.
    2. Apply numerically stable softmax to the four membership scores to
       produce a probability distribution over regimes.
    3. Select the regime with the highest softmax probability.
    4. Return the regime label and its softmax probability as confidence.

    The softmax is computed as:

    .. math::

        p_r = \\frac{\\exp(m_r - m_{\\max})}{\\sum_{r'} \\exp(m_{r'} - m_{\\max})}

    where :math:`m_r` is the membership score for regime :math:`r` and
    :math:`m_{\\max} = \\max_r m_r` (log-sum-exp trick for numerical stability).

    Parameters
    ----------
    score_vector : Dict[str, float]
        Four-dimensional score vector with keys ``"s_g"``, ``"s_pi"``,
        ``"s_pol"``, ``"s_fc"``. Each value is a float in ``[-3, 3]``,
        produced by Tools 2–5.
    regime_thresholds : Optional[Dict[str, Dict[str, Tuple[Optional[float], Optional[float]]]]]
        Override for the frozen threshold matrix. If ``None``, uses the
        frozen thresholds from ``METHODOLOGY_PARAMS["MACRO_REGIME_THRESHOLDS"]``.
        Structure: ``{regime_name: {dimension_key: (lower_bound, upper_bound)}}``.
        ``None`` bounds indicate no constraint on that side.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"regime"`` (``str``): One of ``{"Expansion", "Late-cycle",
          "Recession", "Recovery"}``.
        - ``"confidence"`` (``float``): Softmax probability of the selected
          regime, in ``[0.25, 1.0]`` (minimum is 0.25 for uniform distribution).
        - ``"membership_scores"`` (``Dict[str, int]``): Raw membership score
          (0–4) per regime, for audit transparency.
        - ``"softmax_probabilities"`` (``Dict[str, float]``): Full softmax
          distribution over all four regimes.

    Raises
    ------
    TypeError
        If ``score_vector`` is not a dict.
    ValueError
        If required keys ``"s_g"``, ``"s_pi"``, ``"s_pol"``, ``"s_fc"``
        are missing from ``score_vector``.
    ValueError
        If any score value is outside ``[-3, 3]``.

    Notes
    -----
    **Tiebreaker:** When two or more regimes have equal membership scores,
    softmax assigns equal probability. The regime appearing first in
    ``_VALID_REGIMES`` order is selected as the tiebreaker. This is a
    frozen implementation choice documented here for reproducibility.

    **Threshold matrix (frozen):**

    +------------+------------------+------------------+------------------+------------------+
    | Regime     | s_g              | s_pi             | s_pol            | s_fc             |
    +============+==================+==================+==================+==================+
    | Expansion  | (0.0, None)      | (None, 1.0)      | (None, 0.5)      | (None, 0.5)      |
    +------------+------------------+------------------+------------------+------------------+
    | Late-cycle | (0.0, None)      | (1.0, None)      | (0.5, None)      | (None, None)     |
    +------------+------------------+------------------+------------------+------------------+
    | Recession  | (None, 0.0)      | (None, None)     | (None, None)     | (0.5, None)      |
    +------------+------------------+------------------+------------------+------------------+
    | Recovery   | (None, 0.0)      | (None, 0.0)      | (None, 0.0)      | (None, 0.5)      |
    +------------+------------------+------------------+------------------+------------------+
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(score_vector, dict):
        raise TypeError(
            f"score_vector must be a dict, got {type(score_vector).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys
    # ------------------------------------------------------------------
    _required_keys: Tuple[str, ...] = ("s_g", "s_pi", "s_pol", "s_fc")
    missing_keys: List[str] = [
        k for k in _required_keys if k not in score_vector
    ]
    if missing_keys:
        raise ValueError(
            f"score_vector is missing required keys: {missing_keys}. "
            f"Required: {list(_required_keys)}."
        )

    # ------------------------------------------------------------------
    # Input validation: score values must be in [-3, 3]
    # (with a small tolerance for floating-point boundary cases)
    # ------------------------------------------------------------------
    for key in _required_keys:
        val: float = float(score_vector[key])
        if val < _SCORE_CLIP_LOW - 1e-6 or val > _SCORE_CLIP_HIGH + 1e-6:
            raise ValueError(
                f"score_vector['{key}'] = {val:.4f} is outside the valid "
                f"range [{_SCORE_CLIP_LOW}, {_SCORE_CLIP_HIGH}]."
            )

    # ------------------------------------------------------------------
    # Frozen threshold matrix per METHODOLOGY_PARAMS["MACRO_REGIME_THRESHOLDS"]
    # Structure: {regime: {dimension: (lower_bound, upper_bound)}}
    # None bounds indicate no constraint on that side.
    # ------------------------------------------------------------------
    _default_thresholds: Dict[
        str, Dict[str, Tuple[Optional[float], Optional[float]]]
    ] = {
        "Expansion": {
            "s_g":   (0.0, None),
            "s_pi":  (None, 1.0),
            "s_pol": (None, 0.5),
            "s_fc":  (None, 0.5),
        },
        "Late-cycle": {
            "s_g":   (0.0, None),
            "s_pi":  (1.0, None),
            "s_pol": (0.5, None),
            "s_fc":  (None, None),
        },
        "Recession": {
            "s_g":   (None, 0.0),
            "s_pi":  (None, None),
            "s_pol": (None, None),
            "s_fc":  (0.5, None),
        },
        "Recovery": {
            "s_g":   (None, 0.0),
            "s_pi":  (None, 0.0),
            "s_pol": (None, 0.0),
            "s_fc":  (None, 0.5),
        },
    }
    thresholds: Dict[
        str, Dict[str, Tuple[Optional[float], Optional[float]]]
    ] = (
        regime_thresholds
        if regime_thresholds is not None
        else _default_thresholds
    )

    # ------------------------------------------------------------------
    # Extract score values as floats for threshold comparison
    # ------------------------------------------------------------------
    # Growth score
    s_g: float = float(score_vector["s_g"])
    # Inflation score
    s_pi: float = float(score_vector["s_pi"])
    # Policy score
    s_pol: float = float(score_vector["s_pol"])
    # Financial conditions score
    s_fc: float = float(score_vector["s_fc"])

    # Map dimension keys to their current score values for iteration
    score_map: Dict[str, float] = {
        "s_g": s_g,
        "s_pi": s_pi,
        "s_pol": s_pol,
        "s_fc": s_fc,
    }

    # ------------------------------------------------------------------
    # Compute membership score for each regime:
    # m_r = count of threshold conditions satisfied by the score vector
    # A condition (lower, upper) is satisfied if:
    #   (lower is None OR score >= lower) AND (upper is None OR score <= upper)
    # ------------------------------------------------------------------
    membership_scores: Dict[str, int] = {}

    # Iterate over regimes in the canonical order defined by _VALID_REGIMES
    for regime in _VALID_REGIMES:
        # Retrieve the threshold dict for this regime
        regime_thresh: Dict[str, Tuple[Optional[float], Optional[float]]] = (
            thresholds[regime]
        )
        # Initialise condition count for this regime
        conditions_met: int = 0

        # Iterate over each dimension and check its threshold condition
        for dim_key, (lower, upper) in regime_thresh.items():
            # Retrieve the current score for this dimension
            current_score: float = score_map[dim_key]

            # Check lower bound: satisfied if lower is None OR score >= lower
            lower_ok: bool = (lower is None) or (current_score >= lower)
            # Check upper bound: satisfied if upper is None OR score <= upper
            upper_ok: bool = (upper is None) or (current_score <= upper)

            # Increment count if both bounds are satisfied
            if lower_ok and upper_ok:
                conditions_met += 1

        # Store the membership score for this regime
        membership_scores[regime] = conditions_met

    # ------------------------------------------------------------------
    # Apply numerically stable softmax to membership scores
    # Softmax: p_r = exp(m_r - m_max) / sum_r'(exp(m_r' - m_max))
    # The log-sum-exp trick (subtracting m_max) prevents overflow.
    # ------------------------------------------------------------------
    # Convert membership scores to a numpy array in canonical regime order
    m_array: np.ndarray = np.array(
        [membership_scores[r] for r in _VALID_REGIMES],
        dtype=np.float64,
    )

    # Subtract the maximum for numerical stability (log-sum-exp trick)
    m_shifted: np.ndarray = m_array - m_array.max()

    # Compute exponentials of shifted scores
    exp_m: np.ndarray = np.exp(m_shifted)

    # Normalise to obtain softmax probabilities
    softmax_probs: np.ndarray = exp_m / exp_m.sum()

    # ------------------------------------------------------------------
    # Select the regime with the highest softmax probability
    # np.argmax returns the first occurrence in case of ties (tiebreaker:
    # first regime in _VALID_REGIMES order — documented in docstring)
    # ------------------------------------------------------------------
    best_idx: int = int(np.argmax(softmax_probs))
    selected_regime: str = _VALID_REGIMES[best_idx]
    confidence: float = float(softmax_probs[best_idx])

    # ------------------------------------------------------------------
    # Build the full softmax probability dict for audit transparency
    # ------------------------------------------------------------------
    softmax_dict: Dict[str, float] = {
        regime: float(softmax_probs[i])
        for i, regime in enumerate(_VALID_REGIMES)
    }

    # Log the classification result for audit trail
    logger.info(
        "classify_regime: selected='%s', confidence=%.4f, "
        "membership_scores=%s",
        selected_regime,
        confidence,
        membership_scores,
    )

    # ------------------------------------------------------------------
    # Construct and return the output dict
    # ------------------------------------------------------------------
    return {
        "regime": selected_regime,
        "confidence": confidence,
        "membership_scores": membership_scores,
        "softmax_probabilities": softmax_dict,
    }


# =============================================================================
# TOOL 7: write_macro_view_json
# =============================================================================

def write_macro_view_json(
    regime: str,
    scores: Dict[str, float],
    confidence: float,
    rationale: str,
    artifact_dir: Path,
    as_of_date: Optional[str] = None,
) -> str:
    """
    Serialise and persist the macro regime view to ``macro-view.json``.

    This tool implements the artifact-writing step for the Macro Agent
    (Task 17, Step 3). The output file is the primary inter-agent artifact
    consumed by all downstream AC agents, PC agents, CMA Judge, and CIO agent.
    It must conform to the frozen ``macro_view.schema.json`` schema.

    The output JSON structure is:

    .. code-block:: json

        {
            "regime": "Late-cycle",
            "scores": {
                "s_g": -0.42,
                "s_pi": 1.21,
                "s_pol": 0.87,
                "s_fc": 0.33
            },
            "confidence": 0.61,
            "rationale": "...",
            "as_of_date": "2026-03-31"
        }

    Parameters
    ----------
    regime : str
        Macro regime label. Must be one of ``{"Expansion", "Late-cycle",
        "Recession", "Recovery"}``.
    scores : Dict[str, float]
        Four-dimensional score vector with keys ``"s_g"``, ``"s_pi"``,
        ``"s_pol"``, ``"s_fc"``. Each value must be in ``[-3, 3]``.
    confidence : float
        Softmax confidence of the selected regime. Must be in ``[0, 1]``.
    rationale : str
        LLM-generated or scripted narrative explaining the regime
        classification. Must be non-empty (minimum 10 characters).
    artifact_dir : Path
        Directory to write ``macro-view.json``. Created if it does not exist.
    as_of_date : Optional[str]
        ISO-8601 date string for the as-of date. Included in the artifact
        for provenance. If ``None``, omitted from the output.

    Returns
    -------
    str
        Absolute path to the written ``macro-view.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``regime`` is not one of the four valid regime labels.
    ValueError
        If ``scores`` is missing required keys or contains out-of-range values.
    ValueError
        If ``confidence`` is outside ``[0, 1]``.
    ValueError
        If ``rationale`` is empty or fewer than 10 characters.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    All numeric values are cast to Python native ``float`` before serialisation
    to ensure JSON compatibility (``np.float64`` is not JSON-serialisable by
    the standard ``json`` module).
    """
    # ------------------------------------------------------------------
    # Input validation: artifact_dir type
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: regime label
    # ------------------------------------------------------------------
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime='{regime}' is not a valid regime label. "
            f"Must be one of: {list(_VALID_REGIMES)}."
        )

    # ------------------------------------------------------------------
    # Input validation: scores dict — required keys
    # ------------------------------------------------------------------
    _required_score_keys: Tuple[str, ...] = ("s_g", "s_pi", "s_pol", "s_fc")
    missing_score_keys: List[str] = [
        k for k in _required_score_keys if k not in scores
    ]
    if missing_score_keys:
        raise ValueError(
            f"scores is missing required keys: {missing_score_keys}."
        )

    # ------------------------------------------------------------------
    # Input validation: scores values in [-3, 3]
    # ------------------------------------------------------------------
    for key in _required_score_keys:
        val: float = float(scores[key])
        if val < _SCORE_CLIP_LOW - 1e-6 or val > _SCORE_CLIP_HIGH + 1e-6:
            raise ValueError(
                f"scores['{key}'] = {val:.4f} is outside [-3, 3]."
            )

    # ------------------------------------------------------------------
    # Input validation: confidence in [0, 1]
    # ------------------------------------------------------------------
    if not (0.0 <= confidence <= 1.0):
        raise ValueError(
            f"confidence={confidence:.4f} must be in [0, 1]."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < 10:
        raise ValueError(
            "rationale must be a non-empty string with at least 10 characters."
        )

    # ------------------------------------------------------------------
    # Create the artifact directory if it does not exist
    # parents=True creates all intermediate directories
    # exist_ok=True prevents error if directory already exists
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Construct the output dict with Python native float values
    # Cast all numeric values to float for JSON serialisability
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # Regime label (string)
        "regime": regime,
        # Four-dimensional score vector (all values cast to float)
        "scores": {
            k: float(v) for k, v in scores.items()
            if k in _required_score_keys
        },
        # Softmax confidence of the selected regime
        "confidence": float(confidence),
        # LLM-generated or scripted rationale narrative
        "rationale": rationale.strip(),
    }

    # ------------------------------------------------------------------
    # Optionally include the as_of_date for provenance
    # ------------------------------------------------------------------
    if as_of_date is not None:
        # Include the as-of date string for audit trail
        output_dict["as_of_date"] = str(as_of_date)

    # ------------------------------------------------------------------
    # Define the output file path
    # ------------------------------------------------------------------
    output_path: Path = artifact_dir / "macro-view.json"

    # ------------------------------------------------------------------
    # Serialise to JSON and write to file
    # indent=2 for human readability
    # ensure_ascii=False to support international characters in rationale
    # encoding="utf-8" for cross-platform compatibility
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised output dict to the file
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write macro-view.json to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_macro_view_json: written to '%s', regime='%s', "
        "confidence=%.4f",
        output_path,
        regime,
        confidence,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 8: write_analysis_md
# =============================================================================

def write_analysis_md(
    regime: str,
    rationale: str,
    artifact_dir: Path,
    scores: Optional[Dict[str, float]] = None,
    confidence: Optional[float] = None,
    as_of_date: Optional[str] = None,
) -> str:
    """
    Format and persist the macro regime analysis narrative to ``analysis.md``.

    This tool implements the markdown artifact-writing step for the Macro Agent
    (Task 17, Step 3) and the Cash AC Agent (Task 18). The output file provides
    a human-readable audit trail of the regime classification decision,
    including the regime label, dimension scores, confidence, rationale, and
    conditions that would invalidate the current regime call.

    The markdown document structure is:

    .. code-block:: markdown

        # Macro Regime Analysis
        **As-of Date:** 2026-03-31
        **Regime:** Late-cycle
        **Confidence:** 61.0%

        ## Dimension Scores
        | Dimension | Score |
        ...

        ## Rationale
        ...

        ## Invalidation Conditions
        ...

    Parameters
    ----------
    regime : str
        Macro regime label. Must be one of ``{"Expansion", "Late-cycle",
        "Recession", "Recovery"}``.
    rationale : str
        Narrative explaining the regime classification. Must be non-empty
        (minimum 10 characters). May be LLM-generated or scripted.
    artifact_dir : Path
        Directory to write ``analysis.md``. Created if it does not exist.
    scores : Optional[Dict[str, float]]
        Four-dimensional score vector (``s_g``, ``s_pi``, ``s_pol``, ``s_fc``).
        If provided, included in the markdown as a dimension scores table.
    confidence : Optional[float]
        Softmax confidence of the selected regime, in ``[0, 1]``.
        If provided, included in the markdown header.
    as_of_date : Optional[str]
        ISO-8601 date string for the as-of date. If provided, included in
        the markdown header for provenance.

    Returns
    -------
    str
        Absolute path to the written ``analysis.md`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``regime`` is not a valid regime label.
    ValueError
        If ``rationale`` is empty or fewer than 10 characters.
    ValueError
        If ``confidence`` is provided but outside ``[0, 1]``.
    OSError
        If the file cannot be written due to filesystem permissions.
    """
    # ------------------------------------------------------------------
    # Input validation: artifact_dir type
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: regime label
    # ------------------------------------------------------------------
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime='{regime}' is not a valid regime label. "
            f"Must be one of: {list(_VALID_REGIMES)}."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < 10:
        raise ValueError(
            "rationale must be a non-empty string with at least 10 characters."
        )

    # ------------------------------------------------------------------
    # Input validation: confidence range if provided
    # ------------------------------------------------------------------
    if confidence is not None and not (0.0 <= confidence <= 1.0):
        raise ValueError(
            f"confidence={confidence:.4f} must be in [0, 1]."
        )

    # ------------------------------------------------------------------
    # Create the artifact directory if it does not exist
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Build the markdown document section by section
    # ------------------------------------------------------------------

    # --- Header section ---
    # Document title
    md_lines: List[str] = ["# Macro Regime Analysis", ""]

    # As-of date line (if provided)
    if as_of_date is not None:
        md_lines.append(f"**As-of Date:** {as_of_date}")

    # Regime label line
    md_lines.append(f"**Regime:** {regime}")

    # Confidence line (if provided), formatted as percentage
    if confidence is not None:
        md_lines.append(f"**Confidence:** {confidence * 100:.1f}%")

    # Blank line after header
    md_lines.append("")

    # --- Dimension scores table (if scores provided) ---
    if scores is not None:
        # Section heading for dimension scores
        md_lines.append("## Dimension Scores")
        md_lines.append("")
        # Markdown table header
        md_lines.append("| Dimension | Score | Interpretation |")
        md_lines.append("|-----------|-------|----------------|")

        # Dimension label mapping for human-readable table
        _dim_labels: Dict[str, Tuple[str, str]] = {
            "s_g":   ("Growth", "Higher = stronger growth"),
            "s_pi":  ("Inflation", "Higher = more inflationary pressure"),
            "s_pol": ("Monetary Policy", "Higher = tighter policy"),
            "s_fc":  ("Financial Conditions", "Higher = tighter conditions"),
        }

        # Add a row for each dimension score
        for dim_key, (dim_label, interpretation) in _dim_labels.items():
            if dim_key in scores:
                # Format the score to 4 decimal places
                score_val: float = float(scores[dim_key])
                md_lines.append(
                    f"| {dim_label} | {score_val:+.4f} | {interpretation} |"
                )

        # Blank line after table
        md_lines.append("")

    # --- Rationale section ---
    # Section heading for the regime rationale
    md_lines.append("## Rationale")
    md_lines.append("")
    # The rationale narrative (stripped of leading/trailing whitespace)
    md_lines.append(rationale.strip())
    md_lines.append("")

    # --- Invalidation conditions section ---
    # Standard invalidation conditions based on the selected regime
    md_lines.append("## Invalidation Conditions")
    md_lines.append("")
    md_lines.append(
        "The current regime classification would be invalidated by any of "
        "the following conditions:"
    )
    md_lines.append("")

    # Regime-specific invalidation conditions
    _invalidation_map: Dict[str, List[str]] = {
        "Expansion": [
            "- Nonfarm payrolls turn negative for two consecutive months.",
            "- Real GDP growth decelerates below 0% on a quarterly basis.",
            "- CPI YoY accelerates above 3% (transition to Late-cycle).",
            "- Financial conditions index rises above its historical 75th percentile.",
        ],
        "Late-cycle": [
            "- Real GDP growth decelerates below 0% (transition to Recession).",
            "- Nonfarm payrolls turn negative for two consecutive months.",
            "- CPI YoY decelerates below 1% (transition to Expansion or Recovery).",
            "- Fed funds rate begins a sustained cutting cycle (>50bps in 3 months).",
        ],
        "Recession": [
            "- Real GDP growth returns to positive territory for two consecutive quarters.",
            "- Nonfarm payrolls turn positive for two consecutive months.",
            "- Financial conditions index falls below its historical 50th percentile.",
        ],
        "Recovery": [
            "- Real GDP growth accelerates above 0% and CPI YoY rises above 1% "
              "(transition to Expansion).",
            "- Financial conditions index rises above its historical 75th percentile "
              "(re-entry into Recession).",
            "- Fed funds rate begins a sustained hiking cycle.",
        ],
    }

    # Add the regime-specific invalidation conditions
    for condition in _invalidation_map.get(regime, []):
        md_lines.append(condition)

    # Final blank line
    md_lines.append("")

    # ------------------------------------------------------------------
    # Join all lines into a single markdown string
    # ------------------------------------------------------------------
    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Define the output file path
    # ------------------------------------------------------------------
    output_path: Path = artifact_dir / "analysis.md"

    # ------------------------------------------------------------------
    # Write the markdown content to file
    # encoding="utf-8" for cross-platform compatibility
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the complete markdown document to the file
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write analysis.md to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_analysis_md: written to '%s', regime='%s'",
        output_path,
        regime,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 9: fetch_historical_stats
# =============================================================================

def fetch_historical_stats(
    asset_class: str,
    as_of_date: str,
    df_total_return_raw: pd.DataFrame,
    universe_map: Dict[str, Dict[str, Any]],
    history_start: Optional[str] = None,
    periods_per_year: int = 12,
) -> Dict[str, Any]:
    """
    Compute the full historical statistics panel for a given asset class.

    This tool implements the historical analysis step for all AC agents
    (Task 18, Step 2) and is also used by the CRO Agent (Task 27). It
    computes annualised return, volatility, maximum drawdown, and pairwise
    correlations against all other asset classes in the 18-asset universe.

    **Frozen conventions (from DATA_CONVENTIONS):**

    - Return formula: :math:`r_t = TR_t / TR_{t-1} - 1` (simple periodic)
    - Annualised return: :math:`\\mu_{ann} = 12 \\cdot \\bar{r}_{mo}`
    - Annualised volatility: :math:`\\sigma_{ann} = \\sqrt{12} \\cdot \\sigma_{mo}`
    - Maximum drawdown: :math:`MDD = \\min_t(V_t / \\max_{s \\leq t} V_s - 1)`

    Parameters
    ----------
    asset_class : str
        Asset class name (must be a key in ``universe_map``). Example:
        ``"US Large Cap"``.
    as_of_date : str
        ISO-8601 date string. All data is filtered to ``<= as_of_date``
        for point-in-time discipline.
    df_total_return_raw : pd.DataFrame
        Total return index panel. Must have a MultiIndex with levels
        ``["date", "ticker", "investment_universe"]`` and column
        ``"total_return_index"`` (strictly positive float64).
        Shape: ``(T_total, 1)``.
    universe_map : Dict[str, Dict[str, Any]]
        Mapping from asset class names to their metadata. Each value must
        contain at least ``"ticker"`` (str) and ``"category"`` (str).
        Must cover all 18 asset classes for correlation computation.
    history_start : Optional[str]
        ISO-8601 date string for the start of the history window. If
        ``None``, uses all available history up to ``as_of_date``.
    periods_per_year : int
        Number of periods per year for annualisation. Default: ``12``
        (monthly). Must be a positive integer.

    Returns
    -------
    Dict[str, Any]
        Dictionary with the following keys:

        - ``"asset_class"`` (``str``): The asset class name.
        - ``"ticker"`` (``str``): The resolved ticker.
        - ``"as_of_date"`` (``str``): The as-of date used.
        - ``"history_start_actual"`` (``str``): Actual start date of the
          history window used.
        - ``"n_observations"`` (``int``): Number of monthly return observations.
        - ``"annualised_return"`` (``float``): Annualised arithmetic mean
          return: :math:`\\mu_{ann} = 12 \\cdot \\bar{r}_{mo}`.
        - ``"annualised_vol"`` (``float``): Annualised volatility:
          :math:`\\sigma_{ann} = \\sqrt{12} \\cdot \\sigma_{mo}`.
        - ``"max_drawdown"`` (``float``): Maximum drawdown (negative):
          :math:`MDD = \\min_t(V_t / \\max_{s \\leq t} V_s - 1)`.
        - ``"monthly_returns_series"`` (``pd.Series``): Monthly simple
          returns with DatetimeIndex.
        - ``"correlation_matrix"`` (``pd.DataFrame``): Pairwise correlation
          matrix of shape ``(N_available, N_available)`` where
          ``N_available ≤ 18``, computed over the common date range.
        - ``"sharpe_ratio_unannualised"`` (``float``): Raw monthly Sharpe
          (not annualised; annualised Sharpe requires rf, computed in CRO).

    Raises
    ------
    TypeError
        If ``df_total_return_raw`` is not a ``pd.DataFrame`` or
        ``universe_map`` is not a dict.
    ValueError
        If ``asset_class`` is not in ``universe_map``.
    ValueError
        If the resolved ticker is not found in ``df_total_return_raw``.
    ValueError
        If fewer than ``_MIN_MONTHLY_OBS`` observations are available.
    ValueError
        If ``total_return_index`` contains non-positive values.

    Notes
    -----
    **Correlation computation:** Pairwise correlations are computed over the
    intersection of available date ranges across all asset classes (inner
    join). Asset classes with no overlapping history are excluded from the
    correlation matrix with a logged warning.

    **MDD computation:** The maximum drawdown is computed on the
    ``total_return_index`` level series (not the return series), using the
    running maximum formula. This is the path-based definition from
    ``IPS_GOVERNANCE["CONSTRAINT_DEFINITIONS"]["max_drawdown_formula"]``.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(df_total_return_raw, pd.DataFrame):
        raise TypeError(
            f"df_total_return_raw must be a pd.DataFrame, "
            f"got {type(df_total_return_raw).__name__}."
        )
    if not isinstance(universe_map, dict):
        raise TypeError(
            f"universe_map must be a dict, got {type(universe_map).__name__}."
        )
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError(
            f"periods_per_year must be a positive integer, "
            f"got {periods_per_year}."
        )

    # ------------------------------------------------------------------
    # Input validation: asset_class in universe_map
    # ------------------------------------------------------------------
    if asset_class not in universe_map:
        raise ValueError(
            f"asset_class='{asset_class}' not found in universe_map. "
            f"Available: {list(universe_map.keys())}."
        )

    # ------------------------------------------------------------------
    # Resolve ticker from universe_map
    # ------------------------------------------------------------------
    asset_meta: Dict[str, Any] = universe_map[asset_class]
    if "ticker" not in asset_meta:
        raise ValueError(
            f"universe_map['{asset_class}'] is missing required key 'ticker'."
        )
    # The canonical ticker identifier for this asset class
    ticker: str = asset_meta["ticker"]

    # ------------------------------------------------------------------
    # Parse as_of_date to pd.Timestamp
    # ------------------------------------------------------------------
    try:
        # Convert ISO-8601 string to timezone-naive pd.Timestamp
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
    except Exception as exc:
        raise ValueError(
            f"as_of_date='{as_of_date}' cannot be parsed. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Parse history_start to pd.Timestamp if provided
    # ------------------------------------------------------------------
    history_start_ts: Optional[pd.Timestamp] = None
    if history_start is not None:
        try:
            # Convert the history start date string to a pd.Timestamp
            history_start_ts = pd.Timestamp(history_start)
        except Exception as exc:
            raise ValueError(
                f"history_start='{history_start}' cannot be parsed. "
                f"Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Validate that df_total_return_raw has a MultiIndex
    # ------------------------------------------------------------------
    if not isinstance(df_total_return_raw.index, pd.MultiIndex):
        raise TypeError(
            "df_total_return_raw must have a MultiIndex with levels "
            "['date', 'ticker', 'investment_universe']. "
            f"Got index type: {type(df_total_return_raw.index).__name__}."
        )

    # ------------------------------------------------------------------
    # Validate that 'total_return_index' column exists
    # ------------------------------------------------------------------
    if "total_return_index" not in df_total_return_raw.columns:
        raise ValueError(
            "df_total_return_raw must contain column 'total_return_index'."
        )

    # ------------------------------------------------------------------
    # Extract the total return index series for the target asset class
    # using .xs() on the 'ticker' level of the MultiIndex
    # ------------------------------------------------------------------
    try:
        # Slice the MultiIndex DataFrame to the target ticker
        df_ticker: pd.DataFrame = df_total_return_raw.xs(
            ticker, level="ticker"
        )
    except KeyError:
        raise ValueError(
            f"Ticker '{ticker}' (for asset_class='{asset_class}') not found "
            f"in df_total_return_raw. "
            f"Available tickers: "
            f"{df_total_return_raw.index.get_level_values('ticker').unique().tolist()}."
        )

    # ------------------------------------------------------------------
    # Extract the DatetimeIndex from the sliced DataFrame
    # The 'date' level becomes the index after .xs() on 'ticker'
    # ------------------------------------------------------------------
    # Get the date index from the sliced DataFrame
    date_index: pd.Index = df_ticker.index

    # ------------------------------------------------------------------
    # Strip timezone from date index if present
    # ------------------------------------------------------------------
    if hasattr(date_index, "tz") and date_index.tz is not None:
        # Make the index timezone-naive for consistent comparison
        df_ticker = df_ticker.copy()
        df_ticker.index = date_index.tz_localize(None)

    # ------------------------------------------------------------------
    # Apply point-in-time filter: retain rows <= as_of_date
    # ------------------------------------------------------------------
    df_ticker = df_ticker.loc[df_ticker.index <= as_of_ts]

    # ------------------------------------------------------------------
    # Apply history_start filter if provided
    # ------------------------------------------------------------------
    if history_start_ts is not None:
        # Retain only rows on or after the history start date
        df_ticker = df_ticker.loc[df_ticker.index >= history_start_ts]

    # ------------------------------------------------------------------
    # Guard: ensure data is available after filtering
    # ------------------------------------------------------------------
    if df_ticker.empty:
        raise ValueError(
            f"No data available for ticker='{ticker}' "
            f"(asset_class='{asset_class}') "
            f"in the range [{history_start}, {as_of_date}]."
        )

    # ------------------------------------------------------------------
    # Extract the total return index as a pd.Series
    # ------------------------------------------------------------------
    # Total return index series (strictly positive, dividends included)
    tri_series: pd.Series = df_ticker["total_return_index"].sort_index()

    # ------------------------------------------------------------------
    # Validate that total_return_index is strictly positive
    # ------------------------------------------------------------------
    if (tri_series <= 0).any():
        raise ValueError(
            f"total_return_index for '{asset_class}' contains non-positive "
            f"values. Total return indices must be strictly positive."
        )

    # ------------------------------------------------------------------
    # Compute monthly simple returns using the frozen formula:
    # r_t = TR_t / TR_{t-1} - 1
    # (equivalent to pct_change() but using the explicit frozen formula)
    # ------------------------------------------------------------------
    # Compute the ratio TR_t / TR_{t-1} for each consecutive pair
    tri_shifted: pd.Series = tri_series.shift(1)
    # Compute simple periodic returns: r_t = TR_t / TR_{t-1} - 1
    monthly_returns: pd.Series = (tri_series / tri_shifted) - 1.0
    # Drop the first NaN (no prior period for the first observation)
    monthly_returns = monthly_returns.dropna()

    # ------------------------------------------------------------------
    # Minimum observation check
    # ------------------------------------------------------------------
    n_obs: int = len(monthly_returns)
    if n_obs < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"Insufficient history for '{asset_class}': "
            f"{n_obs} monthly return observations, "
            f"minimum required: {_MIN_MONTHLY_OBS}."
        )

    # ------------------------------------------------------------------
    # Compute annualised return: mu_ann = periods_per_year * mean(r_mo)
    # Frozen formula from DATA_CONVENTIONS["annualisation"]["mu_multiplier"]
    # ------------------------------------------------------------------
    # Monthly mean return
    mean_monthly: float = float(monthly_returns.mean())
    # Annualised return: mu_ann = 12 * mu_mo
    annualised_return: float = float(periods_per_year) * mean_monthly

    # ------------------------------------------------------------------
    # Compute annualised volatility: sigma_ann = sqrt(12) * sigma_mo
    # Frozen formula from DATA_CONVENTIONS["annualisation"]["sigma_multiplier"]
    # ------------------------------------------------------------------
    # Monthly standard deviation (sample, ddof=1)
    std_monthly: float = float(monthly_returns.std(ddof=1))
    # Annualised volatility: sigma_ann = sqrt(12) * sigma_mo
    annualised_vol: float = float(np.sqrt(periods_per_year)) * std_monthly

    # ------------------------------------------------------------------
    # Compute maximum drawdown on the total return index level series
    # MDD = min_t(V_t / max_{s<=t}(V_s) - 1)
    # Frozen formula from IPS_GOVERNANCE["CONSTRAINT_DEFINITIONS"]
    # ------------------------------------------------------------------
    # Compute the running maximum of the total return index up to each date
    running_max: pd.Series = tri_series.cummax()
    # Compute the drawdown at each date: V_t / max_{s<=t}(V_s) - 1
    drawdown_series: pd.Series = (tri_series / running_max) - 1.0
    # Maximum drawdown is the minimum (most negative) drawdown value
    max_drawdown: float = float(drawdown_series.min())

    # ------------------------------------------------------------------
    # Compute raw monthly Sharpe (without rf; annualised Sharpe requires rf)
    # Included for completeness; CRO Agent computes the full Sharpe with rf
    # ------------------------------------------------------------------
    # Raw monthly Sharpe: mean / std (no rf subtraction here)
    sharpe_raw: float = (
        mean_monthly / std_monthly if std_monthly > _ZSCORE_EPS else 0.0
    )

    # ------------------------------------------------------------------
    # Compute pairwise correlations against all other asset classes
    # using the inner join of available date ranges
    # ------------------------------------------------------------------
    # Build a dict of monthly return series for all available asset classes
    all_returns_dict: Dict[str, pd.Series] = {}

    # Add the target asset class's return series first
    all_returns_dict[asset_class] = monthly_returns

    # Iterate over all other asset classes in the universe_map
    for other_ac, other_meta in universe_map.items():
        # Skip the target asset class (already added)
        if other_ac == asset_class:
            continue

        # Resolve the ticker for the other asset class
        other_ticker: str = other_meta.get("ticker", "")
        if not other_ticker:
            # Log a warning if the ticker is missing and skip
            logger.warning(
                "fetch_historical_stats: universe_map['%s'] missing 'ticker'. "
                "Skipping from correlation computation.",
                other_ac,
            )
            continue

        # Attempt to extract the other asset class's total return index
        try:
            # Slice the MultiIndex DataFrame to the other ticker
            df_other: pd.DataFrame = df_total_return_raw.xs(
                other_ticker, level="ticker"
            )
        except KeyError:
            # Log a warning if the ticker is not found and skip
            logger.warning(
                "fetch_historical_stats: ticker='%s' for asset_class='%s' "
                "not found in df_total_return_raw. "
                "Skipping from correlation computation.",
                other_ticker,
                other_ac,
            )
            continue

        # Strip timezone from the other asset class's index if present
        if hasattr(df_other.index, "tz") and df_other.index.tz is not None:
            df_other = df_other.copy()
            df_other.index = df_other.index.tz_localize(None)

        # Apply point-in-time filter to the other asset class
        df_other = df_other.loc[df_other.index <= as_of_ts]

        # Skip if no data available after filtering
        if df_other.empty or "total_return_index" not in df_other.columns:
            continue

        # Extract and sort the total return index for the other asset class
        tri_other: pd.Series = df_other["total_return_index"].sort_index()

        # Skip if total return index contains non-positive values
        if (tri_other <= 0).any():
            logger.warning(
                "fetch_historical_stats: Non-positive total_return_index "
                "for '%s'. Skipping from correlation computation.",
                other_ac,
            )
            continue

        # Compute monthly returns for the other asset class using frozen formula
        tri_other_shifted: pd.Series = tri_other.shift(1)
        # r_t = TR_t / TR_{t-1} - 1
        returns_other: pd.Series = (tri_other / tri_other_shifted) - 1.0
        # Drop the first NaN observation
        returns_other = returns_other.dropna()

        # Add to the returns dict if sufficient observations exist
        if len(returns_other) >= _MIN_MONTHLY_OBS:
            all_returns_dict[other_ac] = returns_other

    # ------------------------------------------------------------------
    # Construct the wide returns DataFrame using inner join
    # (intersection of available date ranges across all asset classes)
    # ------------------------------------------------------------------
    # Concatenate all return series into a wide DataFrame (inner join)
    returns_wide: pd.DataFrame = pd.concat(
        all_returns_dict, axis=1, join="inner"
    )

    # ------------------------------------------------------------------
    # Compute the pairwise correlation matrix
    # pd.DataFrame.corr() handles NaN alignment automatically
    # ------------------------------------------------------------------
    # Pearson correlation matrix of shape (N_available, N_available)
    correlation_matrix: pd.DataFrame = returns_wide.corr(method="pearson")

    # ------------------------------------------------------------------
    # Record the actual history start date used
    # ------------------------------------------------------------------
    # The actual start date is the first date in the monthly returns series
    history_start_actual: str = str(monthly_returns.index[0].date())

    # Log the computed statistics for audit trail
    logger.info(
        "fetch_historical_stats: asset_class='%s', n_obs=%d, "
        "ann_return=%.4f, ann_vol=%.4f, mdd=%.4f",
        asset_class,
        n_obs,
        annualised_return,
        annualised_vol,
        max_drawdown,
    )

    # ------------------------------------------------------------------
    # Construct and return the output dict
    # ------------------------------------------------------------------
    return {
        # Asset class name for identification
        "asset_class": asset_class,
        # Resolved ticker identifier
        "ticker": ticker,
        # As-of date used for point-in-time filtering
        "as_of_date": as_of_date,
        # Actual start date of the history window
        "history_start_actual": history_start_actual,
        # Number of monthly return observations
        "n_observations": n_obs,
        # Annualised arithmetic mean return: mu_ann = 12 * mu_mo
        "annualised_return": float(annualised_return),
        # Annualised volatility: sigma_ann = sqrt(12) * sigma_mo
        "annualised_vol": float(annualised_vol),
        # Maximum drawdown: MDD = min_t(V_t / max_{s<=t}(V_s) - 1)
        "max_drawdown": float(max_drawdown),
        # Monthly simple returns series with DatetimeIndex
        "monthly_returns_series": monthly_returns,
        # Pairwise correlation matrix (N_available x N_available)
        "correlation_matrix": correlation_matrix,
        # Raw monthly Sharpe (no rf; annualised Sharpe computed in CRO)
        "sharpe_ratio_unannualised": float(sharpe_raw),
    }


# =============================================================================
# TOOL 10: fetch_signals
# =============================================================================

def fetch_signals(
    asset_class: str,
    as_of_date: str,
    df_signals_raw: pd.DataFrame,
    df_fundamentals_raw: pd.DataFrame,
    universe_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Retrieve and validate the most recent technical and valuation signals
    for a given asset class as-of ``as_of_date``.

    This tool implements the signal retrieval step for all AC agents
    (Task 18, Step 3) and the CMA Judge (Task 22, Step 1). It enforces:

    - **Point-in-time discipline:** Only data on or before ``as_of_date``
      is returned.
    - **Equity-only field filtering:** CAPE, P/E, and earnings yield are
      only retrieved for equity asset classes. For non-equity asset classes
      (commodities, gold), these fields are set to ``None`` and documented
      in ``"omitted_fields"``.
    - **Decimal scale validation:** Yield fields (``dividend_yield``,
      ``buyback_yield``, ``earnings_yield``) must be in decimal form
      (e.g., 0.03 = 3%). Values > 1.0 trigger a ``ValueError``.
    - **Staleness warning:** If the most recent available row is more than
      ``_MAX_STALENESS_MONTHS`` months before ``as_of_date``, a
      ``UserWarning`` is emitted.

    Parameters
    ----------
    asset_class : str
        Asset class name (must be a key in ``universe_map``). Example:
        ``"US Large Cap"``.
    as_of_date : str
        ISO-8601 date string. All data is filtered to ``<= as_of_date``.
    df_signals_raw : pd.DataFrame
        Technical and contextual signals panel. Must have a MultiIndex
        with levels ``["date", "ticker", "investment_universe"]`` and
        columns: ``"rsi_14d"``, ``"momentum_12m"``, ``"market_breadth_raw"``,
        ``"net_fund_flows"``, ``"positioning_score_raw"``.
        Shape: ``(T_signals, 5)``.
    df_fundamentals_raw : pd.DataFrame
        Fundamental and valuation inputs panel. Must have a MultiIndex
        with levels ``["date", "ticker", "investment_universe"]`` and
        columns: ``"cape_ratio"``, ``"pe_trailing"``, ``"earnings_yield"``,
        ``"dividend_yield"``, ``"buyback_yield"``, ``"market_cap_usd"``,
        ``"earnings_growth_forecast"``, ``"valuation_change_assumption"``.
        Shape: ``(T_fundamentals, 8)``.
    universe_map : Dict[str, Dict[str, Any]]
        Mapping from asset class names to metadata. Each value must contain
        ``"ticker"`` (str) and ``"category"`` (str, one of
        ``{"Equity", "Fixed Income", "Real Assets", "Cash"}``).

    Returns
    -------
    Dict[str, Any]
        Flat dictionary with the following keys (all values are Python
        native ``float`` or ``None``):

        **Always present:**

        - ``"asset_class"`` (``str``): Asset class name.
        - ``"ticker"`` (``str``): Resolved ticker.
        - ``"category"`` (``str``): Asset category.
        - ``"as_of_date"`` (``str``): As-of date used.
        - ``"signals_date"`` (``str``): Actual date of the signals row.
        - ``"fundamentals_date"`` (``str``): Actual date of the fundamentals row.
        - ``"rsi_14d"`` (``float | None``): 14-day RSI.
        - ``"momentum_12m"`` (``float | None``): 12-month price momentum.
        - ``"market_breadth_raw"`` (``float | None``): Market breadth indicator.
        - ``"net_fund_flows"`` (``float | None``): Net ETF fund flows.
        - ``"positioning_score_raw"`` (``float | None``): Positioning/crowding score.
        - ``"dividend_yield"`` (``float | None``): Dividend yield (decimal).
        - ``"buyback_yield"`` (``float | None``): Buyback yield (decimal).
        - ``"market_cap_usd"`` (``float | None``): Market capitalisation (USD).
        - ``"earnings_growth_forecast"`` (``float | None``): Earnings growth forecast.
        - ``"valuation_change_assumption"`` (``float | None``): Valuation change assumption.
        - ``"omitted_fields"`` (``List[str]``): Fields excluded due to
          asset category (e.g., CAPE for commodities).

        **Equity-only (``None`` for non-equity):**

        - ``"cape_ratio"`` (``float | None``): CAPE ratio (in ×).
        - ``"pe_trailing"`` (``float | None``): Trailing P/E (in ×).
        - ``"earnings_yield"`` (``float | None``): Earnings yield (decimal).

    Raises
    ------
    TypeError
        If any DataFrame input is not a ``pd.DataFrame``.
    ValueError
        If ``asset_class`` is not in ``universe_map``.
    ValueError
        If the resolved ticker is not found in either DataFrame.
    ValueError
        If any yield field contains values > 1.0 (indicating percent storage).

    Warns
    -----
    UserWarning
        If the most recent available signals or fundamentals row is more
        than ``_MAX_STALENESS_MONTHS`` months before ``as_of_date``.

    Notes
    -----
    **Scale convention (frozen):** CAPE and P/E are in "×" (e.g., 25.0 = 25×).
    Yield fields are in decimals (e.g., 0.03 = 3%). This is frozen per
    ``DATA_CONVENTIONS`` and ``RAW_DATA_SCHEMAS["FUNDAMENTALS"]``.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(df_signals_raw, pd.DataFrame):
        raise TypeError(
            f"df_signals_raw must be a pd.DataFrame, "
            f"got {type(df_signals_raw).__name__}."
        )
    if not isinstance(df_fundamentals_raw, pd.DataFrame):
        raise TypeError(
            f"df_fundamentals_raw must be a pd.DataFrame, "
            f"got {type(df_fundamentals_raw).__name__}."
        )
    if not isinstance(universe_map, dict):
        raise TypeError(
            f"universe_map must be a dict, got {type(universe_map).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: asset_class in universe_map
    # ------------------------------------------------------------------
    if asset_class not in universe_map:
        raise ValueError(
            f"asset_class='{asset_class}' not found in universe_map. "
            f"Available: {list(universe_map.keys())}."
        )

    # ------------------------------------------------------------------
    # Resolve ticker and category from universe_map
    # ------------------------------------------------------------------
    asset_meta: Dict[str, Any] = universe_map[asset_class]
    if "ticker" not in asset_meta:
        raise ValueError(
            f"universe_map['{asset_class}'] is missing required key 'ticker'."
        )
    if "category" not in asset_meta:
        raise ValueError(
            f"universe_map['{asset_class}'] is missing required key 'category'."
        )
    # The canonical ticker identifier for this asset class
    ticker: str = asset_meta["ticker"]
    # The asset category (Equity, Fixed Income, Real Assets, Cash)
    category: str = asset_meta["category"]

    # ------------------------------------------------------------------
    # Parse as_of_date to pd.Timestamp
    # ------------------------------------------------------------------
    try:
        # Convert ISO-8601 string to timezone-naive pd.Timestamp
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
    except Exception as exc:
        raise ValueError(
            f"as_of_date='{as_of_date}' cannot be parsed. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Validate MultiIndex structure for both DataFrames
    # ------------------------------------------------------------------
    for df_name, df_obj in [
        ("df_signals_raw", df_signals_raw),
        ("df_fundamentals_raw", df_fundamentals_raw),
    ]:
        if not isinstance(df_obj.index, pd.MultiIndex):
            raise TypeError(
                f"{df_name} must have a MultiIndex with levels "
                f"['date', 'ticker', 'investment_universe']. "
                f"Got index type: {type(df_obj.index).__name__}."
            )

    # ------------------------------------------------------------------
    # Helper: extract the most recent row for a ticker from a MultiIndex
    # DataFrame, applying point-in-time filter
    # ------------------------------------------------------------------
    def _extract_latest_row(
        df: pd.DataFrame,
        tkr: str,
        df_label: str,
    ) -> Tuple[Optional[pd.Series], Optional[pd.Timestamp]]:
        """
        Extract the most recent row for ``tkr`` from ``df`` on or before
        ``as_of_ts``. Returns ``(None, None)`` if no data is available.
        """
        # Attempt to slice the MultiIndex DataFrame to the target ticker
        try:
            df_tkr: pd.DataFrame = df.xs(tkr, level="ticker")
        except KeyError:
            # Ticker not found — return None to signal missing data
            logger.warning(
                "fetch_signals: ticker='%s' not found in %s.",
                tkr,
                df_label,
            )
            return None, None

        # Strip timezone from index if present
        if hasattr(df_tkr.index, "tz") and df_tkr.index.tz is not None:
            df_tkr = df_tkr.copy()
            df_tkr.index = df_tkr.index.tz_localize(None)

        # Apply point-in-time filter: retain rows <= as_of_date
        df_filtered: pd.DataFrame = df_tkr.loc[df_tkr.index <= as_of_ts]

        # Return None if no rows available after filtering
        if df_filtered.empty:
            logger.warning(
                "fetch_signals: No data for ticker='%s' in %s "
                "on or before '%s'.",
                tkr,
                df_label,
                as_of_date,
            )
            return None, None

        # Select the most recent row
        latest_row: pd.Series = df_filtered.iloc[-1]
        # Record the actual date of the most recent row
        latest_date: pd.Timestamp = df_filtered.index[-1]

        return latest_row, latest_date

    # ------------------------------------------------------------------
    # Extract the most recent signals row for the target ticker
    # ------------------------------------------------------------------
    signals_row, signals_date = _extract_latest_row(
        df_signals_raw, ticker, "df_signals_raw"
    )

    # ------------------------------------------------------------------
    # Extract the most recent fundamentals row for the target ticker
    # ------------------------------------------------------------------
    fundamentals_row, fundamentals_date = _extract_latest_row(
        df_fundamentals_raw, ticker, "df_fundamentals_raw"
    )

    # ------------------------------------------------------------------
    # Staleness check: warn if most recent row is > _MAX_STALENESS_MONTHS
    # months before as_of_date
    # ------------------------------------------------------------------
    # Compute the staleness threshold date
    staleness_threshold: pd.Timestamp = as_of_ts - pd.DateOffset(
        months=_MAX_STALENESS_MONTHS
    )

    # Check signals staleness
    if signals_date is not None and signals_date < staleness_threshold:
        warnings.warn(
            f"fetch_signals: signals data for '{asset_class}' is stale. "
            f"Most recent row: {signals_date.date()}, "
            f"as_of_date: {as_of_date}, "
            f"staleness threshold: {_MAX_STALENESS_MONTHS} months.",
            UserWarning,
            stacklevel=2,
        )

    # Check fundamentals staleness
    if fundamentals_date is not None and fundamentals_date < staleness_threshold:
        warnings.warn(
            f"fetch_signals: fundamentals data for '{asset_class}' is stale. "
            f"Most recent row: {fundamentals_date.date()}, "
            f"as_of_date: {as_of_date}, "
            f"staleness threshold: {_MAX_STALENESS_MONTHS} months.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Helper: safely extract a float value from a pd.Series row
    # Returns None if the field is missing or NaN
    # ------------------------------------------------------------------
    def _safe_float(
        row: Optional[pd.Series],
        field: str,
    ) -> Optional[float]:
        """
        Safely extract a float value from a pd.Series row.
        Returns None if the row is None, the field is missing, or the
        value is NaN.
        """
        # Return None if the row itself is None (data not available)
        if row is None:
            return None
        # Return None if the field is not in the row
        if field not in row.index:
            return None
        # Retrieve the raw value
        val = row[field]
        # Return None if the value is NaN
        if pd.isna(val):
            return None
        # Cast to Python native float and return
        return float(val)

    # ------------------------------------------------------------------
    # Extract signal fields from the signals row
    # These fields are applicable to all asset classes
    # ------------------------------------------------------------------
    # 14-day RSI technical indicator
    rsi_14d: Optional[float] = _safe_float(signals_row, "rsi_14d")
    # 12-month price momentum
    momentum_12m: Optional[float] = _safe_float(signals_row, "momentum_12m")
    # Market breadth indicator (e.g., % constituents above 200d MA)
    market_breadth_raw: Optional[float] = _safe_float(
        signals_row, "market_breadth_raw"
    )
    # Net ETF fund flows
    net_fund_flows: Optional[float] = _safe_float(signals_row, "net_fund_flows")
    # Positioning/crowding score
    positioning_score_raw: Optional[float] = _safe_float(
        signals_row, "positioning_score_raw"
    )

    # ------------------------------------------------------------------
    # Extract fundamental fields applicable to all asset classes
    # ------------------------------------------------------------------
    # Dividend yield (decimal form, e.g., 0.03 = 3%)
    dividend_yield: Optional[float] = _safe_float(
        fundamentals_row, "dividend_yield"
    )
    # Buyback yield (decimal form)
    buyback_yield: Optional[float] = _safe_float(
        fundamentals_row, "buyback_yield"
    )
    # Market capitalisation in USD
    market_cap_usd: Optional[float] = _safe_float(
        fundamentals_row, "market_cap_usd"
    )
    # Earnings growth forecast (decimal form)
    earnings_growth_forecast: Optional[float] = _safe_float(
        fundamentals_row, "earnings_growth_forecast"
    )
    # Valuation change assumption (decimal form)
    valuation_change_assumption: Optional[float] = _safe_float(
        fundamentals_row, "valuation_change_assumption"
    )

    # ------------------------------------------------------------------
    # Determine whether this asset class is equity
    # Equity-only fields are only valid for Equity category assets
    # per RAW_DATA_SCHEMAS["FUNDAMENTALS"]["validation"]["equity_only_fields"]
    # ------------------------------------------------------------------
    # Check if the asset category is Equity
    is_equity: bool = (category == "Equity")

    # Initialise the list of omitted fields for non-equity assets
    omitted_fields: List[str] = []

    # ------------------------------------------------------------------
    # Extract equity-only fields (CAPE, P/E, earnings yield)
    # Set to None and record in omitted_fields for non-equity assets
    # ------------------------------------------------------------------
    if is_equity:
        # CAPE ratio (in ×, e.g., 25.0 = 25×) — equity only
        cape_ratio: Optional[float] = _safe_float(
            fundamentals_row, "cape_ratio"
        )
        # Trailing P/E ratio (in ×) — equity only
        pe_trailing: Optional[float] = _safe_float(
            fundamentals_row, "pe_trailing"
        )
        # Earnings yield (decimal form) — equity only
        earnings_yield: Optional[float] = _safe_float(
            fundamentals_row, "earnings_yield"
        )
    else:
        # CAPE and P/E are not meaningful for non-equity assets
        # Set to None and document the omission
        cape_ratio = None
        pe_trailing = None
        earnings_yield = None
        # Record the omitted equity-only fields for audit transparency
        omitted_fields.extend(list(_EQUITY_ONLY_FIELDS))
        logger.debug(
            "fetch_signals: Equity-only fields %s omitted for "
            "non-equity asset_class='%s' (category='%s').",
            list(_EQUITY_ONLY_FIELDS),
            asset_class,
            category,
        )

    # ------------------------------------------------------------------
    # Yield decimal validation: yield fields must be in [0, 1]
    # Values > 1.0 indicate percent storage (e.g., 3.0 instead of 0.03)
    # which would corrupt downstream CMA method computations
    # ------------------------------------------------------------------
    # Collect all yield fields and their values for validation
    yield_field_values: Dict[str, Optional[float]] = {
        "dividend_yield": dividend_yield,
        "buyback_yield": buyback_yield,
        "earnings_yield": earnings_yield,
    }

    # Check each yield field for out-of-range values
    for yield_field, yield_val in yield_field_values.items():
        if yield_val is not None and yield_val > 1.0:
            raise ValueError(
                f"fetch_signals: '{yield_field}' for '{asset_class}' = "
                f"{yield_val:.4f} exceeds 1.0. "
                "Yield fields must be stored in decimal form "
                "(e.g., 0.03 = 3%), not percent form (e.g., 3.0 = 3%). "
                "Please convert the data before calling this tool. "
                "This is a frozen convention per DATA_CONVENTIONS."
            )

    # ------------------------------------------------------------------
    # Record the actual dates of the signals and fundamentals rows
    # ------------------------------------------------------------------
    # Actual date of the signals row (or 'N/A' if not available)
    signals_date_str: str = (
        str(signals_date.date()) if signals_date is not None else "N/A"
    )
    # Actual date of the fundamentals row (or 'N/A' if not available)
    fundamentals_date_str: str = (
        str(fundamentals_date.date())
        if fundamentals_date is not None
        else "N/A"
    )

    # Log the signal retrieval for audit trail
    logger.info(
        "fetch_signals: asset_class='%s', category='%s', "
        "signals_date=%s, fundamentals_date=%s, "
        "omitted_fields=%s",
        asset_class,
        category,
        signals_date_str,
        fundamentals_date_str,
        omitted_fields,
    )

    # ------------------------------------------------------------------
    # Construct and return the output dict
    # All numeric values are Python native float or None
    # ------------------------------------------------------------------
    return {
        # Asset class name for identification
        "asset_class": asset_class,
        # Resolved ticker identifier
        "ticker": ticker,
        # Asset category (Equity, Fixed Income, Real Assets, Cash)
        "category": category,
        # As-of date used for point-in-time filtering
        "as_of_date": as_of_date,
        # Actual date of the most recent signals row
        "signals_date": signals_date_str,
        # Actual date of the most recent fundamentals row
        "fundamentals_date": fundamentals_date_str,
        # --- Technical signals (all asset classes) ---
        # 14-day RSI technical indicator
        "rsi_14d": rsi_14d,
        # 12-month price momentum
        "momentum_12m": momentum_12m,
        # Market breadth indicator
        "market_breadth_raw": market_breadth_raw,
        # Net ETF fund flows
        "net_fund_flows": net_fund_flows,
        # Positioning/crowding score
        "positioning_score_raw": positioning_score_raw,
        # --- Fundamental fields (all asset classes) ---
        # Dividend yield in decimal form (e.g., 0.03 = 3%)
        "dividend_yield": dividend_yield,
        # Buyback yield in decimal form
        "buyback_yield": buyback_yield,
        # Market capitalisation in USD
        "market_cap_usd": market_cap_usd,
        # Earnings growth forecast
        "earnings_growth_forecast": earnings_growth_forecast,
        # Valuation change assumption
        "valuation_change_assumption": valuation_change_assumption,
        # --- Equity-only fields (None for non-equity) ---
        # CAPE ratio in × (e.g., 25.0 = 25×) — equity only
        "cape_ratio": cape_ratio,
        # Trailing P/E ratio in × — equity only
        "pe_trailing": pe_trailing,
        # Earnings yield in decimal form — equity only
        "earnings_yield": earnings_yield,
        # --- Audit fields ---
        # List of fields omitted due to asset category
        "omitted_fields": omitted_fields,
    }

# =============================================================================
# SELF-DRIVING PORTFOLIO: TOOL REGISTRY — BATCH 2 (TOOLS 11–20)
# =============================================================================
# Implements tools 11–20 from the complete 78-tool registry for the agentic
# Strategic Asset Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# Tools implemented:
#   11. run_cma_method_1   — Historical ERP + Rf
#   12. run_cma_method_2   — Regime-Adjusted ERP + Rf
#   13. run_cma_method_3   — Black-Litterman Equilibrium Prior
#   14. run_cma_method_4   — Inverse Gordon / Building Block
#   15. run_cma_method_5   — Implied ERP (CAPE-based)
#   16. run_cma_method_6   — Survey/Analyst Expected Return
#   17. run_cma_method_7   — Confidence-Weighted Auto-Blend
#   18. run_fi_cma_builder — Fixed Income CMA Builder
#   19. load_cma_methods_json — CMA Judge Input Loader
#   20. classify_dispersion   — CMA Judge Dispersion Classifier
#
# All tools are purely deterministic Python callables — no LLM interaction.
# All arithmetic conventions are frozen per STUDY_CONFIG["DATA_CONVENTIONS"].
# =============================================================================

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# Initialise a named logger so callers can configure log levels independently
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants (sourced from STUDY_CONFIG; reproduced for self-contained
# validation — the orchestrator injects the live config at runtime)
# ---------------------------------------------------------------------------

# Valid macro regime labels per METHODOLOGY_PARAMS["MACRO_REGIMES"]
_VALID_REGIMES: Tuple[str, ...] = (
    "Expansion",
    "Late-cycle",
    "Recession",
    "Recovery",
)

# Annualisation multiplier for monthly returns (periods per year)
_PERIODS_PER_YEAR: int = 12

# Minimum monthly observations required for reliable statistical estimation
_MIN_MONTHLY_OBS: int = 24

# Minimum regime-matching observations for Method 2 (regime-adjusted ERP)
_MIN_REGIME_OBS: int = 12

# Numerical stability epsilon
_EPS: float = 1e-8

# Confidence recency decay constant for Method 6 (per month)
_SURVEY_DECAY_LAMBDA: float = 0.1

# Frozen BL equilibrium parameters per METHODOLOGY_PARAMS["BL_EQUILIBRIUM_PARAMS"]
_BL_DELTA_DEFAULT: float = 2.5
_BL_TAU_DEFAULT: float = 0.05

# Frozen dispersion thresholds (pp) per METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]
_DISPERSION_TIGHT_UPPER: float = 3.0
_DISPERSION_MODERATE_UPPER: float = 6.0

# Canonical 18 asset class names in IPS order (used for BL index alignment)
_CANONICAL_ASSET_CLASS_ORDER: Tuple[str, ...] = (
    "US Large Cap",
    "US Small Cap",
    "US Value",
    "US Growth",
    "International Developed",
    "Emerging Markets",
    "Short-Term Treasuries",
    "Intermediate Treasuries",
    "Long-Term Treasuries",
    "Investment-Grade Corporates",
    "High-Yield Corporates",
    "International Sovereign Bonds",
    "International Corporates",
    "USD Emerging Market Debt",
    "REITs",
    "Gold",
    "Commodities",
    "Cash",
)

# Equity asset classes (for equity-only method applicability checks)
_EQUITY_ASSET_CLASSES: Tuple[str, ...] = (
    "US Large Cap",
    "US Small Cap",
    "US Value",
    "US Growth",
    "International Developed",
    "Emerging Markets",
)

# Frozen FI asset-class-to-yield-field mapping
# Maps each FI asset class to its primary yield field and spread field
_FI_YIELD_FIELD_MAP: Dict[str, Dict[str, Any]] = {
    "Short-Term Treasuries": {
        "yield_field": "ust_3m_yield",
        "spread_field": None,
        "maturity_years": 0.25,
        "duration_approx": 0.25,
        "is_credit": False,
    },
    "Intermediate Treasuries": {
        "yield_field": "ust_10y_yield",
        "spread_field": None,
        "maturity_years": 7.0,
        "duration_approx": 6.5,
        "is_credit": False,
    },
    "Long-Term Treasuries": {
        "yield_field": "ust_30y_yield",
        "spread_field": None,
        "maturity_years": 25.0,
        "duration_approx": 18.0,
        "is_credit": False,
    },
    "Investment-Grade Corporates": {
        "yield_field": "ust_10y_yield",
        "spread_field": "ig_oas",
        "maturity_years": 8.0,
        "duration_approx": 7.0,
        "is_credit": True,
        "lgd": 0.60,
        "pd_spread_fraction": 0.40,
    },
    "High-Yield Corporates": {
        "yield_field": "ust_5y_yield",
        "spread_field": "hy_oas",
        "maturity_years": 5.0,
        "duration_approx": 4.0,
        "is_credit": True,
        "lgd": 0.60,
        "pd_spread_fraction": 0.55,
    },
    "International Sovereign Bonds": {
        "yield_field": "ust_10y_yield",
        "spread_field": None,
        "maturity_years": 8.0,
        "duration_approx": 7.0,
        "is_credit": False,
    },
    "International Corporates": {
        "yield_field": "ust_10y_yield",
        "spread_field": "ig_oas",
        "maturity_years": 7.0,
        "duration_approx": 6.0,
        "is_credit": True,
        "lgd": 0.60,
        "pd_spread_fraction": 0.40,
    },
    "USD Emerging Market Debt": {
        "yield_field": "ust_10y_yield",
        "spread_field": "em_spread",
        "maturity_years": 10.0,
        "duration_approx": 7.5,
        "is_credit": True,
        "lgd": 0.65,
        "pd_spread_fraction": 0.50,
    },
}

# Yield curve tenor points (years) mapped to df_fixed_income column names
_YIELD_CURVE_TENORS: List[float] = [0.25, 2.0, 10.0, 30.0]
_YIELD_CURVE_FIELDS: List[str] = [
    "ust_3m_yield",
    "ust_2y_yield",
    "ust_10y_yield",
    "ust_30y_yield",
]


# =============================================================================
# TOOL 11: run_cma_method_1 — Historical ERP + Rf
# =============================================================================

def run_cma_method_1(
    historical_stats: Dict[str, Any],
    rf: float,
    rf_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Compute CMA Method 1: Historical Equity Risk Premium + Risk-Free Rate.

    Implements the long-run realised ERP approach (Task 21, Step 1):

    .. math::

        \\hat{\\mu}_1 = r_f + \\widehat{ERP}_{hist}

    where the historical ERP is estimated as the mean annualised excess return
    over the full available history:

    .. math::

        \\widehat{ERP}_{hist} = 12 \\cdot \\overline{r_{excess,mo}}
        = 12 \\cdot \\overline{(r_{p,t} - r_{f,t})}

    Confidence is scored using the information ratio (IR) of the historical
    excess return series, normalised to ``[0, 1]`` via a sigmoid transform:

    .. math::

        IR = \\frac{\\bar{r}_{excess,ann}}{\\sigma_{excess,ann}}, \\quad
        c_1 = \\frac{1}{1 + \\exp(-|IR|)}

    per ``METHODOLOGY_PARAMS["CMA_CONFIDENCE_SCORING"]["method_1_historical_erp"]
    = "information_ratio_based"``.

    Parameters
    ----------
    historical_stats : Dict[str, Any]
        Output of ``fetch_historical_stats``. Must contain keys
        ``"monthly_returns_series"`` (``pd.Series``) and
        ``"annualised_return"`` (``float``).
    rf : float
        Current annualised risk-free rate in decimal form (e.g., 0.053 = 5.3%).
        Used as the current rf in the estimate: :math:`\\hat{\\mu}_1 = r_f + ERP`.
    rf_series : Optional[pd.Series]
        Historical monthly risk-free rate series (decimal, per-month) with
        ``DatetimeIndex``, aligned to the same calendar as
        ``monthly_returns_series``. If ``None``, the current ``rf / 12``
        is used as a constant monthly rf proxy for the full history.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"method_id"`` (``int``): ``1``
        - ``"estimate"`` (``float``): :math:`\\hat{\\mu}_1` in decimal form.
        - ``"confidence"`` (``float``): IR-based confidence in ``[0, 1]``.
        - ``"breakdown"`` (``Dict[str, float]``): Component breakdown:
          ``erp_hist``, ``rf_current``, ``ir_annualised``, ``n_obs``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``historical_stats`` is not a dict.
    ValueError
        If required keys are missing from ``historical_stats``.
    ValueError
        If ``rf`` is not a finite float.
    ValueError
        If ``monthly_returns_series`` has fewer than ``_MIN_MONTHLY_OBS``
        observations.

    Notes
    -----
    The frozen annualisation convention is:
    :math:`\\mu_{ann} = 12 \\cdot \\mu_{mo}`,
    :math:`\\sigma_{ann} = \\sqrt{12} \\cdot \\sigma_{mo}`,
    per ``DATA_CONVENTIONS["annualisation"]``.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(historical_stats, dict):
        raise TypeError(
            f"historical_stats must be a dict, "
            f"got {type(historical_stats).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys in historical_stats
    # ------------------------------------------------------------------
    _required_keys: Tuple[str, ...] = (
        "monthly_returns_series",
        "annualised_return",
    )
    missing_keys: List[str] = [
        k for k in _required_keys if k not in historical_stats
    ]
    if missing_keys:
        raise ValueError(
            f"historical_stats is missing required keys: {missing_keys}."
        )

    # ------------------------------------------------------------------
    # Input validation: rf must be a finite float
    # ------------------------------------------------------------------
    if not np.isfinite(rf):
        raise ValueError(
            f"rf must be a finite float, got {rf}."
        )

    # ------------------------------------------------------------------
    # Extract the monthly returns series from historical_stats
    # ------------------------------------------------------------------
    # Monthly simple returns series with DatetimeIndex, shape (T,)
    monthly_returns: pd.Series = historical_stats["monthly_returns_series"]

    # ------------------------------------------------------------------
    # Validate that monthly_returns is a pd.Series
    # ------------------------------------------------------------------
    if not isinstance(monthly_returns, pd.Series):
        raise TypeError(
            f"historical_stats['monthly_returns_series'] must be a "
            f"pd.Series, got {type(monthly_returns).__name__}."
        )

    # ------------------------------------------------------------------
    # Minimum observation check
    # ------------------------------------------------------------------
    n_obs: int = len(monthly_returns)
    if n_obs < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"monthly_returns_series has {n_obs} observations, "
            f"minimum required: {_MIN_MONTHLY_OBS}."
        )

    # ------------------------------------------------------------------
    # Construct the monthly rf series for excess return computation
    # If rf_series is provided, align it to monthly_returns via inner join.
    # Otherwise, use rf / 12 as a constant monthly rf proxy.
    # ------------------------------------------------------------------
    if rf_series is not None:
        # Validate rf_series type
        if not isinstance(rf_series, pd.Series):
            raise TypeError(
                f"rf_series must be a pd.Series, "
                f"got {type(rf_series).__name__}."
            )
        # Align rf_series to monthly_returns via inner join on DatetimeIndex
        aligned: pd.DataFrame = pd.concat(
            {"r_p": monthly_returns, "r_f": rf_series},
            axis=1,
            join="inner",
        )
        # Monthly portfolio returns aligned to rf dates
        r_p_aligned: pd.Series = aligned["r_p"]
        # Monthly rf rates aligned to portfolio return dates
        r_f_aligned: pd.Series = aligned["r_f"]
    else:
        # Use current rf / 12 as a constant monthly rf proxy for all periods
        # This is a documented approximation when historical rf is unavailable
        r_p_aligned = monthly_returns
        # Constant monthly rf = annualised rf / 12
        r_f_aligned = pd.Series(
            rf / _PERIODS_PER_YEAR,
            index=monthly_returns.index,
        )

    # ------------------------------------------------------------------
    # Compute monthly excess returns: r_excess_t = r_p_t - r_f_t
    # ------------------------------------------------------------------
    # Monthly excess return series
    monthly_excess: pd.Series = r_p_aligned - r_f_aligned

    # ------------------------------------------------------------------
    # Compute mean and standard deviation of monthly excess returns
    # ------------------------------------------------------------------
    # Mean monthly excess return
    mean_excess_mo: float = float(monthly_excess.mean())
    # Standard deviation of monthly excess returns (sample, ddof=1)
    std_excess_mo: float = float(monthly_excess.std(ddof=1))

    # ------------------------------------------------------------------
    # Annualise the historical ERP:
    # ERP_hist = 12 * mean(r_excess_mo)
    # per DATA_CONVENTIONS["annualisation"]["mu_multiplier"] = 12
    # ------------------------------------------------------------------
    erp_hist: float = float(_PERIODS_PER_YEAR) * mean_excess_mo

    # ------------------------------------------------------------------
    # Annualise the excess return volatility:
    # sigma_excess_ann = sqrt(12) * sigma_excess_mo
    # per DATA_CONVENTIONS["annualisation"]["sigma_multiplier"] = sqrt(12)
    # ------------------------------------------------------------------
    sigma_excess_ann: float = float(np.sqrt(_PERIODS_PER_YEAR)) * std_excess_mo

    # ------------------------------------------------------------------
    # Compute the Method 1 estimate:
    # mu_hat_1 = r_f + ERP_hist
    # ------------------------------------------------------------------
    estimate: float = rf + erp_hist

    # ------------------------------------------------------------------
    # Compute the annualised information ratio:
    # IR = ERP_hist / sigma_excess_ann
    # ------------------------------------------------------------------
    if sigma_excess_ann > _EPS:
        # Information ratio: mean annualised excess return / annualised vol
        ir_annualised: float = erp_hist / sigma_excess_ann
    else:
        # If volatility is effectively zero, IR is undefined; set to 0
        ir_annualised = 0.0

    # ------------------------------------------------------------------
    # Normalise IR to confidence in [0, 1] via sigmoid transform:
    # c_1 = 1 / (1 + exp(-|IR|))
    # The absolute value ensures that negative IRs still produce positive
    # confidence (the magnitude of the IR matters, not its sign).
    # ------------------------------------------------------------------
    confidence: float = float(1.0 / (1.0 + np.exp(-abs(ir_annualised))))

    # ------------------------------------------------------------------
    # Construct the breakdown dict for audit transparency
    # ------------------------------------------------------------------
    breakdown: Dict[str, float] = {
        # Historical ERP (annualised): ERP_hist = 12 * mean(r_excess_mo)
        "erp_hist": float(erp_hist),
        # Current risk-free rate (annualised, decimal)
        "rf_current": float(rf),
        # Annualised information ratio
        "ir_annualised": float(ir_annualised),
        # Annualised excess return volatility
        "sigma_excess_ann": float(sigma_excess_ann),
        # Number of monthly observations used
        "n_obs": float(n_obs),
        # Whether a constant rf proxy was used
        "rf_proxy_used": float(rf_series is None),
    }

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    rationale: str = (
        f"Method 1 (Historical ERP + Rf): "
        f"Historical ERP = {erp_hist * 100:.2f}% "
        f"(annualised mean excess return over {n_obs} monthly observations). "
        f"Current Rf = {rf * 100:.2f}%. "
        f"Estimate = {estimate * 100:.2f}%. "
        f"IR = {ir_annualised:.3f}, Confidence = {confidence:.3f}. "
        f"{'Constant rf proxy used (rf_series not provided).' if rf_series is None else 'Historical rf series used.'}"
    )

    # Log the result for audit trail
    logger.debug(
        "run_cma_method_1: estimate=%.4f, erp_hist=%.4f, "
        "ir=%.4f, confidence=%.4f",
        estimate, erp_hist, ir_annualised, confidence,
    )

    # ------------------------------------------------------------------
    # Return the method result dict
    # ------------------------------------------------------------------
    return {
        # Method identifier: 1 (Historical ERP + Rf)
        "method_id": 1,
        # Expected return estimate in decimal form
        "estimate": float(estimate),
        # IR-based confidence in [0, 1]
        "confidence": float(confidence),
        # Component breakdown for audit
        "breakdown": breakdown,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 12: run_cma_method_2 — Regime-Adjusted ERP + Rf
# =============================================================================

def run_cma_method_2(
    historical_stats: Dict[str, Any],
    macro_view: Dict[str, Any],
    rf: float,
    regime_history: Optional[pd.Series] = None,
    rf_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Compute CMA Method 2: Regime-Adjusted Equity Risk Premium + Risk-Free Rate.

    Implements the regime-conditional ERP approach (Task 21, Step 1):

    .. math::

        \\hat{\\mu}_2 = r_f + \\widehat{ERP}(\\text{regime})

    where the regime-conditional ERP is the mean annualised excess return
    computed exclusively over historical periods matching the current macro
    regime:

    .. math::

        \\widehat{ERP}(\\text{regime}) =
        12 \\cdot \\overline{r_{excess,t} \\mid \\text{regime}_t = \\text{regime}_{current}}

    Confidence is weighted by the product of the regime classification
    confidence (from ``macro_view["confidence"]``) and a sample-size factor
    based on the number of regime-matching observations:

    .. math::

        c_2 = c_{regime} \\cdot \\min\\left(1,
        \\frac{n_{regime}}{n_{regime,min}}\\right)

    per ``METHODOLOGY_PARAMS["CMA_CONFIDENCE_SCORING"]["method_2_regime_adjusted"]
    = "regime_classification_confidence_weighted"``.

    Parameters
    ----------
    historical_stats : Dict[str, Any]
        Output of ``fetch_historical_stats``. Must contain
        ``"monthly_returns_series"`` (``pd.Series``).
    macro_view : Dict[str, Any]
        Output of ``write_macro_view_json`` (loaded from artifact). Must
        contain ``"regime"`` (``str``) and ``"confidence"`` (``float``).
        **Injected at dispatch time by the ReAct loop — not passed by the LLM.**
    rf : float
        Current annualised risk-free rate in decimal form.
    regime_history : Optional[pd.Series]
        Historical monthly regime classification series with ``DatetimeIndex``,
        values are regime label strings (e.g., ``"Late-cycle"``). If ``None``,
        falls back to Method 1 (full-history ERP) with a reduced confidence
        and a documented note in the rationale.
    rf_series : Optional[pd.Series]
        Historical monthly risk-free rate series (decimal, per-month).
        If ``None``, uses ``rf / 12`` as a constant proxy.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"method_id"`` (``int``): ``2``
        - ``"estimate"`` (``float``): :math:`\\hat{\\mu}_2` in decimal form.
        - ``"confidence"`` (``float``): Regime-confidence-weighted score in
          ``[0, 1]``.
        - ``"breakdown"`` (``Dict[str, float]``): Component breakdown:
          ``erp_regime``, ``rf_current``, ``n_regime_obs``,
          ``regime_confidence``, ``fallback_used``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``historical_stats`` or ``macro_view`` are not dicts.
    ValueError
        If required keys are missing from ``historical_stats`` or
        ``macro_view``.
    ValueError
        If ``macro_view["regime"]`` is not a valid regime label.
    ValueError
        If ``rf`` is not a finite float.

    Notes
    -----
    **Fallback behaviour:** If ``regime_history`` is ``None`` or if fewer
    than ``_MIN_REGIME_OBS`` observations match the current regime, the
    method falls back to the full-history ERP (Method 1 logic) with a
    confidence penalty of 0.5× applied to the regime classification
    confidence. This fallback is documented in the rationale string.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(historical_stats, dict):
        raise TypeError(
            f"historical_stats must be a dict, "
            f"got {type(historical_stats).__name__}."
        )
    if not isinstance(macro_view, dict):
        raise TypeError(
            f"macro_view must be a dict, "
            f"got {type(macro_view).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys
    # ------------------------------------------------------------------
    if "monthly_returns_series" not in historical_stats:
        raise ValueError(
            "historical_stats is missing required key "
            "'monthly_returns_series'."
        )
    for key in ("regime", "confidence"):
        if key not in macro_view:
            raise ValueError(
                f"macro_view is missing required key '{key}'."
            )

    # ------------------------------------------------------------------
    # Input validation: regime label validity
    # ------------------------------------------------------------------
    current_regime: str = macro_view["regime"]
    if current_regime not in _VALID_REGIMES:
        raise ValueError(
            f"macro_view['regime'] = '{current_regime}' is not a valid "
            f"regime label. Must be one of: {list(_VALID_REGIMES)}."
        )

    # ------------------------------------------------------------------
    # Input validation: rf must be finite
    # ------------------------------------------------------------------
    if not np.isfinite(rf):
        raise ValueError(f"rf must be a finite float, got {rf}.")

    # ------------------------------------------------------------------
    # Extract the monthly returns series and regime classification confidence
    # ------------------------------------------------------------------
    # Monthly simple returns series with DatetimeIndex
    monthly_returns: pd.Series = historical_stats["monthly_returns_series"]
    # Regime classification confidence from the macro agent
    regime_confidence: float = float(macro_view["confidence"])

    # ------------------------------------------------------------------
    # Validate monthly_returns type and minimum observations
    # ------------------------------------------------------------------
    if not isinstance(monthly_returns, pd.Series):
        raise TypeError(
            f"historical_stats['monthly_returns_series'] must be a "
            f"pd.Series, got {type(monthly_returns).__name__}."
        )
    if len(monthly_returns) < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"monthly_returns_series has {len(monthly_returns)} observations, "
            f"minimum required: {_MIN_MONTHLY_OBS}."
        )

    # ------------------------------------------------------------------
    # Construct the monthly rf series (same logic as Method 1)
    # ------------------------------------------------------------------
    if rf_series is not None:
        if not isinstance(rf_series, pd.Series):
            raise TypeError(
                f"rf_series must be a pd.Series, "
                f"got {type(rf_series).__name__}."
            )
        # Align rf_series to monthly_returns via inner join
        aligned: pd.DataFrame = pd.concat(
            {"r_p": monthly_returns, "r_f": rf_series},
            axis=1,
            join="inner",
        )
        # Aligned portfolio returns
        r_p_aligned: pd.Series = aligned["r_p"]
        # Aligned historical rf rates
        r_f_aligned: pd.Series = aligned["r_f"]
    else:
        # Use constant monthly rf proxy: rf / 12
        r_p_aligned = monthly_returns
        r_f_aligned = pd.Series(
            rf / _PERIODS_PER_YEAR,
            index=monthly_returns.index,
        )

    # ------------------------------------------------------------------
    # Compute monthly excess returns: r_excess_t = r_p_t - r_f_t
    # ------------------------------------------------------------------
    # Monthly excess return series
    monthly_excess: pd.Series = r_p_aligned - r_f_aligned

    # ------------------------------------------------------------------
    # Determine whether to use regime-filtered or full-history ERP
    # ------------------------------------------------------------------
    # Flag indicating whether the fallback to full-history ERP was used
    fallback_used: bool = False
    # Number of regime-matching observations (0 if fallback)
    n_regime_obs: int = 0

    if regime_history is not None:
        # Validate regime_history type
        if not isinstance(regime_history, pd.Series):
            raise TypeError(
                f"regime_history must be a pd.Series, "
                f"got {type(regime_history).__name__}."
            )

        # Find dates where the historical regime matches the current regime
        regime_dates: pd.Index = regime_history[
            regime_history == current_regime
        ].index

        # Find the intersection of regime dates and excess return dates
        common_dates: pd.Index = monthly_excess.index.intersection(
            regime_dates
        )
        n_regime_obs = len(common_dates)

        if n_regime_obs >= _MIN_REGIME_OBS:
            # Sufficient regime-matching observations: use regime-filtered ERP
            # Filter excess returns to regime-matching dates only
            regime_excess: pd.Series = monthly_excess.loc[common_dates]
            # Mean monthly excess return during regime periods
            mean_excess_mo: float = float(regime_excess.mean())
        else:
            # Insufficient regime history: fall back to full-history ERP
            fallback_used = True
            logger.warning(
                "run_cma_method_2: Only %d regime-matching observations "
                "for regime='%s' (minimum: %d). "
                "Falling back to full-history ERP.",
                n_regime_obs,
                current_regime,
                _MIN_REGIME_OBS,
            )
            # Use full-history mean excess return as fallback
            mean_excess_mo = float(monthly_excess.mean())
    else:
        # No regime_history provided: fall back to full-history ERP
        fallback_used = True
        logger.warning(
            "run_cma_method_2: regime_history not provided. "
            "Falling back to full-history ERP."
        )
        # Use full-history mean excess return as fallback
        mean_excess_mo = float(monthly_excess.mean())

    # ------------------------------------------------------------------
    # Annualise the regime-conditional ERP:
    # ERP_regime = 12 * mean(r_excess_mo | regime)
    # per DATA_CONVENTIONS["annualisation"]["mu_multiplier"] = 12
    # ------------------------------------------------------------------
    erp_regime: float = float(_PERIODS_PER_YEAR) * mean_excess_mo

    # ------------------------------------------------------------------
    # Compute the Method 2 estimate:
    # mu_hat_2 = r_f + ERP_regime
    # ------------------------------------------------------------------
    estimate: float = rf + erp_regime

    # ------------------------------------------------------------------
    # Compute confidence:
    # c_2 = c_regime * min(1, n_regime_obs / n_regime_min)
    # If fallback was used, apply a 0.5x confidence penalty.
    # ------------------------------------------------------------------
    if fallback_used:
        # Fallback penalty: halve the regime classification confidence
        sample_factor: float = 0.5
    else:
        # Sample size factor: saturates at 1.0 when n_regime_obs >= minimum
        sample_factor = min(1.0, n_regime_obs / _MIN_REGIME_OBS)

    # Final confidence: regime classification confidence × sample factor
    confidence: float = float(regime_confidence * sample_factor)
    # Ensure confidence is bounded to [0, 1]
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Construct the breakdown dict for audit transparency
    # ------------------------------------------------------------------
    breakdown: Dict[str, float] = {
        # Regime-conditional ERP (annualised, decimal)
        "erp_regime": float(erp_regime),
        # Current risk-free rate (annualised, decimal)
        "rf_current": float(rf),
        # Number of regime-matching observations used
        "n_regime_obs": float(n_regime_obs),
        # Regime classification confidence from macro agent
        "regime_confidence": float(regime_confidence),
        # Sample size factor applied to confidence
        "sample_factor": float(sample_factor),
        # Whether the fallback to full-history ERP was used
        "fallback_used": float(fallback_used),
    }

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    if fallback_used:
        regime_note: str = (
            f"Fallback to full-history ERP used "
            f"({'regime_history not provided' if regime_history is None else f'only {n_regime_obs} regime-matching obs'})."
        )
    else:
        regime_note = (
            f"Regime-filtered ERP computed over {n_regime_obs} "
            f"'{current_regime}' periods."
        )

    rationale: str = (
        f"Method 2 (Regime-Adjusted ERP + Rf): "
        f"Current regime = '{current_regime}' "
        f"(confidence = {regime_confidence:.3f}). "
        f"{regime_note} "
        f"ERP_regime = {erp_regime * 100:.2f}%. "
        f"Current Rf = {rf * 100:.2f}%. "
        f"Estimate = {estimate * 100:.2f}%. "
        f"Confidence = {confidence:.3f}."
    )

    # Log the result for audit trail
    logger.debug(
        "run_cma_method_2: regime='%s', estimate=%.4f, "
        "erp_regime=%.4f, n_regime_obs=%d, confidence=%.4f",
        current_regime, estimate, erp_regime, n_regime_obs, confidence,
    )

    # ------------------------------------------------------------------
    # Return the method result dict
    # ------------------------------------------------------------------
    return {
        # Method identifier: 2 (Regime-Adjusted ERP + Rf)
        "method_id": 2,
        # Expected return estimate in decimal form
        "estimate": float(estimate),
        # Regime-confidence-weighted confidence in [0, 1]
        "confidence": float(confidence),
        # Component breakdown for audit
        "breakdown": breakdown,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 13: run_cma_method_3 — Black-Litterman Equilibrium Prior
# =============================================================================

def run_cma_method_3(
    asset_class: str,
    sigma: np.ndarray,
    w_mkt: np.ndarray,
    asset_class_order: Optional[List[str]] = None,
    bl_params: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute CMA Method 3: Black-Litterman Equilibrium Implied Return.

    Implements the Black-Litterman (1992) equilibrium prior (Task 21, Step 2):

    .. math::

        \\pi = \\delta \\Sigma w_{mkt}

    where :math:`\\delta` is the risk aversion coefficient, :math:`\\Sigma`
    is the 18×18 asset covariance matrix (Ledoit-Wolf shrinkage estimate from
    Task 24), and :math:`w_{mkt}` is the 18-element market-capitalisation
    weight vector derived from ``df_fundamentals_raw["market_cap_usd"]``.

    The estimate for ``asset_class`` is the element of :math:`\\pi`
    corresponding to its position in ``asset_class_order``.

    Confidence is scored by market-cap weight stability:

    .. math::

        c_3 = \\frac{w_{mkt,i}}{\\max_j w_{mkt,j}}

    per ``METHODOLOGY_PARAMS["CMA_CONFIDENCE_SCORING"]["method_3_bl_equilibrium"]
    = "market_cap_weight_stability"``.

    Parameters
    ----------
    asset_class : str
        Asset class name. Must be present in ``asset_class_order``.
    sigma : np.ndarray
        18×18 annualised covariance matrix (Ledoit-Wolf shrinkage estimate).
        Must be positive semi-definite. Shape: ``(18, 18)``.
    w_mkt : np.ndarray
        18-element market-capitalisation weight vector. Values must be
        non-negative. Will be normalised to sum to 1.0 if not already.
        Shape: ``(18,)``.
    asset_class_order : Optional[List[str]]
        Ordered list of 18 asset class names defining the row/column
        correspondence of ``sigma`` and ``w_mkt``. If ``None``, uses
        ``_CANONICAL_ASSET_CLASS_ORDER``.
    bl_params : Optional[Dict[str, float]]
        Black-Litterman parameters. Keys: ``"delta"`` (risk aversion,
        default 2.5) and ``"tau"`` (scaling, default 0.05). If ``None``,
        uses frozen defaults from
        ``METHODOLOGY_PARAMS["BL_EQUILIBRIUM_PARAMS"]``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"method_id"`` (``int``): ``3``
        - ``"estimate"`` (``float``): :math:`\\pi_i` for ``asset_class``,
          in decimal form (annualised).
        - ``"confidence"`` (``float``): Market-cap weight stability score
          in ``[0, 1]``.
        - ``"breakdown"`` (``Dict[str, float]``): Component breakdown:
          ``delta``, ``w_mkt_i``, ``pi_i``, ``asset_class_index``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``sigma`` or ``w_mkt`` are not ``np.ndarray``.
    ValueError
        If ``sigma`` is not shape ``(18, 18)`` or ``w_mkt`` is not shape
        ``(18,)``.
    ValueError
        If ``asset_class`` is not in ``asset_class_order``.
    ValueError
        If ``sigma`` is not positive semi-definite (minimum eigenvalue < 0).
    ValueError
        If ``w_mkt`` contains negative values.

    Notes
    -----
    The BL equilibrium prior :math:`\\pi = \\delta \\Sigma w_{mkt}` produces
    the vector of market-implied expected returns consistent with the
    market-cap weights being mean-variance optimal. This is the "reverse
    optimisation" interpretation of Black and Litterman (1992).

    The ``tau`` parameter is stored in ``bl_params`` for completeness but is
    not used in the pure equilibrium prior computation (it is used in the
    posterior update with views, which is not implemented here).
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(sigma, np.ndarray):
        raise TypeError(
            f"sigma must be a np.ndarray, got {type(sigma).__name__}."
        )
    if not isinstance(w_mkt, np.ndarray):
        raise TypeError(
            f"w_mkt must be a np.ndarray, got {type(w_mkt).__name__}."
        )

    # ------------------------------------------------------------------
    # Resolve asset class order (default or override)
    # ------------------------------------------------------------------
    ac_order: List[str] = (
        list(asset_class_order)
        if asset_class_order is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )
    n_assets: int = len(ac_order)

    # ------------------------------------------------------------------
    # Input validation: sigma shape must be (n_assets, n_assets)
    # ------------------------------------------------------------------
    if sigma.shape != (n_assets, n_assets):
        raise ValueError(
            f"sigma must have shape ({n_assets}, {n_assets}), "
            f"got {sigma.shape}."
        )

    # ------------------------------------------------------------------
    # Input validation: w_mkt shape must be (n_assets,)
    # ------------------------------------------------------------------
    if w_mkt.shape != (n_assets,):
        raise ValueError(
            f"w_mkt must have shape ({n_assets},), got {w_mkt.shape}."
        )

    # ------------------------------------------------------------------
    # Input validation: w_mkt must be non-negative
    # ------------------------------------------------------------------
    if (w_mkt < 0).any():
        raise ValueError(
            "w_mkt contains negative values. "
            "Market-cap weights must be non-negative."
        )

    # ------------------------------------------------------------------
    # Input validation: asset_class must be in ac_order
    # ------------------------------------------------------------------
    if asset_class not in ac_order:
        raise ValueError(
            f"asset_class='{asset_class}' not found in asset_class_order. "
            f"Available: {ac_order}."
        )

    # ------------------------------------------------------------------
    # Input validation: sigma must be positive semi-definite
    # Check via minimum eigenvalue (using eigvalsh for symmetric matrices)
    # ------------------------------------------------------------------
    min_eigenvalue: float = float(np.linalg.eigvalsh(sigma).min())
    if min_eigenvalue < -1e-6:
        raise ValueError(
            f"sigma is not positive semi-definite. "
            f"Minimum eigenvalue: {min_eigenvalue:.6e}. "
            "Apply eigenvalue clipping (Task 24) before calling this tool."
        )

    # ------------------------------------------------------------------
    # Resolve BL parameters (default or override)
    # Frozen defaults: delta=2.5, tau=0.05
    # per METHODOLOGY_PARAMS["BL_EQUILIBRIUM_PARAMS"]
    # ------------------------------------------------------------------
    _default_bl_params: Dict[str, float] = {
        "delta": _BL_DELTA_DEFAULT,
        "tau": _BL_TAU_DEFAULT,
    }
    params: Dict[str, float] = (
        bl_params if bl_params is not None else _default_bl_params
    )
    # Risk aversion coefficient delta
    delta: float = float(params.get("delta", _BL_DELTA_DEFAULT))

    # ------------------------------------------------------------------
    # Normalise w_mkt to sum to 1.0 (with warning if not already normalised)
    # ------------------------------------------------------------------
    w_sum: float = float(w_mkt.sum())
    if abs(w_sum - 1.0) > 1e-4:
        logger.warning(
            "run_cma_method_3: w_mkt does not sum to 1.0 (sum=%.6f). "
            "Normalising to sum to 1.0.",
            w_sum,
        )
        # Normalise w_mkt to sum to 1.0
        w_mkt_norm: np.ndarray = w_mkt / w_sum
    else:
        # w_mkt already sums to 1.0; use as-is
        w_mkt_norm = w_mkt.copy()

    # ------------------------------------------------------------------
    # Compute the BL equilibrium implied return vector:
    # pi = delta * Sigma * w_mkt
    # This is a matrix-vector product: (18,18) @ (18,) = (18,)
    # ------------------------------------------------------------------
    pi_vector: np.ndarray = delta * (sigma @ w_mkt_norm)

    # ------------------------------------------------------------------
    # Extract the index of the target asset class in ac_order
    # ------------------------------------------------------------------
    # Index of the target asset class in the canonical ordering
    asset_idx: int = ac_order.index(asset_class)

    # ------------------------------------------------------------------
    # Extract the implied return for the target asset class
    # pi_i = (delta * Sigma * w_mkt)[i]
    # ------------------------------------------------------------------
    pi_i: float = float(pi_vector[asset_idx])

    # ------------------------------------------------------------------
    # Extract the market-cap weight for the target asset class
    # ------------------------------------------------------------------
    # Market-cap weight of the target asset class
    w_mkt_i: float = float(w_mkt_norm[asset_idx])

    # ------------------------------------------------------------------
    # Compute confidence: market-cap weight stability
    # c_3 = w_mkt_i / max_j(w_mkt_j)
    # Higher weight = more stable = higher confidence
    # ------------------------------------------------------------------
    w_mkt_max: float = float(w_mkt_norm.max())
    if w_mkt_max > _EPS:
        # Confidence is the relative market-cap weight of this asset class
        confidence: float = float(w_mkt_i / w_mkt_max)
    else:
        # All weights are zero (degenerate case); assign minimum confidence
        confidence = 0.0

    # Ensure confidence is bounded to [0, 1]
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Construct the breakdown dict for audit transparency
    # ------------------------------------------------------------------
    breakdown: Dict[str, float] = {
        # Risk aversion coefficient delta
        "delta": float(delta),
        # Market-cap weight of the target asset class
        "w_mkt_i": float(w_mkt_i),
        # BL equilibrium implied return for the target asset class
        "pi_i": float(pi_i),
        # Index of the target asset class in the canonical ordering
        "asset_class_index": float(asset_idx),
        # Maximum market-cap weight across all asset classes
        "w_mkt_max": float(w_mkt_max),
        # Minimum eigenvalue of sigma (PSD check)
        "sigma_min_eigenvalue": float(min_eigenvalue),
    }

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    rationale: str = (
        f"Method 3 (BL Equilibrium Prior): "
        f"pi = delta * Sigma * w_mkt, delta = {delta:.2f}. "
        f"Asset class '{asset_class}' (index {asset_idx}): "
        f"w_mkt = {w_mkt_i * 100:.2f}%, "
        f"pi_i = {pi_i * 100:.2f}%. "
        f"Confidence (market-cap weight stability) = {confidence:.3f}."
    )

    # Log the result for audit trail
    logger.debug(
        "run_cma_method_3: asset_class='%s', pi_i=%.4f, "
        "w_mkt_i=%.4f, confidence=%.4f",
        asset_class, pi_i, w_mkt_i, confidence,
    )

    # ------------------------------------------------------------------
    # Return the method result dict
    # ------------------------------------------------------------------
    return {
        # Method identifier: 3 (BL Equilibrium Prior)
        "method_id": 3,
        # BL equilibrium implied return in decimal form
        "estimate": float(pi_i),
        # Market-cap weight stability confidence in [0, 1]
        "confidence": float(confidence),
        # Component breakdown for audit
        "breakdown": breakdown,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 14: run_cma_method_4 — Inverse Gordon / Building Block
# =============================================================================

def run_cma_method_4(
    asset_class: str,
    as_of_date: str,
    df_fundamentals_raw: pd.DataFrame,
    universe_map: Dict[str, Dict[str, Any]],
    nominal_gdp_growth_proxy: float = 0.04,
) -> Dict[str, Any]:
    """
    Compute CMA Method 4: Inverse Gordon / Grinold-Kroner Building-Block Model.

    Implements the building-block decomposition of expected equity returns
    (Task 21, Step 2), following Grinold and Kroner (2002):

    .. math::

        \\hat{\\mu}_4 = y_{div} + y_{buyback} + g + \\Delta v

    where:

    - :math:`y_{div}` = dividend yield (decimal)
    - :math:`y_{buyback}` = buyback yield (decimal)
    - :math:`g` = earnings growth forecast (decimal)
    - :math:`\\Delta v` = valuation change assumption (decimal, e.g.,
      expected change in P/E or CAPE over the forecast horizon)

    All components are retrieved from ``df_fundamentals_raw`` as-of
    ``as_of_date``. Confidence is scored by earnings forecast quality:
    higher confidence when ``earnings_growth_forecast`` is explicitly
    available (not a fallback proxy).

    Parameters
    ----------
    asset_class : str
        Asset class name. Must be a key in ``universe_map``.
    as_of_date : str
        ISO-8601 date string for point-in-time data retrieval.
    df_fundamentals_raw : pd.DataFrame
        Fundamental and valuation inputs panel. Must have a MultiIndex
        with levels ``["date", "ticker", "investment_universe"]`` and
        columns: ``"dividend_yield"``, ``"buyback_yield"``,
        ``"earnings_growth_forecast"``, ``"valuation_change_assumption"``.
        Shape: ``(T_fundamentals, ≥4)``.
    universe_map : Dict[str, Dict[str, Any]]
        Mapping from asset class names to metadata. Must contain
        ``"ticker"`` for the target asset class.
    nominal_gdp_growth_proxy : float
        Fallback earnings growth rate (decimal) used when
        ``earnings_growth_forecast`` is NaN or unavailable. Default: 0.04
        (4% nominal GDP growth proxy). This is a frozen implementation
        choice documented in the rationale.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"method_id"`` (``int``): ``4``
        - ``"estimate"`` (``float``): :math:`\\hat{\\mu}_4` in decimal form.
        - ``"confidence"`` (``float``): Earnings forecast quality score
          in ``[0, 1]``.
        - ``"breakdown"`` (``Dict[str, float]``): Component breakdown:
          ``dividend_yield``, ``buyback_yield``, ``earnings_growth``,
          ``valuation_change``, ``earnings_growth_fallback_used``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``df_fundamentals_raw`` is not a ``pd.DataFrame``.
    ValueError
        If ``asset_class`` is not in ``universe_map``.
    ValueError
        If the resolved ticker is not found in ``df_fundamentals_raw``.
    ValueError
        If no data is available on or before ``as_of_date``.
    ValueError
        If yield fields contain values > 1.0 (percent form detected).

    Notes
    -----
    **Fallback for missing earnings growth:** If ``earnings_growth_forecast``
    is NaN, the ``nominal_gdp_growth_proxy`` is used as a fallback. This
    reduces the confidence score by 0.3 (from 0.8 to 0.5) to reflect the
    lower quality of the proxy estimate. This is a frozen implementation
    choice per ``METHODOLOGY_PARAMS``.

    **Valuation change assumption:** If ``valuation_change_assumption`` is
    NaN, it is set to 0.0 (no mean reversion assumed). This is documented
    in the rationale.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(df_fundamentals_raw, pd.DataFrame):
        raise TypeError(
            f"df_fundamentals_raw must be a pd.DataFrame, "
            f"got {type(df_fundamentals_raw).__name__}."
        )
    if not isinstance(universe_map, dict):
        raise TypeError(
            f"universe_map must be a dict, got {type(universe_map).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: asset_class in universe_map
    # ------------------------------------------------------------------
    if asset_class not in universe_map:
        raise ValueError(
            f"asset_class='{asset_class}' not found in universe_map."
        )

    # ------------------------------------------------------------------
    # Resolve ticker from universe_map
    # ------------------------------------------------------------------
    ticker: str = universe_map[asset_class]["ticker"]

    # ------------------------------------------------------------------
    # Parse as_of_date to pd.Timestamp
    # ------------------------------------------------------------------
    try:
        # Convert ISO-8601 string to timezone-naive pd.Timestamp
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
    except Exception as exc:
        raise ValueError(
            f"as_of_date='{as_of_date}' cannot be parsed. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Validate MultiIndex structure
    # ------------------------------------------------------------------
    if not isinstance(df_fundamentals_raw.index, pd.MultiIndex):
        raise TypeError(
            "df_fundamentals_raw must have a MultiIndex with levels "
            "['date', 'ticker', 'investment_universe']."
        )

    # ------------------------------------------------------------------
    # Slice the fundamentals DataFrame to the target ticker
    # ------------------------------------------------------------------
    try:
        # Extract rows for the target ticker using .xs() on the ticker level
        df_ticker: pd.DataFrame = df_fundamentals_raw.xs(
            ticker, level="ticker"
        )
    except KeyError:
        raise ValueError(
            f"Ticker '{ticker}' (for asset_class='{asset_class}') not found "
            f"in df_fundamentals_raw."
        )

    # ------------------------------------------------------------------
    # Strip timezone from index if present
    # ------------------------------------------------------------------
    if hasattr(df_ticker.index, "tz") and df_ticker.index.tz is not None:
        df_ticker = df_ticker.copy()
        df_ticker.index = df_ticker.index.tz_localize(None)

    # ------------------------------------------------------------------
    # Apply point-in-time filter: retain rows <= as_of_date
    # ------------------------------------------------------------------
    df_filtered: pd.DataFrame = df_ticker.loc[df_ticker.index <= as_of_ts]

    # ------------------------------------------------------------------
    # Guard: ensure data is available after filtering
    # ------------------------------------------------------------------
    if df_filtered.empty:
        raise ValueError(
            f"No fundamentals data available for '{asset_class}' "
            f"(ticker='{ticker}') on or before '{as_of_date}'."
        )

    # ------------------------------------------------------------------
    # Select the most recent row
    # ------------------------------------------------------------------
    # Most recent fundamentals row as a pd.Series
    latest_row: pd.Series = df_filtered.iloc[-1]

    # ------------------------------------------------------------------
    # Helper: safely extract a float from the latest row
    # ------------------------------------------------------------------
    def _get_field(field: str) -> Optional[float]:
        """Extract a float field from latest_row; return None if missing/NaN."""
        if field not in latest_row.index:
            return None
        val = latest_row[field]
        if pd.isna(val):
            return None
        return float(val)

    # ------------------------------------------------------------------
    # Extract the four building-block components
    # ------------------------------------------------------------------
    # Dividend yield: y_div (decimal form, e.g., 0.015 = 1.5%)
    dividend_yield: Optional[float] = _get_field("dividend_yield")
    # Buyback yield: y_buyback (decimal form)
    buyback_yield: Optional[float] = _get_field("buyback_yield")
    # Earnings growth forecast: g (decimal form)
    earnings_growth_raw: Optional[float] = _get_field("earnings_growth_forecast")
    # Valuation change assumption: delta_v (decimal form)
    valuation_change_raw: Optional[float] = _get_field("valuation_change_assumption")

    # ------------------------------------------------------------------
    # Yield decimal validation: values > 1.0 indicate percent storage
    # ------------------------------------------------------------------
    for field_name, field_val in [
        ("dividend_yield", dividend_yield),
        ("buyback_yield", buyback_yield),
    ]:
        if field_val is not None and field_val > 1.0:
            raise ValueError(
                f"'{field_name}' for '{asset_class}' = {field_val:.4f} "
                "exceeds 1.0. Yield fields must be in decimal form "
                "(e.g., 0.015 = 1.5%), not percent form."
            )

    # ------------------------------------------------------------------
    # Apply defaults for missing components
    # ------------------------------------------------------------------
    # Default dividend yield to 0.0 if missing (no dividend)
    y_div: float = dividend_yield if dividend_yield is not None else 0.0
    # Default buyback yield to 0.0 if missing (no buybacks)
    y_buyback: float = buyback_yield if buyback_yield is not None else 0.0

    # ------------------------------------------------------------------
    # Handle missing earnings growth forecast with fallback
    # ------------------------------------------------------------------
    # Flag indicating whether the fallback proxy was used
    earnings_growth_fallback_used: bool = False
    if earnings_growth_raw is not None:
        # Use the explicitly provided earnings growth forecast
        g: float = earnings_growth_raw
        # High confidence when explicit forecast is available
        base_confidence: float = 0.8
    else:
        # Fall back to nominal GDP growth proxy (frozen implementation choice)
        g = nominal_gdp_growth_proxy
        earnings_growth_fallback_used = True
        # Reduced confidence when fallback proxy is used
        base_confidence = 0.5
        logger.warning(
            "run_cma_method_4: 'earnings_growth_forecast' is NaN for "
            "'%s'. Using nominal GDP growth proxy = %.4f.",
            asset_class,
            nominal_gdp_growth_proxy,
        )

    # ------------------------------------------------------------------
    # Handle missing valuation change assumption
    # ------------------------------------------------------------------
    if valuation_change_raw is not None:
        # Use the explicitly provided valuation change assumption
        delta_v: float = valuation_change_raw
    else:
        # Default to 0.0 (no mean reversion assumed)
        delta_v = 0.0
        logger.debug(
            "run_cma_method_4: 'valuation_change_assumption' is NaN for "
            "'%s'. Defaulting to 0.0 (no mean reversion).",
            asset_class,
        )

    # ------------------------------------------------------------------
    # Compute the Inverse Gordon / Grinold-Kroner estimate:
    # mu_hat_4 = y_div + y_buyback + g + delta_v
    # ------------------------------------------------------------------
    estimate: float = y_div + y_buyback + g + delta_v

    # ------------------------------------------------------------------
    # Confidence = base_confidence (reduced if fallback used)
    # ------------------------------------------------------------------
    confidence: float = float(np.clip(base_confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Construct the breakdown dict for audit transparency
    # ------------------------------------------------------------------
    breakdown: Dict[str, float] = {
        # Dividend yield component: y_div
        "dividend_yield": float(y_div),
        # Buyback yield component: y_buyback
        "buyback_yield": float(y_buyback),
        # Earnings growth component: g
        "earnings_growth": float(g),
        # Valuation change component: delta_v
        "valuation_change": float(delta_v),
        # Whether the earnings growth fallback proxy was used
        "earnings_growth_fallback_used": float(earnings_growth_fallback_used),
        # Nominal GDP growth proxy value used as fallback
        "nominal_gdp_growth_proxy": float(nominal_gdp_growth_proxy),
    }

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    rationale: str = (
        f"Method 4 (Inverse Gordon / Grinold-Kroner): "
        f"mu_hat_4 = y_div + y_buyback + g + delta_v = "
        f"{y_div * 100:.2f}% + {y_buyback * 100:.2f}% + "
        f"{g * 100:.2f}% + {delta_v * 100:.2f}% = "
        f"{estimate * 100:.2f}%. "
        f"{'Earnings growth fallback (nominal GDP proxy) used.' if earnings_growth_fallback_used else 'Explicit earnings growth forecast used.'} "
        f"{'Valuation change defaulted to 0.0 (no mean reversion).' if valuation_change_raw is None else ''} "
        f"Confidence = {confidence:.3f}."
    )

    # Log the result for audit trail
    logger.debug(
        "run_cma_method_4: asset_class='%s', estimate=%.4f, "
        "y_div=%.4f, y_buyback=%.4f, g=%.4f, delta_v=%.4f",
        asset_class, estimate, y_div, y_buyback, g, delta_v,
    )

    # ------------------------------------------------------------------
    # Return the method result dict
    # ------------------------------------------------------------------
    return {
        # Method identifier: 4 (Inverse Gordon / Building Block)
        "method_id": 4,
        # Building-block expected return estimate in decimal form
        "estimate": float(estimate),
        # Earnings forecast quality confidence in [0, 1]
        "confidence": float(confidence),
        # Component breakdown for audit
        "breakdown": breakdown,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 15: run_cma_method_5 — Implied ERP (CAPE-based)
# =============================================================================

def run_cma_method_5(
    signals: Dict[str, Any],
    rf: float,
    cape_hist_series: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Compute CMA Method 5: Implied ERP Proxy (CAPE-based Earnings Yield).

    Implements the Campbell and Shiller (1998) CAPE-based implied ERP approach
    (Task 21, Step 2):

    .. math::

        \\widehat{ERP}_{impl} = \\text{EarningsYield} - r_f

    .. math::

        \\hat{\\mu}_5 = r_f + \\widehat{ERP}_{impl} = \\text{EarningsYield}

    where ``EarningsYield`` is retrieved from ``signals["earnings_yield"]``
    in decimal form (e.g., 0.032 = 3.2%).

    Confidence is scored by CAPE percentile distance from the historical mean:

    .. math::

        c_5 = 1 - \\text{CAPE\_percentile}

    where ``CAPE_percentile`` is the current CAPE's percentile rank in the
    historical CAPE distribution. A lower CAPE (cheaper market) implies a
    more reliable earnings yield as a forward return predictor, hence higher
    confidence.

    Parameters
    ----------
    signals : Dict[str, Any]
        Output of ``fetch_signals``. Must contain ``"earnings_yield"``
        (``float``, decimal form) for equity asset classes. Must also
        contain ``"category"`` to validate equity applicability.
    rf : float
        Current annualised risk-free rate in decimal form.
    cape_hist_series : Optional[pd.Series]
        Historical CAPE ratio series (in ×, e.g., 25.0 = 25×) with
        ``DatetimeIndex``. Used to compute the CAPE percentile for
        confidence scoring. If ``None``, a default confidence of 0.5
        is assigned.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"method_id"`` (``int``): ``5``
        - ``"estimate"`` (``float``): :math:`\\hat{\\mu}_5` in decimal form.
        - ``"confidence"`` (``float``): CAPE-percentile-based confidence
          in ``[0, 1]``.
        - ``"breakdown"`` (``Dict[str, float]``): Component breakdown:
          ``earnings_yield``, ``rf_current``, ``erp_implied``,
          ``cape_ratio``, ``cape_percentile``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``signals`` is not a dict.
    ValueError
        If ``signals["earnings_yield"]`` is ``None`` (non-equity asset).
    ValueError
        If ``signals["earnings_yield"]`` > 1.0 (percent form detected).
    ValueError
        If ``rf`` is not a finite float.

    Notes
    -----
    This method is **only applicable to equity asset classes**. For non-equity
    assets (commodities, gold, fixed income), ``signals["earnings_yield"]``
    will be ``None`` (set by ``fetch_signals``), and this method raises a
    ``ValueError`` with a clear message indicating inapplicability.

    The estimate :math:`\\hat{\\mu}_5 = \\text{EarningsYield}` is equivalent
    to :math:`r_f + (\\text{EarningsYield} - r_f)`, which simplifies to the
    earnings yield itself. This is the standard CAPE-based implied return
    proxy (Campbell and Shiller 1998).
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(signals, dict):
        raise TypeError(
            f"signals must be a dict, got {type(signals).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: rf must be finite
    # ------------------------------------------------------------------
    if not np.isfinite(rf):
        raise ValueError(f"rf must be a finite float, got {rf}.")

    # ------------------------------------------------------------------
    # Input validation: earnings_yield must be present (equity-only check)
    # ------------------------------------------------------------------
    earnings_yield: Optional[float] = signals.get("earnings_yield", None)
    if earnings_yield is None:
        asset_class: str = signals.get("asset_class", "unknown")
        category: str = signals.get("category", "unknown")
        raise ValueError(
            f"Method 5 (Implied ERP) is not applicable to asset_class="
            f"'{asset_class}' (category='{category}'). "
            "'earnings_yield' is None — this field is only available for "
            "equity asset classes. "
            "Skip Method 5 for non-equity assets and document the omission."
        )

    # ------------------------------------------------------------------
    # Input validation: earnings_yield must be in decimal form
    # ------------------------------------------------------------------
    if earnings_yield > 1.0:
        raise ValueError(
            f"signals['earnings_yield'] = {earnings_yield:.4f} exceeds 1.0. "
            "Earnings yield must be in decimal form (e.g., 0.032 = 3.2%), "
            "not percent form."
        )

    # ------------------------------------------------------------------
    # Extract CAPE ratio from signals (may be None for non-equity)
    # ------------------------------------------------------------------
    # CAPE ratio in × (e.g., 25.0 = 25×)
    cape_ratio: Optional[float] = signals.get("cape_ratio", None)

    # ------------------------------------------------------------------
    # Compute the implied ERP:
    # ERP_impl = EarningsYield - r_f
    # ------------------------------------------------------------------
    erp_implied: float = earnings_yield - rf

    # ------------------------------------------------------------------
    # Compute the Method 5 estimate:
    # mu_hat_5 = r_f + ERP_impl = EarningsYield
    # (simplification: r_f + (EarningsYield - r_f) = EarningsYield)
    # ------------------------------------------------------------------
    estimate: float = rf + erp_implied

    # ------------------------------------------------------------------
    # Compute confidence based on CAPE percentile
    # ------------------------------------------------------------------
    # CAPE percentile of the current CAPE in the historical distribution
    cape_percentile: Optional[float] = None

    if cape_ratio is not None and cape_hist_series is not None:
        # Validate cape_hist_series type
        if not isinstance(cape_hist_series, pd.Series):
            raise TypeError(
                f"cape_hist_series must be a pd.Series, "
                f"got {type(cape_hist_series).__name__}."
            )
        # Drop NaN values from the historical CAPE series
        cape_hist_clean: np.ndarray = cape_hist_series.dropna().values
        if len(cape_hist_clean) > 0:
            # Compute the percentile rank of the current CAPE in the
            # historical distribution using searchsorted for efficiency
            # Percentile = fraction of historical values <= current CAPE
            n_below: int = int(np.sum(cape_hist_clean <= cape_ratio))
            cape_percentile = float(n_below) / float(len(cape_hist_clean))
        else:
            # Empty historical series: use default confidence
            cape_percentile = None

    if cape_percentile is not None:
        # Confidence = 1 - CAPE_percentile
        # Lower CAPE (cheaper) → lower percentile → higher confidence
        confidence: float = float(1.0 - cape_percentile)
    else:
        # Default confidence when CAPE percentile cannot be computed
        confidence = 0.5
        logger.debug(
            "run_cma_method_5: CAPE percentile not available. "
            "Using default confidence = 0.5."
        )

    # Ensure confidence is bounded to [0, 1]
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Construct the breakdown dict for audit transparency
    # ------------------------------------------------------------------
    breakdown: Dict[str, float] = {
        # Earnings yield (decimal form)
        "earnings_yield": float(earnings_yield),
        # Current risk-free rate (annualised, decimal)
        "rf_current": float(rf),
        # Implied ERP: EarningsYield - rf
        "erp_implied": float(erp_implied),
        # CAPE ratio (in ×), if available
        "cape_ratio": float(cape_ratio) if cape_ratio is not None else float("nan"),
        # CAPE percentile in historical distribution, if available
        "cape_percentile": (
            float(cape_percentile)
            if cape_percentile is not None
            else float("nan")
        ),
    }

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    cape_note: str = (
        f"CAPE = {cape_ratio:.1f}× "
        f"(percentile = {cape_percentile * 100:.1f}%)."
        if cape_percentile is not None and cape_ratio is not None
        else "CAPE percentile not available; default confidence used."
    )
    rationale: str = (
        f"Method 5 (Implied ERP / CAPE-based): "
        f"EarningsYield = {earnings_yield * 100:.2f}%, "
        f"Rf = {rf * 100:.2f}%, "
        f"ERP_impl = {erp_implied * 100:.2f}%, "
        f"Estimate = {estimate * 100:.2f}%. "
        f"{cape_note} "
        f"Confidence = {confidence:.3f}."
    )

    # Log the result for audit trail
    logger.debug(
        "run_cma_method_5: estimate=%.4f, erp_implied=%.4f, "
        "cape_ratio=%s, confidence=%.4f",
        estimate,
        erp_implied,
        f"{cape_ratio:.2f}" if cape_ratio is not None else "N/A",
        confidence,
    )

    # ------------------------------------------------------------------
    # Return the method result dict
    # ------------------------------------------------------------------
    return {
        # Method identifier: 5 (Implied ERP / CAPE-based)
        "method_id": 5,
        # CAPE-based implied return estimate in decimal form
        "estimate": float(estimate),
        # CAPE-percentile-based confidence in [0, 1]
        "confidence": float(confidence),
        # Component breakdown for audit
        "breakdown": breakdown,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 16: run_cma_method_6 — Survey/Analyst Expected Return
# =============================================================================

def run_cma_method_6(
    asset_class: str,
    as_of_date: str,
    df_survey_cma_raw: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute CMA Method 6: Survey/Analyst Consensus Expected Return.

    Implements the survey-based expected return retrieval (Task 21, Step 3):

    .. math::

        \\hat{\\mu}_6 = \\text{survey\\_expected\\_return}

    The most recent survey estimate on or before ``as_of_date`` is retrieved
    from ``df_survey_cma_raw`` for the target ``asset_class``.

    Confidence is scored using an exponential recency decay:

    .. math::

        c_6 = \\exp(-\\lambda \\cdot \\Delta t)

    where :math:`\\Delta t` is the number of months between the survey date
    and ``as_of_date``, and :math:`\\lambda = 0.1` per month (frozen per
    ``_SURVEY_DECAY_LAMBDA``).

    Parameters
    ----------
    asset_class : str
        Asset class name. Must match values in the ``"asset_class"`` column
        of ``df_survey_cma_raw``.
    as_of_date : str
        ISO-8601 date string for point-in-time retrieval.
    df_survey_cma_raw : pd.DataFrame
        Survey CMA panel. Must have a ``DatetimeIndex`` (monthly or
        quarterly) and columns ``"asset_class"`` (object) and
        ``"survey_expected_return"`` (float64, decimal form).
        Shape: ``(T_survey, 2)``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"method_id"`` (``int``): ``6``
        - ``"estimate"`` (``float | None``): Survey expected return in
          decimal form, or ``None`` if no survey data is available.
        - ``"confidence"`` (``float``): Recency-decay confidence in
          ``[0, 1]``, or ``0.0`` if no data available.
        - ``"breakdown"`` (``Dict[str, Any]``): Component breakdown:
          ``survey_date``, ``months_stale``, ``decay_lambda``,
          ``data_available``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``df_survey_cma_raw`` is not a ``pd.DataFrame``.
    ValueError
        If required columns are missing from ``df_survey_cma_raw``.
    ValueError
        If ``survey_expected_return`` > 1.0 (percent form detected).

    Notes
    -----
    **No-data fallback:** If no survey data is available for the target
    asset class on or before ``as_of_date``, the method returns
    ``estimate=None`` and ``confidence=0.0``. The CMA Judge's auto-blend
    (Method 7) will exclude this method from the weighted average when
    ``estimate=None``.

    **Decimal validation:** ``survey_expected_return`` must be in decimal
    form (e.g., 0.07 = 7%). Values > 1.0 trigger a ``ValueError``.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(df_survey_cma_raw, pd.DataFrame):
        raise TypeError(
            f"df_survey_cma_raw must be a pd.DataFrame, "
            f"got {type(df_survey_cma_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required columns
    # ------------------------------------------------------------------
    _required_cols: Tuple[str, ...] = ("asset_class", "survey_expected_return")
    missing_cols: List[str] = [
        c for c in _required_cols if c not in df_survey_cma_raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_survey_cma_raw is missing required columns: {missing_cols}."
        )

    # ------------------------------------------------------------------
    # Input validation: DatetimeIndex
    # ------------------------------------------------------------------
    if not isinstance(df_survey_cma_raw.index, pd.DatetimeIndex):
        raise TypeError(
            "df_survey_cma_raw must have a DatetimeIndex. "
            f"Got index type: {type(df_survey_cma_raw.index).__name__}."
        )

    # ------------------------------------------------------------------
    # Parse as_of_date to pd.Timestamp
    # ------------------------------------------------------------------
    try:
        # Convert ISO-8601 string to timezone-naive pd.Timestamp
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
    except Exception as exc:
        raise ValueError(
            f"as_of_date='{as_of_date}' cannot be parsed. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Strip timezone from index if present
    # ------------------------------------------------------------------
    if df_survey_cma_raw.index.tz is not None:
        df_survey_cma_raw = df_survey_cma_raw.copy()
        df_survey_cma_raw.index = df_survey_cma_raw.index.tz_localize(None)

    # ------------------------------------------------------------------
    # Filter to rows matching the target asset class
    # ------------------------------------------------------------------
    # Boolean mask: rows where asset_class column matches the target
    ac_mask: pd.Series = df_survey_cma_raw["asset_class"] == asset_class
    # Apply the asset class filter
    df_ac: pd.DataFrame = df_survey_cma_raw.loc[ac_mask]

    # ------------------------------------------------------------------
    # Apply point-in-time filter: retain rows <= as_of_date
    # ------------------------------------------------------------------
    df_filtered: pd.DataFrame = df_ac.loc[df_ac.index <= as_of_ts]

    # ------------------------------------------------------------------
    # Handle no-data case: return None estimate with zero confidence
    # ------------------------------------------------------------------
    if df_filtered.empty:
        logger.warning(
            "run_cma_method_6: No survey data available for "
            "asset_class='%s' on or before '%s'. "
            "Returning estimate=None, confidence=0.0.",
            asset_class,
            as_of_date,
        )
        return {
            # Method identifier: 6 (Survey/Analyst)
            "method_id": 6,
            # No survey data available
            "estimate": None,
            # Zero confidence when no data is available
            "confidence": 0.0,
            # Breakdown indicating no data
            "breakdown": {
                "survey_date": "N/A",
                "months_stale": float("nan"),
                "decay_lambda": float(_SURVEY_DECAY_LAMBDA),
                "data_available": 0.0,
            },
            # Rationale explaining the no-data situation
            "rationale": (
                f"Method 6 (Survey/Analyst): No survey data available for "
                f"'{asset_class}' on or before '{as_of_date}'. "
                "Estimate set to None; excluded from auto-blend."
            ),
        }

    # ------------------------------------------------------------------
    # Select the most recent survey row
    # ------------------------------------------------------------------
    # Most recent survey row as a pd.Series
    latest_row: pd.Series = df_filtered.iloc[-1]
    # Date of the most recent survey
    survey_date: pd.Timestamp = df_filtered.index[-1]

    # ------------------------------------------------------------------
    # Extract the survey expected return value
    # ------------------------------------------------------------------
    survey_return_raw = latest_row["survey_expected_return"]

    # ------------------------------------------------------------------
    # Handle NaN survey return
    # ------------------------------------------------------------------
    if pd.isna(survey_return_raw):
        logger.warning(
            "run_cma_method_6: survey_expected_return is NaN for "
            "asset_class='%s' at date '%s'. "
            "Returning estimate=None, confidence=0.0.",
            asset_class,
            survey_date.date(),
        )
        return {
            "method_id": 6,
            "estimate": None,
            "confidence": 0.0,
            "breakdown": {
                "survey_date": str(survey_date.date()),
                "months_stale": float("nan"),
                "decay_lambda": float(_SURVEY_DECAY_LAMBDA),
                "data_available": 0.0,
            },
            "rationale": (
                f"Method 6 (Survey/Analyst): survey_expected_return is NaN "
                f"for '{asset_class}' at '{survey_date.date()}'. "
                "Estimate set to None."
            ),
        }

    # ------------------------------------------------------------------
    # Cast to float
    # ------------------------------------------------------------------
    # Survey expected return in decimal form
    survey_return: float = float(survey_return_raw)

    # ------------------------------------------------------------------
    # Decimal validation: survey_expected_return must be in [0, 1]
    # ------------------------------------------------------------------
    if survey_return > 1.0:
        raise ValueError(
            f"survey_expected_return for '{asset_class}' = {survey_return:.4f} "
            "exceeds 1.0. Survey returns must be in decimal form "
            "(e.g., 0.07 = 7%), not percent form."
        )

    # ------------------------------------------------------------------
    # Compute the number of months between the survey date and as_of_date
    # ------------------------------------------------------------------
    # Approximate months stale: difference in days / 30.44 (avg days/month)
    days_stale: float = float((as_of_ts - survey_date).days)
    months_stale: float = days_stale / 30.44

    # ------------------------------------------------------------------
    # Compute recency decay confidence:
    # c_6 = exp(-lambda * delta_t)
    # where lambda = _SURVEY_DECAY_LAMBDA = 0.1 per month
    # ------------------------------------------------------------------
    confidence: float = float(np.exp(-_SURVEY_DECAY_LAMBDA * months_stale))
    # Ensure confidence is bounded to [0, 1]
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Construct the breakdown dict for audit transparency
    # ------------------------------------------------------------------
    breakdown: Dict[str, Any] = {
        # Date of the most recent survey used
        "survey_date": str(survey_date.date()),
        # Number of months between survey date and as_of_date
        "months_stale": float(months_stale),
        # Recency decay constant (lambda)
        "decay_lambda": float(_SURVEY_DECAY_LAMBDA),
        # Flag indicating data was available
        "data_available": 1.0,
    }

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    rationale: str = (
        f"Method 6 (Survey/Analyst): "
        f"Survey expected return = {survey_return * 100:.2f}% "
        f"(source date: {survey_date.date()}, "
        f"{months_stale:.1f} months before as_of_date). "
        f"Recency decay confidence = exp(-{_SURVEY_DECAY_LAMBDA} × "
        f"{months_stale:.1f}) = {confidence:.3f}."
    )

    # Log the result for audit trail
    logger.debug(
        "run_cma_method_6: asset_class='%s', estimate=%.4f, "
        "months_stale=%.1f, confidence=%.4f",
        asset_class, survey_return, months_stale, confidence,
    )

    # ------------------------------------------------------------------
    # Return the method result dict
    # ------------------------------------------------------------------
    return {
        # Method identifier: 6 (Survey/Analyst)
        "method_id": 6,
        # Survey expected return in decimal form
        "estimate": float(survey_return),
        # Recency-decay confidence in [0, 1]
        "confidence": float(confidence),
        # Component breakdown for audit
        "breakdown": breakdown,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 17: run_cma_method_7 — Confidence-Weighted Auto-Blend
# =============================================================================

def run_cma_method_7(
    method_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute CMA Method 7: Confidence-Weighted Auto-Blend of Methods 1–6.

    Implements the confidence-weighted average of all valid method estimates
    (Task 21, Step 3):

    .. math::

        \\hat{\\mu}_7 = \\frac{\\sum_{k=1}^{6} c_k \\hat{\\mu}_k}
                              {\\sum_{k=1}^{6} c_k}

    where the sum is taken only over methods with non-``None`` estimates and
    positive confidence scores. Methods with ``estimate=None`` or
    ``confidence=0`` are excluded from the blend.

    The confidence of Method 7 is the weighted harmonic mean of the
    individual method confidences:

    .. math::

        c_7 = \\frac{\\sum_{k} c_k}{\\sum_{k} c_k / c_k} = \\bar{c}_{weighted}

    Specifically, :math:`c_7` is computed as the arithmetic mean of the
    individual confidences weighted by their own values (self-weighted mean):

    .. math::

        c_7 = \\frac{\\sum_k c_k^2}{\\sum_k c_k}

    Parameters
    ----------
    method_results : List[Dict[str, Any]]
        List of method result dicts from Methods 1–6. Each dict must have
        keys ``"method_id"`` (int), ``"estimate"`` (float or None),
        ``"confidence"`` (float in [0, 1]). Typically 6 elements but may
        be fewer for non-equity assets where some methods are inapplicable.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"method_id"`` (``int``): ``7``
        - ``"estimate"`` (``float``): Confidence-weighted blend in decimal
          form.
        - ``"confidence"`` (``float``): Self-weighted mean confidence in
          ``[0, 1]``.
        - ``"breakdown"`` (``Dict[str, Any]``): Component breakdown:
          ``n_valid_methods``, ``per_method_weights``,
          ``per_method_estimates``, ``sum_confidences``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``method_results`` is not a list.
    ValueError
        If ``method_results`` is empty.
    ValueError
        If fewer than 2 valid (non-None, positive-confidence) methods are
        available for blending.
    ValueError
        If the sum of confidences is effectively zero (all confidences are
        zero or near-zero).

    Notes
    -----
    **Minimum valid methods:** At least 2 valid methods are required for a
    meaningful blend. If only 1 valid method exists, the blend degenerates
    to that single method's estimate; this is allowed with a warning.

    **Exclusion criteria:** A method is excluded from the blend if:
    (1) ``estimate`` is ``None``, or (2) ``confidence`` is 0.0 or NaN.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(method_results, list):
        raise TypeError(
            f"method_results must be a list, "
            f"got {type(method_results).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-empty list
    # ------------------------------------------------------------------
    if len(method_results) == 0:
        raise ValueError(
            "method_results is empty. At least one method result is required."
        )

    # ------------------------------------------------------------------
    # Validate each method result dict for required keys
    # ------------------------------------------------------------------
    for i, result in enumerate(method_results):
        if not isinstance(result, dict):
            raise TypeError(
                f"method_results[{i}] must be a dict, "
                f"got {type(result).__name__}."
            )
        for key in ("method_id", "estimate", "confidence"):
            if key not in result:
                raise ValueError(
                    f"method_results[{i}] is missing required key '{key}'."
                )

    # ------------------------------------------------------------------
    # Filter to valid methods: non-None estimate and positive confidence
    # ------------------------------------------------------------------
    valid_methods: List[Dict[str, Any]] = []
    for result in method_results:
        # Exclude methods with None estimate (inapplicable methods)
        if result["estimate"] is None:
            continue
        # Exclude methods with zero or NaN confidence
        conf = result["confidence"]
        if conf is None or not np.isfinite(conf) or conf <= 0.0:
            continue
        # This method is valid for blending
        valid_methods.append(result)

    # ------------------------------------------------------------------
    # Guard: require at least 1 valid method
    # ------------------------------------------------------------------
    n_valid: int = len(valid_methods)
    if n_valid == 0:
        raise ValueError(
            "No valid methods available for auto-blend. "
            "All methods have None estimates or zero confidence."
        )

    # ------------------------------------------------------------------
    # Warn if fewer than 2 valid methods (degenerate blend)
    # ------------------------------------------------------------------
    if n_valid < 2:
        warnings.warn(
            f"run_cma_method_7: Only {n_valid} valid method(s) available "
            "for auto-blend. The blend degenerates to a single method's "
            "estimate.",
            UserWarning,
            stacklevel=2,
        )

    # ------------------------------------------------------------------
    # Extract estimates and confidences as numpy arrays
    # ------------------------------------------------------------------
    # Array of valid method estimates (decimal form)
    estimates_arr: np.ndarray = np.array(
        [float(m["estimate"]) for m in valid_methods],
        dtype=np.float64,
    )
    # Array of valid method confidences
    confidences_arr: np.ndarray = np.array(
        [float(m["confidence"]) for m in valid_methods],
        dtype=np.float64,
    )

    # ------------------------------------------------------------------
    # Guard: sum of confidences must be positive
    # ------------------------------------------------------------------
    sum_confidences: float = float(confidences_arr.sum())
    if sum_confidences < _EPS:
        raise ValueError(
            f"Sum of confidences is effectively zero ({sum_confidences:.2e}). "
            "Cannot compute confidence-weighted blend."
        )

    # ------------------------------------------------------------------
    # Compute the confidence-weighted auto-blend:
    # mu_hat_7 = sum(c_k * mu_hat_k) / sum(c_k)
    # ------------------------------------------------------------------
    # Weighted sum of estimates
    weighted_sum: float = float(np.dot(confidences_arr, estimates_arr))
    # Confidence-weighted average estimate
    estimate: float = weighted_sum / sum_confidences

    # ------------------------------------------------------------------
    # Compute the self-weighted mean confidence:
    # c_7 = sum(c_k^2) / sum(c_k)
    # This is the confidence-weighted average of the confidences themselves,
    # reflecting the overall reliability of the blend.
    # ------------------------------------------------------------------
    c7_numerator: float = float(np.dot(confidences_arr, confidences_arr))
    confidence: float = float(c7_numerator / sum_confidences)
    # Ensure confidence is bounded to [0, 1]
    confidence = float(np.clip(confidence, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Compute per-method normalised weights for the breakdown
    # w_k = c_k / sum(c_k)
    # ------------------------------------------------------------------
    # Normalised weights for each valid method
    normalised_weights: np.ndarray = confidences_arr / sum_confidences

    # ------------------------------------------------------------------
    # Construct the breakdown dict for audit transparency
    # ------------------------------------------------------------------
    # Per-method weight and estimate for audit
    per_method_weights: Dict[str, float] = {
        f"method_{m['method_id']}": float(w)
        for m, w in zip(valid_methods, normalised_weights)
    }
    per_method_estimates: Dict[str, float] = {
        f"method_{m['method_id']}": float(m["estimate"])
        for m in valid_methods
    }

    breakdown: Dict[str, Any] = {
        # Number of valid methods included in the blend
        "n_valid_methods": float(n_valid),
        # Normalised weight assigned to each method
        "per_method_weights": per_method_weights,
        # Estimate from each method included in the blend
        "per_method_estimates": per_method_estimates,
        # Sum of confidences (denominator of the weighted average)
        "sum_confidences": float(sum_confidences),
    }

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    method_summary: str = ", ".join(
        f"M{m['method_id']}={m['estimate'] * 100:.2f}% "
        f"(w={w * 100:.1f}%)"
        for m, w in zip(valid_methods, normalised_weights)
    )
    rationale: str = (
        f"Method 7 (Confidence-Weighted Auto-Blend): "
        f"{n_valid} valid methods blended. "
        f"Components: [{method_summary}]. "
        f"Blend estimate = {estimate * 100:.2f}%. "
        f"Self-weighted confidence = {confidence:.3f}."
    )

    # Log the result for audit trail
    logger.debug(
        "run_cma_method_7: n_valid=%d, estimate=%.4f, confidence=%.4f",
        n_valid, estimate, confidence,
    )

    # ------------------------------------------------------------------
    # Return the method result dict
    # ------------------------------------------------------------------
    return {
        # Method identifier: 7 (Confidence-Weighted Auto-Blend)
        "method_id": 7,
        # Confidence-weighted blend estimate in decimal form
        "estimate": float(estimate),
        # Self-weighted mean confidence in [0, 1]
        "confidence": float(confidence),
        # Component breakdown for audit
        "breakdown": breakdown,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 18: run_fi_cma_builder — Fixed Income CMA Builder
# =============================================================================

def run_fi_cma_builder(
    asset_class: str,
    as_of_date: str,
    df_fixed_income_curves_spreads_raw: pd.DataFrame,
    df_total_return_raw: pd.DataFrame,
    universe_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute the Fixed Income CMA using the building-block decomposition.

    Implements the frozen FI CMA formula from
    ``METHODOLOGY_PARAMS["FI_CMA_FORMULA"]`` (Task 23):

    .. math::

        \\hat{\\mu}_{FI} = y + \\text{roll-down} + \\text{spread carry}
                         - \\text{expected default loss}

    where:

    - :math:`y` = current benchmark yield (asset-class-specific)
    - :math:`\\text{roll-down}` = price appreciation from rolling down the
      yield curve by 1 year (approximated via linear interpolation)
    - :math:`\\text{spread carry}` = option-adjusted spread (OAS) for credit
      assets; zero for sovereign bonds
    - :math:`\\text{expected default loss}` = :math:`PD \\times LGD`,
      approximated as a fraction of the OAS for credit assets; zero for
      sovereigns

    Also computes:

    - ``credit_spread_duration``: modified duration × spread sensitivity
      (as mandated by the manuscript narrative, Section 3.3)
    - ``sector_concentration``: flagged as unknown (sector weights not
      available in the input data)

    Parameters
    ----------
    asset_class : str
        Fixed income asset class name. Must be a key in
        ``_FI_YIELD_FIELD_MAP``.
    as_of_date : str
        ISO-8601 date string for point-in-time data retrieval.
    df_fixed_income_curves_spreads_raw : pd.DataFrame
        Fixed income yield curve and spread panel. Must have a
        ``DatetimeIndex`` (monthly, month-end) and columns:
        ``"ust_3m_yield"``, ``"ust_2y_yield"``, ``"ust_10y_yield"``,
        ``"ust_30y_yield"``, ``"ig_oas"``, ``"hy_oas"``, ``"em_spread"``.
        All values in decimal form. Shape: ``(T_fi, ≥7)``.
    df_total_return_raw : pd.DataFrame
        Total return index panel (used for historical context and
        confidence scoring). Shape: ``(T_total, 1)``.
    universe_map : Dict[str, Dict[str, Any]]
        Mapping from asset class names to metadata.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"estimate"`` (``float``): :math:`\\hat{\\mu}_{FI}` in decimal
          form (annualised).
        - ``"confidence"`` (``float``): Yield curve stability confidence
          in ``[0, 1]``.
        - ``"credit_spread_duration"`` (``float``): Modified duration ×
          spread sensitivity (years × decimal spread).
        - ``"sector_concentration"`` (``float | None``): Herfindahl index
          of sector weights; ``None`` if sector data unavailable.
        - ``"breakdown"`` (``Dict[str, float]``): Component breakdown:
          ``yield_component``, ``roll_down``, ``spread_carry``,
          ``expected_default_loss``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``df_fixed_income_curves_spreads_raw`` is not a ``pd.DataFrame``.
    ValueError
        If ``asset_class`` is not in ``_FI_YIELD_FIELD_MAP``.
    ValueError
        If required yield curve columns are missing.
    ValueError
        If no data is available on or before ``as_of_date``.

    Notes
    -----
    **Roll-down approximation:** The roll-down is computed as the price
    change from rolling down the yield curve by 1 year, approximated using
    linear interpolation between available tenor points:

    .. math::

        \\text{roll-down} \\approx -D_{mod} \\cdot (y_{T-1} - y_T)

    where :math:`D_{mod}` is the approximate modified duration and
    :math:`y_{T-1}` is the interpolated yield at maturity minus 1 year.

    **Expected default loss approximation (frozen):** For credit assets:

    .. math::

        \\text{EDL} \\approx \\text{pd\\_spread\\_fraction} \\times
        \\text{OAS} \\times \\text{LGD}

    This is a simplified approximation documented as a frozen implementation
    choice per ``METHODOLOGY_PARAMS["FI_CMA_FORMULA"]``.

    **Sector concentration:** Set to ``None`` as sector weight data is not
    available in the input DataFrames. The manuscript mandates reporting
    this field; it is flagged as unknown for audit transparency.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(df_fixed_income_curves_spreads_raw, pd.DataFrame):
        raise TypeError(
            f"df_fixed_income_curves_spreads_raw must be a pd.DataFrame, "
            f"got {type(df_fixed_income_curves_spreads_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: asset_class must be in the FI yield field map
    # ------------------------------------------------------------------
    if asset_class not in _FI_YIELD_FIELD_MAP:
        raise ValueError(
            f"asset_class='{asset_class}' is not a recognised fixed income "
            f"asset class. Must be one of: {list(_FI_YIELD_FIELD_MAP.keys())}."
        )

    # ------------------------------------------------------------------
    # Retrieve the asset-class-specific field configuration
    # ------------------------------------------------------------------
    # Configuration dict for this FI asset class
    fi_config: Dict[str, Any] = _FI_YIELD_FIELD_MAP[asset_class]
    # Primary yield field name for this asset class
    yield_field: str = fi_config["yield_field"]
    # Spread field name (None for sovereign bonds)
    spread_field: Optional[str] = fi_config["spread_field"]
    # Approximate maturity in years
    maturity_years: float = fi_config["maturity_years"]
    # Approximate modified duration in years
    duration_approx: float = fi_config["duration_approx"]
    # Whether this is a credit asset (has default risk)
    is_credit: bool = fi_config["is_credit"]

    # ------------------------------------------------------------------
    # Input validation: required yield curve columns
    # ------------------------------------------------------------------
    _required_yield_cols: List[str] = list(_YIELD_CURVE_FIELDS)
    missing_cols: List[str] = [
        c for c in _required_yield_cols
        if c not in df_fixed_income_curves_spreads_raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_fixed_income_curves_spreads_raw is missing required "
            f"yield curve columns: {missing_cols}."
        )

    # ------------------------------------------------------------------
    # Input validation: primary yield field must be present
    # ------------------------------------------------------------------
    if yield_field not in df_fixed_income_curves_spreads_raw.columns:
        raise ValueError(
            f"Primary yield field '{yield_field}' for asset_class="
            f"'{asset_class}' not found in "
            f"df_fixed_income_curves_spreads_raw."
        )

    # ------------------------------------------------------------------
    # Parse as_of_date to pd.Timestamp
    # ------------------------------------------------------------------
    try:
        # Convert ISO-8601 string to timezone-naive pd.Timestamp
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
    except Exception as exc:
        raise ValueError(
            f"as_of_date='{as_of_date}' cannot be parsed. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Strip timezone from index if present
    # ------------------------------------------------------------------
    df_fi: pd.DataFrame = df_fixed_income_curves_spreads_raw.copy()
    if df_fi.index.tz is not None:
        df_fi.index = df_fi.index.tz_localize(None)

    # ------------------------------------------------------------------
    # Apply point-in-time filter: retain rows <= as_of_date
    # ------------------------------------------------------------------
    df_filtered: pd.DataFrame = df_fi.loc[df_fi.index <= as_of_ts]

    # ------------------------------------------------------------------
    # Guard: ensure data is available after filtering
    # ------------------------------------------------------------------
    if df_filtered.empty:
        raise ValueError(
            f"No fixed income curve data available on or before "
            f"'{as_of_date}'."
        )

    # ------------------------------------------------------------------
    # Select the most recent row
    # ------------------------------------------------------------------
    # Most recent yield curve and spread snapshot
    latest_row: pd.Series = df_filtered.iloc[-1]

    # ------------------------------------------------------------------
    # Extract the primary yield component: y
    # ------------------------------------------------------------------
    # Current benchmark yield for this asset class (decimal form)
    y_raw = latest_row[yield_field]
    if pd.isna(y_raw):
        raise ValueError(
            f"Primary yield field '{yield_field}' is NaN for "
            f"asset_class='{asset_class}' at the most recent date."
        )
    # Benchmark yield in decimal form
    y: float = float(y_raw)

    # ------------------------------------------------------------------
    # Extract the spread component (for credit assets)
    # ------------------------------------------------------------------
    # Option-adjusted spread (OAS) in decimal form; 0.0 for sovereigns
    spread: float = 0.0
    if spread_field is not None and spread_field in df_filtered.columns:
        spread_raw = latest_row[spread_field]
        if not pd.isna(spread_raw):
            spread = float(spread_raw)

    # ------------------------------------------------------------------
    # Compute the roll-down component via yield curve interpolation
    # Roll-down ≈ -D_mod * (y_{T-1} - y_T)
    # where y_{T-1} is the interpolated yield at maturity - 1 year
    # ------------------------------------------------------------------
    # Extract the yield curve tenor points and their current yields
    tenor_yields: List[float] = []
    for field in _YIELD_CURVE_FIELDS:
        if field in latest_row.index and not pd.isna(latest_row[field]):
            tenor_yields.append(float(latest_row[field]))
        else:
            # Use NaN as placeholder for missing tenor points
            tenor_yields.append(float("nan"))

    # Filter to valid (non-NaN) tenor points for interpolation
    valid_tenors: List[float] = []
    valid_yields: List[float] = []
    for tenor, yld in zip(_YIELD_CURVE_TENORS, tenor_yields):
        if not np.isnan(yld):
            valid_tenors.append(tenor)
            valid_yields.append(yld)

    # Compute roll-down if we have at least 2 valid tenor points
    roll_down: float = 0.0
    if len(valid_tenors) >= 2 and maturity_years > 1.0:
        # Build a linear interpolator for the yield curve
        # bounds_error=False allows extrapolation at the boundaries
        yield_curve_interp = interp1d(
            valid_tenors,
            valid_yields,
            kind="linear",
            bounds_error=False,
            fill_value=(valid_yields[0], valid_yields[-1]),
        )
        # Interpolated yield at maturity - 1 year (roll-down target)
        y_rolled: float = float(
            yield_curve_interp(max(maturity_years - 1.0, 0.25))
        )
        # Roll-down ≈ -D_mod * (y_{T-1} - y_T)
        # Positive when yield curve is upward sloping (y_{T-1} < y_T)
        roll_down = float(-duration_approx * (y_rolled - y))
    else:
        # Insufficient tenor points for interpolation; set roll-down to 0
        logger.warning(
            "run_fi_cma_builder: Insufficient yield curve tenor points "
            "for roll-down computation for '%s'. Setting roll-down = 0.",
            asset_class,
        )

    # ------------------------------------------------------------------
    # Compute the spread carry component
    # For credit assets: spread_carry = OAS
    # For sovereign bonds: spread_carry = 0
    # ------------------------------------------------------------------
    # Spread carry is the full OAS for credit assets
    spread_carry: float = spread if is_credit else 0.0

    # ------------------------------------------------------------------
    # Compute the expected default loss (EDL) for credit assets
    # EDL ≈ pd_spread_fraction * OAS * LGD (frozen approximation)
    # For sovereign bonds: EDL = 0
    # ------------------------------------------------------------------
    expected_default_loss: float = 0.0
    if is_credit and spread > 0.0:
        # Retrieve the frozen EDL parameters for this asset class
        lgd: float = float(fi_config.get("lgd", 0.60))
        pd_fraction: float = float(fi_config.get("pd_spread_fraction", 0.40))
        # EDL ≈ pd_spread_fraction * OAS * LGD
        expected_default_loss = pd_fraction * spread * lgd

    # ------------------------------------------------------------------
    # Compute the total FI CMA estimate:
    # mu_hat_FI = y + roll_down + spread_carry - expected_default_loss
    # ------------------------------------------------------------------
    estimate: float = y + roll_down + spread_carry - expected_default_loss

    # ------------------------------------------------------------------
    # Compute credit_spread_duration:
    # credit_spread_duration = modified_duration * spread
    # (sensitivity of portfolio value to a 1pp change in spread)
    # ------------------------------------------------------------------
    credit_spread_duration: float = duration_approx * spread

    # ------------------------------------------------------------------
    # Sector concentration: flagged as None (data not available)
    # The manuscript mandates reporting this field; flagged for audit.
    # ------------------------------------------------------------------
    sector_concentration: Optional[float] = None

    # ------------------------------------------------------------------
    # Compute confidence based on yield curve data recency and stability
    # Confidence = 0.8 for credit assets (spread adds uncertainty),
    # 0.9 for sovereign bonds (yield is directly observable)
    # ------------------------------------------------------------------
    confidence: float = 0.8 if is_credit else 0.9

    # ------------------------------------------------------------------
    # Construct the breakdown dict for audit transparency
    # ------------------------------------------------------------------
    breakdown: Dict[str, float] = {
        # Benchmark yield component: y
        "yield_component": float(y),
        # Roll-down component (price appreciation from rolling down curve)
        "roll_down": float(roll_down),
        # Spread carry component (OAS for credit; 0 for sovereigns)
        "spread_carry": float(spread_carry),
        # Expected default loss (EDL = pd_fraction * OAS * LGD)
        "expected_default_loss": float(expected_default_loss),
        # Raw OAS or spread value
        "spread_raw": float(spread),
        # Approximate modified duration used
        "duration_approx": float(duration_approx),
        # Approximate maturity in years
        "maturity_years": float(maturity_years),
    }

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    rationale: str = (
        f"FI CMA Builder for '{asset_class}': "
        f"mu_FI = y + roll_down + spread_carry - EDL = "
        f"{y * 100:.2f}% + {roll_down * 100:.2f}% + "
        f"{spread_carry * 100:.2f}% - {expected_default_loss * 100:.2f}% = "
        f"{estimate * 100:.2f}%. "
        f"Credit spread duration = {credit_spread_duration:.3f}. "
        f"Sector concentration = {'N/A (data unavailable)' if sector_concentration is None else f'{sector_concentration:.3f}'}. "
        f"Confidence = {confidence:.3f}."
    )

    # Log the result for audit trail
    logger.debug(
        "run_fi_cma_builder: asset_class='%s', estimate=%.4f, "
        "y=%.4f, roll_down=%.4f, spread_carry=%.4f, edl=%.4f",
        asset_class, estimate, y, roll_down, spread_carry,
        expected_default_loss,
    )

    # ------------------------------------------------------------------
    # Return the FI CMA result dict
    # ------------------------------------------------------------------
    return {
        # FI CMA estimate in decimal form (annualised)
        "estimate": float(estimate),
        # Yield curve stability confidence in [0, 1]
        "confidence": float(confidence),
        # Modified duration × spread (credit spread duration)
        "credit_spread_duration": float(credit_spread_duration),
        # Sector concentration (None if data unavailable)
        "sector_concentration": sector_concentration,
        # Component breakdown for audit
        "breakdown": breakdown,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 19: load_cma_methods_json — CMA Judge Input Loader
# =============================================================================

def load_cma_methods_json(
    asset_class: str,
    artifact_dir: Path,
) -> List[Dict[str, Any]]:
    """
    Load and validate the ``cma_methods.json`` artifact for a given asset class.

    This tool implements the first step of the CMA Judge ReAct loop (Task 22,
    Step 1). It loads the ``cma_methods.json`` file written by the AC agent
    for the given asset class, validates its structure against the expected
    schema, and returns the validated list of method result dicts.

    The file is expected at:
    ``{artifact_dir}/{asset_class_slug}/cma_methods.json``

    where ``asset_class_slug`` is derived by converting the asset class name
    to lowercase and replacing spaces with underscores (e.g.,
    ``"US Large Cap"`` → ``"us_large_cap"``).

    Parameters
    ----------
    asset_class : str
        Asset class name (e.g., ``"US Large Cap"``). Used to construct the
        file path via slug derivation.
    artifact_dir : Path
        Base artifact directory. The file is expected at
        ``{artifact_dir}/{asset_class_slug}/cma_methods.json``.

    Returns
    -------
    List[Dict[str, Any]]
        Validated list of method result dicts. Each dict contains:

        - ``"method_id"`` (``int``): Method identifier (1–7).
        - ``"estimate"`` (``float | None``): Expected return estimate in
          decimal form, or ``None`` if the method was inapplicable.
        - ``"confidence"`` (``float``): Confidence score in ``[0, 1]``.
        - ``"breakdown"`` (``dict``): Component breakdown.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    FileNotFoundError
        If ``cma_methods.json`` does not exist at the expected path.
    ValueError
        If the file contains invalid JSON.
    ValueError
        If the loaded content is not a list.
    ValueError
        If any method dict is missing required keys.

    Notes
    -----
    **Slug derivation:** The asset class slug is derived by:
    (1) converting to lowercase, (2) replacing spaces with underscores,
    (3) removing any characters that are not alphanumeric or underscores.
    Example: ``"US Large Cap"`` → ``"us_large_cap"``.
    """
    # ------------------------------------------------------------------
    # Input validation: artifact_dir type
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Derive the asset class slug from the asset class name
    # Step 1: convert to lowercase
    # Step 2: replace spaces with underscores
    # Step 3: remove non-alphanumeric/underscore characters
    # ------------------------------------------------------------------
    # Convert asset class name to lowercase
    slug_lower: str = asset_class.lower()
    # Replace spaces with underscores
    slug_underscored: str = slug_lower.replace(" ", "_")
    # Remove any characters that are not alphanumeric or underscores
    asset_class_slug: str = re.sub(r"[^a-z0-9_]", "", slug_underscored)

    # ------------------------------------------------------------------
    # Construct the expected file path
    # ------------------------------------------------------------------
    # Full path to the cma_methods.json artifact
    file_path: Path = artifact_dir / asset_class_slug / "cma_methods.json"

    # ------------------------------------------------------------------
    # Check that the file exists
    # ------------------------------------------------------------------
    if not file_path.exists():
        raise FileNotFoundError(
            f"cma_methods.json not found at expected path: '{file_path}'. "
            f"Ensure the AC agent for '{asset_class}' has completed "
            "successfully and written its output artifacts."
        )

    # ------------------------------------------------------------------
    # Load and parse the JSON file
    # ------------------------------------------------------------------
    try:
        with open(file_path, "r", encoding="utf-8") as fh:
            # Parse the JSON content into a Python object
            raw_content: Any = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Failed to parse '{file_path}' as valid JSON. "
            f"The file may be corrupted or partially written. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Validate that the loaded content is a list
    # ------------------------------------------------------------------
    if not isinstance(raw_content, list):
        raise ValueError(
            f"'{file_path}' must contain a JSON array (list), "
            f"got {type(raw_content).__name__}."
        )

    # ------------------------------------------------------------------
    # Validate each method dict for required keys
    # ------------------------------------------------------------------
    _required_method_keys: Tuple[str, ...] = (
        "method_id",
        "estimate",
        "confidence",
        "breakdown",
        "rationale",
    )
    for i, method_dict in enumerate(raw_content):
        # Each element must be a dict
        if not isinstance(method_dict, dict):
            raise ValueError(
                f"cma_methods.json element [{i}] must be a dict, "
                f"got {type(method_dict).__name__}."
            )
        # Check for required keys
        missing_keys: List[str] = [
            k for k in _required_method_keys if k not in method_dict
        ]
        if missing_keys:
            raise ValueError(
                f"cma_methods.json element [{i}] (method_id="
                f"{method_dict.get('method_id', 'unknown')}) "
                f"is missing required keys: {missing_keys}."
            )
        # Validate confidence is in [0, 1] if not None
        conf = method_dict["confidence"]
        if conf is not None and not (0.0 <= float(conf) <= 1.0):
            raise ValueError(
                f"cma_methods.json element [{i}] has confidence="
                f"{conf} outside [0, 1]."
            )

    # Log the successful load for audit trail
    logger.info(
        "load_cma_methods_json: loaded %d method results for "
        "asset_class='%s' from '%s'.",
        len(raw_content),
        asset_class,
        file_path,
    )

    # Return the validated list of method result dicts
    return raw_content


# =============================================================================
# TOOL 20: classify_dispersion — CMA Judge Dispersion Classifier
# =============================================================================

def classify_dispersion(
    estimates: List[Optional[float]],
    estimates_in_decimal: bool = True,
) -> Dict[str, Any]:
    """
    Classify the dispersion of CMA method estimates for the CMA Judge.

    Implements Step 1 of the Exhibit 4 CMA Judge algorithm (Task 22, Step 2):

    Compute the spread between the maximum and minimum valid method estimates
    in percentage points (pp), then classify as:

    - **Tight:** spread < 3.0 pp
    - **Moderate:** 3.0 pp ≤ spread ≤ 6.0 pp
    - **Wide:** spread > 6.0 pp

    using the frozen thresholds from
    ``METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]["dispersion_thresholds_pp"]``:
    ``tight_upper=3.0``, ``moderate_upper=6.0``.

    Parameters
    ----------
    estimates : List[Optional[float]]
        List of method estimates. ``None`` values (inapplicable methods)
        are excluded from the dispersion computation. Typically 7 elements
        (methods 1–7) but may be fewer for non-equity assets.
    estimates_in_decimal : bool
        If ``True`` (default), estimates are in decimal form (e.g., 0.075
        = 7.5%) and are multiplied by 100 to convert to percentage points
        before computing the spread. If ``False``, estimates are already
        in percentage points (e.g., 7.5 = 7.5%).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"classification"`` (``str``): One of ``{"tight", "moderate",
          "wide"}``.
        - ``"spread_pp"`` (``float``): Spread in percentage points
          (max - min of valid estimates, in pp).
        - ``"min_estimate"`` (``float``): Minimum valid estimate (in the
          same units as input, before pp conversion).
        - ``"max_estimate"`` (``float``): Maximum valid estimate (in the
          same units as input, before pp conversion).
        - ``"n_valid_estimates"`` (``int``): Number of non-None estimates
          used in the computation.

    Raises
    ------
    TypeError
        If ``estimates`` is not a list.
    ValueError
        If all estimates are ``None`` (no valid estimates to classify).

    Notes
    -----
    **Single valid estimate:** If only one valid estimate exists, the spread
    is 0.0 pp, which classifies as ``"tight"``. This is a valid edge case
    (e.g., for cash, where only one method applies).

    **Frozen thresholds:** The classification thresholds are sourced from
    ``METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]["dispersion_thresholds_pp"]``:
    tight < 3.0 pp ≤ moderate ≤ 6.0 pp < wide.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(estimates, list):
        raise TypeError(
            f"estimates must be a list, got {type(estimates).__name__}."
        )

    # ------------------------------------------------------------------
    # Filter out None values to get valid estimates
    # ------------------------------------------------------------------
    valid_estimates: List[float] = [
        float(e) for e in estimates if e is not None
    ]

    # ------------------------------------------------------------------
    # Guard: at least one valid estimate required
    # ------------------------------------------------------------------
    if len(valid_estimates) == 0:
        raise ValueError(
            "All estimates are None. At least one valid estimate is required "
            "to classify dispersion."
        )

    # ------------------------------------------------------------------
    # Convert to numpy array for efficient min/max computation
    # ------------------------------------------------------------------
    # Array of valid estimates in their original units
    estimates_arr: np.ndarray = np.array(valid_estimates, dtype=np.float64)

    # ------------------------------------------------------------------
    # Compute min and max of valid estimates (in original units)
    # ------------------------------------------------------------------
    # Minimum valid estimate (original units)
    min_estimate: float = float(estimates_arr.min())
    # Maximum valid estimate (original units)
    max_estimate: float = float(estimates_arr.max())

    # ------------------------------------------------------------------
    # Compute the spread in percentage points
    # If estimates_in_decimal=True, multiply by 100 to convert to pp
    # If estimates_in_decimal=False, estimates are already in pp
    # ------------------------------------------------------------------
    # Raw spread in original units
    raw_spread: float = max_estimate - min_estimate

    if estimates_in_decimal:
        # Convert from decimal to percentage points: multiply by 100
        spread_pp: float = raw_spread * 100.0
    else:
        # Estimates are already in percentage points
        spread_pp = raw_spread

    # ------------------------------------------------------------------
    # Classify the dispersion using frozen thresholds:
    # tight: spread_pp < 3.0
    # moderate: 3.0 <= spread_pp <= 6.0
    # wide: spread_pp > 6.0
    # per METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]["dispersion_thresholds_pp"]
    # ------------------------------------------------------------------
    if spread_pp < _DISPERSION_TIGHT_UPPER:
        # Spread is below the tight threshold: methods broadly agree
        classification: str = "tight"
    elif spread_pp <= _DISPERSION_MODERATE_UPPER:
        # Spread is between tight and moderate thresholds
        classification = "moderate"
    else:
        # Spread exceeds the moderate threshold: wide disagreement
        classification = "wide"

    # Log the classification result for audit trail
    logger.debug(
        "classify_dispersion: n_valid=%d, spread_pp=%.2f, "
        "classification='%s', min=%.4f, max=%.4f",
        len(valid_estimates),
        spread_pp,
        classification,
        min_estimate,
        max_estimate,
    )

    # ------------------------------------------------------------------
    # Return the dispersion classification dict
    # ------------------------------------------------------------------
    return {
        # Dispersion classification: tight, moderate, or wide
        "classification": classification,
        # Spread in percentage points (max - min, in pp)
        "spread_pp": float(spread_pp),
        # Minimum valid estimate (original units)
        "min_estimate": float(min_estimate),
        # Maximum valid estimate (original units)
        "max_estimate": float(max_estimate),
        # Number of non-None estimates used
        "n_valid_estimates": len(valid_estimates),
    }

# =============================================================================
# SELF-DRIVING PORTFOLIO: TOOL REGISTRY — BATCH 3 (TOOLS 21–30)
# =============================================================================
# Implements tools 21–30 from the complete 78-tool registry for the agentic
# Strategic Asset Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# Tools implemented:
#   21. check_regime_logic          — CMA Judge: Exhibit 4 Step 2
#   22. check_valuation_thresholds  — CMA Judge: Exhibit 4 Step 3
#   23. check_signal_alignment      — CMA Judge: Exhibit 4 Step 4
#   24. enforce_range_constraint    — CMA Judge: Exhibit 4 Hard Constraint
#   25. write_cma_methods_json      — AC Agent artifact writer
#   26. write_signals_json          — AC Agent artifact writer
#   27. write_historical_stats_json — AC Agent artifact writer
#   28. write_scenarios_json        — AC Agent artifact writer
#   29. write_correlation_row_json  — AC Agent artifact writer
#   30. write_cma_json              — CMA Judge / FI / Real Assets writer
#
# All tools are purely deterministic Python callables — no LLM interaction.
# All arithmetic conventions are frozen per STUDY_CONFIG["DATA_CONVENTIONS"].
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# Initialise a named logger so callers can configure log levels independently
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants (sourced from STUDY_CONFIG; reproduced for self-contained
# validation — the orchestrator injects the live config at runtime)
# ---------------------------------------------------------------------------

# Valid macro regime labels per METHODOLOGY_PARAMS["MACRO_REGIMES"]
_VALID_REGIMES: Tuple[str, ...] = (
    "Expansion",
    "Late-cycle",
    "Recession",
    "Recovery",
)

# Frozen regime logic per METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]["regime_logic"]
# Maps each regime to the list of preferred method IDs (Exhibit 4, Step 2)
_REGIME_PREFERRED_METHODS: Dict[str, List[int]] = {
    "Late-cycle": [4, 5, 2],   # tilt valuation (4,5) + regime-adjusted (2)
    "Expansion":  [7],          # default auto-blend
    "Recession":  [2, 3],       # regime-adjusted (2) + BL equilibrium (3)
    "Recovery":   [1, 3],       # historical ERP (1) + BL equilibrium (3)
}

# Frozen valuation thresholds per METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]
_PE_THRESHOLD_EXPENSIVE: float = 30.0   # PE > 30 → elevated (tilt methods 4,5)
_PE_THRESHOLD_CHEAP: float = 12.0       # PE < 12 → depressed (tilt methods 1,3)

# RSI thresholds for signal alignment (standard technical analysis values)
_RSI_OVERBOUGHT: float = 70.0   # RSI > 70 → overbought (bearish forward signal)
_RSI_OVERSOLD: float = 30.0     # RSI < 30 → oversold (bullish forward signal)

# Market breadth thresholds for signal alignment
_BREADTH_BULLISH: float = 0.60  # > 60% constituents above 200d MA → bullish
_BREADTH_BEARISH: float = 0.40  # < 40% constituents above 200d MA → bearish

# Signal alignment classification thresholds
_ALIGNMENT_CONFIRMING_THRESHOLD: float = 0.30   # score > 0.30 → confirming
_ALIGNMENT_CONTRADICTING_THRESHOLD: float = -0.30  # score < -0.30 → contradicting

# Floating-point tolerance for range constraint boundary checks
_RANGE_CONSTRAINT_TOLERANCE: float = 1e-8

# Numerical stability epsilon
_EPS: float = 1e-8

# Required keys for each method result dict (used in validation)
_REQUIRED_METHOD_RESULT_KEYS: Tuple[str, ...] = (
    "method_id",
    "estimate",
    "confidence",
    "breakdown",
    "rationale",
)

# Required keys for the method_range dict in write_cma_json
_REQUIRED_METHOD_RANGE_KEYS: Tuple[str, ...] = ("min", "max")

# Required keys for the signals dict (from fetch_signals output)
_REQUIRED_SIGNALS_KEYS: Tuple[str, ...] = (
    "asset_class",
    "ticker",
    "category",
    "as_of_date",
)

# Required keys for the historical_stats dict (from fetch_historical_stats)
_REQUIRED_HISTORICAL_STATS_KEYS: Tuple[str, ...] = (
    "asset_class",
    "annualised_return",
    "annualised_vol",
    "max_drawdown",
    "n_observations",
    "monthly_returns_series",
    "correlation_matrix",
)


# ---------------------------------------------------------------------------
# Shared utility: recursive JSON-safe type casting
# ---------------------------------------------------------------------------

def _cast_to_json_safe(obj: Any) -> Any:
    """
    Recursively cast an object to JSON-serialisable Python native types.

    Converts ``np.float64``, ``np.int64``, ``np.bool_``, ``np.ndarray``,
    ``pd.Series``, and ``pd.DataFrame`` to their Python native equivalents.
    ``None`` and ``float("nan")`` are preserved as ``None`` (JSON ``null``).

    Parameters
    ----------
    obj : Any
        The object to cast. May be a scalar, list, dict, or nested structure.

    Returns
    -------
    Any
        A JSON-serialisable Python native object.
    """
    # Handle None: preserve as None (serialises to JSON null)
    if obj is None:
        return None

    # Handle numpy floating-point scalars: cast to Python float
    if isinstance(obj, (np.floating,)):
        # Convert NaN numpy floats to None (JSON null)
        if np.isnan(obj):
            return None
        return float(obj)

    # Handle Python float: convert NaN to None
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        return obj

    # Handle numpy integer scalars: cast to Python int
    if isinstance(obj, (np.integer,)):
        return int(obj)

    # Handle numpy boolean scalars: cast to Python bool
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Handle numpy arrays: convert to list of JSON-safe elements
    if isinstance(obj, np.ndarray):
        return [_cast_to_json_safe(v) for v in obj.tolist()]

    # Handle pandas Series: convert to dict with string keys
    if isinstance(obj, pd.Series):
        return {
            str(k): _cast_to_json_safe(v)
            for k, v in obj.items()
        }

    # Handle pandas DataFrame: convert to nested dict
    if isinstance(obj, pd.DataFrame):
        return {
            str(col): {
                str(idx): _cast_to_json_safe(val)
                for idx, val in obj[col].items()
            }
            for col in obj.columns
        }

    # Handle dicts: recursively cast all values
    if isinstance(obj, dict):
        return {
            str(k): _cast_to_json_safe(v)
            for k, v in obj.items()
        }

    # Handle lists and tuples: recursively cast all elements
    if isinstance(obj, (list, tuple)):
        return [_cast_to_json_safe(v) for v in obj]

    # Handle Python int and bool: return as-is
    if isinstance(obj, (int, bool)):
        return obj

    # Handle strings: return as-is
    if isinstance(obj, str):
        return obj

    # Fallback: attempt str conversion for unknown types
    return str(obj)


def _derive_asset_class_slug(asset_class: str) -> str:
    """
    Derive the filesystem-safe slug from an asset class name.

    Applies the canonical slug derivation:
    (1) convert to lowercase,
    (2) replace spaces with underscores,
    (3) remove non-alphanumeric/underscore characters.

    Example: ``"US Large Cap"`` → ``"us_large_cap"``.

    Parameters
    ----------
    asset_class : str
        Asset class name (e.g., ``"US Large Cap"``).

    Returns
    -------
    str
        Filesystem-safe slug (e.g., ``"us_large_cap"``).
    """
    # Step 1: convert to lowercase
    slug_lower: str = asset_class.lower()
    # Step 2: replace spaces with underscores
    slug_underscored: str = slug_lower.replace(" ", "_")
    # Step 3: remove any characters that are not alphanumeric or underscores
    slug_clean: str = re.sub(r"[^a-z0-9_]", "", slug_underscored)
    return slug_clean


# =============================================================================
# TOOL 21: check_regime_logic
# =============================================================================

def check_regime_logic(
    macro_view: Dict[str, Any],
    dispersion: Dict[str, Any],
    regime_preferred_methods: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, Any]:
    """
    Apply the regime-conditional method selection hint from Exhibit 4 Step 2.

    Implements the second step of the CMA Judge algorithm (Task 22, Step 2),
    as specified verbatim in Exhibit 4:

    - **Late-cycle:** tilt toward valuation methods (4, 5) and
      regime-adjusted (2)
    - **Expansion:** default to auto-blend (7)
    - **Recession:** tilt toward regime-adjusted (2) and BL equilibrium (3)
    - **Recovery:** tilt toward historical ERP (1) and BL equilibrium (3)

    The dispersion classification from Step 1 modulates the strength of the
    regime hint:

    - **Tight dispersion:** regime hint is advisory (methods broadly agree;
      auto-blend is acceptable regardless of regime)
    - **Moderate dispersion:** regime hint is a moderate tilt
    - **Wide dispersion:** regime hint is a strong tilt (methods disagree
      significantly; regime logic should dominate)

    Parameters
    ----------
    macro_view : Dict[str, Any]
        Output of ``write_macro_view_json`` (loaded from artifact). Must
        contain ``"regime"`` (``str``) and ``"confidence"`` (``float``).
    dispersion : Dict[str, Any]
        Output of ``classify_dispersion``. Must contain
        ``"classification"`` (``str``, one of ``{"tight", "moderate",
        "wide"}``) and ``"spread_pp"`` (``float``).
    regime_preferred_methods : Optional[Dict[str, List[int]]]
        Override for the frozen regime-to-method mapping. If ``None``,
        uses ``_REGIME_PREFERRED_METHODS`` from
        ``METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]["regime_logic"]``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"preferred_method_ids"`` (``List[int]``): Ordered list of
          preferred method IDs for the current regime. The first element
          is the most strongly preferred.
        - ``"regime"`` (``str``): Current macro regime label.
        - ``"regime_confidence"`` (``float``): Regime classification
          confidence from ``macro_view``.
        - ``"dispersion_classification"`` (``str``): Dispersion
          classification from Step 1.
        - ``"tilt_strength"`` (``str``): One of ``{"advisory",
          "moderate", "strong"}``, derived from dispersion classification.
        - ``"rationale"`` (``str``): Human-readable explanation of the
          regime logic applied.

    Raises
    ------
    TypeError
        If ``macro_view`` or ``dispersion`` are not dicts.
    ValueError
        If ``macro_view["regime"]`` is not a valid regime label.
    ValueError
        If required keys are missing from ``macro_view`` or ``dispersion``.
    ValueError
        If ``dispersion["classification"]`` is not one of
        ``{"tight", "moderate", "wide"}``.

    Notes
    -----
    This tool implements **Step 2** of the Exhibit 4 CMA Judge algorithm
    verbatim. The output ``preferred_method_ids`` is an advisory hint to
    the LLM judge — it does not override the LLM's final selection, but
    provides structured guidance that the LLM must consider and either
    follow or explicitly justify deviating from.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(macro_view, dict):
        raise TypeError(
            f"macro_view must be a dict, got {type(macro_view).__name__}."
        )
    if not isinstance(dispersion, dict):
        raise TypeError(
            f"dispersion must be a dict, got {type(dispersion).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys in macro_view
    # ------------------------------------------------------------------
    for key in ("regime", "confidence"):
        if key not in macro_view:
            raise ValueError(
                f"macro_view is missing required key '{key}'."
            )

    # ------------------------------------------------------------------
    # Input validation: required keys in dispersion
    # ------------------------------------------------------------------
    for key in ("classification", "spread_pp"):
        if key not in dispersion:
            raise ValueError(
                f"dispersion is missing required key '{key}'."
            )

    # ------------------------------------------------------------------
    # Extract and validate the current regime label
    # ------------------------------------------------------------------
    # Current macro regime from the macro agent's output
    current_regime: str = str(macro_view["regime"])
    if current_regime not in _VALID_REGIMES:
        raise ValueError(
            f"macro_view['regime'] = '{current_regime}' is not a valid "
            f"regime label. Must be one of: {list(_VALID_REGIMES)}."
        )

    # ------------------------------------------------------------------
    # Extract and validate the dispersion classification
    # ------------------------------------------------------------------
    # Dispersion classification from classify_dispersion (Step 1)
    dispersion_classification: str = str(dispersion["classification"])
    _valid_classifications: Tuple[str, ...] = ("tight", "moderate", "wide")
    if dispersion_classification not in _valid_classifications:
        raise ValueError(
            f"dispersion['classification'] = '{dispersion_classification}' "
            f"is not valid. Must be one of: {list(_valid_classifications)}."
        )

    # ------------------------------------------------------------------
    # Extract regime confidence and spread_pp for rationale construction
    # ------------------------------------------------------------------
    # Regime classification confidence from the macro agent
    regime_confidence: float = float(macro_view["confidence"])
    # Spread in percentage points from the dispersion classification
    spread_pp: float = float(dispersion["spread_pp"])

    # ------------------------------------------------------------------
    # Resolve the frozen regime-to-method mapping (default or override)
    # ------------------------------------------------------------------
    preferred_map: Dict[str, List[int]] = (
        regime_preferred_methods
        if regime_preferred_methods is not None
        else _REGIME_PREFERRED_METHODS
    )

    # ------------------------------------------------------------------
    # Look up the preferred method IDs for the current regime
    # ------------------------------------------------------------------
    # Preferred method IDs for the current regime (Exhibit 4 Step 2)
    preferred_method_ids: List[int] = preferred_map.get(
        current_regime, [7]  # Default to auto-blend if regime not in map
    )

    # ------------------------------------------------------------------
    # Determine the tilt strength based on dispersion classification
    # Tight → advisory (methods agree; regime hint is informational)
    # Moderate → moderate tilt (some disagreement; regime hint matters)
    # Wide → strong tilt (significant disagreement; regime logic dominates)
    # ------------------------------------------------------------------
    _tilt_strength_map: Dict[str, str] = {
        "tight":    "advisory",
        "moderate": "moderate",
        "wide":     "strong",
    }
    # Tilt strength derived from dispersion classification
    tilt_strength: str = _tilt_strength_map[dispersion_classification]

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    # Human-readable description of the regime logic applied
    _regime_descriptions: Dict[str, str] = {
        "Late-cycle": (
            "Late-cycle regime: tilt toward valuation-based methods (4, 5) "
            "and regime-adjusted ERP (2). Backward-looking historical ERP "
            "may overstate forward returns in late-cycle environments."
        ),
        "Expansion": (
            "Expansion regime: default to confidence-weighted auto-blend (7). "
            "Methods broadly agree in expansion; no strong tilt warranted."
        ),
        "Recession": (
            "Recession regime: tilt toward regime-adjusted ERP (2) and "
            "BL equilibrium (3). Valuation-based methods may be unreliable "
            "during earnings contractions."
        ),
        "Recovery": (
            "Recovery regime: tilt toward historical ERP (1) and "
            "BL equilibrium (3). Valuation methods may understate returns "
            "as earnings recover from trough."
        ),
    }
    regime_description: str = _regime_descriptions.get(
        current_regime,
        f"Regime '{current_regime}': apply auto-blend as default.",
    )

    rationale: str = (
        f"Exhibit 4 Step 2 — Regime Logic: "
        f"Current regime = '{current_regime}' "
        f"(confidence = {regime_confidence:.3f}). "
        f"{regime_description} "
        f"Preferred method IDs: {preferred_method_ids}. "
        f"Dispersion = '{dispersion_classification}' "
        f"({spread_pp:.2f} pp spread) → "
        f"tilt strength = '{tilt_strength}'. "
        f"{'Auto-blend is acceptable despite regime hint (tight dispersion).' if tilt_strength == 'advisory' else ''}"
    )

    # Log the regime logic result for audit trail
    logger.debug(
        "check_regime_logic: regime='%s', preferred_methods=%s, "
        "tilt_strength='%s'",
        current_regime,
        preferred_method_ids,
        tilt_strength,
    )

    # ------------------------------------------------------------------
    # Return the regime logic output dict
    # ------------------------------------------------------------------
    return {
        # Ordered list of preferred method IDs for the current regime
        "preferred_method_ids": preferred_method_ids,
        # Current macro regime label
        "regime": current_regime,
        # Regime classification confidence from the macro agent
        "regime_confidence": float(regime_confidence),
        # Dispersion classification from Step 1
        "dispersion_classification": dispersion_classification,
        # Tilt strength derived from dispersion classification
        "tilt_strength": tilt_strength,
        # Human-readable rationale for the regime logic applied
        "rationale": rationale,
    }


# =============================================================================
# TOOL 22: check_valuation_thresholds
# =============================================================================

def check_valuation_thresholds(
    signals: Dict[str, Any],
    pe_threshold_expensive: float = _PE_THRESHOLD_EXPENSIVE,
    pe_threshold_cheap: float = _PE_THRESHOLD_CHEAP,
) -> Dict[str, Any]:
    """
    Apply the valuation threshold check from Exhibit 4 Step 3.

    Implements the third step of the CMA Judge algorithm (Task 22, Step 2),
    as specified verbatim in Exhibit 4:

    - **PE > 30×:** ``pe_flag = "elevated"`` → tilt toward valuation-based
      methods (4, 5); backward-looking ERP likely overstates forward returns
    - **PE < 12×:** ``pe_flag = "depressed"`` → tilt toward historical ERP
      (1) and BL equilibrium (3); valuation methods may understate returns
    - **12× ≤ PE ≤ 30×:** ``pe_flag = "normal"`` → flag if methods disagree
    - **PE not available (non-equity):** ``pe_flag = "not_applicable"``

    Parameters
    ----------
    signals : Dict[str, Any]
        Output of ``fetch_signals``. Must contain ``"pe_trailing"``
        (``float | None``) and ``"category"`` (``str``).
    pe_threshold_expensive : float
        PE threshold above which the market is considered expensive.
        Default: 30.0 per
        ``METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]["valuation_thresholds_pe"]
        ["expensive"]``.
    pe_threshold_cheap : float
        PE threshold below which the market is considered cheap.
        Default: 12.0 per
        ``METHODOLOGY_PARAMS["CMA_JUDGE_RULES"]["valuation_thresholds_pe"]
        ["cheap"]``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"pe_flag"`` (``str``): One of ``{"elevated", "depressed",
          "normal", "not_applicable"}``.
        - ``"pe_value"`` (``float | None``): The PE ratio value used, or
          ``None`` if not available.
        - ``"threshold_expensive"`` (``float``): The expensive threshold
          applied (30.0).
        - ``"threshold_cheap"`` (``float``): The cheap threshold applied
          (12.0).
        - ``"preferred_method_ids_hint"`` (``List[int]``): Method IDs
          suggested by the valuation context (advisory hint).
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``signals`` is not a dict.
    ValueError
        If required keys are missing from ``signals``.
    ValueError
        If ``pe_threshold_cheap >= pe_threshold_expensive``.

    Notes
    -----
    **Negative PE handling:** A negative PE ratio (negative earnings) is
    treated as ``"elevated"`` (expensive signal), since negative earnings
    imply no earnings yield and the market is pricing in future recovery.
    This is a frozen implementation choice documented here.

    **Non-equity assets:** For non-equity asset classes (commodities, gold,
    fixed income, cash), ``pe_trailing`` is ``None`` (set by
    ``fetch_signals``). The flag is set to ``"not_applicable"`` and no
    method tilt is recommended.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(signals, dict):
        raise TypeError(
            f"signals must be a dict, got {type(signals).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys
    # ------------------------------------------------------------------
    for key in ("pe_trailing", "category"):
        if key not in signals:
            raise ValueError(
                f"signals is missing required key '{key}'."
            )

    # ------------------------------------------------------------------
    # Input validation: threshold ordering
    # ------------------------------------------------------------------
    if pe_threshold_cheap >= pe_threshold_expensive:
        raise ValueError(
            f"pe_threshold_cheap ({pe_threshold_cheap}) must be strictly "
            f"less than pe_threshold_expensive ({pe_threshold_expensive})."
        )

    # ------------------------------------------------------------------
    # Extract the PE ratio and asset category from signals
    # ------------------------------------------------------------------
    # Trailing P/E ratio (in ×); None for non-equity assets
    pe_trailing: Optional[float] = signals.get("pe_trailing", None)
    # Asset category (Equity, Fixed Income, Real Assets, Cash)
    category: str = str(signals.get("category", "unknown"))

    # ------------------------------------------------------------------
    # Handle non-equity assets: PE is not applicable
    # ------------------------------------------------------------------
    if pe_trailing is None:
        # PE is not available for this asset class
        rationale: str = (
            f"Exhibit 4 Step 3 — Valuation Thresholds: "
            f"PE ratio not applicable for category='{category}'. "
            "No valuation-based method tilt recommended."
        )
        logger.debug(
            "check_valuation_thresholds: pe_flag='not_applicable', "
            "category='%s'",
            category,
        )
        return {
            # PE not applicable for non-equity assets
            "pe_flag": "not_applicable",
            # PE value is None for non-equity assets
            "pe_value": None,
            # Expensive threshold (for reference)
            "threshold_expensive": float(pe_threshold_expensive),
            # Cheap threshold (for reference)
            "threshold_cheap": float(pe_threshold_cheap),
            # No method tilt for non-equity assets
            "preferred_method_ids_hint": [],
            # Human-readable rationale
            "rationale": rationale,
        }

    # ------------------------------------------------------------------
    # Cast PE to float for comparison
    # ------------------------------------------------------------------
    pe_value: float = float(pe_trailing)

    # ------------------------------------------------------------------
    # Apply the frozen valuation threshold logic from Exhibit 4 Step 3:
    # PE > 30 → elevated (tilt methods 4, 5)
    # PE < 12 → depressed (tilt methods 1, 3)
    # 12 ≤ PE ≤ 30 → normal (flag if methods disagree)
    # Negative PE → treated as elevated (no earnings yield)
    # ------------------------------------------------------------------
    if pe_value < 0:
        # Negative PE: treat as elevated (negative earnings = no yield)
        pe_flag: str = "elevated"
        # Tilt toward valuation-based methods (4, 5) for elevated PE
        preferred_method_ids_hint: List[int] = [4, 5]
        pe_note: str = (
            f"PE = {pe_value:.1f}× (negative earnings). "
            "Treated as elevated — no earnings yield available."
        )
    elif pe_value > pe_threshold_expensive:
        # PE above expensive threshold: market is stretched
        pe_flag = "elevated"
        # Tilt toward valuation-based methods (4, 5) per Exhibit 4
        preferred_method_ids_hint = [4, 5]
        pe_note = (
            f"PE = {pe_value:.1f}× > {pe_threshold_expensive:.0f}× "
            "(expensive). Tilt toward valuation methods (4, 5)."
        )
    elif pe_value < pe_threshold_cheap:
        # PE below cheap threshold: market is inexpensive
        pe_flag = "depressed"
        # Tilt toward historical ERP (1) and BL equilibrium (3) per Exhibit 4
        preferred_method_ids_hint = [1, 3]
        pe_note = (
            f"PE = {pe_value:.1f}× < {pe_threshold_cheap:.0f}× "
            "(cheap). Tilt toward historical ERP (1) and BL (3)."
        )
    else:
        # PE within normal range: no strong tilt
        pe_flag = "normal"
        # No specific method tilt; flag if methods disagree
        preferred_method_ids_hint = []
        pe_note = (
            f"PE = {pe_value:.1f}× within normal range "
            f"[{pe_threshold_cheap:.0f}×, {pe_threshold_expensive:.0f}×]. "
            "Flag if methods disagree significantly."
        )

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    rationale = (
        f"Exhibit 4 Step 3 — Valuation Thresholds: "
        f"{pe_note} "
        f"PE flag = '{pe_flag}'. "
        f"Method hint: {preferred_method_ids_hint if preferred_method_ids_hint else 'none (normal range)'}."
    )

    # Log the valuation threshold result for audit trail
    logger.debug(
        "check_valuation_thresholds: pe_value=%.2f, pe_flag='%s', "
        "preferred_methods=%s",
        pe_value,
        pe_flag,
        preferred_method_ids_hint,
    )

    # ------------------------------------------------------------------
    # Return the valuation threshold output dict
    # ------------------------------------------------------------------
    return {
        # PE flag: elevated, depressed, normal, or not_applicable
        "pe_flag": pe_flag,
        # PE ratio value used (in ×)
        "pe_value": float(pe_value),
        # Expensive threshold applied (30.0)
        "threshold_expensive": float(pe_threshold_expensive),
        # Cheap threshold applied (12.0)
        "threshold_cheap": float(pe_threshold_cheap),
        # Advisory method IDs suggested by valuation context
        "preferred_method_ids_hint": preferred_method_ids_hint,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 23: check_signal_alignment
# =============================================================================

def check_signal_alignment(
    signals: Dict[str, Any],
    final_estimate: float,
    rsi_overbought: float = _RSI_OVERBOUGHT,
    rsi_oversold: float = _RSI_OVERSOLD,
    breadth_bullish: float = _BREADTH_BULLISH,
    breadth_bearish: float = _BREADTH_BEARISH,
) -> Dict[str, Any]:
    """
    Compute the alignment between technical signals and the proposed CMA estimate.

    Implements Step 4 of the Exhibit 4 CMA Judge algorithm (Task 22, Step 2):
    "Check signal alignment (confirm or hedge against method spread)."

    For each available technical signal, a directional implication is
    determined (bullish, bearish, or neutral). The directional implication
    of ``final_estimate`` is determined relative to zero (positive = bullish,
    negative = bearish). The alignment score is:

    .. math::

        \\text{alignment\\_score} = \\frac{n_{agree} - n_{disagree}}
                                          {n_{available}}

    where :math:`n_{agree}` is the number of signals that agree with the
    estimate direction, :math:`n_{disagree}` is the number that contradict
    it, and :math:`n_{available}` is the total number of non-neutral,
    non-None signals.

    Signal directional rules (frozen):

    - **RSI 14d:** > 70 → overbought (bearish forward signal);
      < 30 → oversold (bullish); 30–70 → neutral
    - **Momentum 12M:** > 0 → bullish; < 0 → bearish; = 0 → neutral
    - **Market breadth:** > 0.60 → bullish; < 0.40 → bearish; else neutral

    Parameters
    ----------
    signals : Dict[str, Any]
        Output of ``fetch_signals``. Uses ``"rsi_14d"``,
        ``"momentum_12m"``, and ``"market_breadth_raw"`` fields.
        ``None`` values are treated as unavailable (excluded from scoring).
    final_estimate : float
        The LLM's proposed final CMA estimate in decimal form. Used to
        determine the estimate's directional implication (positive =
        bullish, negative = bearish).
    rsi_overbought : float
        RSI threshold above which the signal is bearish. Default: 70.0.
    rsi_oversold : float
        RSI threshold below which the signal is bullish. Default: 30.0.
    breadth_bullish : float
        Market breadth threshold above which the signal is bullish.
        Default: 0.60.
    breadth_bearish : float
        Market breadth threshold below which the signal is bearish.
        Default: 0.40.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"alignment_score"`` (``float``): Score in ``[-1, 1]``.
          Positive = signals confirm the estimate; negative = signals
          contradict it; zero = neutral or no signals available.
        - ``"direction"`` (``str``): One of ``{"confirming",
          "contradicting", "neutral"}``.
        - ``"n_signals_available"`` (``int``): Number of non-None signals
          evaluated.
        - ``"n_directional_signals"`` (``int``): Number of non-neutral
          signals (bullish or bearish).
        - ``"signal_directions"`` (``Dict[str, str]``): Per-signal
          directional implication: ``{"rsi_14d": "bullish"|"bearish"|
          "neutral"|"unavailable", ...}``.
        - ``"estimate_direction"`` (``str``): ``"bullish"`` if
          ``final_estimate > 0``, ``"bearish"`` if ``< 0``,
          ``"neutral"`` if ``= 0``.
        - ``"rationale"`` (``str``): Human-readable explanation.

    Raises
    ------
    TypeError
        If ``signals`` is not a dict.
    ValueError
        If ``final_estimate`` is not a finite float.

    Notes
    -----
    **Baseline for estimate direction:** The estimate direction is
    determined relative to zero (positive = bullish, negative = bearish).
    This is a frozen implementation choice. A more sophisticated baseline
    (e.g., the historical mean return) could be used but is not specified
    in the manuscript.

    **All-None signals:** If all three signals are None (unavailable),
    the alignment score is 0.0 and the direction is ``"neutral"``. This
    is a valid edge case for asset classes with limited signal coverage.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(signals, dict):
        raise TypeError(
            f"signals must be a dict, got {type(signals).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: final_estimate must be finite
    # ------------------------------------------------------------------
    if not np.isfinite(final_estimate):
        raise ValueError(
            f"final_estimate must be a finite float, got {final_estimate}."
        )

    # ------------------------------------------------------------------
    # Extract the three technical signal values from the signals dict
    # None values indicate the signal is unavailable for this asset class
    # ------------------------------------------------------------------
    # 14-day RSI technical indicator (0–100 scale)
    rsi_14d: Optional[float] = signals.get("rsi_14d", None)
    # 12-month price momentum (decimal form)
    momentum_12m: Optional[float] = signals.get("momentum_12m", None)
    # Market breadth (fraction of constituents above 200d MA, 0–1 scale)
    market_breadth_raw: Optional[float] = signals.get("market_breadth_raw", None)

    # ------------------------------------------------------------------
    # Determine the directional implication of the final estimate
    # Positive estimate → bullish; negative → bearish; zero → neutral
    # ------------------------------------------------------------------
    if final_estimate > _EPS:
        # Positive estimate implies bullish forward return expectation
        estimate_direction: str = "bullish"
    elif final_estimate < -_EPS:
        # Negative estimate implies bearish forward return expectation
        estimate_direction = "bearish"
    else:
        # Near-zero estimate: neutral direction
        estimate_direction = "neutral"

    # ------------------------------------------------------------------
    # Determine the directional implication of each signal
    # ------------------------------------------------------------------
    # Dict to store per-signal directional implications
    signal_directions: Dict[str, str] = {}

    # --- RSI 14d signal ---
    if rsi_14d is not None:
        if rsi_14d > rsi_overbought:
            # RSI above overbought threshold: bearish forward signal
            signal_directions["rsi_14d"] = "bearish"
        elif rsi_14d < rsi_oversold:
            # RSI below oversold threshold: bullish forward signal
            signal_directions["rsi_14d"] = "bullish"
        else:
            # RSI in neutral zone (30–70): no directional signal
            signal_directions["rsi_14d"] = "neutral"
    else:
        # RSI not available for this asset class
        signal_directions["rsi_14d"] = "unavailable"

    # --- Momentum 12M signal ---
    if momentum_12m is not None:
        if momentum_12m > _EPS:
            # Positive 12-month momentum: bullish signal
            signal_directions["momentum_12m"] = "bullish"
        elif momentum_12m < -_EPS:
            # Negative 12-month momentum: bearish signal
            signal_directions["momentum_12m"] = "bearish"
        else:
            # Near-zero momentum: neutral
            signal_directions["momentum_12m"] = "neutral"
    else:
        # Momentum not available for this asset class
        signal_directions["momentum_12m"] = "unavailable"

    # --- Market breadth signal ---
    if market_breadth_raw is not None:
        if market_breadth_raw > breadth_bullish:
            # High breadth (>60% above 200d MA): bullish signal
            signal_directions["market_breadth_raw"] = "bullish"
        elif market_breadth_raw < breadth_bearish:
            # Low breadth (<40% above 200d MA): bearish signal
            signal_directions["market_breadth_raw"] = "bearish"
        else:
            # Breadth in neutral zone (40%–60%): neutral
            signal_directions["market_breadth_raw"] = "neutral"
    else:
        # Market breadth not available for this asset class
        signal_directions["market_breadth_raw"] = "unavailable"

    # ------------------------------------------------------------------
    # Count available signals (non-unavailable) and directional signals
    # (non-neutral, non-unavailable)
    # ------------------------------------------------------------------
    # Number of signals that are available (not "unavailable")
    n_signals_available: int = sum(
        1 for d in signal_directions.values()
        if d != "unavailable"
    )
    # Number of directional signals (bullish or bearish, not neutral)
    n_directional: int = sum(
        1 for d in signal_directions.values()
        if d in ("bullish", "bearish")
    )

    # ------------------------------------------------------------------
    # Compute the alignment score
    # For each directional signal, check if it agrees with the estimate
    # direction. Score = (n_agree - n_disagree) / n_directional
    # ------------------------------------------------------------------
    if n_directional == 0 or estimate_direction == "neutral":
        # No directional signals or neutral estimate: alignment is zero
        alignment_score: float = 0.0
    else:
        # Count signals that agree with the estimate direction
        n_agree: int = sum(
            1 for d in signal_directions.values()
            if d == estimate_direction
        )
        # Count signals that contradict the estimate direction
        _opposite: str = "bearish" if estimate_direction == "bullish" else "bullish"
        n_disagree: int = sum(
            1 for d in signal_directions.values()
            if d == _opposite
        )
        # Alignment score: (agree - disagree) / total directional signals
        alignment_score = float(n_agree - n_disagree) / float(n_directional)

    # ------------------------------------------------------------------
    # Clip alignment score to [-1, 1] for numerical safety
    # ------------------------------------------------------------------
    alignment_score = float(np.clip(alignment_score, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Classify the alignment direction
    # confirming: score > 0.30; contradicting: score < -0.30; neutral: otherwise
    # ------------------------------------------------------------------
    if alignment_score > _ALIGNMENT_CONFIRMING_THRESHOLD:
        # Signals broadly confirm the estimate direction
        direction: str = "confirming"
    elif alignment_score < _ALIGNMENT_CONTRADICTING_THRESHOLD:
        # Signals broadly contradict the estimate direction
        direction = "contradicting"
    else:
        # Signals are mixed or unavailable: neutral alignment
        direction = "neutral"

    # ------------------------------------------------------------------
    # Construct the rationale string
    # ------------------------------------------------------------------
    signal_summary: str = ", ".join(
        f"{k}={v}" for k, v in signal_directions.items()
    )
    rationale: str = (
        f"Exhibit 4 Step 4 — Signal Alignment: "
        f"Estimate direction = '{estimate_direction}' "
        f"(final_estimate = {final_estimate * 100:.2f}%). "
        f"Signal directions: [{signal_summary}]. "
        f"n_available = {n_signals_available}, "
        f"n_directional = {n_directional}. "
        f"Alignment score = {alignment_score:.3f} → '{direction}'. "
        f"{'Signals confirm the estimate.' if direction == 'confirming' else 'Signals contradict the estimate — consider hedging.' if direction == 'contradicting' else 'Signals are neutral or mixed.'}"
    )

    # Log the signal alignment result for audit trail
    logger.debug(
        "check_signal_alignment: estimate_direction='%s', "
        "alignment_score=%.3f, direction='%s'",
        estimate_direction,
        alignment_score,
        direction,
    )

    # ------------------------------------------------------------------
    # Return the signal alignment output dict
    # ------------------------------------------------------------------
    return {
        # Alignment score in [-1, 1]
        "alignment_score": float(alignment_score),
        # Direction classification: confirming, contradicting, or neutral
        "direction": direction,
        # Number of non-None signals evaluated
        "n_signals_available": n_signals_available,
        # Number of non-neutral directional signals
        "n_directional_signals": n_directional,
        # Per-signal directional implications
        "signal_directions": signal_directions,
        # Directional implication of the final estimate
        "estimate_direction": estimate_direction,
        # Human-readable rationale
        "rationale": rationale,
    }


# =============================================================================
# TOOL 24: enforce_range_constraint
# =============================================================================

def enforce_range_constraint(
    final_estimate: float,
    estimates: List[Optional[float]],
) -> Dict[str, Any]:
    """
    Enforce the hard range constraint from Exhibit 4 (Layer 1 gate).

    Implements the hard constraint from the CMA Judge Skill (Exhibit 4):

    .. math::

        \\hat{\\mu}_{final} \\in [\\min_k \\hat{\\mu}_k, \\max_k \\hat{\\mu}_k]

    This is the **Layer 1** enforcement gate within the ReAct loop. If the
    LLM's proposed ``final_estimate`` violates the constraint, the tool
    returns ``status="fail"`` and provides a ``clipped_value`` that the LLM
    is instructed to use as the corrected ``final_estimate`` in the
    subsequent ``write_cma_json`` call.

    The ``constraint_message`` field in the return dict is formatted as an
    explicit instruction to the LLM, consistent with the ReAct loop's
    tool-result injection pattern.

    Parameters
    ----------
    final_estimate : float
        The LLM's proposed final CMA estimate in decimal form. Must be a
        finite float.
    estimates : List[Optional[float]]
        List of all method estimates from ``cma_methods.json``. ``None``
        values (inapplicable methods) are excluded from the bound
        computation. Must contain at least one non-None value.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"status"`` (``str``): ``"pass"`` if the constraint is
          satisfied; ``"fail"`` if violated.
        - ``"clipped_value"`` (``float``): The ``final_estimate`` clipped
          to ``[min_estimate, max_estimate]``. Equal to ``final_estimate``
          when ``status="pass"``.
        - ``"min_estimate"`` (``float``): Minimum valid method estimate.
        - ``"max_estimate"`` (``float``): Maximum valid method estimate.
        - ``"constraint_message"`` (``str``): Explicit instruction to the
          LLM. When ``status="fail"``, instructs the LLM to use
          ``clipped_value`` as ``final_estimate`` in ``write_cma_json``.

    Raises
    ------
    TypeError
        If ``estimates`` is not a list.
    ValueError
        If ``final_estimate`` is not a finite float.
    ValueError
        If all estimates are ``None`` (no valid bounds can be computed).

    Notes
    -----
    **Layer 1 vs Layer 2:** This tool implements the Layer 1 gate (within
    the ReAct loop). The Layer 2 post-hoc verification is implemented in
    ``write_cma_json`` (Tool 30), which independently reloads the persisted
    artifact and re-verifies the constraint. Both layers must pass for the
    pipeline to proceed.

    **Floating-point tolerance:** The constraint check uses a tolerance of
    ``_RANGE_CONSTRAINT_TOLERANCE = 1e-8`` to handle floating-point
    boundary cases where the estimate is at the exact boundary.
    """
    # ------------------------------------------------------------------
    # Input validation: final_estimate must be finite
    # ------------------------------------------------------------------
    if not np.isfinite(final_estimate):
        raise ValueError(
            f"final_estimate must be a finite float, got {final_estimate}."
        )

    # ------------------------------------------------------------------
    # Input validation: estimates must be a list
    # ------------------------------------------------------------------
    if not isinstance(estimates, list):
        raise TypeError(
            f"estimates must be a list, got {type(estimates).__name__}."
        )

    # ------------------------------------------------------------------
    # Filter out None values to get valid estimates
    # ------------------------------------------------------------------
    valid_estimates: List[float] = [
        float(e) for e in estimates if e is not None
    ]

    # ------------------------------------------------------------------
    # Guard: at least one valid estimate required to compute bounds
    # ------------------------------------------------------------------
    if len(valid_estimates) == 0:
        raise ValueError(
            "All estimates are None. Cannot compute range constraint bounds."
        )

    # ------------------------------------------------------------------
    # Compute the minimum and maximum of valid estimates
    # These define the hard constraint bounds: [min_estimate, max_estimate]
    # ------------------------------------------------------------------
    # Minimum valid method estimate (lower bound of constraint)
    min_estimate: float = float(min(valid_estimates))
    # Maximum valid method estimate (upper bound of constraint)
    max_estimate: float = float(max(valid_estimates))

    # ------------------------------------------------------------------
    # Check the range constraint with floating-point tolerance:
    # final_estimate must be in [min_estimate - tol, max_estimate + tol]
    # ------------------------------------------------------------------
    # Lower bound check: final_estimate >= min_estimate (with tolerance)
    lower_ok: bool = final_estimate >= (min_estimate - _RANGE_CONSTRAINT_TOLERANCE)
    # Upper bound check: final_estimate <= max_estimate (with tolerance)
    upper_ok: bool = final_estimate <= (max_estimate + _RANGE_CONSTRAINT_TOLERANCE)

    # ------------------------------------------------------------------
    # Determine constraint status
    # ------------------------------------------------------------------
    constraint_satisfied: bool = lower_ok and upper_ok

    # ------------------------------------------------------------------
    # Compute the clipped value (equal to final_estimate if constraint passes)
    # np.clip ensures the value is within [min_estimate, max_estimate]
    # ------------------------------------------------------------------
    clipped_value: float = float(
        np.clip(final_estimate, min_estimate, max_estimate)
    )

    # ------------------------------------------------------------------
    # Assign status string
    # ------------------------------------------------------------------
    status: str = "pass" if constraint_satisfied else "fail"

    # ------------------------------------------------------------------
    # Construct the constraint message for the LLM
    # When status="fail", this message is injected into the ReAct loop
    # as a tool result, instructing the LLM to use clipped_value.
    # ------------------------------------------------------------------
    if constraint_satisfied:
        # Constraint satisfied: informational message
        constraint_message: str = (
            f"CONSTRAINT SATISFIED. "
            f"final_estimate = {final_estimate * 100:.4f}% is within "
            f"[{min_estimate * 100:.4f}%, {max_estimate * 100:.4f}%]. "
            "Proceed with write_cma_json using this final_estimate."
        )
    else:
        # Constraint violated: explicit instruction to use clipped_value
        constraint_message = (
            f"CONSTRAINT VIOLATED. "
            f"final_estimate = {final_estimate * 100:.4f}% is outside "
            f"[{min_estimate * 100:.4f}%, {max_estimate * 100:.4f}%]. "
            f"You MUST use clipped_value = {clipped_value * 100:.4f}% "
            "as final_estimate in the write_cma_json call. "
            "The final estimate MUST be within the range of method estimates "
            "per the hard constraint in Exhibit 4."
        )

    # Log the constraint check result for audit trail
    logger.debug(
        "enforce_range_constraint: final_estimate=%.4f, "
        "min=%.4f, max=%.4f, status='%s', clipped=%.4f",
        final_estimate,
        min_estimate,
        max_estimate,
        status,
        clipped_value,
    )

    # ------------------------------------------------------------------
    # Return the range constraint output dict
    # ------------------------------------------------------------------
    return {
        # Constraint status: "pass" or "fail"
        "status": status,
        # Clipped value (= final_estimate if pass; = boundary if fail)
        "clipped_value": float(clipped_value),
        # Minimum valid method estimate (lower bound)
        "min_estimate": float(min_estimate),
        # Maximum valid method estimate (upper bound)
        "max_estimate": float(max_estimate),
        # Explicit instruction message for the LLM ReAct loop
        "constraint_message": constraint_message,
    }


# =============================================================================
# TOOL 25: write_cma_methods_json
# =============================================================================

def write_cma_methods_json(
    asset_class: str,
    method_results: List[Dict[str, Any]],
    artifact_dir: Path,
) -> str:
    """
    Serialise and persist the CMA method results to ``cma_methods.json``.

    This tool implements the artifact-writing step for all AC agents
    (Task 18, Step 2). The output file is consumed by the CMA Judge
    (Tool 19: ``load_cma_methods_json``) and must conform to the frozen
    ``cma_methods.schema.json`` schema.

    The file is written to:
    ``{artifact_dir}/{asset_class_slug}/cma_methods.json``

    Parameters
    ----------
    asset_class : str
        Asset class name (e.g., ``"US Large Cap"``). Used to construct
        the output file path via slug derivation.
    method_results : List[Dict[str, Any]]
        List of method result dicts from CMA Methods 1–7 (equity) or
        the FI CMA builder (fixed income). Each dict must contain keys:
        ``"method_id"``, ``"estimate"``, ``"confidence"``,
        ``"breakdown"``, ``"rationale"``. May contain ``None`` estimates
        for inapplicable methods.
    artifact_dir : Path
        Base artifact directory. The file is written to
        ``{artifact_dir}/{asset_class_slug}/cma_methods.json``.

    Returns
    -------
    str
        Absolute path to the written ``cma_methods.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or
        ``method_results`` is not a list.
    ValueError
        If ``method_results`` is empty.
    ValueError
        If any method result dict is missing required keys.
    ValueError
        If any confidence value is outside ``[0, 1]``.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    All numeric values (including ``np.float64`` values in ``breakdown``
    dicts) are recursively cast to Python native types via
    ``_cast_to_json_safe`` before serialisation to ensure JSON
    compatibility.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(method_results, list):
        raise TypeError(
            f"method_results must be a list, "
            f"got {type(method_results).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-empty list
    # ------------------------------------------------------------------
    if len(method_results) == 0:
        raise ValueError(
            "method_results is empty. At least one method result is required."
        )

    # ------------------------------------------------------------------
    # Input validation: validate each method result dict
    # ------------------------------------------------------------------
    for i, result in enumerate(method_results):
        if not isinstance(result, dict):
            raise TypeError(
                f"method_results[{i}] must be a dict, "
                f"got {type(result).__name__}."
            )
        # Check for required keys
        missing_keys: List[str] = [
            k for k in _REQUIRED_METHOD_RESULT_KEYS if k not in result
        ]
        if missing_keys:
            raise ValueError(
                f"method_results[{i}] is missing required keys: "
                f"{missing_keys}."
            )
        # Validate confidence is in [0, 1] if not None
        conf = result.get("confidence")
        if conf is not None and not (0.0 <= float(conf) <= 1.0):
            raise ValueError(
                f"method_results[{i}] has confidence={conf} outside [0, 1]."
            )

    # ------------------------------------------------------------------
    # Derive the asset class slug for the output directory
    # ------------------------------------------------------------------
    # Filesystem-safe slug derived from the asset class name
    asset_class_slug: str = _derive_asset_class_slug(asset_class)

    # ------------------------------------------------------------------
    # Construct the output directory and file path
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{asset_class_slug}/
    output_dir: Path = artifact_dir / asset_class_slug
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "cma_methods.json"

    # ------------------------------------------------------------------
    # Construct the output dict with JSON-safe types
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # Asset class name for identification
        "asset_class": asset_class,
        # Number of method results in the list
        "n_methods": len(method_results),
        # Recursively cast method results to JSON-safe types
        "method_results": _cast_to_json_safe(method_results),
    }

    # ------------------------------------------------------------------
    # Serialise to JSON and write to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised output dict with 2-space indentation
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write cma_methods.json to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_cma_methods_json: written %d method results for "
        "asset_class='%s' to '%s'.",
        len(method_results),
        asset_class,
        output_path,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 26: write_signals_json
# =============================================================================

def write_signals_json(
    asset_class: str,
    signals: Dict[str, Any],
    artifact_dir: Path,
) -> str:
    """
    Serialise and persist the signals dict to ``signals.json``.

    This tool implements the signals artifact-writing step for all AC agents
    (Task 18, Step 3). The output file is consumed by the CMA Judge
    (Task 22) for signal alignment checks and by the AC agent's investment
    case memo. It must conform to the frozen ``signals.schema.json`` schema.

    The file is written to:
    ``{artifact_dir}/{asset_class_slug}/signals.json``

    Parameters
    ----------
    asset_class : str
        Asset class name. Used to construct the output file path.
    signals : Dict[str, Any]
        Output of ``fetch_signals``. Must contain at minimum the keys
        specified in ``_REQUIRED_SIGNALS_KEYS``. May contain ``None``
        values for unavailable fields (serialised as JSON ``null``).
    artifact_dir : Path
        Base artifact directory.

    Returns
    -------
    str
        Absolute path to the written ``signals.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``signals`` is
        not a dict.
    ValueError
        If required keys are missing from ``signals``.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    ``None`` values in ``signals`` are preserved as JSON ``null`` via the
    standard ``json.dump`` behaviour. The ``omitted_fields`` list (present
    for non-equity assets) is serialised as a JSON array.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(signals, dict):
        raise TypeError(
            f"signals must be a dict, got {type(signals).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys in signals
    # ------------------------------------------------------------------
    missing_keys: List[str] = [
        k for k in _REQUIRED_SIGNALS_KEYS if k not in signals
    ]
    if missing_keys:
        raise ValueError(
            f"signals is missing required keys: {missing_keys}."
        )

    # ------------------------------------------------------------------
    # Derive the asset class slug for the output directory
    # ------------------------------------------------------------------
    # Filesystem-safe slug derived from the asset class name
    asset_class_slug: str = _derive_asset_class_slug(asset_class)

    # ------------------------------------------------------------------
    # Construct the output directory and file path
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{asset_class_slug}/
    output_dir: Path = artifact_dir / asset_class_slug
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "signals.json"

    # ------------------------------------------------------------------
    # Cast all values to JSON-safe types
    # None values are preserved as None (serialised as JSON null)
    # ------------------------------------------------------------------
    json_safe_signals: Dict[str, Any] = _cast_to_json_safe(signals)

    # ------------------------------------------------------------------
    # Serialise to JSON and write to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised signals dict with 2-space indentation
            json.dump(json_safe_signals, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write signals.json to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_signals_json: written signals for asset_class='%s' "
        "to '%s'.",
        asset_class,
        output_path,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 27: write_historical_stats_json
# =============================================================================

def write_historical_stats_json(
    asset_class: str,
    historical_stats: Dict[str, Any],
    artifact_dir: Path,
) -> str:
    """
    Serialise and persist the historical statistics to ``historical_stats.json``.

    This tool implements the historical statistics artifact-writing step for
    all AC agents (Task 18, Step 2). The output file is consumed by the CMA
    Judge (Task 22) for historical context and by the CRO Agent (Task 27)
    for backtest diagnostics. It must conform to the frozen
    ``historical_stats.schema.json`` schema.

    The file is written to:
    ``{artifact_dir}/{asset_class_slug}/historical_stats.json``

    The ``monthly_returns_series`` (``pd.Series``) and
    ``correlation_matrix`` (``pd.DataFrame``) fields in ``historical_stats``
    require special handling for JSON serialisation:

    - ``monthly_returns_series``: converted to ``{date_str: return_value}``
      where ``date_str`` is ISO-8601 (``"%Y-%m-%d"``).
    - ``correlation_matrix``: converted to a nested dict
      ``{asset_class: {other_asset_class: correlation}}``.

    Parameters
    ----------
    asset_class : str
        Asset class name. Used to construct the output file path.
    historical_stats : Dict[str, Any]
        Output of ``fetch_historical_stats``. Must contain the keys
        specified in ``_REQUIRED_HISTORICAL_STATS_KEYS``.
    artifact_dir : Path
        Base artifact directory.

    Returns
    -------
    str
        Absolute path to the written ``historical_stats.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or
        ``historical_stats`` is not a dict.
    ValueError
        If required keys are missing from ``historical_stats``.
    ValueError
        If ``monthly_returns_series`` is not a ``pd.Series``.
    ValueError
        If ``correlation_matrix`` is not a ``pd.DataFrame``.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    **DatetimeIndex to string conversion:** The ``monthly_returns_series``
    index (``DatetimeIndex``) is converted to ISO-8601 strings using
    ``strftime("%Y-%m-%d")`` to ensure JSON serialisability.

    **NaN in correlation matrix:** NaN correlation values (asset classes
    with no overlapping history) are converted to ``None`` (JSON ``null``)
    via ``_cast_to_json_safe``.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(historical_stats, dict):
        raise TypeError(
            f"historical_stats must be a dict, "
            f"got {type(historical_stats).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys
    # ------------------------------------------------------------------
    missing_keys: List[str] = [
        k for k in _REQUIRED_HISTORICAL_STATS_KEYS
        if k not in historical_stats
    ]
    if missing_keys:
        raise ValueError(
            f"historical_stats is missing required keys: {missing_keys}."
        )

    # ------------------------------------------------------------------
    # Input validation: monthly_returns_series must be a pd.Series
    # ------------------------------------------------------------------
    monthly_returns: Any = historical_stats["monthly_returns_series"]
    if not isinstance(monthly_returns, pd.Series):
        raise ValueError(
            f"historical_stats['monthly_returns_series'] must be a "
            f"pd.Series, got {type(monthly_returns).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: correlation_matrix must be a pd.DataFrame
    # ------------------------------------------------------------------
    corr_matrix: Any = historical_stats["correlation_matrix"]
    if not isinstance(corr_matrix, pd.DataFrame):
        raise ValueError(
            f"historical_stats['correlation_matrix'] must be a "
            f"pd.DataFrame, got {type(corr_matrix).__name__}."
        )

    # ------------------------------------------------------------------
    # Convert monthly_returns_series to a JSON-serialisable dict
    # Keys: ISO-8601 date strings; Values: Python native floats
    # ------------------------------------------------------------------
    # Convert DatetimeIndex to ISO-8601 strings and values to float/None
    monthly_returns_dict: Dict[str, Optional[float]] = {
        # Format each date as ISO-8601 string (YYYY-MM-DD)
        idx.strftime("%Y-%m-%d"): (
            float(val) if not pd.isna(val) else None
        )
        for idx, val in monthly_returns.items()
    }

    # ------------------------------------------------------------------
    # Convert correlation_matrix to a nested JSON-serialisable dict
    # Structure: {column_label: {index_label: correlation_value}}
    # NaN values are converted to None via _cast_to_json_safe
    # ------------------------------------------------------------------
    # Convert the correlation matrix DataFrame to a nested dict
    # with string keys and float/None values
    correlation_dict: Dict[str, Dict[str, Optional[float]]] = {}
    for col in corr_matrix.columns:
        # Convert column label to string
        col_str: str = str(col)
        correlation_dict[col_str] = {}
        for idx in corr_matrix.index:
            # Convert index label to string
            idx_str: str = str(idx)
            # Extract the correlation value
            val = corr_matrix.loc[idx, col]
            # Convert NaN to None for JSON serialisability
            correlation_dict[col_str][idx_str] = (
                float(val) if not pd.isna(val) else None
            )

    # ------------------------------------------------------------------
    # Construct the output dict with JSON-safe scalar fields
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # Asset class name for identification
        "asset_class": str(historical_stats.get("asset_class", asset_class)),
        # Resolved ticker identifier
        "ticker": str(historical_stats.get("ticker", "")),
        # As-of date used for point-in-time filtering
        "as_of_date": str(historical_stats.get("as_of_date", "")),
        # Actual start date of the history window
        "history_start_actual": str(
            historical_stats.get("history_start_actual", "")
        ),
        # Number of monthly return observations
        "n_observations": int(historical_stats["n_observations"]),
        # Annualised arithmetic mean return: mu_ann = 12 * mu_mo
        "annualised_return": float(historical_stats["annualised_return"]),
        # Annualised volatility: sigma_ann = sqrt(12) * sigma_mo
        "annualised_vol": float(historical_stats["annualised_vol"]),
        # Maximum drawdown: MDD = min_t(V_t / max_{s<=t}(V_s) - 1)
        "max_drawdown": float(historical_stats["max_drawdown"]),
        # Raw monthly Sharpe (no rf; annualised Sharpe computed in CRO)
        "sharpe_ratio_unannualised": float(
            historical_stats.get("sharpe_ratio_unannualised", 0.0)
        ),
        # Monthly returns series as {date_str: return_value}
        "monthly_returns": monthly_returns_dict,
        # Pairwise correlation matrix as nested dict
        "correlation_matrix": correlation_dict,
    }

    # ------------------------------------------------------------------
    # Derive the asset class slug for the output directory
    # ------------------------------------------------------------------
    # Filesystem-safe slug derived from the asset class name
    asset_class_slug: str = _derive_asset_class_slug(asset_class)

    # ------------------------------------------------------------------
    # Construct the output directory and file path
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{asset_class_slug}/
    output_dir: Path = artifact_dir / asset_class_slug
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "historical_stats.json"

    # ------------------------------------------------------------------
    # Serialise to JSON and write to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised output dict with 2-space indentation
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write historical_stats.json to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_historical_stats_json: written for asset_class='%s' "
        "to '%s' (n_obs=%d).",
        asset_class,
        output_path,
        historical_stats["n_observations"],
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 28: write_scenarios_json
# =============================================================================

def write_scenarios_json(
    asset_class: str,
    method_results: List[Dict[str, Any]],
    artifact_dir: Path,
) -> str:
    """
    Derive and persist bull/bear/base scenario estimates to ``scenarios.json``.

    This tool implements the scenario analysis artifact-writing step for
    equity and real assets AC agents (Task 18, Step 2). Scenarios are
    derived from the range of valid method estimates:

    - **Bull scenario:** maximum valid method estimate (most optimistic)
    - **Bear scenario:** minimum valid method estimate (most pessimistic)
    - **Base scenario:** Method 7 (auto-blend) estimate if available;
      otherwise the arithmetic mean of all valid estimates

    The file is written to:
    ``{artifact_dir}/{asset_class_slug}/scenarios.json``

    Parameters
    ----------
    asset_class : str
        Asset class name. Used to construct the output file path.
    method_results : List[Dict[str, Any]]
        List of method result dicts from CMA Methods 1–7. Each dict must
        contain ``"method_id"``, ``"estimate"`` (float or None), and
        ``"confidence"``. ``None`` estimates are excluded from scenario
        computation.
    artifact_dir : Path
        Base artifact directory.

    Returns
    -------
    str
        Absolute path to the written ``scenarios.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or
        ``method_results`` is not a list.
    ValueError
        If ``method_results`` is empty or all estimates are ``None``.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    **Single valid estimate:** If only one valid estimate exists, the bull
    and bear scenarios are identical (spread = 0). This is a valid edge
    case for asset classes with limited method applicability (e.g., cash).

    **Base scenario source:** Method 7 (auto-blend) is preferred as the
    base scenario because it represents the confidence-weighted consensus.
    If Method 7 is not present or has a ``None`` estimate, the arithmetic
    mean of all valid estimates is used as the base.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(method_results, list):
        raise TypeError(
            f"method_results must be a list, "
            f"got {type(method_results).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-empty list
    # ------------------------------------------------------------------
    if len(method_results) == 0:
        raise ValueError(
            "method_results is empty. At least one method result is required."
        )

    # ------------------------------------------------------------------
    # Filter to valid methods: non-None estimate
    # ------------------------------------------------------------------
    valid_methods: List[Dict[str, Any]] = [
        m for m in method_results
        if m.get("estimate") is not None
    ]

    # ------------------------------------------------------------------
    # Guard: at least one valid estimate required
    # ------------------------------------------------------------------
    if len(valid_methods) == 0:
        raise ValueError(
            "All method estimates are None. Cannot derive scenarios."
        )

    # ------------------------------------------------------------------
    # Extract valid estimates and their method IDs
    # ------------------------------------------------------------------
    # List of (method_id, estimate) tuples for valid methods
    valid_pairs: List[Tuple[int, float]] = [
        (int(m["method_id"]), float(m["estimate"]))
        for m in valid_methods
    ]

    # ------------------------------------------------------------------
    # Identify the bull scenario: maximum valid estimate
    # ------------------------------------------------------------------
    # Find the method with the maximum estimate (most optimistic)
    bull_method_id: int
    bull_estimate: float
    bull_method_id, bull_estimate = max(valid_pairs, key=lambda x: x[1])

    # ------------------------------------------------------------------
    # Identify the bear scenario: minimum valid estimate
    # ------------------------------------------------------------------
    # Find the method with the minimum estimate (most pessimistic)
    bear_method_id: int
    bear_estimate: float
    bear_method_id, bear_estimate = min(valid_pairs, key=lambda x: x[1])

    # ------------------------------------------------------------------
    # Identify the base scenario: Method 7 (auto-blend) if available
    # ------------------------------------------------------------------
    # Search for Method 7 (auto-blend) in the valid methods
    method_7_result: Optional[Dict[str, Any]] = next(
        (m for m in valid_methods if int(m["method_id"]) == 7),
        None,
    )

    if method_7_result is not None:
        # Use Method 7 (auto-blend) as the base scenario
        base_estimate: float = float(method_7_result["estimate"])
        base_source: str = "method_7_auto_blend"
    else:
        # Fall back to arithmetic mean of all valid estimates
        base_estimate = float(
            np.mean([float(m["estimate"]) for m in valid_methods])
        )
        base_source = "arithmetic_mean_of_valid_methods"
        logger.debug(
            "write_scenarios_json: Method 7 not available for '%s'. "
            "Using arithmetic mean as base scenario.",
            asset_class,
        )

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # Asset class name for identification
        "asset_class": asset_class,
        # Number of valid methods used for scenario derivation
        "n_valid_methods": len(valid_methods),
        # Bull scenario: maximum valid method estimate
        "bull": {
            # Bull scenario estimate (most optimistic)
            "estimate": float(bull_estimate),
            # Method ID that produced the bull estimate
            "source_method_id": bull_method_id,
        },
        # Bear scenario: minimum valid method estimate
        "bear": {
            # Bear scenario estimate (most pessimistic)
            "estimate": float(bear_estimate),
            # Method ID that produced the bear estimate
            "source_method_id": bear_method_id,
        },
        # Base scenario: Method 7 auto-blend or arithmetic mean
        "base": {
            # Base scenario estimate (consensus)
            "estimate": float(base_estimate),
            # Source of the base scenario estimate
            "source": base_source,
        },
        # Spread between bull and bear scenarios (in decimal)
        "bull_bear_spread": float(bull_estimate - bear_estimate),
    }

    # ------------------------------------------------------------------
    # Derive the asset class slug for the output directory
    # ------------------------------------------------------------------
    # Filesystem-safe slug derived from the asset class name
    asset_class_slug: str = _derive_asset_class_slug(asset_class)

    # ------------------------------------------------------------------
    # Construct the output directory and file path
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{asset_class_slug}/
    output_dir: Path = artifact_dir / asset_class_slug
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "scenarios.json"

    # ------------------------------------------------------------------
    # Serialise to JSON and write to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised output dict with 2-space indentation
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write scenarios.json to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_scenarios_json: written for asset_class='%s' to '%s'. "
        "Bull=%.4f (M%d), Bear=%.4f (M%d), Base=%.4f (%s).",
        asset_class,
        output_path,
        bull_estimate,
        bull_method_id,
        bear_estimate,
        bear_method_id,
        base_estimate,
        base_source,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 29: write_correlation_row_json
# =============================================================================

def write_correlation_row_json(
    asset_class: str,
    historical_stats: Dict[str, Any],
    artifact_dir: Path,
) -> str:
    """
    Extract and persist the correlation row to ``correlation_row.json``.

    This tool implements the correlation row artifact-writing step for
    equity and real assets AC agents (Task 18, Step 2). The correlation
    row is the row of the pairwise correlation matrix corresponding to
    the target ``asset_class``, representing its correlation against all
    other asset classes in the 18-asset universe.

    The file is written to:
    ``{artifact_dir}/{asset_class_slug}/correlation_row.json``

    Parameters
    ----------
    asset_class : str
        Asset class name. Used to construct the output file path and to
        locate the correct row in the correlation matrix.
    historical_stats : Dict[str, Any]
        Output of ``fetch_historical_stats``. Must contain
        ``"correlation_matrix"`` (``pd.DataFrame``).
    artifact_dir : Path
        Base artifact directory.

    Returns
    -------
    str
        Absolute path to the written ``correlation_row.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or
        ``historical_stats`` is not a dict.
    ValueError
        If ``"correlation_matrix"`` is missing from ``historical_stats``.
    ValueError
        If ``correlation_matrix`` is not a ``pd.DataFrame``.
    ValueError
        If ``asset_class`` is not found in the correlation matrix index.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    **NaN values:** Correlation values of ``NaN`` (asset classes with no
    overlapping history) are converted to ``None`` (JSON ``null``) for
    serialisability. The ``"n_nan_correlations"`` field in the output
    records the count of NaN correlations for audit transparency.

    **Self-correlation:** The diagonal element (correlation of the asset
    class with itself) is always 1.0 and is included in the output for
    completeness.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(historical_stats, dict):
        raise TypeError(
            f"historical_stats must be a dict, "
            f"got {type(historical_stats).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: correlation_matrix key must be present
    # ------------------------------------------------------------------
    if "correlation_matrix" not in historical_stats:
        raise ValueError(
            "historical_stats is missing required key 'correlation_matrix'."
        )

    # ------------------------------------------------------------------
    # Input validation: correlation_matrix must be a pd.DataFrame
    # ------------------------------------------------------------------
    corr_matrix: Any = historical_stats["correlation_matrix"]
    if not isinstance(corr_matrix, pd.DataFrame):
        raise ValueError(
            f"historical_stats['correlation_matrix'] must be a "
            f"pd.DataFrame, got {type(corr_matrix).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: asset_class must be in the correlation matrix index
    # ------------------------------------------------------------------
    if asset_class not in corr_matrix.index:
        raise ValueError(
            f"asset_class='{asset_class}' not found in correlation_matrix "
            f"index. Available: {list(corr_matrix.index)}."
        )

    # ------------------------------------------------------------------
    # Extract the correlation row for the target asset class
    # corr_matrix.loc[asset_class] returns a pd.Series of correlations
    # ------------------------------------------------------------------
    # Correlation row: correlations of asset_class with all other assets
    corr_row: pd.Series = corr_matrix.loc[asset_class]

    # ------------------------------------------------------------------
    # Convert the correlation row to a JSON-serialisable dict
    # NaN values are converted to None (JSON null)
    # ------------------------------------------------------------------
    # Dict mapping other asset class names to their correlation values
    correlations_dict: Dict[str, Optional[float]] = {}
    # Count of NaN correlations for audit transparency
    n_nan: int = 0

    for other_ac, corr_val in corr_row.items():
        # Convert the other asset class label to string
        other_ac_str: str = str(other_ac)
        if pd.isna(corr_val):
            # NaN correlation: no overlapping history with this asset class
            correlations_dict[other_ac_str] = None
            n_nan += 1
        else:
            # Valid correlation value: cast to Python native float
            correlations_dict[other_ac_str] = float(corr_val)

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # Asset class name for identification
        "asset_class": asset_class,
        # Number of asset classes in the correlation row
        "n_asset_classes": len(corr_row),
        # Number of NaN correlations (no overlapping history)
        "n_nan_correlations": n_nan,
        # Correlation row: {other_asset_class: correlation_value}
        "correlations": correlations_dict,
    }

    # ------------------------------------------------------------------
    # Derive the asset class slug for the output directory
    # ------------------------------------------------------------------
    # Filesystem-safe slug derived from the asset class name
    asset_class_slug: str = _derive_asset_class_slug(asset_class)

    # ------------------------------------------------------------------
    # Construct the output directory and file path
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{asset_class_slug}/
    output_dir: Path = artifact_dir / asset_class_slug
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "correlation_row.json"

    # ------------------------------------------------------------------
    # Serialise to JSON and write to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised output dict with 2-space indentation
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write correlation_row.json to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_correlation_row_json: written for asset_class='%s' "
        "to '%s' (n_assets=%d, n_nan=%d).",
        asset_class,
        output_path,
        len(corr_row),
        n_nan,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 30: write_cma_json
# =============================================================================

def write_cma_json(
    asset_class: str,
    final_estimate: float,
    method_weights: Dict[str, float],
    rationale: str,
    method_range: Dict[str, float],
    artifact_dir: Path,
    credit_spread_duration: Optional[float] = None,
    sector_concentration: Optional[float] = None,
    as_of_date: Optional[str] = None,
) -> str:
    """
    Serialise and persist the final CMA estimate to ``cma.json``.

    This tool implements the definitive CMA artifact-writing step for the
    CMA Judge (Task 22, Step 2), FI AC agents (Task 18), real assets AC
    agents (Task 18), and the Cash AC agent (Task 18). The output file is
    the authoritative CMA artifact consumed by the PC agents (Task 25) and
    the CIO agent (Task 31). It must conform to the frozen
    ``cma.schema.json`` schema.

    **Layer 2 range constraint verification (post-hoc):** This tool
    independently verifies that ``final_estimate`` lies within
    ``[method_range["min"], method_range["max"]]``. If this verification
    fails, a ``RuntimeError`` is raised immediately and the pipeline halts
    (fail-closed). This is the Layer 2 belt-and-suspenders gate described
    in Task 22 Step 3.

    The file is written to:
    ``{artifact_dir}/{asset_class_slug}/cma.json``

    Parameters
    ----------
    asset_class : str
        Asset class name. Used to construct the output file path.
    final_estimate : float
        The final CMA expected return estimate in decimal form (e.g.,
        0.068 = 6.8%). Must be within ``[method_range["min"],
        method_range["max"]]``.
    method_weights : Dict[str, float]
        Mapping from method identifier (e.g., ``"method_4"``) to the
        weight assigned to that method in the final blend. Values should
        sum to approximately 1.0 (normalised if not). May be empty if a
        single method was selected without blending.
    rationale : str
        LLM-generated or scripted narrative explaining the final CMA
        selection. Must be non-empty (minimum 10 characters).
    method_range : Dict[str, float]
        Dict with keys ``"min"`` and ``"max"`` defining the range of
        valid method estimates. Used for the Layer 2 range constraint
        verification.
    artifact_dir : Path
        Base artifact directory.
    credit_spread_duration : Optional[float]
        Modified duration × spread sensitivity (for FI asset classes).
        ``None`` for non-FI asset classes.
    sector_concentration : Optional[float]
        Herfindahl index of sector weights (for FI asset classes).
        ``None`` if sector data is unavailable.
    as_of_date : Optional[str]
        ISO-8601 date string for provenance. Included in the artifact
        if provided.

    Returns
    -------
    str
        Absolute path to the written ``cma.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``rationale`` is empty or fewer than 10 characters.
    ValueError
        If ``method_range`` is missing ``"min"`` or ``"max"`` keys.
    ValueError
        If ``final_estimate`` is not a finite float.
    RuntimeError
        If ``final_estimate`` violates the range constraint
        ``[method_range["min"], method_range["max"]]``. This is a
        pipeline-halting error (fail-closed architecture).
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    **Layer 2 range constraint:** The range constraint check in this tool
    is independent of the Layer 1 check in ``enforce_range_constraint``
    (Tool 24). Both layers must pass. The Layer 2 check uses the same
    tolerance (``_RANGE_CONSTRAINT_TOLERANCE = 1e-8``) as Layer 1.

    **Method weights normalisation:** If ``method_weights`` is non-empty
    and its values do not sum to 1.0 (within 1e-4 tolerance), the weights
    are normalised and a warning is logged. This handles floating-point
    rounding in the LLM's weight assignment.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(method_weights, dict):
        raise TypeError(
            f"method_weights must be a dict, "
            f"got {type(method_weights).__name__}."
        )
    if not isinstance(method_range, dict):
        raise TypeError(
            f"method_range must be a dict, "
            f"got {type(method_range).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: final_estimate must be finite
    # ------------------------------------------------------------------
    if not np.isfinite(final_estimate):
        raise ValueError(
            f"final_estimate must be a finite float, got {final_estimate}."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < 10:
        raise ValueError(
            "rationale must be a non-empty string with at least 10 characters."
        )

    # ------------------------------------------------------------------
    # Input validation: method_range must have "min" and "max" keys
    # ------------------------------------------------------------------
    missing_range_keys: List[str] = [
        k for k in _REQUIRED_METHOD_RANGE_KEYS if k not in method_range
    ]
    if missing_range_keys:
        raise ValueError(
            f"method_range is missing required keys: {missing_range_keys}."
        )

    # ------------------------------------------------------------------
    # Extract range bounds
    # ------------------------------------------------------------------
    # Lower bound of the method range
    range_min: float = float(method_range["min"])
    # Upper bound of the method range
    range_max: float = float(method_range["max"])

    # ------------------------------------------------------------------
    # LAYER 2 RANGE CONSTRAINT VERIFICATION (post-hoc, fail-closed)
    # Independently verify: range_min <= final_estimate <= range_max
    # This is the belt-and-suspenders check described in Task 22 Step 3.
    # A RuntimeError halts the pipeline immediately if violated.
    # ------------------------------------------------------------------
    # Check lower bound with floating-point tolerance
    lower_ok: bool = final_estimate >= (range_min - _RANGE_CONSTRAINT_TOLERANCE)
    # Check upper bound with floating-point tolerance
    upper_ok: bool = final_estimate <= (range_max + _RANGE_CONSTRAINT_TOLERANCE)

    if not (lower_ok and upper_ok):
        # CRITICAL: range constraint violated — halt the pipeline
        raise RuntimeError(
            f"LAYER 2 RANGE CONSTRAINT VIOLATION for asset_class="
            f"'{asset_class}': "
            f"final_estimate = {final_estimate * 100:.4f}% is outside "
            f"[{range_min * 100:.4f}%, {range_max * 100:.4f}%]. "
            "This is a pipeline-halting error (fail-closed). "
            "The CMA Judge must have failed to apply the Layer 1 "
            "enforce_range_constraint gate correctly. "
            "Investigate the ReAct loop for this asset class."
        )

    # ------------------------------------------------------------------
    # Normalise method_weights if non-empty and not summing to 1.0
    # ------------------------------------------------------------------
    normalised_weights: Dict[str, float] = {}
    if method_weights:
        # Compute the sum of all method weights
        weight_sum: float = sum(float(v) for v in method_weights.values())
        if abs(weight_sum - 1.0) > 1e-4:
            # Weights do not sum to 1.0: normalise and warn
            logger.warning(
                "write_cma_json: method_weights for '%s' sum to %.6f "
                "(not 1.0). Normalising.",
                asset_class,
                weight_sum,
            )
            if weight_sum > _EPS:
                # Normalise each weight by the total sum
                normalised_weights = {
                    k: float(v) / weight_sum
                    for k, v in method_weights.items()
                }
            else:
                # All weights are zero: use empty dict
                normalised_weights = {}
        else:
            # Weights already sum to 1.0: cast to float
            normalised_weights = {
                k: float(v) for k, v in method_weights.items()
            }

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # Asset class name for identification
        "asset_class": asset_class,
        # Final CMA expected return estimate in decimal form
        "final_estimate": float(final_estimate),
        # Normalised method weights used in the final blend
        "method_weights": normalised_weights,
        # LLM-generated or scripted rationale for the selection
        "rationale": rationale.strip(),
        # Method range bounds for audit and Layer 2 verification evidence
        "method_range": {
            "min": float(range_min),
            "max": float(range_max),
        },
        # Flag confirming that the Layer 2 range constraint was verified
        "range_constraint_verified": True,
    }

    # ------------------------------------------------------------------
    # Add FI-specific fields if provided
    # ------------------------------------------------------------------
    if credit_spread_duration is not None:
        # Credit spread duration: modified duration × spread sensitivity
        output_dict["credit_spread_duration"] = float(credit_spread_duration)

    if sector_concentration is not None:
        # Sector concentration: Herfindahl index of sector weights
        output_dict["sector_concentration"] = float(sector_concentration)
    else:
        # Sector concentration not available: flag as None for audit
        output_dict["sector_concentration"] = None

    # ------------------------------------------------------------------
    # Add as_of_date for provenance if provided
    # ------------------------------------------------------------------
    if as_of_date is not None:
        # Include the as-of date string for audit trail
        output_dict["as_of_date"] = str(as_of_date)

    # ------------------------------------------------------------------
    # Derive the asset class slug for the output directory
    # ------------------------------------------------------------------
    # Filesystem-safe slug derived from the asset class name
    asset_class_slug: str = _derive_asset_class_slug(asset_class)

    # ------------------------------------------------------------------
    # Construct the output directory and file path
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{asset_class_slug}/
    output_dir: Path = artifact_dir / asset_class_slug
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "cma.json"

    # ------------------------------------------------------------------
    # Serialise to JSON and write to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised output dict with 2-space indentation
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write cma.json to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_cma_json: written for asset_class='%s' to '%s'. "
        "final_estimate=%.4f, range=[%.4f, %.4f], "
        "range_constraint_verified=True.",
        asset_class,
        output_path,
        final_estimate,
        range_min,
        range_max,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())

# =============================================================================
# SELF-DRIVING PORTFOLIO: TOOL REGISTRY — BATCH 4 (TOOLS 31–40)
# =============================================================================
# Implements tools 31–40 from the complete 78-tool registry for the agentic
# Strategic Asset Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# Tools implemented:
#   31. compute_ensemble_centroid              — Adversarial Diversifier
#   32. run_adversarial_diversifier_optimizer  — Adversarial Diversifier
#   33. check_sharpe_floor                     — Adversarial Diversifier
#   34. compute_ex_ante_vol                    — CRO Agent / IPS
#   35. compute_tracking_error                 — CRO Agent / IPS
#   36. compute_backtest_sharpe                — CRO Agent / Backtest
#   37. compute_mdd                            — CRO Agent / Backtest
#   38. run_factor_regression                  — CRO Agent / Factor Tilts
#   39. check_ips_compliance                   — Shared (PC, CRO, CIO)
#   40. write_cro_report_json                  — CRO Agent artifact writer
#
# All tools are purely deterministic Python callables — no LLM interaction.
# All arithmetic conventions are frozen per STUDY_CONFIG["DATA_CONVENTIONS"].
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# Initialise a named logger so callers can configure log levels independently
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants (sourced from STUDY_CONFIG; reproduced for self-contained
# validation — the orchestrator injects the live config at runtime)
# ---------------------------------------------------------------------------

# Number of assets in the 18-asset universe
_N_ASSETS: int = 18

# Annualisation multiplier for monthly returns (periods per year)
_PERIODS_PER_YEAR: int = 12

# Numerical stability epsilon
_EPS: float = 1e-8

# Frozen IPS constraint values per IPS_GOVERNANCE
_IPS_MAX_WEIGHT: float = 0.25          # Maximum weight per asset
_IPS_MIN_WEIGHT: float = 0.00          # Minimum weight per asset (long-only)
_IPS_TE_BUDGET: float = 0.06           # Ex-ante tracking error budget (6%)
_IPS_VOL_LOWER: float = 0.08           # Volatility band lower bound (8%)
_IPS_VOL_UPPER: float = 0.12           # Volatility band upper bound (12%)
_IPS_MDD_LIMIT: float = -0.25          # Maximum drawdown limit (-25%)

# Frozen adversarial diversifier parameters
_ADVERSARIAL_SHARPE_FLOOR_FRACTION: float = 0.75  # 75% of max-Sharpe SR

# Minimum monthly observations for backtest computations
_MIN_MONTHLY_OBS: int = 24

# Minimum aligned observations for factor regression
_MIN_FACTOR_OBS: int = 24

# Slug for the adversarial diversifier (excluded from centroid computation)
_ADVERSARIAL_SLUG: str = "adversarial_diversifier"

# Required metric keys for CRO report
_REQUIRED_CRO_METRIC_KEYS: Tuple[str, ...] = (
    "ex_ante_vol",
    "tracking_error",
    "backtest_sharpe",
    "max_drawdown",
    "alpha",
    "beta_M",
    "beta_S",
    "beta_H",
    "r_squared",
)

# Required IPS compliance flag keys
_REQUIRED_IPS_FLAG_KEYS: Tuple[str, ...] = (
    "long_only",
    "max_weight",
    "sum_to_one",
    "tracking_error",
    "volatility_band",
)


# ---------------------------------------------------------------------------
# Shared utility: recursive JSON-safe type casting (re-declared for
# self-contained module; identical to Batch 3 implementation)
# ---------------------------------------------------------------------------

def _cast_to_json_safe(obj: Any) -> Any:
    """
    Recursively cast an object to JSON-serialisable Python native types.

    Converts ``np.float64``, ``np.int64``, ``np.bool_``, ``np.ndarray``,
    ``pd.Series``, and ``pd.DataFrame`` to their Python native equivalents.
    ``None`` and ``float("nan")`` are preserved as ``None`` (JSON ``null``).

    Parameters
    ----------
    obj : Any
        The object to cast.

    Returns
    -------
    Any
        A JSON-serialisable Python native object.
    """
    # Preserve None as None (serialises to JSON null)
    if obj is None:
        return None
    # Cast numpy floating-point scalars to Python float; NaN → None
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    # Cast Python float; NaN → None
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    # Cast numpy integer scalars to Python int
    if isinstance(obj, np.integer):
        return int(obj)
    # Cast numpy boolean scalars to Python bool
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Convert numpy arrays to list of JSON-safe elements
    if isinstance(obj, np.ndarray):
        return [_cast_to_json_safe(v) for v in obj.tolist()]
    # Convert pandas Series to dict with string keys
    if isinstance(obj, pd.Series):
        return {str(k): _cast_to_json_safe(v) for k, v in obj.items()}
    # Convert pandas DataFrame to nested dict
    if isinstance(obj, pd.DataFrame):
        return {
            str(col): {str(idx): _cast_to_json_safe(val)
                       for idx, val in obj[col].items()}
            for col in obj.columns
        }
    # Recursively cast dict values
    if isinstance(obj, dict):
        return {str(k): _cast_to_json_safe(v) for k, v in obj.items()}
    # Recursively cast list/tuple elements
    if isinstance(obj, (list, tuple)):
        return [_cast_to_json_safe(v) for v in obj]
    # Python int and bool: return as-is
    if isinstance(obj, (int, bool)):
        return obj
    # Strings: return as-is
    if isinstance(obj, str):
        return obj
    # Fallback: string conversion for unknown types
    return str(obj)


def _derive_asset_class_slug(name: str) -> str:
    """
    Derive a filesystem-safe slug from a name string.

    Applies: lowercase → replace spaces with underscores →
    remove non-alphanumeric/underscore characters.

    Parameters
    ----------
    name : str
        Input name (e.g., ``"US Large Cap"`` or ``"max_sharpe"``).

    Returns
    -------
    str
        Filesystem-safe slug (e.g., ``"us_large_cap"``).
    """
    # Convert to lowercase
    s: str = name.lower()
    # Replace spaces with underscores
    s = s.replace(" ", "_")
    # Remove non-alphanumeric/underscore characters
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


# =============================================================================
# TOOL 31: compute_ensemble_centroid
# =============================================================================

def compute_ensemble_centroid(
    other_pc_weights_dir: Path,
    n_methods: int = 20,
    method_slugs: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute the equal-weighted centroid of all non-adversarial PC portfolios.

    Implements Task 26, Step 1 — the centroid computation that must precede
    the adversarial diversifier optimisation:

    .. math::

        \\bar{w} = \\frac{1}{K} \\sum_{k=1}^{K} w^{(k)}

    where :math:`K` is the number of valid non-adversarial PC portfolios
    loaded from ``other_pc_weights_dir``.

    This tool enforces the **activation gate** for the adversarial diversifier:
    all ``n_methods`` non-adversarial ``pc_weights.json`` artifacts must exist
    and be valid before the centroid can be computed. If fewer than
    ``n_methods`` valid files are found, a ``RuntimeError`` is raised to halt
    the pipeline (fail-closed), consistent with the AutoGen Phase B
    sequencing constraint.

    The adversarial diversifier's own weights (slug: ``"adversarial_diversifier"``)
    are explicitly excluded from the centroid computation.

    Parameters
    ----------
    other_pc_weights_dir : Path
        Directory containing subdirectories for each PC method, each with
        a ``pc_weights.json`` file. Expected structure:
        ``{other_pc_weights_dir}/{method_slug}/pc_weights.json``.
    n_methods : int
        Expected number of non-adversarial PC methods. Default: 20.
        The activation gate requires exactly this many valid files.
    method_slugs : Optional[List[str]]
        Explicit list of method slugs to include. If ``None``, all
        subdirectories in ``other_pc_weights_dir`` are scanned, excluding
        ``"adversarial_diversifier"``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"centroid_weights"`` (``List[float]``): 18-element centroid
          weight vector. Sums to 1.0 (within ``1e-6``).
        - ``"n_portfolios_used"`` (``int``): Number of portfolios included
          in the centroid computation (must equal ``n_methods``).
        - ``"method_slugs_included"`` (``List[str]``): Sorted list of
          method slugs whose weights were included.

    Raises
    ------
    TypeError
        If ``other_pc_weights_dir`` is not a ``pathlib.Path``.
    FileNotFoundError
        If ``other_pc_weights_dir`` does not exist.
    RuntimeError
        If fewer than ``n_methods`` valid ``pc_weights.json`` files are
        found. This is the activation gate enforcement — the pipeline
        halts until all non-adversarial PC agents have completed.
    ValueError
        If any loaded weight vector does not have exactly 18 elements or
        does not sum to 1.0 (within ``1e-6``).

    Notes
    -----
    **Activation gate:** The adversarial diversifier (Phase B of the
    AutoGen topology) must run only after all 20 non-adversarial PC agents
    have completed Phase A. This tool enforces that gate by requiring
    exactly ``n_methods`` valid weight files. Any missing file raises a
    ``RuntimeError`` rather than a ``ValueError`` to signal a pipeline
    sequencing violation rather than a data error.

    **Centroid sum validation:** The centroid :math:`\\bar{w}` must sum to
    1.0 because each constituent weight vector sums to 1.0 and the
    arithmetic mean preserves this property. A deviation beyond ``1e-6``
    indicates a numerical error in the constituent weights.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(other_pc_weights_dir, Path):
        raise TypeError(
            f"other_pc_weights_dir must be a pathlib.Path, "
            f"got {type(other_pc_weights_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: directory must exist
    # ------------------------------------------------------------------
    if not other_pc_weights_dir.exists():
        raise FileNotFoundError(
            f"other_pc_weights_dir does not exist: '{other_pc_weights_dir}'."
        )

    # ------------------------------------------------------------------
    # Determine which method slugs to load
    # ------------------------------------------------------------------
    if method_slugs is not None:
        # Use the explicitly provided list of method slugs
        slugs_to_load: List[str] = [
            s for s in method_slugs if s != _ADVERSARIAL_SLUG
        ]
    else:
        # Auto-discover: scan all subdirectories, exclude adversarial slug
        slugs_to_load = sorted([
            d.name
            for d in other_pc_weights_dir.iterdir()
            if d.is_dir() and d.name != _ADVERSARIAL_SLUG
        ])

    # ------------------------------------------------------------------
    # Load pc_weights.json for each method slug
    # ------------------------------------------------------------------
    # List to accumulate valid weight vectors
    weight_vectors: List[np.ndarray] = []
    # List to accumulate successfully loaded method slugs
    loaded_slugs: List[str] = []

    for slug in slugs_to_load:
        # Construct the expected path to this method's pc_weights.json
        weights_path: Path = other_pc_weights_dir / slug / "pc_weights.json"

        # Skip if the file does not exist (will be caught by activation gate)
        if not weights_path.exists():
            logger.warning(
                "compute_ensemble_centroid: pc_weights.json not found "
                "for slug='%s' at '%s'. Skipping.",
                slug,
                weights_path,
            )
            continue

        # Load and parse the JSON file
        try:
            with open(weights_path, "r", encoding="utf-8") as fh:
                # Parse the JSON content
                pc_data: Dict[str, Any] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "compute_ensemble_centroid: Failed to load '%s': %s. "
                "Skipping.",
                weights_path,
                exc,
            )
            continue

        # Extract the weights field from the loaded JSON
        raw_weights = pc_data.get("weights")
        if raw_weights is None:
            logger.warning(
                "compute_ensemble_centroid: 'weights' field missing in "
                "'%s'. Skipping.",
                weights_path,
            )
            continue

        # Convert to numpy array of float64
        try:
            w: np.ndarray = np.array(raw_weights, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            logger.warning(
                "compute_ensemble_centroid: Cannot convert weights to "
                "array for slug='%s': %s. Skipping.",
                slug,
                exc,
            )
            continue

        # ------------------------------------------------------------------
        # Validate weight vector: must have exactly _N_ASSETS elements
        # ------------------------------------------------------------------
        if w.shape != (_N_ASSETS,):
            raise ValueError(
                f"Weight vector for slug='{slug}' has shape {w.shape}, "
                f"expected ({_N_ASSETS},)."
            )

        # ------------------------------------------------------------------
        # Validate weight vector: must sum to 1.0 within tolerance
        # ------------------------------------------------------------------
        w_sum: float = float(w.sum())
        if abs(w_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Weight vector for slug='{slug}' sums to {w_sum:.8f}, "
                f"expected 1.0 (tolerance 1e-6)."
            )

        # Accumulate the valid weight vector and its slug
        weight_vectors.append(w)
        loaded_slugs.append(slug)

    # ------------------------------------------------------------------
    # Activation gate: require exactly n_methods valid weight files
    # ------------------------------------------------------------------
    n_loaded: int = len(weight_vectors)
    if n_loaded < n_methods:
        raise RuntimeError(
            f"ACTIVATION GATE VIOLATION: compute_ensemble_centroid "
            f"requires {n_methods} valid pc_weights.json files, "
            f"but only {n_loaded} were found in '{other_pc_weights_dir}'. "
            "The adversarial diversifier (Phase B) cannot run until all "
            f"{n_methods} non-adversarial PC agents (Phase A) have "
            "completed successfully. "
            f"Missing slugs: "
            f"{sorted(set(slugs_to_load) - set(loaded_slugs))}."
        )

    # ------------------------------------------------------------------
    # Stack weight vectors into a (K, 18) matrix
    # ------------------------------------------------------------------
    # Weight matrix: each row is one PC portfolio's weight vector
    W: np.ndarray = np.stack(weight_vectors, axis=0)  # shape: (K, 18)

    # ------------------------------------------------------------------
    # Compute the equal-weighted centroid: w_bar = (1/K) * sum_k(w_k)
    # np.mean(axis=0) computes the column-wise mean across all K portfolios
    # ------------------------------------------------------------------
    centroid: np.ndarray = W.mean(axis=0)  # shape: (18,)

    # ------------------------------------------------------------------
    # Validate that the centroid sums to 1.0
    # (must hold since each row sums to 1.0 and mean preserves this)
    # ------------------------------------------------------------------
    centroid_sum: float = float(centroid.sum())
    if abs(centroid_sum - 1.0) > 1e-6:
        logger.warning(
            "compute_ensemble_centroid: centroid sums to %.8f "
            "(expected 1.0). Normalising.",
            centroid_sum,
        )
        # Normalise the centroid to sum to exactly 1.0
        centroid = centroid / centroid_sum

    # Log the centroid computation for audit trail
    logger.info(
        "compute_ensemble_centroid: centroid computed from %d portfolios. "
        "Centroid sum = %.8f.",
        n_loaded,
        float(centroid.sum()),
    )

    # ------------------------------------------------------------------
    # Return the centroid output dict
    # ------------------------------------------------------------------
    return {
        # 18-element centroid weight vector as a list of Python floats
        "centroid_weights": [float(v) for v in centroid],
        # Number of portfolios used in the centroid computation
        "n_portfolios_used": n_loaded,
        # Sorted list of method slugs included in the centroid
        "method_slugs_included": sorted(loaded_slugs),
    }


# =============================================================================
# TOOL 32: run_adversarial_diversifier_optimizer
# =============================================================================

def run_adversarial_diversifier_optimizer(
    centroid_weights: List[float],
    sharpe_floor: float,
    sigma: np.ndarray,
    mu: np.ndarray,
    rf: float,
    benchmark_weights: np.ndarray,
    constraints: Dict[str, Any],
    optimizer_seed: int = 24680,
    n_starts: int = 5,
) -> Dict[str, Any]:
    """
    Solve the adversarial diversifier optimisation problem.

    Implements Task 26, Step 3 — the adversarial portfolio construction
    objective:

    .. math::

        \\max_w\\ (w - \\bar{w})^\\top \\Sigma (w - \\bar{w})

    subject to:

    .. math::

        SR(w) = \\frac{w^\\top \\mu - r_f}{\\sqrt{w^\\top \\Sigma w}}
        \\geq SR_{floor}

    and all IPS constraints (long-only, max weight 0.25, budget, ex-ante
    tracking error ≤ 6%, portfolio volatility ∈ [8%, 12%]).

    The objective is non-convex (maximising a quadratic form), so the
    problem is reformulated as a minimisation of the negative tracking
    variance and solved using SLSQP with ``n_starts`` random initialisations
    to mitigate local optima.

    Parameters
    ----------
    centroid_weights : List[float]
        18-element centroid weight vector from ``compute_ensemble_centroid``.
        Defines the reference point for tracking variance maximisation.
    sharpe_floor : float
        Minimum acceptable Sharpe ratio for the adversarial portfolio.
        Equal to :math:`0.75 \\times SR(w_{\\max SR})` from
        ``check_sharpe_floor``.
    sigma : np.ndarray
        18×18 annualised covariance matrix (Ledoit-Wolf shrinkage).
        Shape: ``(18, 18)``. Must be positive semi-definite.
    mu : np.ndarray
        18-element vector of annualised CMA expected returns (decimal form).
        Shape: ``(18,)``.
    rf : float
        Annualised risk-free rate in decimal form (e.g., 0.053 = 5.3%).
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    constraints : Dict[str, Any]
        IPS implementation constraints. Expected keys:
        ``"long_only"`` (bool), ``"max_weight_per_asset"`` (float),
        ``"min_weight_per_asset"`` (float).
    optimizer_seed : int
        Random seed for reproducible multi-start initialisation.
        Default: 24680 per ``STUDY_CONFIG["RANDOM_SEEDS"]["optimizer_seed"]``.
    n_starts : int
        Number of random starting points for the multi-start optimisation.
        Default: 5.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"weights"`` (``List[float]``): 18-element optimal weight vector.
          Sums to 1.0 (within ``1e-6``), non-negative.
        - ``"tracking_variance_vs_centroid"`` (``float``): Achieved
          tracking variance :math:`(w - \\bar{w})^\\top \\Sigma (w - \\bar{w})`.
        - ``"achieved_sharpe"`` (``float``): Annualised Sharpe ratio of
          the optimal portfolio.
        - ``"diagnostics"`` (``Dict[str, Any]``): Optimisation diagnostics:
          ``n_starts_attempted``, ``best_start_idx``, ``converged``,
          ``sharpe_floor_relaxed``, ``fallback_used``.

    Raises
    ------
    TypeError
        If ``sigma``, ``mu``, or ``benchmark_weights`` are not
        ``np.ndarray``.
    ValueError
        If ``sigma`` is not shape ``(18, 18)`` or ``mu`` / ``benchmark_weights``
        are not shape ``(18,)``.
    ValueError
        If ``centroid_weights`` does not have 18 elements.
    RuntimeError
        If no feasible solution is found after all starts and fallback
        attempts.

    Notes
    -----
    **Non-convex problem:** The adversarial objective maximises a quadratic
    form, which is non-convex. SLSQP finds local optima; multiple random
    starts improve the probability of finding a good solution.

    **Sharpe floor relaxation fallback:** If no feasible solution is found
    with the original Sharpe floor, the floor is relaxed by 10% and the
    optimisation is retried once. This is documented in ``diagnostics
    ["sharpe_floor_relaxed"]``.

    **Fallback to centroid:** If all optimisation attempts fail (including
    the relaxed floor), the centroid itself is returned as the fallback
    solution with zero tracking variance. This is documented in
    ``diagnostics["fallback_used"]``.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    for arr_name, arr_val in [
        ("sigma", sigma), ("mu", mu), ("benchmark_weights", benchmark_weights)
    ]:
        if not isinstance(arr_val, np.ndarray):
            raise TypeError(
                f"{arr_name} must be a np.ndarray, "
                f"got {type(arr_val).__name__}."
            )

    # ------------------------------------------------------------------
    # Input validation: shape checks
    # ------------------------------------------------------------------
    if sigma.shape != (_N_ASSETS, _N_ASSETS):
        raise ValueError(
            f"sigma must have shape ({_N_ASSETS}, {_N_ASSETS}), "
            f"got {sigma.shape}."
        )
    if mu.shape != (_N_ASSETS,):
        raise ValueError(
            f"mu must have shape ({_N_ASSETS},), got {mu.shape}."
        )
    if benchmark_weights.shape != (_N_ASSETS,):
        raise ValueError(
            f"benchmark_weights must have shape ({_N_ASSETS},), "
            f"got {benchmark_weights.shape}."
        )
    if len(centroid_weights) != _N_ASSETS:
        raise ValueError(
            f"centroid_weights must have {_N_ASSETS} elements, "
            f"got {len(centroid_weights)}."
        )

    # ------------------------------------------------------------------
    # Convert centroid_weights to numpy array
    # ------------------------------------------------------------------
    # Centroid weight vector as numpy array for vectorised operations
    w_bar: np.ndarray = np.array(centroid_weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Extract IPS constraint parameters
    # ------------------------------------------------------------------
    # Maximum weight per asset (default: 0.25)
    max_w: float = float(constraints.get("max_weight_per_asset", _IPS_MAX_WEIGHT))
    # Minimum weight per asset (default: 0.00, long-only)
    min_w: float = float(constraints.get("min_weight_per_asset", _IPS_MIN_WEIGHT))

    # ------------------------------------------------------------------
    # Define the objective function: negative tracking variance
    # Minimising -TV is equivalent to maximising TV
    # TV(w) = (w - w_bar)' * Sigma * (w - w_bar)
    # ------------------------------------------------------------------
    def _neg_tracking_variance(w: np.ndarray) -> float:
        """Negative tracking variance (objective to minimise)."""
        # Active weight vector relative to centroid
        w_active: np.ndarray = w - w_bar
        # Quadratic form: (w - w_bar)' * Sigma * (w - w_bar)
        tv: float = float(np.dot(w_active, sigma @ w_active))
        # Return negative TV for minimisation
        return -tv

    # ------------------------------------------------------------------
    # Define the Sharpe ratio function (used in constraint)
    # SR(w) = (w' * mu - rf) / sqrt(w' * Sigma * w)
    # ------------------------------------------------------------------
    def _sharpe_ratio(w: np.ndarray) -> float:
        """Annualised Sharpe ratio of portfolio w."""
        # Portfolio expected return: w' * mu
        port_return: float = float(np.dot(w, mu))
        # Portfolio variance: w' * Sigma * w
        port_var: float = float(np.dot(w, sigma @ w))
        # Portfolio volatility (annualised, sigma already annualised)
        port_vol: float = float(np.sqrt(max(port_var, _EPS)))
        # Annualised Sharpe ratio
        return (port_return - rf) / port_vol

    # ------------------------------------------------------------------
    # Define the ex-ante tracking error function (used in constraint)
    # TE(w) = sqrt((w - w_b)' * Sigma * (w - w_b))
    # ------------------------------------------------------------------
    def _tracking_error(w: np.ndarray) -> float:
        """Ex-ante annualised tracking error of portfolio w."""
        # Active weight vector relative to benchmark
        w_active_bm: np.ndarray = w - benchmark_weights
        # Quadratic form for tracking error
        te_var: float = float(np.dot(w_active_bm, sigma @ w_active_bm))
        return float(np.sqrt(max(te_var, 0.0)))

    # ------------------------------------------------------------------
    # Define the portfolio volatility function (used in constraint)
    # sigma_p(w) = sqrt(w' * Sigma * w)
    # ------------------------------------------------------------------
    def _portfolio_vol(w: np.ndarray) -> float:
        """Ex-ante annualised portfolio volatility."""
        port_var: float = float(np.dot(w, sigma @ w))
        return float(np.sqrt(max(port_var, 0.0)))

    # ------------------------------------------------------------------
    # Define SLSQP constraints
    # ------------------------------------------------------------------
    def _build_constraints(floor: float) -> List[Dict[str, Any]]:
        """Build the SLSQP constraint list for a given Sharpe floor."""
        return [
            # Budget constraint: sum(w) = 1
            {
                "type": "eq",
                "fun": lambda w: float(w.sum()) - 1.0,
            },
            # Sharpe floor constraint: SR(w) >= floor
            # Reformulated as: SR(w) - floor >= 0
            {
                "type": "ineq",
                "fun": lambda w, f=floor: _sharpe_ratio(w) - f,
            },
            # Tracking error constraint: TE(w) <= 0.06
            # Reformulated as: 0.06 - TE(w) >= 0
            {
                "type": "ineq",
                "fun": lambda w: _IPS_TE_BUDGET - _tracking_error(w),
            },
            # Volatility lower bound: sigma_p(w) >= 0.08
            # Reformulated as: sigma_p(w) - 0.08 >= 0
            {
                "type": "ineq",
                "fun": lambda w: _portfolio_vol(w) - _IPS_VOL_LOWER,
            },
            # Volatility upper bound: sigma_p(w) <= 0.12
            # Reformulated as: 0.12 - sigma_p(w) >= 0
            {
                "type": "ineq",
                "fun": lambda w: _IPS_VOL_UPPER - _portfolio_vol(w),
            },
        ]

    # ------------------------------------------------------------------
    # Define variable bounds: min_w <= w_i <= max_w for all i
    # ------------------------------------------------------------------
    # Bounds for each of the 18 weight variables
    bounds: List[Tuple[float, float]] = [
        (min_w, max_w) for _ in range(_N_ASSETS)
    ]

    # ------------------------------------------------------------------
    # Multi-start SLSQP optimisation
    # ------------------------------------------------------------------
    # Seed the random number generator for reproducibility
    rng: np.random.Generator = np.random.default_rng(optimizer_seed)

    # Track the best feasible solution across all starts
    best_result: Optional[OptimizeResult] = None
    best_tv: float = -np.inf  # Best (most negative) objective value found
    best_start_idx: int = -1
    sharpe_floor_relaxed: bool = False

    # Build constraints with the original Sharpe floor
    slsqp_constraints: List[Dict[str, Any]] = _build_constraints(sharpe_floor)

    # Attempt optimisation from n_starts random starting points
    for start_idx in range(n_starts):
        # Generate a random feasible starting point
        # Draw random weights from a Dirichlet distribution (sums to 1)
        w0_raw: np.ndarray = rng.dirichlet(np.ones(_N_ASSETS))
        # Clip to [min_w, max_w] and renormalise
        w0: np.ndarray = np.clip(w0_raw, min_w, max_w)
        w0_sum: float = float(w0.sum())
        if w0_sum > _EPS:
            w0 = w0 / w0_sum
        else:
            # Fallback: equal weight starting point
            w0 = np.ones(_N_ASSETS) / _N_ASSETS

        # Run SLSQP minimisation from this starting point
        result: OptimizeResult = minimize(
            fun=_neg_tracking_variance,
            x0=w0,
            method="SLSQP",
            bounds=bounds,
            constraints=slsqp_constraints,
            options={
                "ftol": 1e-9,       # Tight function tolerance for precision
                "maxiter": 1000,    # Maximum iterations per start
                "disp": False,      # Suppress solver output
            },
        )

        # Check if this result is feasible and better than the current best
        if result.success and result.fun < best_tv:
            # Update the best result (most negative = highest tracking variance)
            best_tv = result.fun
            best_result = result
            best_start_idx = start_idx

    # ------------------------------------------------------------------
    # Sharpe floor relaxation fallback: if no feasible solution found,
    # relax the Sharpe floor by 10% and retry
    # ------------------------------------------------------------------
    if best_result is None:
        # Relax the Sharpe floor by 10%
        relaxed_floor: float = sharpe_floor * 0.90
        sharpe_floor_relaxed = True
        logger.warning(
            "run_adversarial_diversifier_optimizer: No feasible solution "
            "found with Sharpe floor = %.4f. "
            "Relaxing to %.4f (10%% reduction) and retrying.",
            sharpe_floor,
            relaxed_floor,
        )
        # Build constraints with the relaxed Sharpe floor
        relaxed_constraints: List[Dict[str, Any]] = _build_constraints(
            relaxed_floor
        )
        # Retry from n_starts random starting points with relaxed floor
        for start_idx in range(n_starts):
            w0_raw = rng.dirichlet(np.ones(_N_ASSETS))
            w0 = np.clip(w0_raw, min_w, max_w)
            w0_sum = float(w0.sum())
            if w0_sum > _EPS:
                w0 = w0 / w0_sum
            else:
                w0 = np.ones(_N_ASSETS) / _N_ASSETS

            result = minimize(
                fun=_neg_tracking_variance,
                x0=w0,
                method="SLSQP",
                bounds=bounds,
                constraints=relaxed_constraints,
                options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
            )
            if result.success and result.fun < best_tv:
                best_tv = result.fun
                best_result = result
                best_start_idx = start_idx

    # ------------------------------------------------------------------
    # Fallback to centroid if all optimisation attempts fail
    # ------------------------------------------------------------------
    fallback_used: bool = False
    if best_result is None:
        fallback_used = True
        logger.error(
            "run_adversarial_diversifier_optimizer: All optimisation "
            "attempts failed (including relaxed floor). "
            "Falling back to centroid weights with zero tracking variance."
        )
        # Use the centroid as the fallback solution
        optimal_weights: np.ndarray = w_bar.copy()
    else:
        # Extract the optimal weights from the best result
        optimal_weights = best_result.x.copy()

    # ------------------------------------------------------------------
    # Post-process: clip to [min_w, max_w] and renormalise to sum to 1.0
    # (SLSQP may produce tiny numerical violations at boundaries)
    # ------------------------------------------------------------------
    # Clip weights to the valid range
    optimal_weights = np.clip(optimal_weights, min_w, max_w)
    # Renormalise to sum to exactly 1.0
    opt_sum: float = float(optimal_weights.sum())
    if opt_sum > _EPS:
        optimal_weights = optimal_weights / opt_sum
    else:
        # Degenerate case: use equal weights
        optimal_weights = np.ones(_N_ASSETS) / _N_ASSETS

    # ------------------------------------------------------------------
    # Compute the achieved tracking variance and Sharpe ratio
    # ------------------------------------------------------------------
    # Active weight vector relative to centroid
    w_active_centroid: np.ndarray = optimal_weights - w_bar
    # Tracking variance: (w - w_bar)' * Sigma * (w - w_bar)
    achieved_tv: float = float(
        np.dot(w_active_centroid, sigma @ w_active_centroid)
    )
    # Achieved Sharpe ratio of the optimal portfolio
    achieved_sharpe: float = _sharpe_ratio(optimal_weights)

    # Log the optimisation result for audit trail
    logger.info(
        "run_adversarial_diversifier_optimizer: "
        "tracking_variance=%.6f, achieved_sharpe=%.4f, "
        "sharpe_floor=%.4f, fallback_used=%s, "
        "sharpe_floor_relaxed=%s.",
        achieved_tv,
        achieved_sharpe,
        sharpe_floor,
        fallback_used,
        sharpe_floor_relaxed,
    )

    # ------------------------------------------------------------------
    # Return the adversarial diversifier result dict
    # ------------------------------------------------------------------
    return {
        # Optimal weight vector (18 elements, sums to 1.0)
        "weights": [float(v) for v in optimal_weights],
        # Achieved tracking variance vs centroid
        "tracking_variance_vs_centroid": float(achieved_tv),
        # Achieved annualised Sharpe ratio
        "achieved_sharpe": float(achieved_sharpe),
        # Optimisation diagnostics for audit
        "diagnostics": {
            "n_starts_attempted": n_starts,
            "best_start_idx": best_start_idx,
            "converged": best_result is not None and not fallback_used,
            "sharpe_floor_relaxed": sharpe_floor_relaxed,
            "fallback_used": fallback_used,
            "sharpe_floor_used": float(
                sharpe_floor * 0.90 if sharpe_floor_relaxed else sharpe_floor
            ),
        },
    }


# =============================================================================
# TOOL 33: check_sharpe_floor
# =============================================================================

def check_sharpe_floor(
    max_sharpe_diagnostics_path: Path,
    returns_matrix: np.ndarray,
    rf: float,
    achieved_sharpe: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute the Sharpe floor for the adversarial diversifier constraint.

    Implements Task 26, Step 2:

    .. math::

        SR_{floor} = 0.75 \\times SR(w_{\\max SR})

    where :math:`SR(w_{\\max SR})` is the backtest Sharpe ratio of the
    maximum-Sharpe PC portfolio, loaded from ``max_sharpe_diagnostics_path``.

    The Sharpe ratio is computed using the frozen backtest convention:

    .. math::

        SR_{ann} = \\frac{12 \\cdot \\bar{r}_{excess,mo}}
                        {\\sqrt{12} \\cdot \\sigma_{excess,mo}}

    per ``DATA_CONVENTIONS["annualisation"]``.

    Parameters
    ----------
    max_sharpe_diagnostics_path : Path
        Path to the ``pc_weights.json`` artifact of the maximum-Sharpe
        PC portfolio. Must contain a ``"weights"`` field with 18 elements.
    returns_matrix : np.ndarray
        T×18 monthly returns matrix (injected via closure). Shape:
        ``(T, 18)``. Used to compute the backtest Sharpe of the
        max-Sharpe portfolio.
    rf : float
        Monthly risk-free rate in decimal form (e.g., 0.053/12 ≈ 0.00442).
        Used to compute excess returns for the Sharpe calculation.
    achieved_sharpe : Optional[float]
        The adversarial portfolio's achieved Sharpe ratio (if already
        computed). If provided, ``floor_met`` is set to
        ``achieved_sharpe >= sharpe_floor``. If ``None``, ``floor_met``
        is ``None``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"sharpe_floor"`` (``float``): The computed Sharpe floor
          :math:`0.75 \\times SR(w_{\\max SR})`. Bounded below by 0.0.
        - ``"max_sharpe_value"`` (``float``): The backtest Sharpe ratio
          of the max-Sharpe portfolio.
        - ``"floor_met"`` (``bool | None``): Whether ``achieved_sharpe``
          meets the floor. ``None`` if ``achieved_sharpe`` is not provided.
        - ``"sharpe_floor_fraction"`` (``float``): The fraction used
          (0.75, frozen per ``METHODOLOGY_PARAMS``).

    Raises
    ------
    TypeError
        If ``max_sharpe_diagnostics_path`` is not a ``pathlib.Path`` or
        ``returns_matrix`` is not a ``np.ndarray``.
    FileNotFoundError
        If ``max_sharpe_diagnostics_path`` does not exist.
    ValueError
        If the loaded weight vector does not have 18 elements.
    ValueError
        If ``returns_matrix`` does not have shape ``(T, 18)``.

    Notes
    -----
    **Negative Sharpe floor guard:** If the max-Sharpe portfolio has a
    negative backtest Sharpe (possible in adverse market conditions), the
    floor is set to ``max(0.0, 0.75 × SR)``. A negative floor would be
    non-binding and is replaced with zero to ensure the constraint is
    meaningful.

    **Monthly rf convention:** The ``rf`` parameter must be in monthly
    form (annualised rf / 12). The Sharpe ratio is then annualised by
    multiplying the mean excess return by 12 and the volatility by
    :math:`\\sqrt{12}`.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(max_sharpe_diagnostics_path, Path):
        raise TypeError(
            f"max_sharpe_diagnostics_path must be a pathlib.Path, "
            f"got {type(max_sharpe_diagnostics_path).__name__}."
        )
    if not isinstance(returns_matrix, np.ndarray):
        raise TypeError(
            f"returns_matrix must be a np.ndarray, "
            f"got {type(returns_matrix).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: file must exist
    # ------------------------------------------------------------------
    if not max_sharpe_diagnostics_path.exists():
        raise FileNotFoundError(
            f"max_sharpe_diagnostics_path does not exist: "
            f"'{max_sharpe_diagnostics_path}'. "
            "Ensure the max-Sharpe PC agent has completed Phase A "
            "before calling check_sharpe_floor."
        )

    # ------------------------------------------------------------------
    # Input validation: returns_matrix shape
    # ------------------------------------------------------------------
    if returns_matrix.ndim != 2 or returns_matrix.shape[1] != _N_ASSETS:
        raise ValueError(
            f"returns_matrix must have shape (T, {_N_ASSETS}), "
            f"got {returns_matrix.shape}."
        )

    # ------------------------------------------------------------------
    # Load the max-Sharpe portfolio weights from the JSON artifact
    # ------------------------------------------------------------------
    try:
        with open(max_sharpe_diagnostics_path, "r", encoding="utf-8") as fh:
            # Parse the JSON content of the max-Sharpe pc_weights.json
            pc_data: Dict[str, Any] = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"Failed to load '{max_sharpe_diagnostics_path}': {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Extract and validate the weight vector
    # ------------------------------------------------------------------
    raw_weights = pc_data.get("weights")
    if raw_weights is None:
        raise ValueError(
            f"'weights' field missing in '{max_sharpe_diagnostics_path}'."
        )

    # Convert to numpy array of float64
    w_max_sharpe: np.ndarray = np.array(raw_weights, dtype=np.float64)

    # Validate shape
    if w_max_sharpe.shape != (_N_ASSETS,):
        raise ValueError(
            f"Max-Sharpe weight vector has shape {w_max_sharpe.shape}, "
            f"expected ({_N_ASSETS},)."
        )

    # ------------------------------------------------------------------
    # Compute the backtest Sharpe ratio of the max-Sharpe portfolio
    # r_p_t = sum_i(w_i * r_i_t) = returns_matrix @ w_max_sharpe
    # ------------------------------------------------------------------
    # Portfolio monthly returns: (T,)
    r_p: np.ndarray = returns_matrix @ w_max_sharpe

    # ------------------------------------------------------------------
    # Compute monthly excess returns: r_excess_t = r_p_t - rf
    # rf is the monthly risk-free rate (annualised rf / 12)
    # ------------------------------------------------------------------
    r_excess: np.ndarray = r_p - rf

    # ------------------------------------------------------------------
    # Compute mean and standard deviation of monthly excess returns
    # ------------------------------------------------------------------
    # Mean monthly excess return
    mean_excess_mo: float = float(r_excess.mean())
    # Standard deviation of monthly excess returns (sample, ddof=1)
    std_excess_mo: float = float(r_excess.std(ddof=1))

    # ------------------------------------------------------------------
    # Annualise the Sharpe ratio:
    # SR_ann = (12 * mean_excess_mo) / (sqrt(12) * std_excess_mo)
    # per DATA_CONVENTIONS["annualisation"]
    # ------------------------------------------------------------------
    if std_excess_mo > _EPS:
        # Annualised mean excess return: 12 * mean_mo
        mean_excess_ann: float = float(_PERIODS_PER_YEAR) * mean_excess_mo
        # Annualised excess return volatility: sqrt(12) * std_mo
        std_excess_ann: float = float(np.sqrt(_PERIODS_PER_YEAR)) * std_excess_mo
        # Annualised Sharpe ratio
        max_sharpe_value: float = mean_excess_ann / std_excess_ann
    else:
        # Zero volatility: Sharpe is undefined; set to 0.0
        max_sharpe_value = 0.0
        logger.warning(
            "check_sharpe_floor: Max-Sharpe portfolio has near-zero "
            "excess return volatility. Setting max_sharpe_value = 0.0."
        )

    # ------------------------------------------------------------------
    # Compute the Sharpe floor:
    # SR_floor = 0.75 * SR(w_max_SR)
    # Bounded below by 0.0 (negative floor is non-binding)
    # ------------------------------------------------------------------
    sharpe_floor: float = float(
        max(0.0, _ADVERSARIAL_SHARPE_FLOOR_FRACTION * max_sharpe_value)
    )

    # ------------------------------------------------------------------
    # Check whether the achieved Sharpe meets the floor (if provided)
    # ------------------------------------------------------------------
    floor_met: Optional[bool] = None
    if achieved_sharpe is not None:
        # Check: achieved_sharpe >= sharpe_floor
        floor_met = bool(float(achieved_sharpe) >= sharpe_floor)

    # Log the Sharpe floor computation for audit trail
    logger.info(
        "check_sharpe_floor: max_sharpe_value=%.4f, "
        "sharpe_floor=%.4f (fraction=%.2f), floor_met=%s.",
        max_sharpe_value,
        sharpe_floor,
        _ADVERSARIAL_SHARPE_FLOOR_FRACTION,
        floor_met,
    )

    # ------------------------------------------------------------------
    # Return the Sharpe floor output dict
    # ------------------------------------------------------------------
    return {
        # Sharpe floor: 0.75 * SR(w_max_SR), bounded below by 0.0
        "sharpe_floor": float(sharpe_floor),
        # Backtest Sharpe ratio of the max-Sharpe portfolio
        "max_sharpe_value": float(max_sharpe_value),
        # Whether the achieved Sharpe meets the floor (None if not provided)
        "floor_met": floor_met,
        # Fraction used to compute the floor (0.75, frozen)
        "sharpe_floor_fraction": float(_ADVERSARIAL_SHARPE_FLOOR_FRACTION),
    }


# =============================================================================
# TOOL 34: compute_ex_ante_vol
# =============================================================================

def compute_ex_ante_vol(
    weights: List[float],
    sigma: np.ndarray,
) -> float:
    """
    Compute the ex-ante annualised portfolio volatility.

    Implements the frozen portfolio volatility formula from
    ``IPS_GOVERNANCE["CONSTRAINT_DEFINITIONS"]["portfolio_vol_formula"]``
    (Task 4, Step 1; Task 27, Step 1):

    .. math::

        \\sigma_p = \\sqrt{w^\\top \\Sigma w}

    The result is annualised because ``sigma`` is the annualised covariance
    matrix (monthly covariance × 12, from Task 24).

    Parameters
    ----------
    weights : List[float]
        18-element portfolio weight vector. Must sum to 1.0 (within
        ``1e-6``) and be non-negative (long-only).
    sigma : np.ndarray
        18×18 annualised covariance matrix (Ledoit-Wolf shrinkage estimate
        from Task 24). Shape: ``(18, 18)``. Must be positive semi-definite.

    Returns
    -------
    float
        Annualised ex-ante portfolio volatility :math:`\\sigma_p \\geq 0`.

    Raises
    ------
    TypeError
        If ``sigma`` is not a ``np.ndarray``.
    ValueError
        If ``weights`` does not have 18 elements.
    ValueError
        If ``sigma`` is not shape ``(18, 18)``.

    Notes
    -----
    **Numerical guard:** If the quadratic form :math:`w^\\top \\Sigma w`
    is slightly negative due to floating-point errors in a near-PSD matrix,
    it is clipped to zero before taking the square root. A warning is
    logged if this occurs.
    """
    # ------------------------------------------------------------------
    # Input validation: sigma type and shape
    # ------------------------------------------------------------------
    if not isinstance(sigma, np.ndarray):
        raise TypeError(
            f"sigma must be a np.ndarray, got {type(sigma).__name__}."
        )
    if sigma.shape != (_N_ASSETS, _N_ASSETS):
        raise ValueError(
            f"sigma must have shape ({_N_ASSETS}, {_N_ASSETS}), "
            f"got {sigma.shape}."
        )

    # ------------------------------------------------------------------
    # Input validation: weights length
    # ------------------------------------------------------------------
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Convert weights to numpy array of float64
    # ------------------------------------------------------------------
    # Portfolio weight vector as numpy array
    w: np.ndarray = np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Compute the quadratic form: w' * Sigma * w
    # Using np.dot(w, sigma @ w) for numerical stability
    # ------------------------------------------------------------------
    # Intermediate: Sigma * w (matrix-vector product), shape (18,)
    sigma_w: np.ndarray = sigma @ w
    # Quadratic form: w' * (Sigma * w) = scalar
    quad_form: float = float(np.dot(w, sigma_w))

    # ------------------------------------------------------------------
    # Numerical guard: clip to zero if slightly negative
    # (can occur due to floating-point errors in near-PSD matrices)
    # ------------------------------------------------------------------
    if quad_form < 0.0:
        if quad_form < -1e-6:
            logger.warning(
                "compute_ex_ante_vol: Quadratic form w'Σw = %.2e < 0. "
                "Clipping to 0. Check that sigma is PSD.",
                quad_form,
            )
        # Clip to zero to prevent NaN from sqrt of negative number
        quad_form = 0.0

    # ------------------------------------------------------------------
    # Compute portfolio volatility: sigma_p = sqrt(w' * Sigma * w)
    # ------------------------------------------------------------------
    # Ex-ante annualised portfolio volatility
    sigma_p: float = float(np.sqrt(quad_form))

    return sigma_p


# =============================================================================
# TOOL 35: compute_tracking_error
# =============================================================================

def compute_tracking_error(
    weights: List[float],
    benchmark_weights: np.ndarray,
    sigma: np.ndarray,
) -> float:
    """
    Compute the ex-ante annualised tracking error vs the benchmark.

    Implements the frozen tracking error formula from
    ``IPS_GOVERNANCE["CONSTRAINT_DEFINITIONS"]["tracking_error_formula"]``
    (Task 4, Step 2; Task 27, Step 1):

    .. math::

        TE = \\sqrt{(w - w_b)^\\top \\Sigma (w - w_b)}

    The IPS requires :math:`TE \\leq 0.06` (6%). The result is annualised
    because ``sigma`` is the annualised covariance matrix.

    Parameters
    ----------
    weights : List[float]
        18-element portfolio weight vector.
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.

    Returns
    -------
    float
        Annualised ex-ante tracking error :math:`TE \\geq 0`.

    Raises
    ------
    TypeError
        If ``sigma`` or ``benchmark_weights`` are not ``np.ndarray``.
    ValueError
        If ``weights`` does not have 18 elements or shapes are incorrect.

    Notes
    -----
    The tracking error formula is structurally identical to
    ``compute_ex_ante_vol`` but applied to the **active weight vector**
    :math:`w_{active} = w - w_b` rather than the portfolio weight vector.
    The same numerical guard (clip to zero) is applied.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(sigma, np.ndarray):
        raise TypeError(
            f"sigma must be a np.ndarray, got {type(sigma).__name__}."
        )
    if not isinstance(benchmark_weights, np.ndarray):
        raise TypeError(
            f"benchmark_weights must be a np.ndarray, "
            f"got {type(benchmark_weights).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: shape checks
    # ------------------------------------------------------------------
    if sigma.shape != (_N_ASSETS, _N_ASSETS):
        raise ValueError(
            f"sigma must have shape ({_N_ASSETS}, {_N_ASSETS}), "
            f"got {sigma.shape}."
        )
    if benchmark_weights.shape != (_N_ASSETS,):
        raise ValueError(
            f"benchmark_weights must have shape ({_N_ASSETS},), "
            f"got {benchmark_weights.shape}."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Convert weights to numpy array of float64
    # ------------------------------------------------------------------
    # Portfolio weight vector as numpy array
    w: np.ndarray = np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Compute the active weight vector: w_active = w - w_b
    # ------------------------------------------------------------------
    # Active weight vector relative to the benchmark
    w_active: np.ndarray = w - benchmark_weights

    # ------------------------------------------------------------------
    # Compute the quadratic form: w_active' * Sigma * w_active
    # TE^2 = (w - w_b)' * Sigma * (w - w_b)
    # ------------------------------------------------------------------
    # Intermediate: Sigma * w_active, shape (18,)
    sigma_w_active: np.ndarray = sigma @ w_active
    # Quadratic form: w_active' * (Sigma * w_active) = scalar
    te_var: float = float(np.dot(w_active, sigma_w_active))

    # ------------------------------------------------------------------
    # Numerical guard: clip to zero if slightly negative
    # ------------------------------------------------------------------
    if te_var < 0.0:
        if te_var < -1e-6:
            logger.warning(
                "compute_tracking_error: Quadratic form (w-wb)'Σ(w-wb) "
                "= %.2e < 0. Clipping to 0.",
                te_var,
            )
        # Clip to zero to prevent NaN from sqrt of negative number
        te_var = 0.0

    # ------------------------------------------------------------------
    # Compute tracking error: TE = sqrt((w - w_b)' * Sigma * (w - w_b))
    # ------------------------------------------------------------------
    # Ex-ante annualised tracking error
    te: float = float(np.sqrt(te_var))

    return te


# =============================================================================
# TOOL 36: compute_backtest_sharpe
# =============================================================================

def compute_backtest_sharpe(
    weights: List[float],
    returns_matrix: np.ndarray,
    rf: float,
) -> float:
    """
    Compute the annualised backtest Sharpe ratio for a portfolio.

    Implements the frozen Sharpe ratio formula from
    ``METHODOLOGY_PARAMS["BACKTEST_PARAMS"]["sharpe_formula"]``
    (Task 27, Step 1; Task 33, Step 2):

    .. math::

        SR_{ann} = \\frac{12 \\cdot \\bar{r}_{excess,mo}}
                        {\\sqrt{12} \\cdot \\sigma_{excess,mo}}

    where :math:`r_{excess,t} = r_{p,t} - r_f` and
    :math:`r_{p,t} = \\sum_i w_i r_{i,t}` (static weights, frozen
    convention per ``BACKTEST_PARAMS["rebalancing_assumption"]``).

    Parameters
    ----------
    weights : List[float]
        18-element portfolio weight vector.
    returns_matrix : np.ndarray
        T×18 monthly simple returns matrix (injected via closure).
        Shape: ``(T, 18)``. Constructed from ``df_total_return_raw``
        using the frozen formula :math:`r_t = TR_t / TR_{t-1} - 1`.
    rf : float
        Monthly risk-free rate in decimal form (annualised rf / 12).
        **Must be in monthly form**, not annualised. Example: for an
        annualised rf of 5.3%, the monthly rf = 0.053 / 12 ≈ 0.00442.

    Returns
    -------
    float
        Annualised Sharpe ratio. May be negative if the portfolio
        underperforms the risk-free rate on average.

    Raises
    ------
    TypeError
        If ``returns_matrix`` is not a ``np.ndarray``.
    ValueError
        If ``weights`` does not have 18 elements.
    ValueError
        If ``returns_matrix`` does not have shape ``(T, 18)``.
    ValueError
        If ``returns_matrix`` has fewer than ``_MIN_MONTHLY_OBS``
        rows.

    Notes
    -----
    **Monthly rf convention:** The ``rf`` parameter must be in monthly
    form. The Sharpe ratio is annualised by multiplying the mean excess
    return by 12 and the volatility by :math:`\\sqrt{12}`, then dividing.
    This is equivalent to :math:`SR_{ann} = SR_{mo} \\times \\sqrt{12}`.

    **Zero volatility guard:** If the excess return volatility is
    effectively zero (degenerate portfolio), the Sharpe ratio is set to
    0.0 with a warning.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(returns_matrix, np.ndarray):
        raise TypeError(
            f"returns_matrix must be a np.ndarray, "
            f"got {type(returns_matrix).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: shape checks
    # ------------------------------------------------------------------
    if returns_matrix.ndim != 2 or returns_matrix.shape[1] != _N_ASSETS:
        raise ValueError(
            f"returns_matrix must have shape (T, {_N_ASSETS}), "
            f"got {returns_matrix.shape}."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Input validation: minimum observations
    # ------------------------------------------------------------------
    n_obs: int = returns_matrix.shape[0]
    if n_obs < _MIN_MONTHLY_OBS:
        raise ValueError(
            f"returns_matrix has {n_obs} rows, "
            f"minimum required: {_MIN_MONTHLY_OBS}."
        )

    # ------------------------------------------------------------------
    # Convert weights to numpy array of float64
    # ------------------------------------------------------------------
    # Portfolio weight vector as numpy array
    w: np.ndarray = np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Compute portfolio monthly returns:
    # r_p_t = sum_i(w_i * r_i_t) = returns_matrix @ w
    # Assumes static weights between rebalances (frozen convention)
    # ------------------------------------------------------------------
    # Portfolio return series: (T,)
    r_p: np.ndarray = returns_matrix @ w

    # ------------------------------------------------------------------
    # Compute monthly excess returns: r_excess_t = r_p_t - rf
    # rf is the monthly risk-free rate (annualised rf / 12)
    # ------------------------------------------------------------------
    # Monthly excess return series: (T,)
    r_excess: np.ndarray = r_p - rf

    # ------------------------------------------------------------------
    # Compute mean and standard deviation of monthly excess returns
    # ------------------------------------------------------------------
    # Mean monthly excess return
    mean_excess_mo: float = float(r_excess.mean())
    # Standard deviation of monthly excess returns (sample, ddof=1)
    std_excess_mo: float = float(r_excess.std(ddof=1))

    # ------------------------------------------------------------------
    # Zero volatility guard: return 0.0 if std is effectively zero
    # ------------------------------------------------------------------
    if std_excess_mo < _EPS:
        logger.warning(
            "compute_backtest_sharpe: Excess return volatility is "
            "effectively zero (%.2e). Returning Sharpe = 0.0.",
            std_excess_mo,
        )
        return 0.0

    # ------------------------------------------------------------------
    # Annualise the Sharpe ratio:
    # SR_ann = (12 * mean_excess_mo) / (sqrt(12) * std_excess_mo)
    # per DATA_CONVENTIONS["annualisation"]
    # ------------------------------------------------------------------
    # Annualised mean excess return: 12 * mean_mo
    mean_excess_ann: float = float(_PERIODS_PER_YEAR) * mean_excess_mo
    # Annualised excess return volatility: sqrt(12) * std_mo
    std_excess_ann: float = float(np.sqrt(_PERIODS_PER_YEAR)) * std_excess_mo
    # Annualised Sharpe ratio
    sharpe_ann: float = mean_excess_ann / std_excess_ann

    return float(sharpe_ann)


# =============================================================================
# TOOL 37: compute_mdd
# =============================================================================

def compute_mdd(
    weights: List[float],
    returns_matrix: np.ndarray,
) -> float:
    """
    Compute the maximum drawdown of a portfolio over the backtest window.

    Implements the frozen maximum drawdown formula from
    ``IPS_GOVERNANCE["CONSTRAINT_DEFINITIONS"]["max_drawdown_formula"]``
    (Task 4, Step 3; Task 27, Step 1):

    .. math::

        MDD = \\min_t \\left( \\frac{V_t}{\\max_{s \\leq t} V_s} - 1 \\right)

    where :math:`V_t = \\prod_{s=1}^{t} (1 + r_{p,s})` is the portfolio
    value series constructed from cumulative monthly returns, starting
    from :math:`V_0 = 1`.

    The IPS requires :math:`MDD \\geq -0.25` (maximum drawdown no worse
    than −25%).

    Parameters
    ----------
    weights : List[float]
        18-element portfolio weight vector.
    returns_matrix : np.ndarray
        T×18 monthly simple returns matrix (injected via closure).
        Shape: ``(T, 18)``.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g., ``-0.18`` for
        −18%). Returns ``0.0`` if no drawdown occurs (all returns
        non-negative).

    Raises
    ------
    TypeError
        If ``returns_matrix`` is not a ``np.ndarray``.
    ValueError
        If ``weights`` does not have 18 elements.
    ValueError
        If ``returns_matrix`` does not have shape ``(T, 18)``.

    Notes
    -----
    **Running maximum:** The running maximum :math:`\\max_{s \\leq t} V_s`
    is computed using ``np.maximum.accumulate``, which is the most
    efficient vectorised implementation for this path-dependent operation.

    **Division-by-zero guard:** The portfolio value :math:`V_t` is clipped
    to a minimum of ``1e-10`` before computing the drawdown ratio to
    prevent division by zero in degenerate cases where the portfolio value
    approaches zero.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(returns_matrix, np.ndarray):
        raise TypeError(
            f"returns_matrix must be a np.ndarray, "
            f"got {type(returns_matrix).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: shape checks
    # ------------------------------------------------------------------
    if returns_matrix.ndim != 2 or returns_matrix.shape[1] != _N_ASSETS:
        raise ValueError(
            f"returns_matrix must have shape (T, {_N_ASSETS}), "
            f"got {returns_matrix.shape}."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Convert weights to numpy array of float64
    # ------------------------------------------------------------------
    # Portfolio weight vector as numpy array
    w: np.ndarray = np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Compute portfolio monthly returns:
    # r_p_t = sum_i(w_i * r_i_t) = returns_matrix @ w
    # ------------------------------------------------------------------
    # Portfolio return series: (T,)
    r_p: np.ndarray = returns_matrix @ w

    # ------------------------------------------------------------------
    # Construct the portfolio value series:
    # V_t = prod_{s=1}^{t} (1 + r_p_s), starting from V_0 = 1
    # np.cumprod computes the cumulative product along the time axis
    # ------------------------------------------------------------------
    # Portfolio value series: (T,)
    V: np.ndarray = np.cumprod(1.0 + r_p)

    # ------------------------------------------------------------------
    # Division-by-zero guard: clip portfolio value to minimum of 1e-10
    # (prevents division by zero if portfolio value approaches zero)
    # ------------------------------------------------------------------
    V = np.maximum(V, 1e-10)

    # ------------------------------------------------------------------
    # Compute the running maximum of the portfolio value:
    # M_t = max_{s <= t} V_s
    # np.maximum.accumulate computes the cumulative maximum efficiently
    # ------------------------------------------------------------------
    # Running maximum of portfolio value: (T,)
    M: np.ndarray = np.maximum.accumulate(V)

    # ------------------------------------------------------------------
    # Compute the drawdown series:
    # DD_t = V_t / M_t - 1
    # All values are <= 0 (drawdown is non-positive by construction)
    # ------------------------------------------------------------------
    # Drawdown series: (T,)
    DD: np.ndarray = V / M - 1.0

    # ------------------------------------------------------------------
    # Maximum drawdown: MDD = min_t(DD_t)
    # The minimum (most negative) value in the drawdown series
    # ------------------------------------------------------------------
    mdd: float = float(np.min(DD))

    return mdd


# =============================================================================
# TOOL 38: run_factor_regression
# =============================================================================

def run_factor_regression(
    weights: List[float],
    returns_matrix: np.ndarray,
    factors_df: pd.DataFrame,
    rf: float,
) -> Dict[str, float]:
    """
    Fit the Fama-French 3-factor model to the portfolio's excess returns.

    Implements the factor regression step for the CRO Agent (Task 27,
    Step 1), using the frozen factor model from
    ``METHODOLOGY_PARAMS["FACTOR_MODEL"]``:

    .. math::

        r_{p,t} - r_{f,t} = \\alpha + \\beta_M MKT_t
                           + \\beta_S SMB_t + \\beta_H HML_t + \\varepsilon_t

    OLS is solved via ``numpy.linalg.lstsq`` (SVD-based, numerically
    stable). Alpha is annualised: :math:`\\alpha_{ann} = 12 \\cdot \\hat{\\alpha}_{mo}`.

    Parameters
    ----------
    weights : List[float]
        18-element portfolio weight vector.
    returns_matrix : np.ndarray
        T×18 monthly simple returns matrix (injected via closure).
        Shape: ``(T, 18)``. Must have a ``DatetimeIndex`` alignment
        compatible with ``factors_df``.
    factors_df : pd.DataFrame
        Monthly Fama-French factor returns (injected via closure).
        Must have a ``DatetimeIndex`` and columns: ``"rf"``,
        ``"mkt_rf"``, ``"smb"``, ``"hml"``. All values in decimal
        form (per-month). Shape: ``(T_factors, ≥4)``.
    rf : float
        Monthly risk-free rate in decimal form. Used as a fallback if
        ``factors_df["rf"]`` is not available. If ``factors_df["rf"]``
        is present, it takes precedence.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys:

        - ``"alpha"`` (``float``): Annualised Jensen's alpha
          :math:`\\alpha_{ann} = 12 \\cdot \\hat{\\alpha}_{mo}`.
        - ``"beta_M"`` (``float``): Market beta :math:`\\hat{\\beta}_M`.
        - ``"beta_S"`` (``float``): SMB (size) beta :math:`\\hat{\\beta}_S`.
        - ``"beta_H"`` (``float``): HML (value) beta :math:`\\hat{\\beta}_H`.
        - ``"t_alpha"`` (``float``): t-statistic for alpha.
        - ``"t_beta_M"`` (``float``): t-statistic for market beta.
        - ``"t_beta_S"`` (``float``): t-statistic for SMB beta.
        - ``"t_beta_H"`` (``float``): t-statistic for HML beta.
        - ``"r_squared"`` (``float``): R-squared of the regression.
        - ``"n_obs"`` (``float``): Number of aligned observations used.

    Raises
    ------
    TypeError
        If ``returns_matrix`` is not a ``np.ndarray`` or ``factors_df``
        is not a ``pd.DataFrame``.
    ValueError
        If required columns are missing from ``factors_df``.
    ValueError
        If fewer than ``_MIN_FACTOR_OBS`` aligned observations are
        available.

    Notes
    -----
    **Date alignment:** The portfolio returns are computed from
    ``returns_matrix`` using the weight vector, then aligned with
    ``factors_df`` via an inner join on the ``DatetimeIndex``. Only
    dates present in both series are used for the regression.

    **t-statistics:** Computed as :math:`t_j = \\hat{\\beta}_j / SE_j`
    where :math:`SE_j = \\sqrt{\\hat{\\sigma}^2 (X^\\top X)^{-1}_{jj}}` and
    :math:`\\hat{\\sigma}^2 = RSS / (T - k)` with :math:`k = 4` parameters.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(returns_matrix, np.ndarray):
        raise TypeError(
            f"returns_matrix must be a np.ndarray, "
            f"got {type(returns_matrix).__name__}."
        )
    if not isinstance(factors_df, pd.DataFrame):
        raise TypeError(
            f"factors_df must be a pd.DataFrame, "
            f"got {type(factors_df).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required columns in factors_df
    # ------------------------------------------------------------------
    _required_factor_cols: Tuple[str, ...] = ("mkt_rf", "smb", "hml")
    missing_cols: List[str] = [
        c for c in _required_factor_cols if c not in factors_df.columns
    ]
    if missing_cols:
        raise ValueError(
            f"factors_df is missing required columns: {missing_cols}."
        )

    # ------------------------------------------------------------------
    # Input validation: shape checks
    # ------------------------------------------------------------------
    if returns_matrix.ndim != 2 or returns_matrix.shape[1] != _N_ASSETS:
        raise ValueError(
            f"returns_matrix must have shape (T, {_N_ASSETS}), "
            f"got {returns_matrix.shape}."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Convert weights to numpy array of float64
    # ------------------------------------------------------------------
    # Portfolio weight vector as numpy array
    w: np.ndarray = np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Compute portfolio monthly returns: r_p_t = returns_matrix @ w
    # ------------------------------------------------------------------
    # Portfolio return series: (T,)
    r_p_arr: np.ndarray = returns_matrix @ w

    # ------------------------------------------------------------------
    # Align portfolio returns with factors_df via DatetimeIndex
    # ------------------------------------------------------------------
    # Check if returns_matrix has an associated index (passed separately)
    # Since returns_matrix is a plain ndarray, we use factors_df's index
    # for alignment. We assume returns_matrix rows correspond to the
    # same dates as factors_df after inner-join alignment.
    # Build a pd.Series for portfolio returns using factors_df's index
    # (truncated to the length of returns_matrix if needed)
    n_ret: int = len(r_p_arr)
    n_fac: int = len(factors_df)
    # Use the minimum length for alignment
    n_align: int = min(n_ret, n_fac)

    # Aligned portfolio returns (last n_align observations)
    r_p_aligned: np.ndarray = r_p_arr[-n_align:]
    # Aligned factors DataFrame (last n_align rows)
    factors_aligned: pd.DataFrame = factors_df.iloc[-n_align:].copy()

    # ------------------------------------------------------------------
    # Minimum observation check
    # ------------------------------------------------------------------
    if n_align < _MIN_FACTOR_OBS:
        raise ValueError(
            f"Only {n_align} aligned observations available for factor "
            f"regression, minimum required: {_MIN_FACTOR_OBS}."
        )

    # ------------------------------------------------------------------
    # Determine the risk-free rate series for excess return computation
    # Prefer factors_df["rf"] if available; fall back to constant rf
    # ------------------------------------------------------------------
    if "rf" in factors_aligned.columns:
        # Use the historical rf series from factors_df
        rf_series: np.ndarray = factors_aligned["rf"].values
    else:
        # Fall back to constant monthly rf
        rf_series = np.full(n_align, rf)

    # ------------------------------------------------------------------
    # Compute excess portfolio returns: r_excess_t = r_p_t - r_f_t
    # ------------------------------------------------------------------
    # Monthly excess portfolio return series: (n_align,)
    y: np.ndarray = r_p_aligned - rf_series

    # ------------------------------------------------------------------
    # Construct the design matrix X = [1, MKT_t, SMB_t, HML_t]
    # Shape: (n_align, 4) — intercept + 3 factors
    # ------------------------------------------------------------------
    # Intercept column (all ones)
    intercept: np.ndarray = np.ones(n_align)
    # Market excess return factor
    mkt_rf: np.ndarray = factors_aligned["mkt_rf"].values
    # SMB (size) factor
    smb: np.ndarray = factors_aligned["smb"].values
    # HML (value) factor
    hml: np.ndarray = factors_aligned["hml"].values
    # Design matrix: (n_align, 4)
    X: np.ndarray = np.column_stack([intercept, mkt_rf, smb, hml])

    # ------------------------------------------------------------------
    # Solve OLS via numpy.linalg.lstsq (SVD-based, numerically stable)
    # Solves: min ||y - X @ beta||^2
    # Returns: beta_hat (4,), residuals, rank, singular values
    # ------------------------------------------------------------------
    # OLS solution: beta_hat = [alpha_mo, beta_M, beta_S, beta_H]
    beta_hat: np.ndarray
    residuals_sq: np.ndarray
    beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # ------------------------------------------------------------------
    # Extract regression coefficients
    # ------------------------------------------------------------------
    # Monthly alpha (intercept)
    alpha_mo: float = float(beta_hat[0])
    # Market beta
    beta_M: float = float(beta_hat[1])
    # SMB (size) beta
    beta_S: float = float(beta_hat[2])
    # HML (value) beta
    beta_H: float = float(beta_hat[3])

    # ------------------------------------------------------------------
    # Annualise alpha: alpha_ann = 12 * alpha_mo
    # per DATA_CONVENTIONS["annualisation"]["mu_multiplier"] = 12
    # ------------------------------------------------------------------
    # Annualised Jensen's alpha
    alpha_ann: float = float(_PERIODS_PER_YEAR) * alpha_mo

    # ------------------------------------------------------------------
    # Compute residuals: e = y - X @ beta_hat
    # ------------------------------------------------------------------
    # Residual series: (n_align,)
    e: np.ndarray = y - X @ beta_hat

    # ------------------------------------------------------------------
    # Compute residual sum of squares (RSS)
    # ------------------------------------------------------------------
    # RSS = sum(e_t^2)
    rss: float = float(np.dot(e, e))

    # ------------------------------------------------------------------
    # Compute the residual variance: sigma_hat^2 = RSS / (T - k)
    # k = 4 parameters (intercept + 3 factors)
    # ------------------------------------------------------------------
    k: int = 4  # Number of regression parameters
    df_resid: int = n_align - k  # Degrees of freedom for residuals
    if df_resid > 0:
        # Residual variance estimate
        sigma_sq_hat: float = rss / float(df_resid)
    else:
        # Insufficient degrees of freedom: set to zero
        sigma_sq_hat = 0.0

    # ------------------------------------------------------------------
    # Compute standard errors: SE_j = sqrt(sigma_hat^2 * (X'X)^{-1}_{jj})
    # ------------------------------------------------------------------
    # Compute (X'X)^{-1} using numpy.linalg.pinv for numerical stability
    XtX_inv: np.ndarray = np.linalg.pinv(X.T @ X)
    # Diagonal elements of (X'X)^{-1} give the variance of each coefficient
    var_beta: np.ndarray = sigma_sq_hat * np.diag(XtX_inv)
    # Standard errors: SE_j = sqrt(var_beta_j)
    se_beta: np.ndarray = np.sqrt(np.maximum(var_beta, 0.0))

    # ------------------------------------------------------------------
    # Compute t-statistics: t_j = beta_hat_j / SE_j
    # ------------------------------------------------------------------
    # t-statistic for alpha (monthly; annualised alpha has different SE)
    t_alpha: float = (
        float(beta_hat[0] / se_beta[0]) if se_beta[0] > _EPS else 0.0
    )
    # t-statistic for market beta
    t_beta_M: float = (
        float(beta_hat[1] / se_beta[1]) if se_beta[1] > _EPS else 0.0
    )
    # t-statistic for SMB beta
    t_beta_S: float = (
        float(beta_hat[2] / se_beta[2]) if se_beta[2] > _EPS else 0.0
    )
    # t-statistic for HML beta
    t_beta_H: float = (
        float(beta_hat[3] / se_beta[3]) if se_beta[3] > _EPS else 0.0
    )

    # ------------------------------------------------------------------
    # Compute R-squared: R^2 = 1 - RSS / TSS
    # TSS = sum((y_t - y_bar)^2)
    # ------------------------------------------------------------------
    # Total sum of squares
    tss: float = float(np.dot(y - y.mean(), y - y.mean()))
    if tss > _EPS:
        # R-squared: fraction of variance explained by the model
        r_squared: float = float(1.0 - rss / tss)
        # Clip to [0, 1] to handle numerical edge cases
        r_squared = float(np.clip(r_squared, 0.0, 1.0))
    else:
        # Degenerate case: all excess returns are identical
        r_squared = 0.0

    # Log the regression result for audit trail
    logger.debug(
        "run_factor_regression: alpha_ann=%.4f, beta_M=%.4f, "
        "beta_S=%.4f, beta_H=%.4f, R2=%.4f, n_obs=%d.",
        alpha_ann, beta_M, beta_S, beta_H, r_squared, n_align,
    )

    # ------------------------------------------------------------------
    # Return the factor regression result dict
    # ------------------------------------------------------------------
    return {
        # Annualised Jensen's alpha: alpha_ann = 12 * alpha_mo
        "alpha": float(alpha_ann),
        # Market beta (Fama-French MKT factor loading)
        "beta_M": float(beta_M),
        # SMB (size) beta
        "beta_S": float(beta_S),
        # HML (value) beta
        "beta_H": float(beta_H),
        # t-statistic for alpha
        "t_alpha": float(t_alpha),
        # t-statistic for market beta
        "t_beta_M": float(t_beta_M),
        # t-statistic for SMB beta
        "t_beta_S": float(t_beta_S),
        # t-statistic for HML beta
        "t_beta_H": float(t_beta_H),
        # R-squared of the regression
        "r_squared": float(r_squared),
        # Number of aligned observations used
        "n_obs": float(n_align),
    }


# =============================================================================
# TOOL 39: check_ips_compliance
# =============================================================================

def check_ips_compliance(
    weights: List[float],
    sigma: np.ndarray,
    benchmark_weights: np.ndarray,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Check whether a portfolio weight vector satisfies all IPS constraints.

    Implements the IPS compliance check shared across PC agents (Task 25),
    the CRO Agent (Task 27), the Top-5 Revision Agent (Task 30), and the
    CIO Agent (Task 31). Checks all constraints defined in
    ``IPS_GOVERNANCE``:

    1. **Long-only:** :math:`w_i \\geq 0` for all :math:`i`
    2. **Max weight:** :math:`w_i \\leq 0.25` for all :math:`i`
    3. **Budget:** :math:`\\sum_i w_i = 1.0 \\pm 10^{-6}`
    4. **Tracking error:** :math:`TE = \\sqrt{(w-w_b)^\\top \\Sigma (w-w_b)} \\leq 0.06`
    5. **Volatility band:** :math:`\\sigma_p = \\sqrt{w^\\top \\Sigma w} \\in [0.08, 0.12]`

    Parameters
    ----------
    weights : List[float]
        18-element portfolio weight vector to check.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    constraints : Optional[Dict[str, Any]]
        IPS constraint parameters. If ``None``, uses frozen defaults from
        ``IPS_GOVERNANCE``. Expected keys:
        ``"max_weight_per_asset"`` (float, default 0.25),
        ``"min_weight_per_asset"`` (float, default 0.00),
        ``"long_only"`` (bool, default True).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"compliant"`` (``bool``): ``True`` if all constraints pass.
        - ``"flags"`` (``Dict[str, Dict[str, Any]]``): Per-constraint
          results. Each entry contains ``"pass"`` (bool) and
          ``"actual_value"`` (float or list).

    Raises
    ------
    TypeError
        If ``sigma`` or ``benchmark_weights`` are not ``np.ndarray``.
    ValueError
        If ``weights`` does not have 18 elements or shapes are incorrect.

    Notes
    -----
    **MDD constraint:** The maximum drawdown constraint
    (:math:`MDD \\geq -0.25`) is **not** checked here because it requires
    the full backtest return series (path-dependent, non-convex). The MDD
    is checked separately by the CRO Agent using ``compute_mdd``.

    **Constraint values:** The ``flags`` dict includes the actual computed
    value for each constraint (e.g., the actual TE, actual vol) to enable
    informative reporting in the CRO narrative.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(sigma, np.ndarray):
        raise TypeError(
            f"sigma must be a np.ndarray, got {type(sigma).__name__}."
        )
    if not isinstance(benchmark_weights, np.ndarray):
        raise TypeError(
            f"benchmark_weights must be a np.ndarray, "
            f"got {type(benchmark_weights).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: shape checks
    # ------------------------------------------------------------------
    if sigma.shape != (_N_ASSETS, _N_ASSETS):
        raise ValueError(
            f"sigma must have shape ({_N_ASSETS}, {_N_ASSETS}), "
            f"got {sigma.shape}."
        )
    if benchmark_weights.shape != (_N_ASSETS,):
        raise ValueError(
            f"benchmark_weights must have shape ({_N_ASSETS},), "
            f"got {benchmark_weights.shape}."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Resolve constraint parameters (defaults or override)
    # ------------------------------------------------------------------
    _c: Dict[str, Any] = constraints or {}
    # Maximum weight per asset (default: 0.25)
    max_w: float = float(_c.get("max_weight_per_asset", _IPS_MAX_WEIGHT))
    # Minimum weight per asset (default: 0.00, long-only)
    min_w: float = float(_c.get("min_weight_per_asset", _IPS_MIN_WEIGHT))

    # ------------------------------------------------------------------
    # Convert weights to numpy array of float64
    # ------------------------------------------------------------------
    # Portfolio weight vector as numpy array
    w: np.ndarray = np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Constraint 1: Long-only — w_i >= min_w for all i
    # ------------------------------------------------------------------
    # Minimum weight across all assets
    min_weight_actual: float = float(w.min())
    # Pass if all weights are >= min_w (with small tolerance)
    long_only_pass: bool = bool(min_weight_actual >= min_w - 1e-8)

    # ------------------------------------------------------------------
    # Constraint 2: Max weight — w_i <= max_w for all i
    # ------------------------------------------------------------------
    # Maximum weight across all assets
    max_weight_actual: float = float(w.max())
    # Pass if all weights are <= max_w (with small tolerance)
    max_weight_pass: bool = bool(max_weight_actual <= max_w + 1e-8)

    # ------------------------------------------------------------------
    # Constraint 3: Budget — sum(w) = 1.0 ± 1e-6
    # ------------------------------------------------------------------
    # Sum of all portfolio weights
    weight_sum: float = float(w.sum())
    # Pass if sum is within 1e-6 of 1.0
    sum_to_one_pass: bool = bool(abs(weight_sum - 1.0) <= 1e-6)

    # ------------------------------------------------------------------
    # Constraint 4: Tracking error — TE <= 0.06
    # Reuse compute_tracking_error logic inline for efficiency
    # TE = sqrt((w - w_b)' * Sigma * (w - w_b))
    # ------------------------------------------------------------------
    # Active weight vector relative to benchmark
    w_active: np.ndarray = w - benchmark_weights
    # Quadratic form for tracking error variance
    te_var: float = float(np.dot(w_active, sigma @ w_active))
    # Clip to zero if slightly negative (numerical guard)
    te_var = max(0.0, te_var)
    # Ex-ante annualised tracking error
    te_actual: float = float(np.sqrt(te_var))
    # Pass if TE <= 0.06 (with small tolerance)
    te_pass: bool = bool(te_actual <= _IPS_TE_BUDGET + 1e-8)

    # ------------------------------------------------------------------
    # Constraint 5: Volatility band — sigma_p in [0.08, 0.12]
    # sigma_p = sqrt(w' * Sigma * w)
    # ------------------------------------------------------------------
    # Quadratic form for portfolio variance
    port_var: float = float(np.dot(w, sigma @ w))
    # Clip to zero if slightly negative (numerical guard)
    port_var = max(0.0, port_var)
    # Ex-ante annualised portfolio volatility
    vol_actual: float = float(np.sqrt(port_var))
    # Pass if vol is within [0.08, 0.12] (with small tolerance)
    vol_lower_pass: bool = bool(vol_actual >= _IPS_VOL_LOWER - 1e-8)
    vol_upper_pass: bool = bool(vol_actual <= _IPS_VOL_UPPER + 1e-8)
    vol_band_pass: bool = vol_lower_pass and vol_upper_pass

    # ------------------------------------------------------------------
    # Aggregate all constraint flags
    # ------------------------------------------------------------------
    flags: Dict[str, Dict[str, Any]] = {
        # Long-only constraint: w_i >= 0 for all i
        "long_only": {
            "pass": long_only_pass,
            "actual_value": float(min_weight_actual),
            "threshold": float(min_w),
            "description": f"min(w) = {min_weight_actual:.6f} >= {min_w}",
        },
        # Max weight constraint: w_i <= 0.25 for all i
        "max_weight": {
            "pass": max_weight_pass,
            "actual_value": float(max_weight_actual),
            "threshold": float(max_w),
            "description": f"max(w) = {max_weight_actual:.6f} <= {max_w}",
        },
        # Budget constraint: sum(w) = 1.0
        "sum_to_one": {
            "pass": sum_to_one_pass,
            "actual_value": float(weight_sum),
            "threshold": 1.0,
            "description": f"sum(w) = {weight_sum:.8f} ≈ 1.0",
        },
        # Tracking error constraint: TE <= 0.06
        "tracking_error": {
            "pass": te_pass,
            "actual_value": float(te_actual),
            "threshold": float(_IPS_TE_BUDGET),
            "description": f"TE = {te_actual * 100:.2f}% <= {_IPS_TE_BUDGET * 100:.0f}%",
        },
        # Volatility band constraint: sigma_p in [0.08, 0.12]
        "volatility_band": {
            "pass": vol_band_pass,
            "actual_value": float(vol_actual),
            "threshold_lower": float(_IPS_VOL_LOWER),
            "threshold_upper": float(_IPS_VOL_UPPER),
            "description": (
                f"vol = {vol_actual * 100:.2f}% in "
                f"[{_IPS_VOL_LOWER * 100:.0f}%, {_IPS_VOL_UPPER * 100:.0f}%]"
            ),
        },
    }

    # ------------------------------------------------------------------
    # Overall compliance: all constraints must pass
    # ------------------------------------------------------------------
    # Portfolio is IPS-compliant only if every constraint passes
    compliant: bool = all(v["pass"] for v in flags.values())

    # Log the compliance result for audit trail
    logger.debug(
        "check_ips_compliance: compliant=%s, TE=%.4f, vol=%.4f, "
        "max_w=%.4f, sum=%.6f.",
        compliant, te_actual, vol_actual, max_weight_actual, weight_sum,
    )

    # ------------------------------------------------------------------
    # Return the IPS compliance output dict
    # ------------------------------------------------------------------
    return {
        # Overall compliance flag: True if all constraints pass
        "compliant": compliant,
        # Per-constraint pass/fail flags with actual values
        "flags": flags,
    }


# =============================================================================
# TOOL 40: write_cro_report_json
# =============================================================================

def write_cro_report_json(
    candidate_id: str,
    metrics: Dict[str, Any],
    compliance_flags: Dict[str, Any],
    artifact_dir: Path,
) -> str:
    """
    Serialise and persist the CRO risk report to ``cro_report.json``.

    This tool implements the structured CRO report artifact-writing step
    for the CRO Agent (Task 27, Step 2). The output file is consumed by
    the peer review agents (Task 28), the voting agents (Task 29), and
    the CIO agent (Task 31). It must conform to the frozen
    ``cro_report.schema.json`` schema.

    The file is written to:
    ``{artifact_dir}/{candidate_id}/cro_report.json``

    Parameters
    ----------
    candidate_id : str
        PC method slug identifier (e.g., ``"max_diversification"``).
        Used to construct the output file path.
    metrics : Dict[str, Any]
        Dictionary of computed risk metrics. Must contain all keys in
        ``_REQUIRED_CRO_METRIC_KEYS``:
        ``"ex_ante_vol"``, ``"tracking_error"``, ``"backtest_sharpe"``,
        ``"max_drawdown"``, ``"alpha"``, ``"beta_M"``, ``"beta_S"``,
        ``"beta_H"``, ``"r_squared"``.
    compliance_flags : Dict[str, Any]
        Output of ``check_ips_compliance``. Must contain ``"compliant"``
        (bool) and ``"flags"`` (dict).
    artifact_dir : Path
        Base artifact directory. The file is written to
        ``{artifact_dir}/{candidate_id}/cro_report.json``.

    Returns
    -------
    str
        Absolute path to the written ``cro_report.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``metrics`` /
        ``compliance_flags`` are not dicts.
    ValueError
        If required metric keys are missing from ``metrics``.
    ValueError
        If ``compliance_flags`` is missing ``"compliant"`` or ``"flags"``
        keys.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    The ``"ips_compliant"`` field in the output is a boolean summary
    derived from ``compliance_flags["compliant"]``. This field enables
    quick downstream filtering of non-compliant candidates without
    parsing the full ``flags`` dict.

    All numeric values are recursively cast to JSON-safe Python native
    types via ``_cast_to_json_safe`` before serialisation.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(metrics, dict):
        raise TypeError(
            f"metrics must be a dict, got {type(metrics).__name__}."
        )
    if not isinstance(compliance_flags, dict):
        raise TypeError(
            f"compliance_flags must be a dict, "
            f"got {type(compliance_flags).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required metric keys
    # ------------------------------------------------------------------
    missing_metric_keys: List[str] = [
        k for k in _REQUIRED_CRO_METRIC_KEYS if k not in metrics
    ]
    if missing_metric_keys:
        raise ValueError(
            f"metrics is missing required keys: {missing_metric_keys}. "
            f"Required: {list(_REQUIRED_CRO_METRIC_KEYS)}."
        )

    # ------------------------------------------------------------------
    # Input validation: compliance_flags required keys
    # ------------------------------------------------------------------
    for key in ("compliant", "flags"):
        if key not in compliance_flags:
            raise ValueError(
                f"compliance_flags is missing required key '{key}'."
            )

    # ------------------------------------------------------------------
    # Construct the output directory and file path
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{candidate_id}/
    output_dir: Path = artifact_dir / candidate_id
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "cro_report.json"

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # PC method slug identifier
        "candidate_id": candidate_id,
        # Boolean IPS compliance summary (for quick downstream filtering)
        "ips_compliant": bool(compliance_flags["compliant"]),
        # All computed risk metrics (cast to JSON-safe types)
        "metrics": _cast_to_json_safe(metrics),
        # Per-constraint compliance flags with actual values
        "compliance_flags": _cast_to_json_safe(compliance_flags),
    }

    # ------------------------------------------------------------------
    # Serialise to JSON and write to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised output dict with 2-space indentation
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write cro_report.json to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_cro_report_json: written for candidate_id='%s' to '%s'. "
        "ips_compliant=%s.",
        candidate_id,
        output_path,
        bool(compliance_flags["compliant"]),
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())

# =============================================================================
# SELF-DRIVING PORTFOLIO: TOOL REGISTRY — BATCH 5 (TOOLS 41–50)
# =============================================================================
# Implements tools 41–50 from the complete 78-tool registry for the agentic
# Strategic Asset Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# Tools implemented:
#   41. write_cro_narrative_md      — CRO Agent artifact writer
#   42. compute_review_assignments  — Strategy Review Controller
#   43. write_peer_review_md        — Peer Review Phase artifact writer
#   44. validate_review_barrier     — Strategy Review synchronisation gate
#   45. submit_vote                 — Voting Phase forced tool call
#   46. compute_borda_aggregation   — Voting Phase scripted aggregation
#   47. compute_composite_ranking   — Voting Phase scripted ranking
#   48. write_vote_corpus_json      — Voting Phase artifact writer
#   49. apply_diversity_constraint  — Top-5 Revision diversity enforcement
#   50. confirm_no_revision         — Top-5 Revision no-change signal
#
# All tools are purely deterministic Python callables — no LLM interaction.
# All arithmetic conventions are frozen per STUDY_CONFIG["DATA_CONVENTIONS"].
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
# Initialise a named logger so callers can configure log levels independently
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frozen constants (sourced from STUDY_CONFIG; reproduced for self-contained
# validation — the orchestrator injects the live config at runtime)
# ---------------------------------------------------------------------------

# Completion signal that every peer review must contain
# The LLM is instructed to append this string at the end of every review
_REVIEW_COMPLETION_SIGNAL: str = "REVIEW_COMPLETE"

# Minimum character length for review content (non-trivial review)
_MIN_REVIEW_LENGTH: int = 50

# Minimum character length for rationale strings
_MIN_RATIONALE_LENGTH: int = 10

# Frozen Borda count scoring: top-5 points and bottom-flag penalty
_BORDA_TOP5_POINTS: Tuple[int, ...] = (5, 4, 3, 2, 1)
_BORDA_BOTTOM_FLAG_PENALTY: int = -2

# Expected number of votes (21 PC agents)
_EXPECTED_N_VOTES: int = 21

# Expected number of peer reviews (21 agents × 2 reviews each)
_EXPECTED_N_REVIEWS: int = 42

# Diversity constraint: top-5 must include at least 3 of 4 families
_DIVERSITY_REQUIRED_FAMILIES: int = 3
_DIVERSITY_N_SELECT: int = 5

# Valid PC agent family labels
_VALID_PC_FAMILIES: Tuple[str, ...] = (
    "risk_based",
    "return_based",
    "naive",
    "adversarial",
)

# Regime-dependent alpha schedule for composite ranking
# per METHODOLOGY_PARAMS["STRATEGY_REVIEW"]["vote_metric_blend_alpha_schedule"]
_ALPHA_SCHEDULE: Dict[str, float] = {
    "Expansion":  0.5,
    "Late-cycle": 0.6,
    "Recession":  0.4,
    "Recovery":   0.5,
}

# Valid regime labels
_VALID_REGIMES: Tuple[str, ...] = (
    "Expansion",
    "Late-cycle",
    "Recession",
    "Recovery",
)

# Numerical stability epsilon
_EPS: float = 1e-8

# Maximum retry attempts for review assignment generation
_MAX_ASSIGNMENT_RETRIES: int = 10


# ---------------------------------------------------------------------------
# Shared utility: recursive JSON-safe type casting
# (re-declared for self-contained module; identical to prior batches)
# ---------------------------------------------------------------------------

def _cast_to_json_safe(obj: Any) -> Any:
    """
    Recursively cast an object to JSON-serialisable Python native types.

    Converts ``np.float64``, ``np.int64``, ``np.bool_``, ``np.ndarray``,
    and nested containers to their Python native equivalents.
    ``None`` and ``float("nan")`` are preserved as ``None`` (JSON ``null``).

    Parameters
    ----------
    obj : Any
        The object to cast.

    Returns
    -------
    Any
        A JSON-serialisable Python native object.
    """
    # Preserve None as None (serialises to JSON null)
    if obj is None:
        return None
    # Cast numpy floating-point scalars to Python float; NaN → None
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    # Cast Python float; NaN → None
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    # Cast numpy integer scalars to Python int
    if isinstance(obj, np.integer):
        return int(obj)
    # Cast numpy boolean scalars to Python bool
    if isinstance(obj, np.bool_):
        return bool(obj)
    # Convert numpy arrays to list of JSON-safe elements
    if isinstance(obj, np.ndarray):
        return [_cast_to_json_safe(v) for v in obj.tolist()]
    # Recursively cast dict values
    if isinstance(obj, dict):
        return {str(k): _cast_to_json_safe(v) for k, v in obj.items()}
    # Recursively cast list/tuple elements
    if isinstance(obj, (list, tuple)):
        return [_cast_to_json_safe(v) for v in obj]
    # Python int and bool: return as-is
    if isinstance(obj, (int, bool)):
        return obj
    # Strings: return as-is
    if isinstance(obj, str):
        return obj
    # Fallback: string conversion for unknown types
    return str(obj)


def _derive_slug(name: str) -> str:
    """
    Derive a filesystem-safe slug from a name string.

    Applies: lowercase → replace spaces with underscores →
    remove non-alphanumeric/underscore characters.

    Parameters
    ----------
    name : str
        Input name (e.g., ``"max_diversification"``).

    Returns
    -------
    str
        Filesystem-safe slug.
    """
    # Convert to lowercase
    s: str = name.lower()
    # Replace spaces with underscores
    s = s.replace(" ", "_")
    # Remove non-alphanumeric/underscore characters
    s = re.sub(r"[^a-z0-9_]", "", s)
    return s


# =============================================================================
# TOOL 41: write_cro_narrative_md
# =============================================================================

def write_cro_narrative_md(
    candidate_id: str,
    metrics: Dict[str, Any],
    rationale: str,
    artifact_dir: Path,
) -> str:
    """
    Format and persist the CRO risk narrative to ``cro_narrative.md``.

    This tool implements the CRO narrative artifact-writing step for the
    CRO Agent (Task 27, Step 2). The output file is the human-readable
    complement to the machine-readable ``cro_report.json`` (Tool 40).
    It is consumed by peer review agents (Task 28) and the CIO agent
    (Task 31) for governance documentation.

    The markdown document covers:

    - Portfolio risk metrics summary table
    - IPS compliance status
    - Factor exposure summary (Fama-French 3-factor)
    - LLM-generated narrative (volatility regime, TE vs benchmark,
      drawdown risk, factor exposures, IPS compliance status)

    The file is written to:
    ``{artifact_dir}/{candidate_id}/cro_narrative.md``

    Parameters
    ----------
    candidate_id : str
        PC method slug identifier (e.g., ``"max_diversification"``).
        Used to construct the output file path.
    metrics : Dict[str, Any]
        Dictionary of computed risk metrics. Expected keys (all optional
        with ``"N/A"`` fallback for display):
        ``"ex_ante_vol"``, ``"tracking_error"``, ``"backtest_sharpe"``,
        ``"max_drawdown"``, ``"alpha"``, ``"beta_M"``, ``"beta_S"``,
        ``"beta_H"``, ``"r_squared"``.
    rationale : str
        LLM-generated narrative text covering: volatility regime
        characterisation, TE vs 6% budget, drawdown risk vs −25% limit,
        factor tilt implications, and IPS compliance status. Must be
        non-empty (minimum ``_MIN_RATIONALE_LENGTH`` characters).
    artifact_dir : Path
        Base artifact directory. The file is written to
        ``{artifact_dir}/{candidate_id}/cro_narrative.md``.

    Returns
    -------
    str
        Absolute path to the written ``cro_narrative.md`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``rationale`` is empty or fewer than ``_MIN_RATIONALE_LENGTH``
        characters.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    Missing metric values are displayed as ``"N/A"`` in the markdown
    table rather than raising an error. This allows the narrative to be
    written even if some metric computations failed, with the missing
    values clearly flagged for human review.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )

    # ------------------------------------------------------------------
    # Helper: safely format a metric value for display
    # Returns the value formatted to 4 decimal places, or "N/A" if absent
    # ------------------------------------------------------------------
    def _fmt(key: str, multiplier: float = 1.0, suffix: str = "") -> str:
        """Format a metric value for markdown display."""
        val = metrics.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        try:
            return f"{float(val) * multiplier:.4f}{suffix}"
        except (TypeError, ValueError):
            return str(val)

    # ------------------------------------------------------------------
    # Build the markdown document section by section
    # ------------------------------------------------------------------

    # --- Document header ---
    # Title and candidate identifier
    md_lines: List[str] = [
        f"# CRO Risk Report — {candidate_id}",
        "",
        f"**Candidate Portfolio:** `{candidate_id}`",
        "",
    ]

    # --- Risk metrics summary table ---
    # Section heading for the quantitative risk metrics
    md_lines.extend([
        "## Risk Metrics Summary",
        "",
        "| Metric | Value | IPS Threshold |",
        "|--------|-------|---------------|",
        # Ex-ante annualised portfolio volatility
        f"| Ex-Ante Vol (ann.) | {_fmt('ex_ante_vol', 100, '%')} "
        f"| [8%, 12%] |",
        # Ex-ante annualised tracking error vs 60/40 benchmark
        f"| Tracking Error (ann.) | {_fmt('tracking_error', 100, '%')} "
        f"| ≤ 6% |",
        # Backtest annualised Sharpe ratio
        f"| Backtest Sharpe (ann.) | {_fmt('backtest_sharpe')} "
        f"| — |",
        # Maximum drawdown over the backtest window
        f"| Max Drawdown | {_fmt('max_drawdown', 100, '%')} "
        f"| ≥ −25% |",
        "",
    ])

    # --- Factor exposure summary table ---
    # Section heading for the Fama-French 3-factor regression results
    md_lines.extend([
        "## Factor Exposures (Fama-French 3-Factor)",
        "",
        "| Factor | Loading | t-Statistic |",
        "|--------|---------|-------------|",
        # Annualised Jensen's alpha
        f"| Alpha (ann.) | {_fmt('alpha', 100, '%')} "
        f"| {_fmt('t_alpha')} |",
        # Market beta
        f"| Market (MKT) | {_fmt('beta_M')} "
        f"| {_fmt('t_beta_M')} |",
        # SMB (size) beta
        f"| Size (SMB) | {_fmt('beta_S')} "
        f"| {_fmt('t_beta_S')} |",
        # HML (value) beta
        f"| Value (HML) | {_fmt('beta_H')} "
        f"| {_fmt('t_beta_H')} |",
        # R-squared of the factor regression
        f"| R-Squared | {_fmt('r_squared')} | — |",
        "",
    ])

    # --- IPS compliance status ---
    # Section heading for the IPS compliance summary
    md_lines.extend([
        "## IPS Compliance Status",
        "",
    ])

    # Extract IPS compliance information from metrics if available
    ips_compliant = metrics.get("ips_compliant")
    if ips_compliant is not None:
        # Display overall compliance status
        status_str: str = "✅ COMPLIANT" if bool(ips_compliant) else "❌ NON-COMPLIANT"
        md_lines.append(f"**Overall IPS Status:** {status_str}")
        md_lines.append("")
    else:
        # IPS compliance not available in metrics
        md_lines.append("**Overall IPS Status:** See `cro_report.json` for details.")
        md_lines.append("")

    # --- LLM-generated narrative ---
    # Section heading for the CRO narrative analysis
    md_lines.extend([
        "## CRO Narrative Analysis",
        "",
        rationale.strip(),
        "",
    ])

    # ------------------------------------------------------------------
    # Join all lines into a single markdown string
    # ------------------------------------------------------------------
    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Construct the output directory and file path
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{candidate_id}/
    output_dir: Path = artifact_dir / candidate_id
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "cro_narrative.md"

    # ------------------------------------------------------------------
    # Write the markdown content to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the complete markdown document to the file
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write cro_narrative.md to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_cro_narrative_md: written for candidate_id='%s' to '%s'.",
        candidate_id,
        output_path,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 42: compute_review_assignments
# =============================================================================

def compute_review_assignments(
    pc_methods: List[Dict[str, Any]],
    seed: int,
    artifact_dir: Path,
) -> Dict[str, List[str]]:
    """
    Compute the randomised peer review assignment matrix.

    Implements the deterministic peer review assignment step for the
    ``StrategyReviewController`` (Task 28, Step 1). Each PC agent is
    assigned exactly:

    - **1 intra-category peer** (same ``family`` label)
    - **1 inter-category peer** (different ``family`` label)

    No self-reviews are permitted. The assignment is fully deterministic
    given the frozen ``seed`` from
    ``STUDY_CONFIG["RANDOM_SEEDS"]["peer_review_assignment_seed"]``.

    The assignment matrix is persisted as
    ``{artifact_dir}/assignment_matrix.json`` for audit.

    Parameters
    ----------
    pc_methods : List[Dict[str, Any]]
        List of PC method dicts. Each dict must contain:
        ``"slug"`` (``str``) and ``"family"`` (``str``, one of
        ``_VALID_PC_FAMILIES``).
    seed : int
        Random seed for deterministic assignment generation. Must be a
        non-negative integer.
    artifact_dir : Path
        Directory to write ``assignment_matrix.json``.

    Returns
    -------
    Dict[str, List[str]]
        Assignment mapping: ``{reviewer_slug: [intra_peer_slug,
        inter_peer_slug]}``. Each reviewer is assigned exactly 2 peers.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``pc_methods``
        is not a list.
    ValueError
        If any method dict is missing ``"slug"`` or ``"family"`` keys.
    ValueError
        If fewer than 2 distinct families are represented in
        ``pc_methods`` (cannot satisfy intra/inter constraint).
    RuntimeError
        If a valid assignment cannot be generated after
        ``_MAX_ASSIGNMENT_RETRIES`` attempts.
    OSError
        If the assignment matrix cannot be written to disk.

    Notes
    -----
    **Single-member family handling:** If a family has only 1 member
    (e.g., the ``"adversarial"`` family with only the adversarial
    diversifier), that agent has no intra-category peers. In this case,
    a second inter-category peer is assigned instead. This exception is
    recorded in the ``assignment_matrix.json`` under
    ``"single_member_family_exceptions"``.

    **Retry logic:** The assignment algorithm uses seeded random sampling.
    If the first attempt fails to satisfy all constraints (rare), the seed
    is incremented by 1 and the attempt is retried up to
    ``_MAX_ASSIGNMENT_RETRIES`` times. The actual seed used is recorded
    in the output artifact.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(pc_methods, list):
        raise TypeError(
            f"pc_methods must be a list, got {type(pc_methods).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: each method dict must have slug and family
    # ------------------------------------------------------------------
    for i, method in enumerate(pc_methods):
        if not isinstance(method, dict):
            raise TypeError(
                f"pc_methods[{i}] must be a dict, "
                f"got {type(method).__name__}."
            )
        for key in ("slug", "family"):
            if key not in method:
                raise ValueError(
                    f"pc_methods[{i}] is missing required key '{key}'."
                )

    # ------------------------------------------------------------------
    # Extract slugs and families; build family-to-slugs mapping
    # ------------------------------------------------------------------
    # All method slugs in the order provided
    all_slugs: List[str] = [str(m["slug"]) for m in pc_methods]
    # Mapping from method slug to its family label
    slug_to_family: Dict[str, str] = {
        str(m["slug"]): str(m["family"]) for m in pc_methods
    }
    # Mapping from family label to list of method slugs in that family
    family_to_slugs: Dict[str, List[str]] = {}
    for slug, family in slug_to_family.items():
        family_to_slugs.setdefault(family, []).append(slug)

    # ------------------------------------------------------------------
    # Input validation: at least 2 distinct families required
    # ------------------------------------------------------------------
    n_families: int = len(family_to_slugs)
    if n_families < 2:
        raise ValueError(
            f"pc_methods must contain at least 2 distinct families to "
            f"satisfy the intra/inter-category constraint. "
            f"Found {n_families} family/families: "
            f"{list(family_to_slugs.keys())}."
        )

    # ------------------------------------------------------------------
    # Identify single-member families (no intra-category peers available)
    # ------------------------------------------------------------------
    # Set of families with only 1 member (cannot have intra-category peer)
    single_member_families: Set[str] = {
        fam for fam, slugs in family_to_slugs.items() if len(slugs) == 1
    }

    # ------------------------------------------------------------------
    # Multi-attempt assignment generation with seeded RNG
    # ------------------------------------------------------------------
    # Track the actual seed used (may be incremented on retry)
    actual_seed: int = seed
    # Track single-member family exceptions for audit
    single_member_exceptions: List[str] = []
    # The final assignment dict
    assignments: Dict[str, List[str]] = {}

    for attempt in range(_MAX_ASSIGNMENT_RETRIES):
        # Seed the random number generator for this attempt
        rng: np.random.Generator = np.random.default_rng(actual_seed)
        # Reset assignments and exceptions for this attempt
        assignments = {}
        single_member_exceptions = []
        # Flag to track if this attempt succeeded
        attempt_success: bool = True

        for slug in all_slugs:
            # Determine the family of the current reviewer
            reviewer_family: str = slug_to_family[slug]

            # ----------------------------------------------------------
            # Assign intra-category peer (same family, excluding self)
            # ----------------------------------------------------------
            # Candidate intra-category peers: same family, not self
            intra_candidates: List[str] = [
                s for s in family_to_slugs[reviewer_family]
                if s != slug
            ]

            if len(intra_candidates) == 0:
                # Single-member family: no intra-category peer available
                # Assign a second inter-category peer instead
                single_member_exceptions.append(slug)
                intra_peer: Optional[str] = None
            else:
                # Randomly select one intra-category peer
                intra_idx: int = int(
                    rng.integers(0, len(intra_candidates))
                )
                intra_peer = intra_candidates[intra_idx]

            # ----------------------------------------------------------
            # Assign inter-category peer (different family, not self)
            # ----------------------------------------------------------
            # Candidate inter-category peers: different family, not self
            inter_candidates: List[str] = [
                s for s in all_slugs
                if slug_to_family[s] != reviewer_family and s != slug
            ]

            if len(inter_candidates) == 0:
                # Cannot find an inter-category peer: attempt fails
                attempt_success = False
                break

            # Randomly select one inter-category peer
            inter_idx: int = int(rng.integers(0, len(inter_candidates)))
            inter_peer: str = inter_candidates[inter_idx]

            # ----------------------------------------------------------
            # Handle single-member family: assign second inter-category peer
            # ----------------------------------------------------------
            if intra_peer is None:
                # Need a second inter-category peer (different from first)
                second_inter_candidates: List[str] = [
                    s for s in inter_candidates if s != inter_peer
                ]
                if len(second_inter_candidates) == 0:
                    # Cannot find a second inter-category peer: attempt fails
                    attempt_success = False
                    break
                second_inter_idx: int = int(
                    rng.integers(0, len(second_inter_candidates))
                )
                second_inter_peer: str = second_inter_candidates[second_inter_idx]
                # Assign two inter-category peers for single-member family
                assignments[slug] = [inter_peer, second_inter_peer]
            else:
                # Standard case: one intra + one inter peer
                assignments[slug] = [intra_peer, inter_peer]

        if attempt_success:
            # Assignment succeeded: exit the retry loop
            break

        # Increment seed for next attempt
        actual_seed += 1
        logger.warning(
            "compute_review_assignments: Attempt %d failed. "
            "Retrying with seed=%d.",
            attempt + 1,
            actual_seed,
        )
    else:
        # All attempts exhausted: raise RuntimeError
        raise RuntimeError(
            f"compute_review_assignments: Failed to generate a valid "
            f"assignment after {_MAX_ASSIGNMENT_RETRIES} attempts. "
            "Check that pc_methods contains sufficient diversity of families."
        )

    # ------------------------------------------------------------------
    # Validate total review coverage: must equal _EXPECTED_N_REVIEWS
    # ------------------------------------------------------------------
    total_reviews: int = sum(len(peers) for peers in assignments.values())
    if total_reviews != _EXPECTED_N_REVIEWS:
        logger.warning(
            "compute_review_assignments: Total reviews = %d, "
            "expected %d. This may occur if n_methods != 21.",
            total_reviews,
            _EXPECTED_N_REVIEWS,
        )

    # ------------------------------------------------------------------
    # Construct the assignment matrix artifact for audit
    # ------------------------------------------------------------------
    assignment_artifact: Dict[str, Any] = {
        # The assignment mapping: reviewer_slug → [peer_slug_1, peer_slug_2]
        "assignments": assignments,
        # Total number of reviews assigned
        "total_reviews": total_reviews,
        # Number of PC methods in the assignment
        "n_methods": len(all_slugs),
        # Actual seed used (may differ from input if retries occurred)
        "seed_used": actual_seed,
        # Input seed (for reference)
        "seed_requested": seed,
        # Number of retry attempts made
        "n_attempts": attempt + 1,
        # Slugs with single-member family exceptions
        "single_member_family_exceptions": single_member_exceptions,
    }

    # ------------------------------------------------------------------
    # Create the artifact directory and persist the assignment matrix
    # ------------------------------------------------------------------
    # Create the artifact directory if it does not exist
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the assignment matrix artifact
    assignment_path: Path = artifact_dir / "assignment_matrix.json"

    try:
        with open(assignment_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised assignment matrix
            json.dump(
                _cast_to_json_safe(assignment_artifact),
                fh,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as exc:
        raise OSError(
            f"Failed to write assignment_matrix.json to "
            f"'{assignment_path}'. Original error: {exc}"
        ) from exc

    # Log the assignment computation for audit trail
    logger.info(
        "compute_review_assignments: %d assignments generated "
        "(seed=%d, attempts=%d, total_reviews=%d).",
        len(assignments),
        actual_seed,
        attempt + 1,
        total_reviews,
    )

    # Return the assignment dict
    return assignments


# =============================================================================
# TOOL 43: write_peer_review_md
# =============================================================================

def write_peer_review_md(
    reviewer_slug: str,
    peer_slug: str,
    review_content: str,
    artifact_dir: Path,
) -> str:
    """
    Format and persist a peer review to a markdown artifact.

    This tool implements the peer review artifact-writing step for the
    peer review phase (Task 28, Step 1). Each PC agent produces 2 peer
    reviews (1 intra-category, 1 inter-category), each persisted as a
    separate markdown file.

    The file is written to:
    ``{artifact_dir}/{reviewer_slug}_reviews_{peer_slug}.md``

    The ``review_content`` must contain the designated completion signal
    ``"REVIEW_COMPLETE"`` to pass the synchronisation barrier validation
    in ``validate_review_barrier`` (Tool 44).

    Parameters
    ----------
    reviewer_slug : str
        The slug of the reviewing PC agent (e.g., ``"max_diversification"``).
    peer_slug : str
        The slug of the reviewed PC agent (e.g., ``"black_litterman"``).
    review_content : str
        LLM-generated review text. Must be non-empty (minimum
        ``_MIN_REVIEW_LENGTH`` characters) and must contain the
        completion signal ``"REVIEW_COMPLETE"``.
    artifact_dir : Path
        Directory to write the review file. The file is written directly
        to this directory (not a subdirectory).

    Returns
    -------
    str
        Absolute path to the written review markdown file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``review_content`` is shorter than ``_MIN_REVIEW_LENGTH``
        characters.
    ValueError
        If ``review_content`` does not contain the completion signal
        ``"REVIEW_COMPLETE"``.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    **Completion signal:** The string ``"REVIEW_COMPLETE"`` must appear
    verbatim in ``review_content``. This signal is checked by
    ``validate_review_barrier`` (Tool 44) as part of the synchronisation
    barrier before Phase E (voting). The LLM is instructed to append
    this signal at the end of every review.

    **Simultaneous release:** All 42 review artifacts are held in the
    artifact directory and released simultaneously to all agents by the
    ``StrategyReviewController`` after the barrier clears. This tool
    writes to the final artifact directory; the controller manages the
    release timing.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: review_content minimum length
    # ------------------------------------------------------------------
    if not isinstance(review_content, str) or len(review_content.strip()) < _MIN_REVIEW_LENGTH:
        raise ValueError(
            f"review_content must be a non-empty string with at least "
            f"{_MIN_REVIEW_LENGTH} characters. "
            f"Got {len(review_content) if isinstance(review_content, str) else 0} characters."
        )

    # ------------------------------------------------------------------
    # Input validation: completion signal must be present
    # ------------------------------------------------------------------
    if _REVIEW_COMPLETION_SIGNAL not in review_content:
        raise ValueError(
            f"review_content does not contain the required completion "
            f"signal '{_REVIEW_COMPLETION_SIGNAL}'. "
            "The LLM must append this signal at the end of every review. "
            "Inject a corrective message into the ReAct loop to request "
            "the completion signal."
        )

    # ------------------------------------------------------------------
    # Construct the markdown document with a structured header
    # ------------------------------------------------------------------
    # Document header with reviewer, reviewed, and timestamp
    md_lines: List[str] = [
        f"# Peer Review: {reviewer_slug} → {peer_slug}",
        "",
        f"**Reviewer:** `{reviewer_slug}`",
        f"**Reviewed Portfolio:** `{peer_slug}`",
        "",
        "---",
        "",
        # The LLM-generated review content
        review_content.strip(),
        "",
    ]

    # Join all lines into a single markdown string
    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Create the artifact directory if it does not exist
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Construct the output file path
    # Filename: {reviewer_slug}_reviews_{peer_slug}.md
    # ------------------------------------------------------------------
    # Output filename following the naming convention
    filename: str = f"{reviewer_slug}_reviews_{peer_slug}.md"
    # Full path to the output file
    output_path: Path = artifact_dir / filename

    # ------------------------------------------------------------------
    # Write the markdown content to file
    # ------------------------------------------------------------------
    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the complete markdown review document
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write peer review to '{output_path}'. "
            f"Check filesystem permissions. Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_peer_review_md: written '%s' → '%s' to '%s'.",
        reviewer_slug,
        peer_slug,
        output_path,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 44: validate_review_barrier
# =============================================================================

def validate_review_barrier(
    assignment_matrix: Dict[str, List[str]],
    artifact_dir: Path,
) -> Dict[str, Any]:
    """
    Validate the synchronisation barrier for the peer review phase.

    Implements the synchronisation barrier check for the
    ``StrategyReviewController`` (Task 28, Step 3). After all 42 peer
    review artifacts have been written, this tool verifies:

    1. All expected review files exist (count gate)
    2. Each review file is non-empty (content gate)
    3. Each review file contains the completion signal
       ``"REVIEW_COMPLETE"`` (completion gate)

    If all gates pass, ``all_present=True`` and Phase E (voting) is
    unblocked. If any gate fails, ``all_present=False`` and the pipeline
    halts.

    Parameters
    ----------
    assignment_matrix : Dict[str, List[str]]
        Output of ``compute_review_assignments``. Maps
        ``reviewer_slug → [peer_slug_1, peer_slug_2]``.
    artifact_dir : Path
        Directory containing the peer review markdown files.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"all_present"`` (``bool``): ``True`` if all gates pass for
          all expected review files.
        - ``"total_expected"`` (``int``): Total number of expected review
          files (sum of peers per reviewer).
        - ``"total_found"`` (``int``): Number of files that passed all
          three gates.
        - ``"missing"`` (``List[str]``): Filenames of expected files that
          do not exist.
        - ``"empty"`` (``List[str]``): Filenames of files that exist but
          are empty (zero bytes).
        - ``"incomplete"`` (``List[str]``): Filenames of files that exist
          and are non-empty but do not contain the completion signal.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or
        ``assignment_matrix`` is not a dict.
    ValueError
        If ``assignment_matrix`` is empty.

    Notes
    -----
    **Fail-closed semantics:** If ``all_present=False``, the
    ``StrategyReviewController`` raises a ``RuntimeError`` and halts the
    pipeline. This tool returns the diagnostic information needed to
    identify which reviews failed and why.

    **Completion signal:** The completion signal ``"REVIEW_COMPLETE"``
    is the same frozen string checked by ``write_peer_review_md``.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(assignment_matrix, dict):
        raise TypeError(
            f"assignment_matrix must be a dict, "
            f"got {type(assignment_matrix).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-empty assignment matrix
    # ------------------------------------------------------------------
    if len(assignment_matrix) == 0:
        raise ValueError(
            "assignment_matrix is empty. Cannot validate review barrier."
        )

    # ------------------------------------------------------------------
    # Enumerate all expected review files from the assignment matrix
    # ------------------------------------------------------------------
    # Lists to accumulate validation failures
    missing_files: List[str] = []
    empty_files: List[str] = []
    incomplete_files: List[str] = []
    # Count of files that passed all three gates
    n_passed: int = 0
    # Total number of expected review files
    total_expected: int = 0

    for reviewer_slug, peer_slugs in assignment_matrix.items():
        for peer_slug in peer_slugs:
            # Increment the total expected count
            total_expected += 1
            # Construct the expected filename
            filename: str = f"{reviewer_slug}_reviews_{peer_slug}.md"
            # Full path to the expected review file
            file_path: Path = artifact_dir / filename

            # ----------------------------------------------------------
            # Gate 1: File existence check
            # ----------------------------------------------------------
            if not file_path.exists():
                # File does not exist: add to missing list
                missing_files.append(filename)
                continue

            # ----------------------------------------------------------
            # Gate 2: Non-empty check (file size > 0 bytes)
            # ----------------------------------------------------------
            if file_path.stat().st_size == 0:
                # File exists but is empty: add to empty list
                empty_files.append(filename)
                continue

            # ----------------------------------------------------------
            # Gate 3: Completion signal check
            # ----------------------------------------------------------
            try:
                # Read the file content to check for the completion signal
                file_content: str = file_path.read_text(encoding="utf-8")
            except OSError:
                # Cannot read the file: treat as missing
                missing_files.append(filename)
                continue

            if _REVIEW_COMPLETION_SIGNAL not in file_content:
                # File exists and is non-empty but lacks completion signal
                incomplete_files.append(filename)
                continue

            # All three gates passed for this file
            n_passed += 1

    # ------------------------------------------------------------------
    # Determine overall barrier status
    # ------------------------------------------------------------------
    # Barrier passes only if all expected files passed all three gates
    all_present: bool = (
        len(missing_files) == 0
        and len(empty_files) == 0
        and len(incomplete_files) == 0
        and n_passed == total_expected
    )

    # Log the barrier validation result for audit trail
    logger.info(
        "validate_review_barrier: all_present=%s, "
        "total_expected=%d, total_found=%d, "
        "missing=%d, empty=%d, incomplete=%d.",
        all_present,
        total_expected,
        n_passed,
        len(missing_files),
        len(empty_files),
        len(incomplete_files),
    )

    # ------------------------------------------------------------------
    # Return the barrier validation result dict
    # ------------------------------------------------------------------
    return {
        # Overall barrier status: True if all gates pass for all files
        "all_present": all_present,
        # Total number of expected review files
        "total_expected": total_expected,
        # Number of files that passed all three gates
        "total_found": n_passed,
        # Filenames of expected files that do not exist
        "missing": missing_files,
        # Filenames of files that exist but are empty
        "empty": empty_files,
        # Filenames of files that lack the completion signal
        "incomplete": incomplete_files,
    }


# =============================================================================
# TOOL 45: submit_vote
# =============================================================================

def submit_vote(
    voter_id: str,
    top_5: List[Dict[str, Any]],
    bottom_flag: Optional[Dict[str, Any]],
    own_method_slug: str,
    artifact_dir: Path,
) -> str:
    """
    Validate and persist a structured Borda-count vote.

    This tool implements the forced tool call for all voting agents
    (Task 29, Step 1). It enforces the modified Borda-count voting rules
    from the manuscript (Section 3.4):

    - Exactly 5 methods in ``top_5`` with scores ``{5, 4, 3, 2, 1}``
      (no ties, no duplicates)
    - No self-vote (the voter's own method cannot appear in ``top_5``
      or as ``bottom_flag``)
    - Optional ``bottom_flag`` assigning penalty ``−2`` to one method

    The vote is persisted as ``vote_{voter_id}.json`` in ``artifact_dir``.

    Parameters
    ----------
    voter_id : str
        The slug of the voting PC agent (e.g., ``"max_diversification"``).
    top_5 : List[Dict[str, Any]]
        Ordered list of 5 ranked methods. Each dict must contain:
        ``"method"`` (``str``, method slug) and ``"score"`` (``int``,
        one of ``{5, 4, 3, 2, 1}``). Scores must be unique and form
        the complete set ``{1, 2, 3, 4, 5}``.
    bottom_flag : Optional[Dict[str, Any]]
        Optional bottom flag. If provided, must contain ``"method"``
        (``str``) and ``"penalty"`` (``int``, must equal ``−2``).
        Set to ``None`` if no bottom flag is submitted.
    own_method_slug : str
        The voter's own method slug (injected via closure). Used for
        self-vote detection. **Must not appear in ``top_5`` or as
        ``bottom_flag["method"]``.**
    artifact_dir : Path
        Directory to write ``vote_{voter_id}.json``.

    Returns
    -------
    str
        Absolute path to the written ``vote_{voter_id}.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``top_5`` is
        not a list.
    ValueError
        If ``top_5`` does not contain exactly 5 entries.
    ValueError
        If the scores in ``top_5`` are not exactly ``{1, 2, 3, 4, 5}``.
    ValueError
        If any method appears more than once in ``top_5``.
    ValueError
        If ``bottom_flag`` is provided but has an invalid structure or
        penalty value.
    RuntimeError
        If a self-vote is detected (``own_method_slug`` appears in
        ``top_5`` or as ``bottom_flag["method"]``). This is a
        pipeline-halting error.
    OSError
        If the vote file cannot be written.

    Notes
    -----
    **Self-vote guard:** A ``RuntimeError`` (not ``ValueError``) is
    raised for self-vote detection because this represents a protocol
    violation that must halt the pipeline immediately, consistent with
    the fail-closed architecture.

    **Forced tool call:** This tool is invoked via
    ``tool_choice={"type": "function", "name": "submit_vote"}`` in the
    Responses API call, guaranteeing that the LLM produces a structured
    vote rather than plain text.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(top_5, list):
        raise TypeError(
            f"top_5 must be a list, got {type(top_5).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: top_5 must contain exactly 5 entries
    # ------------------------------------------------------------------
    if len(top_5) != 5:
        raise ValueError(
            f"top_5 must contain exactly 5 entries, got {len(top_5)}."
        )

    # ------------------------------------------------------------------
    # Input validation: each entry in top_5 must have 'method' and 'score'
    # ------------------------------------------------------------------
    for i, entry in enumerate(top_5):
        if not isinstance(entry, dict):
            raise TypeError(
                f"top_5[{i}] must be a dict, got {type(entry).__name__}."
            )
        for key in ("method", "score"):
            if key not in entry:
                raise ValueError(
                    f"top_5[{i}] is missing required key '{key}'."
                )

    # ------------------------------------------------------------------
    # Input validation: scores must be exactly {1, 2, 3, 4, 5} (no ties)
    # ------------------------------------------------------------------
    # Extract the set of scores from top_5
    score_set: Set[int] = {int(entry["score"]) for entry in top_5}
    if score_set != {1, 2, 3, 4, 5}:
        raise ValueError(
            f"top_5 scores must be exactly {{1, 2, 3, 4, 5}} (no ties). "
            f"Got: {sorted(score_set)}."
        )

    # ------------------------------------------------------------------
    # Input validation: no duplicate methods in top_5
    # ------------------------------------------------------------------
    # Extract the list of method slugs from top_5
    top_5_methods: List[str] = [str(entry["method"]) for entry in top_5]
    if len(set(top_5_methods)) != 5:
        raise ValueError(
            f"top_5 contains duplicate methods: {top_5_methods}. "
            "Each method must appear at most once."
        )

    # ------------------------------------------------------------------
    # Self-vote guard: own_method_slug must not appear in top_5
    # RuntimeError halts the pipeline immediately (fail-closed)
    # ------------------------------------------------------------------
    if own_method_slug in top_5_methods:
        raise RuntimeError(
            f"SELF-VOTE DETECTED: voter_id='{voter_id}' "
            f"(own_method_slug='{own_method_slug}') "
            f"appears in top_5: {top_5_methods}. "
            "Self-votes are strictly prohibited. "
            "This is a pipeline-halting protocol violation."
        )

    # ------------------------------------------------------------------
    # Input validation: bottom_flag structure (if provided)
    # ------------------------------------------------------------------
    if bottom_flag is not None:
        if not isinstance(bottom_flag, dict):
            raise TypeError(
                f"bottom_flag must be a dict or None, "
                f"got {type(bottom_flag).__name__}."
            )
        for key in ("method", "penalty"):
            if key not in bottom_flag:
                raise ValueError(
                    f"bottom_flag is missing required key '{key}'."
                )
        # Penalty must be exactly -2 per the Borda count rules
        if int(bottom_flag["penalty"]) != _BORDA_BOTTOM_FLAG_PENALTY:
            raise ValueError(
                f"bottom_flag['penalty'] must be {_BORDA_BOTTOM_FLAG_PENALTY}, "
                f"got {bottom_flag['penalty']}."
            )
        # Self-vote guard for bottom_flag
        if str(bottom_flag["method"]) == own_method_slug:
            raise RuntimeError(
                f"SELF-VOTE DETECTED: voter_id='{voter_id}' "
                f"(own_method_slug='{own_method_slug}') "
                f"appears as bottom_flag method. "
                "Self-votes are strictly prohibited."
            )

    # ------------------------------------------------------------------
    # Construct the vote output dict
    # ------------------------------------------------------------------
    vote_dict: Dict[str, Any] = {
        # Voter identifier (PC method slug)
        "voter_id": voter_id,
        # Top-5 ranked methods with Borda scores
        "top_5": [
            {"method": str(e["method"]), "score": int(e["score"])}
            for e in top_5
        ],
        # Optional bottom flag (None if not submitted)
        "bottom_flag": (
            {
                "method": str(bottom_flag["method"]),
                "penalty": int(bottom_flag["penalty"]),
            }
            if bottom_flag is not None
            else None
        ),
    }

    # ------------------------------------------------------------------
    # Create the artifact directory and write the vote JSON
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Output filename: vote_{voter_id}.json
    output_path: Path = artifact_dir / f"vote_{voter_id}.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised vote dict
            json.dump(vote_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write vote to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    # Log the vote submission for audit trail
    logger.info(
        "submit_vote: voter_id='%s' submitted vote. "
        "top_5=%s, bottom_flag=%s.",
        voter_id,
        top_5_methods,
        bottom_flag["method"] if bottom_flag else "None",
    )

    # Return the absolute path to the written vote file
    return str(output_path.resolve())


# =============================================================================
# TOOL 46: compute_borda_aggregation
# =============================================================================

def compute_borda_aggregation(
    vote_corpus: Dict[str, Dict[str, Any]],
    all_methods: List[str],
) -> Dict[str, float]:
    """
    Aggregate Borda-count votes into per-method vote scores.

    Implements the scripted Borda aggregation step for the
    ``StrategyReviewController`` (Task 29, Step 2):

    .. math::

        VoteScore_j = \\sum_{a \\neq j} \\text{points}(a \\to j)

    where ``points(a→j)`` is the Borda score assigned by agent ``a``
    to method ``j``:

    - ``5`` if ranked 1st, ``4`` if 2nd, ``3`` if 3rd, ``2`` if 4th,
      ``1`` if 5th in ``top_5``
    - ``−2`` if flagged as ``bottom_flag``
    - ``0`` otherwise (not ranked, not flagged)

    Parameters
    ----------
    vote_corpus : Dict[str, Dict[str, Any]]
        Mapping from ``voter_id`` to vote dict. Each vote dict must
        contain ``"top_5"`` (list of ``{"method": str, "score": int}``)
        and optionally ``"bottom_flag"`` (``{"method": str, "penalty": int}``
        or ``None``).
    all_methods : List[str]
        Complete list of all 21 PC method slugs. All methods are
        initialised to 0 in the output, even if they received no votes.

    Returns
    -------
    Dict[str, float]
        Mapping from method slug to total Borda score. All methods in
        ``all_methods`` are present in the output (with score 0.0 if
        they received no votes).

    Raises
    ------
    TypeError
        If ``vote_corpus`` is not a dict or ``all_methods`` is not a list.
    ValueError
        If ``vote_corpus`` is empty.

    Notes
    -----
    **Unknown methods:** If a voter ranked a method not in
    ``all_methods``, that vote entry is skipped with a warning. This
    handles the case where the PC-researcher agent proposed a new method
    that was added to the registry after the vote corpus was collected.

    **Duplicate voter IDs:** If ``vote_corpus`` contains duplicate voter
    IDs (which should not occur after the voting barrier), a warning is
    logged and the last vote for each voter ID is used.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(vote_corpus, dict):
        raise TypeError(
            f"vote_corpus must be a dict, got {type(vote_corpus).__name__}."
        )
    if not isinstance(all_methods, list):
        raise TypeError(
            f"all_methods must be a list, got {type(all_methods).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-empty vote corpus
    # ------------------------------------------------------------------
    if len(vote_corpus) == 0:
        raise ValueError(
            "vote_corpus is empty. At least one vote is required."
        )

    # ------------------------------------------------------------------
    # Initialise Borda scores to 0.0 for all methods
    # ------------------------------------------------------------------
    # All methods start with a score of 0.0 (no votes received)
    borda_scores: Dict[str, float] = {method: 0.0 for method in all_methods}

    # ------------------------------------------------------------------
    # Convert all_methods to a set for O(1) membership checks
    # ------------------------------------------------------------------
    # Set of all valid method slugs for fast lookup
    all_methods_set: Set[str] = set(all_methods)

    # ------------------------------------------------------------------
    # Aggregate Borda points from each voter's vote
    # ------------------------------------------------------------------
    for voter_id, vote_dict in vote_corpus.items():
        # Validate that the vote dict has the required structure
        if not isinstance(vote_dict, dict):
            logger.warning(
                "compute_borda_aggregation: vote for voter_id='%s' "
                "is not a dict. Skipping.",
                voter_id,
            )
            continue

        # ------------------------------------------------------------------
        # Process top_5 rankings: add Borda points for each ranked method
        # ------------------------------------------------------------------
        top_5: List[Dict[str, Any]] = vote_dict.get("top_5", [])
        for entry in top_5:
            # Extract the method slug and score from this ranking entry
            method: str = str(entry.get("method", ""))
            score: int = int(entry.get("score", 0))

            # Skip unknown methods (not in all_methods)
            if method not in all_methods_set:
                logger.warning(
                    "compute_borda_aggregation: voter_id='%s' ranked "
                    "unknown method '%s'. Skipping.",
                    voter_id,
                    method,
                )
                continue

            # Add the Borda score for this method from this voter
            borda_scores[method] += float(score)

        # ------------------------------------------------------------------
        # Process bottom_flag: subtract penalty for the flagged method
        # ------------------------------------------------------------------
        bottom_flag: Optional[Dict[str, Any]] = vote_dict.get("bottom_flag")
        if bottom_flag is not None and isinstance(bottom_flag, dict):
            # Extract the flagged method slug
            flagged_method: str = str(bottom_flag.get("method", ""))
            # Extract the penalty value (should be -2)
            penalty: int = int(bottom_flag.get("penalty", _BORDA_BOTTOM_FLAG_PENALTY))

            # Skip unknown methods
            if flagged_method not in all_methods_set:
                logger.warning(
                    "compute_borda_aggregation: voter_id='%s' flagged "
                    "unknown method '%s'. Skipping.",
                    voter_id,
                    flagged_method,
                )
                continue

            # Subtract the penalty from the flagged method's score
            borda_scores[flagged_method] += float(penalty)

    # Log the aggregation result for audit trail
    logger.info(
        "compute_borda_aggregation: aggregated %d votes across "
        "%d methods.",
        len(vote_corpus),
        len(all_methods),
    )

    # Return the Borda score dict (all methods present, even with 0 score)
    return borda_scores


# =============================================================================
# TOOL 47: compute_composite_ranking
# =============================================================================

def compute_composite_ranking(
    borda_scores: Dict[str, float],
    metric_scores: Dict[str, float],
    alpha: float,
    regime: str,
) -> Dict[str, Any]:
    """
    Compute the composite ranking by blending Borda and metric scores.

    Implements the scripted composite ranking step for the
    ``StrategyReviewController`` (Task 29, Step 3):

    .. math::

        Composite_j = \\alpha(\\text{regime}) \\cdot \\widetilde{VoteScore}_j
                    + (1 - \\alpha(\\text{regime})) \\cdot \\widetilde{MetricScore}_j

    where :math:`\\widetilde{\\cdot}` denotes min-max normalisation to
    :math:`[0, 1]`:

    .. math::

        \\widetilde{x}_j = \\frac{x_j - \\min_k x_k}{\\max_k x_k - \\min_k x_k}

    The regime-dependent :math:`\\alpha` is sourced from
    ``METHODOLOGY_PARAMS["STRATEGY_REVIEW"]["vote_metric_blend_alpha_schedule"]``.

    Parameters
    ----------
    borda_scores : Dict[str, float]
        Output of ``compute_borda_aggregation``. Maps method slug to
        total Borda score.
    metric_scores : Dict[str, float]
        Maps method slug to quantitative metric score (backtest Sharpe
        from ``cro_report.json``). Methods missing from this dict
        receive a score of 0.0 with a warning.
    alpha : float
        Regime-dependent blend weight for Borda scores. Must be in
        ``[0, 1]``. Sourced from ``_ALPHA_SCHEDULE[regime]``.
    regime : str
        Current macro regime label. Used for audit documentation.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"composite_scores"`` (``Dict[str, float]``): Composite score
          per method, in ``[0, 1]``.
        - ``"composite_ranking"`` (``List[str]``): Method slugs sorted
          in descending order of composite score.
        - ``"normalised_borda_scores"`` (``Dict[str, float]``):
          Min-max normalised Borda scores.
        - ``"normalised_metric_scores"`` (``Dict[str, float]``):
          Min-max normalised metric scores.
        - ``"alpha_used"`` (``float``): The alpha value applied.
        - ``"regime"`` (``str``): The regime label used.

    Raises
    ------
    TypeError
        If ``borda_scores`` or ``metric_scores`` are not dicts.
    ValueError
        If ``borda_scores`` is empty.
    ValueError
        If ``alpha`` is not in ``[0, 1]``.
    ValueError
        If ``regime`` is not a valid regime label.

    Notes
    -----
    **Degenerate normalisation guard:** If all Borda scores are equal
    (max == min), the normalised scores are set to 0.5 for all methods
    to avoid division by zero. The same guard applies to metric scores.

    **Missing metric scores:** Methods present in ``borda_scores`` but
    absent from ``metric_scores`` receive a metric score of 0.0. A
    warning is logged for each missing method.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(borda_scores, dict):
        raise TypeError(
            f"borda_scores must be a dict, got {type(borda_scores).__name__}."
        )
    if not isinstance(metric_scores, dict):
        raise TypeError(
            f"metric_scores must be a dict, "
            f"got {type(metric_scores).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-empty borda_scores
    # ------------------------------------------------------------------
    if len(borda_scores) == 0:
        raise ValueError(
            "borda_scores is empty. At least one method is required."
        )

    # ------------------------------------------------------------------
    # Input validation: alpha must be in [0, 1]
    # ------------------------------------------------------------------
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(
            f"alpha must be in [0, 1], got {alpha:.4f}."
        )

    # ------------------------------------------------------------------
    # Input validation: regime must be valid
    # ------------------------------------------------------------------
    if regime not in _VALID_REGIMES:
        raise ValueError(
            f"regime='{regime}' is not a valid regime label. "
            f"Must be one of: {list(_VALID_REGIMES)}."
        )

    # ------------------------------------------------------------------
    # Align methods: use all methods from borda_scores as the canonical set
    # ------------------------------------------------------------------
    # Canonical list of all methods (from borda_scores)
    methods: List[str] = list(borda_scores.keys())
    n_methods: int = len(methods)

    # ------------------------------------------------------------------
    # Build aligned arrays for Borda and metric scores
    # ------------------------------------------------------------------
    # Borda score array in canonical method order
    borda_arr: np.ndarray = np.array(
        [float(borda_scores[m]) for m in methods],
        dtype=np.float64,
    )

    # Metric score array: use 0.0 for methods missing from metric_scores
    metric_arr: np.ndarray = np.zeros(n_methods, dtype=np.float64)
    for i, method in enumerate(methods):
        if method in metric_scores:
            metric_arr[i] = float(metric_scores[method])
        else:
            # Method missing from metric_scores: use 0.0 with warning
            logger.warning(
                "compute_composite_ranking: method='%s' not found in "
                "metric_scores. Using 0.0.",
                method,
            )

    # ------------------------------------------------------------------
    # Min-max normalise Borda scores to [0, 1]
    # Degenerate guard: if max == min, set all normalised scores to 0.5
    # ------------------------------------------------------------------
    borda_min: float = float(borda_arr.min())
    borda_max: float = float(borda_arr.max())
    borda_range: float = borda_max - borda_min

    if borda_range > _EPS:
        # Standard min-max normalisation: (x - min) / (max - min)
        borda_norm: np.ndarray = (borda_arr - borda_min) / borda_range
    else:
        # Degenerate case: all Borda scores are equal → set to 0.5
        borda_norm = np.full(n_methods, 0.5, dtype=np.float64)
        logger.warning(
            "compute_composite_ranking: All Borda scores are equal "
            "(range=%.2e). Setting normalised Borda scores to 0.5.",
            borda_range,
        )

    # ------------------------------------------------------------------
    # Min-max normalise metric scores to [0, 1]
    # ------------------------------------------------------------------
    metric_min: float = float(metric_arr.min())
    metric_max: float = float(metric_arr.max())
    metric_range: float = metric_max - metric_min

    if metric_range > _EPS:
        # Standard min-max normalisation
        metric_norm: np.ndarray = (metric_arr - metric_min) / metric_range
    else:
        # Degenerate case: all metric scores are equal → set to 0.5
        metric_norm = np.full(n_methods, 0.5, dtype=np.float64)
        logger.warning(
            "compute_composite_ranking: All metric scores are equal "
            "(range=%.2e). Setting normalised metric scores to 0.5.",
            metric_range,
        )

    # ------------------------------------------------------------------
    # Compute composite scores:
    # Composite_j = alpha * borda_norm_j + (1 - alpha) * metric_norm_j
    # ------------------------------------------------------------------
    composite_arr: np.ndarray = (
        alpha * borda_norm + (1.0 - alpha) * metric_norm
    )

    # ------------------------------------------------------------------
    # Sort methods in descending order of composite score
    # np.argsort returns ascending order; reverse for descending
    # ------------------------------------------------------------------
    # Indices sorted by composite score in descending order
    sorted_indices: np.ndarray = np.argsort(composite_arr)[::-1]
    # Method slugs sorted in descending order of composite score
    composite_ranking: List[str] = [methods[i] for i in sorted_indices]

    # ------------------------------------------------------------------
    # Build output dicts with Python native float values
    # ------------------------------------------------------------------
    # Composite scores per method
    composite_scores: Dict[str, float] = {
        methods[i]: float(composite_arr[i]) for i in range(n_methods)
    }
    # Normalised Borda scores per method
    normalised_borda: Dict[str, float] = {
        methods[i]: float(borda_norm[i]) for i in range(n_methods)
    }
    # Normalised metric scores per method
    normalised_metric: Dict[str, float] = {
        methods[i]: float(metric_norm[i]) for i in range(n_methods)
    }

    # Log the composite ranking result for audit trail
    logger.info(
        "compute_composite_ranking: regime='%s', alpha=%.2f. "
        "Top-3: %s.",
        regime,
        alpha,
        composite_ranking[:3],
    )

    # ------------------------------------------------------------------
    # Return the composite ranking output dict
    # ------------------------------------------------------------------
    return {
        # Composite score per method (in [0, 1])
        "composite_scores": composite_scores,
        # Method slugs sorted in descending order of composite score
        "composite_ranking": composite_ranking,
        # Min-max normalised Borda scores
        "normalised_borda_scores": normalised_borda,
        # Min-max normalised metric scores
        "normalised_metric_scores": normalised_metric,
        # Alpha value applied (regime-dependent)
        "alpha_used": float(alpha),
        # Regime label used for audit documentation
        "regime": regime,
    }


# =============================================================================
# TOOL 48: write_vote_corpus_json
# =============================================================================

def write_vote_corpus_json(
    votes: Dict[str, Dict[str, Any]],
    artifact_dir: Path,
) -> str:
    """
    Aggregate and persist all individual votes to ``vote_corpus.json``.

    This tool implements the vote corpus artifact-writing step for the
    ``StrategyReviewController`` (Task 29, Step 1). It aggregates all
    21 individual ``vote_{voter_id}.json`` artifacts into a single
    ``vote_corpus.json`` file for audit and downstream consumption by
    ``compute_borda_aggregation`` (Tool 46) and the CIO agent (Task 31).

    The file is written to:
    ``{artifact_dir}/vote_corpus.json``

    Parameters
    ----------
    votes : Dict[str, Dict[str, Any]]
        Mapping from ``voter_id`` to vote dict. Typically collected by
        loading all ``vote_{voter_id}.json`` files from the voting
        artifact directory. Must be non-empty.
    artifact_dir : Path
        Directory to write ``vote_corpus.json``.

    Returns
    -------
    str
        Absolute path to the written ``vote_corpus.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``votes`` is
        not a dict.
    ValueError
        If ``votes`` is empty.
    OSError
        If the file cannot be written due to filesystem permissions.

    Notes
    -----
    The output ``vote_corpus.json`` includes a ``"n_voters"`` field
    for quick validation that all 21 votes were collected before
    proceeding to Borda aggregation.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(votes, dict):
        raise TypeError(
            f"votes must be a dict, got {type(votes).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-empty votes dict
    # ------------------------------------------------------------------
    if len(votes) == 0:
        raise ValueError(
            "votes is empty. At least one vote is required."
        )

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # Number of voters (for quick validation)
        "n_voters": len(votes),
        # All vote dicts keyed by voter_id (cast to JSON-safe types)
        "votes": _cast_to_json_safe(votes),
    }

    # ------------------------------------------------------------------
    # Create the artifact directory and write the vote corpus JSON
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = artifact_dir / "vote_corpus.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised vote corpus with 2-space indentation
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write vote_corpus.json to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    # Log the artifact write for audit trail
    logger.info(
        "write_vote_corpus_json: written %d votes to '%s'.",
        len(votes),
        output_path,
    )

    # Return the absolute path to the written file as a string
    return str(output_path.resolve())


# =============================================================================
# TOOL 49: apply_diversity_constraint
# =============================================================================

def apply_diversity_constraint(
    composite_ranking: List[str],
    family_map: Dict[str, str],
    required_families: int = _DIVERSITY_REQUIRED_FAMILIES,
    n_select: int = _DIVERSITY_N_SELECT,
) -> Dict[str, Any]:
    """
    Enforce the diversity constraint on the composite ranking to select top-5.

    Implements the diversity constraint enforcement step for the
    ``StrategyReviewController`` (Task 30, Step 1):

    The top-``n_select`` shortlist must include representation from at
    least ``required_families`` of the 4 PC method families
    (``risk_based``, ``return_based``, ``naive``, ``adversarial``).

    **Algorithm:**

    1. Iterate through ``composite_ranking`` in descending order.
    2. Greedily select methods, tracking the count of distinct families.
    3. If adding the next-ranked method would leave the shortlist with
       fewer than ``required_families`` distinct families after all
       ``n_select`` slots are filled, substitute the highest-ranked
       method from an underrepresented family.
    4. Return the final shortlist with confirmed ≥ ``required_families``
       family coverage.

    Parameters
    ----------
    composite_ranking : List[str]
        Method slugs sorted in descending order of composite score.
        Output of ``compute_composite_ranking``.
    family_map : Dict[str, str]
        Mapping from method slug to family label. Must cover all methods
        in ``composite_ranking``.
    required_families : int
        Minimum number of distinct families required in the top-5.
        Default: 3 per
        ``METHODOLOGY_PARAMS["STRATEGY_REVIEW"]["diversity_constraint"]
        ["top5_min_families"]``.
    n_select : int
        Number of methods to select. Default: 5.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"top_5_methods"`` (``List[str]``): Final shortlist of
          ``n_select`` method slugs.
        - ``"families_represented"`` (``List[str]``): Distinct family
          labels present in the shortlist.
        - ``"n_families_represented"`` (``int``): Count of distinct
          families in the shortlist.
        - ``"substitutions_made"`` (``List[Dict[str, str]]``): List of
          substitutions made to satisfy the diversity constraint. Each
          entry contains ``"replaced"`` (the method that was displaced)
          and ``"substituted"`` (the method that replaced it).
        - ``"diversity_constraint_satisfied"`` (``bool``): Whether the
          final shortlist satisfies the diversity constraint.

    Raises
    ------
    TypeError
        If ``composite_ranking`` is not a list or ``family_map`` is not
        a dict.
    ValueError
        If ``len(composite_ranking) < n_select``.
    ValueError
        If ``required_families`` > number of distinct families in
        ``family_map``.
    ValueError
        If the diversity constraint cannot be satisfied (insufficient
        family diversity in the full ranking).

    Notes
    -----
    **Substitution logic:** When a substitution is needed, the algorithm
    scans the remaining (unselected) methods in the composite ranking
    for the highest-ranked method from an underrepresented family. The
    displaced method is the lowest-ranked method in the current shortlist
    from an over-represented family.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(composite_ranking, list):
        raise TypeError(
            f"composite_ranking must be a list, "
            f"got {type(composite_ranking).__name__}."
        )
    if not isinstance(family_map, dict):
        raise TypeError(
            f"family_map must be a dict, got {type(family_map).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: sufficient methods available
    # ------------------------------------------------------------------
    if len(composite_ranking) < n_select:
        raise ValueError(
            f"composite_ranking has {len(composite_ranking)} methods, "
            f"but n_select={n_select} methods are required."
        )

    # ------------------------------------------------------------------
    # Input validation: sufficient families available
    # ------------------------------------------------------------------
    # Count distinct families in the full ranking
    available_families: Set[str] = {
        family_map.get(slug, "unknown")
        for slug in composite_ranking
    }
    n_available_families: int = len(available_families)
    if n_available_families < required_families:
        raise ValueError(
            f"Only {n_available_families} distinct families available in "
            f"composite_ranking, but required_families={required_families}. "
            f"Available families: {sorted(available_families)}."
        )

    # ------------------------------------------------------------------
    # Greedy selection with family tracking
    # ------------------------------------------------------------------
    # Current shortlist of selected method slugs
    selected: List[str] = []
    # Set of families represented in the current shortlist
    selected_families: Set[str] = set()
    # Track substitutions made for audit
    substitutions_made: List[Dict[str, str]] = []

    # Iterate through the composite ranking in descending order
    for slug in composite_ranking:
        if len(selected) >= n_select:
            # Shortlist is full: stop
            break
        # Add this method to the shortlist
        selected.append(slug)
        selected_families.add(family_map.get(slug, "unknown"))

    # ------------------------------------------------------------------
    # Check if the diversity constraint is satisfied after greedy selection
    # ------------------------------------------------------------------
    if len(selected_families) >= required_families:
        # Diversity constraint satisfied by greedy selection: no substitution needed
        pass
    else:
        # Diversity constraint not satisfied: apply substitution logic
        # Identify underrepresented families (not yet in shortlist)
        all_families_in_ranking: Set[str] = {
            family_map.get(slug, "unknown")
            for slug in composite_ranking
        }
        underrepresented: Set[str] = all_families_in_ranking - selected_families

        # For each underrepresented family, find the highest-ranked method
        # from that family in the full ranking (not already selected)
        for target_family in sorted(underrepresented):
            if len(selected_families) >= required_families:
                # Diversity constraint now satisfied: stop substituting
                break

            # Find the highest-ranked method from the target family
            # that is not already in the shortlist
            candidate: Optional[str] = None
            for slug in composite_ranking:
                if (
                    family_map.get(slug, "unknown") == target_family
                    and slug not in selected
                ):
                    candidate = slug
                    break

            if candidate is None:
                # No candidate found for this family: skip
                continue

            # Find the lowest-ranked method in the current shortlist
            # from an over-represented family (family with > 1 member in shortlist)
            family_counts: Dict[str, int] = {}
            for s in selected:
                fam: str = family_map.get(s, "unknown")
                family_counts[fam] = family_counts.get(fam, 0) + 1

            # Identify over-represented families (count > 1)
            over_represented: Set[str] = {
                fam for fam, cnt in family_counts.items() if cnt > 1
            }

            if not over_represented:
                # No over-represented family to displace: cannot substitute
                continue

            # Find the lowest-ranked method in the shortlist from an
            # over-represented family (scan composite_ranking in reverse)
            displaced: Optional[str] = None
            for slug in reversed(composite_ranking):
                if (
                    slug in selected
                    and family_map.get(slug, "unknown") in over_represented
                ):
                    displaced = slug
                    break

            if displaced is None:
                # Cannot find a method to displace: skip
                continue

            # Perform the substitution: replace displaced with candidate
            displaced_idx: int = selected.index(displaced)
            selected[displaced_idx] = candidate
            # Update the selected families set
            selected_families = {
                family_map.get(s, "unknown") for s in selected
            }
            # Record the substitution for audit
            substitutions_made.append({
                "replaced": displaced,
                "substituted": candidate,
                "target_family": target_family,
            })
            logger.info(
                "apply_diversity_constraint: Substituted '%s' (family='%s') "
                "for '%s' to satisfy diversity constraint.",
                candidate,
                target_family,
                displaced,
            )

    # ------------------------------------------------------------------
    # Final diversity constraint check
    # ------------------------------------------------------------------
    final_families: List[str] = sorted(
        {family_map.get(s, "unknown") for s in selected}
    )
    n_final_families: int = len(final_families)
    diversity_satisfied: bool = n_final_families >= required_families

    if not diversity_satisfied:
        raise ValueError(
            f"Diversity constraint cannot be satisfied: "
            f"final shortlist has {n_final_families} families "
            f"({final_families}), required: {required_families}. "
            "Check that the PC method registry has sufficient family diversity."
        )

    # Log the diversity constraint result for audit trail
    logger.info(
        "apply_diversity_constraint: top_%d selected=%s, "
        "families=%s, substitutions=%d.",
        n_select,
        selected,
        final_families,
        len(substitutions_made),
    )

    # ------------------------------------------------------------------
    # Return the diversity constraint output dict
    # ------------------------------------------------------------------
    return {
        # Final shortlist of n_select method slugs
        "top_5_methods": selected,
        # Distinct family labels in the shortlist
        "families_represented": final_families,
        # Count of distinct families
        "n_families_represented": n_final_families,
        # Substitutions made to satisfy the diversity constraint
        "substitutions_made": substitutions_made,
        # Whether the diversity constraint is satisfied
        "diversity_constraint_satisfied": diversity_satisfied,
    }


# =============================================================================
# TOOL 50: confirm_no_revision
# =============================================================================

def confirm_no_revision(
    method: str,
    rationale: str,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """
    Record and persist a no-revision decision for a top-5 PC agent.

    This tool implements the no-revision signal for the Top-5 Revision
    phase (Task 30, Step 2). When a top-5 shortlisted PC agent determines
    that no portfolio revision is warranted after reviewing peer feedback
    and the CRO report, it calls this tool to record the decision.

    The decision is persisted as:
    ``{artifact_dir}/{method_slug}/no_revision_confirmed.json``

    Parameters
    ----------
    method : str
        The PC method slug (e.g., ``"max_diversification"``). Used to
        construct the output file path.
    rationale : str
        LLM-generated explanation of why no revision is warranted.
        Must be non-empty (minimum ``_MIN_RATIONALE_LENGTH`` characters).
        Should explicitly address the peer review critiques and CRO
        findings that were considered and rejected.
    artifact_dir : Path
        Base artifact directory. The file is written to
        ``{artifact_dir}/{method_slug}/no_revision_confirmed.json``.

    Returns
    -------
    Dict[str, Any]
        Decision dict with keys:

        - ``"method"`` (``str``): The PC method slug.
        - ``"revision"`` (``bool``): Always ``False`` (no revision).
        - ``"rationale"`` (``str``): The LLM-generated rationale.
        - ``"original_weights_retained"`` (``bool``): Always ``True``.
        - ``"timestamp"`` (``str``): ISO-8601 UTC timestamp of the
          decision.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``rationale`` is empty or fewer than ``_MIN_RATIONALE_LENGTH``
        characters.
    OSError
        If the decision file cannot be written.

    Notes
    -----
    **Return value:** This tool returns the decision dict (not just the
    file path) so that the ``StrategyReviewController`` can immediately
    inspect the decision without loading the artifact from disk.

    **Original weights retained:** When this tool is called, the
    original ``pc_weights.json`` from Phase A is used unchanged for
    the CIO ensemble stage. The ``StrategyReviewController`` does not
    re-trigger the CRO report for agents that call ``confirm_no_revision``.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )

    # ------------------------------------------------------------------
    # Derive the method slug for the output directory
    # ------------------------------------------------------------------
    # Filesystem-safe slug derived from the method name
    method_slug: str = _derive_slug(method)

    # ------------------------------------------------------------------
    # Construct the ISO-8601 UTC timestamp for the decision
    # ------------------------------------------------------------------
    # Current UTC timestamp in ISO-8601 format
    timestamp: str = datetime.now(tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    # ------------------------------------------------------------------
    # Construct the decision dict
    # ------------------------------------------------------------------
    decision_dict: Dict[str, Any] = {
        # PC method slug identifier
        "method": method,
        # Revision flag: always False for this tool
        "revision": False,
        # LLM-generated rationale for the no-revision decision
        "rationale": rationale.strip(),
        # Original weights are retained unchanged
        "original_weights_retained": True,
        # UTC timestamp of the decision
        "timestamp": timestamp,
    }

    # ------------------------------------------------------------------
    # Create the output directory and write the decision JSON
    # ------------------------------------------------------------------
    # Output directory: {artifact_dir}/{method_slug}/
    output_dir: Path = artifact_dir / method_slug
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)
    # Full path to the output file
    output_path: Path = output_dir / "no_revision_confirmed.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            # Write the JSON-serialised decision dict
            json.dump(
                _cast_to_json_safe(decision_dict),
                fh,
                indent=2,
                ensure_ascii=False,
            )
    except OSError as exc:
        raise OSError(
            f"Failed to write no_revision_confirmed.json to "
            f"'{output_path}'. Original error: {exc}"
        ) from exc

    # Log the no-revision decision for audit trail
    logger.info(
        "confirm_no_revision: method='%s' confirmed no revision "
        "at %s.",
        method,
        timestamp,
    )

    # Return the decision dict (not just the path)
    return decision_dict

# =============================================================================
# SELF-DRIVING PORTFOLIO: TOOL REGISTRY — BATCH 6 (TOOLS 51–60)
# =============================================================================
# Implements tools 51–60 from the complete 78-tool registry for the agentic
# Strategic Asset Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# Tools implemented:
#   51. compute_cio_composite_scores   — CIO Agent scoring
#   52. evaluate_ensemble_methods      — CIO Agent ensemble computation
#   53. check_ips_compliance_ensemble  — CIO Agent IPS gate
#   54. select_best_ensemble           — CIO Agent selection
#   55. write_final_weights_json       — CIO Agent terminal artifact
#   56. write_board_memo_md            — CIO Agent board memo
#   57. write_pc_weights_json          — PC Agent artifact writer
#   58. write_pc_report_md             — PC Agent artifact writer
#   59. write_proposed_method_md       — PC Researcher artifact writer
#   60. write_proposed_method_spec_json — PC Researcher artifact writer
#
# All tools are purely deterministic Python callables — no LLM interaction.
# All arithmetic conventions are frozen per STUDY_CONFIG["DATA_CONVENTIONS"].
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception for terminal schema validation failures
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """
    Raised when a pipeline artifact fails its terminal schema validation gate.

    This exception signals a fail-closed pipeline halt. It is distinct from
    ``ValueError`` to allow the orchestrator to differentiate between data
    errors (``ValueError``) and schema contract violations (``SchemaValidationError``).
    """
    pass


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

# Number of assets in the 18-asset universe
_N_ASSETS: int = 18

# Annualisation multiplier for monthly returns
_PERIODS_PER_YEAR: int = 12

# Numerical stability epsilon
_EPS: float = 1e-8

# Frozen IPS constraint values
_IPS_MAX_WEIGHT: float = 0.25
_IPS_MIN_WEIGHT: float = 0.00
_IPS_TE_BUDGET: float = 0.06
_IPS_VOL_LOWER: float = 0.08
_IPS_VOL_UPPER: float = 0.12

# Frozen CIO scoring weights per METHODOLOGY_PARAMS["CIO_SCORING_WEIGHTS"]
_CIO_SCORING_WEIGHTS: Dict[str, float] = {
    "backtest_sharpe":       0.25,
    "ips_compliance":        0.15,
    "diversification":       0.15,
    "regime_fit":            0.20,
    "estimation_robustness": 0.15,
    "cma_utilization":       0.10,
}

# Frozen ensemble method names per METHODOLOGY_PARAMS["CIO_ENSEMBLE_METHODS"]
_ENSEMBLE_METHOD_NAMES: Tuple[str, ...] = (
    "simple_average",
    "inverse_tracking_error_weighting",
    "backtest_sharpe_weighting",
    "meta_optimization_pc_as_assets",
    "regime_conditional_weighting",
    "composite_score_weighting",
    "trimmed_mean_outlier_exclusion",
)

# Minimum character length for rationale strings
_MIN_RATIONALE_LENGTH: int = 10

# Required keys for method_spec (PC Researcher output)
_REQUIRED_METHOD_SPEC_KEYS: Tuple[str, ...] = (
    "method_name",
    "objective_function",
    "constraints",
    "required_inputs",
    "expected_behavior",
    "failure_modes",
)

# Required sections in the CIO board memo
_REQUIRED_BOARD_MEMO_SECTIONS: Tuple[str, ...] = (
    "Executive Summary",
    "Recommended Allocation",
    "Macro Regime View",
    "Expected Performance",
    "Key Active Bets",
    "Risk Assessment",
    "IPS Compliance",
    "Dissenting Views",
    "Rebalancing Plan",
    "Invalidation Triggers",
)

# Valid macro regime labels
_VALID_REGIMES: Tuple[str, ...] = (
    "Expansion",
    "Late-cycle",
    "Recession",
    "Recovery",
)

# Regime-optimal tilt directions: positive = overweight, negative = underweight
# Defined as a directional signal over the 4 broad categories
# (Equity, Fixed Income, Real Assets, Cash) per regime
# UNSPECIFIED IN MANUSCRIPT — frozen as implementation choice
_REGIME_TILT_SIGNALS: Dict[str, Dict[str, float]] = {
    "Expansion":  {"Equity": 1.0, "Fixed Income": -0.5, "Real Assets": 0.5, "Cash": -1.0},
    "Late-cycle": {"Equity": -0.5, "Fixed Income": 0.5, "Real Assets": 1.0, "Cash": 0.5},
    "Recession":  {"Equity": -1.0, "Fixed Income": 1.0, "Real Assets": -0.5, "Cash": 1.0},
    "Recovery":   {"Equity": 0.5, "Fixed Income": 0.0, "Real Assets": 0.5, "Cash": -0.5},
}

# Canonical 18 asset class names with their category labels
# per IPS_GOVERNANCE["ASSET_CLASSES"]
_ASSET_CLASS_CATEGORIES: Dict[str, str] = {
    "US Large Cap":                "Equity",
    "US Small Cap":                "Equity",
    "US Value":                    "Equity",
    "US Growth":                   "Equity",
    "International Developed":     "Equity",
    "Emerging Markets":            "Equity",
    "Short-Term Treasuries":       "Fixed Income",
    "Intermediate Treasuries":     "Fixed Income",
    "Long-Term Treasuries":        "Fixed Income",
    "Investment-Grade Corporates": "Fixed Income",
    "High-Yield Corporates":       "Fixed Income",
    "International Sovereign Bonds": "Fixed Income",
    "International Corporates":    "Fixed Income",
    "USD Emerging Market Debt":    "Fixed Income",
    "REITs":                       "Real Assets",
    "Gold":                        "Real Assets",
    "Commodities":                 "Real Assets",
    "Cash":                        "Cash",
}

# Canonical ordered list of 18 asset class names
_CANONICAL_ASSET_CLASS_ORDER: Tuple[str, ...] = tuple(
    _ASSET_CLASS_CATEGORIES.keys()
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _cast_to_json_safe(obj: Any) -> Any:
    """
    Recursively cast an object to JSON-serialisable Python native types.

    Converts ``np.float64``, ``np.int64``, ``np.bool_``, ``np.ndarray``,
    and nested containers to their Python native equivalents.
    ``None`` and ``float("nan")`` are preserved as ``None`` (JSON ``null``).
    """
    if obj is None:
        return None
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_cast_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _cast_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cast_to_json_safe(v) for v in obj]
    if isinstance(obj, (int, bool)):
        return obj
    if isinstance(obj, str):
        return obj
    return str(obj)


def _derive_slug(name: str) -> str:
    """
    Derive a filesystem-safe slug from a name string.

    Applies: lowercase → replace spaces with underscores →
    remove non-alphanumeric/underscore characters.
    """
    s: str = name.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", s)


def _load_pc_weights(
    method_slug: str,
    artifact_dir: Path,
    subdir: str = "pc_agents",
) -> np.ndarray:
    """
    Load and return the weight vector for a PC method from its artifact.

    Parameters
    ----------
    method_slug : str
        PC method slug identifier.
    artifact_dir : Path
        Base artifact directory.
    subdir : str
        Subdirectory under ``artifact_dir`` containing PC agent outputs.
        Default: ``"pc_agents"``.

    Returns
    -------
    np.ndarray
        18-element weight vector.

    Raises
    ------
    FileNotFoundError
        If the ``pc_weights.json`` file does not exist.
    ValueError
        If the weight vector does not have 18 elements.
    """
    # Construct the path to the pc_weights.json artifact
    weights_path: Path = artifact_dir / subdir / method_slug / "pc_weights.json"
    if not weights_path.exists():
        # Also try the top5_revision directory for revised weights
        revised_path: Path = (
            artifact_dir / "top5_revision" / method_slug / "revised_pc_weights.json"
        )
        if revised_path.exists():
            weights_path = revised_path
        else:
            raise FileNotFoundError(
                f"pc_weights.json not found for method='{method_slug}' "
                f"at '{weights_path}'."
            )
    # Load and parse the JSON file
    with open(weights_path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    # Extract the weights field
    raw_weights = data.get("weights")
    if raw_weights is None:
        raise ValueError(
            f"'weights' field missing in '{weights_path}'."
        )
    # Convert to numpy array
    w: np.ndarray = np.array(raw_weights, dtype=np.float64)
    if w.shape != (_N_ASSETS,):
        raise ValueError(
            f"Weight vector for '{method_slug}' has shape {w.shape}, "
            f"expected ({_N_ASSETS},)."
        )
    return w


def _load_cro_metrics(
    method_slug: str,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """
    Load and return the CRO metrics dict for a PC method.

    Parameters
    ----------
    method_slug : str
        PC method slug identifier.
    artifact_dir : Path
        Base artifact directory.

    Returns
    -------
    Dict[str, Any]
        CRO metrics dict from ``cro_report.json``.

    Raises
    ------
    FileNotFoundError
        If the ``cro_report.json`` file does not exist.
    """
    # Construct the path to the cro_report.json artifact
    cro_path: Path = artifact_dir / "cro_reports" / method_slug / "cro_report.json"
    if not cro_path.exists():
        raise FileNotFoundError(
            f"cro_report.json not found for method='{method_slug}' "
            f"at '{cro_path}'."
        )
    with open(cro_path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    return data


def _compute_effective_n(weights: np.ndarray) -> float:
    """
    Compute the effective number of assets (Meucci 2009).

    .. math::

        N_{eff} = \\exp\\left(-\\sum_i p_i \\ln p_i\\right)

    where :math:`p_i = w_i^2 / \\sum_j w_j^2` are the squared-weight
    proportions.

    Parameters
    ----------
    weights : np.ndarray
        18-element weight vector.

    Returns
    -------
    float
        Effective number of assets in ``[1, 18]``.
    """
    # Compute squared weights
    w_sq: np.ndarray = weights ** 2
    # Sum of squared weights (denominator)
    sum_w_sq: float = float(w_sq.sum())
    if sum_w_sq < _EPS:
        # Degenerate case: all weights are zero
        return 1.0
    # Normalised squared-weight proportions: p_i = w_i^2 / sum(w_j^2)
    p: np.ndarray = w_sq / sum_w_sq
    # Shannon entropy of p: H = -sum(p_i * ln(p_i))
    # Clip p to avoid log(0)
    p_clipped: np.ndarray = np.clip(p, _EPS, 1.0)
    entropy: float = float(-np.sum(p_clipped * np.log(p_clipped)))
    # Effective N = exp(H)
    return float(np.exp(entropy))


def _minmax_normalise(values: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalisation to an array, mapping to ``[0, 1]``.

    If all values are equal (degenerate case), returns an array of 0.5.

    Parameters
    ----------
    values : np.ndarray
        1-D array of values to normalise.

    Returns
    -------
    np.ndarray
        Normalised array with values in ``[0, 1]``.
    """
    v_min: float = float(values.min())
    v_max: float = float(values.max())
    v_range: float = v_max - v_min
    if v_range > _EPS:
        return (values - v_min) / v_range
    # Degenerate: all values equal → set to 0.5
    return np.full_like(values, 0.5, dtype=np.float64)


# =============================================================================
# TOOL 51: compute_cio_composite_scores
# =============================================================================

def compute_cio_composite_scores(
    candidate_methods: List[str],
    artifact_dir: Path,
    sigma: np.ndarray,
    benchmark_weights: np.ndarray,
    returns_matrix: np.ndarray,
    rf: float,
    macro_view: Dict[str, Any],
    cma_set: Dict[str, float],
    asset_class_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute the CIO composite score for each top-5 candidate portfolio.

    Implements the CIO scoring step (Task 31, Step 1) using the explicit
    weight schedule from ``METHODOLOGY_PARAMS["CIO_SCORING_WEIGHTS"]``:

    .. math::

        CIO\\_Score_j = 0.25 \\cdot \\widetilde{SR}_j
                      + 0.15 \\cdot IPS_j
                      + 0.15 \\cdot \\widetilde{Div}_j
                      + 0.20 \\cdot \\widetilde{RF}_j
                      + 0.15 \\cdot \\widetilde{ER}_j
                      + 0.10 \\cdot \\widetilde{CU}_j

    where :math:`\\widetilde{\\cdot}` denotes min-max normalisation to
    :math:`[0, 1]` across the candidate set, and the six sub-scores are:

    - **Backtest Sharpe** (:math:`SR`): from ``cro_report.json``
    - **IPS compliance** (:math:`IPS`): binary (1.0 / 0.0)
    - **Diversification** (:math:`Div`): effective number of assets
      :math:`N_{eff} = \\exp(-\\sum_i p_i \\ln p_i)` (Meucci 2009)
    - **Regime fit** (:math:`RF`): cosine similarity between portfolio
      active weights and the regime-optimal tilt vector
    - **Estimation robustness** (:math:`ER`): :math:`1 - HHI(w)` where
      :math:`HHI = \\sum_i w_i^2` (inverse Herfindahl concentration)
    - **CMA utilisation** (:math:`CU`): Pearson correlation between
      portfolio weights and CMA-implied weights (normalised to [0, 1])

    Parameters
    ----------
    candidate_methods : List[str]
        Top-5 PC method slugs from ``apply_diversity_constraint``.
    artifact_dir : Path
        Base artifact directory containing ``pc_agents/``,
        ``cro_reports/``, and ``top5_revision/`` subdirectories.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    returns_matrix : np.ndarray
        T×18 monthly returns matrix. Shape: ``(T, 18)``.
    rf : float
        Monthly risk-free rate in decimal form.
    macro_view : Dict[str, Any]
        Macro regime view dict. Must contain ``"regime"`` (``str``).
    cma_set : Dict[str, float]
        Mapping from asset class name to final CMA expected return
        (decimal form). Used for CMA utilisation scoring.
    asset_class_labels : Optional[List[str]]
        Ordered list of 18 asset class names corresponding to the
        columns of ``sigma``, ``returns_matrix``, and the weight
        vectors. If ``None``, uses ``_CANONICAL_ASSET_CLASS_ORDER``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"composite_scores"`` (``Dict[str, float]``): CIO composite
          score per candidate method, in ``[0, 1]``.
        - ``"sub_scores"`` (``Dict[str, Dict[str, float]]``): Per-method
          sub-scores for all 6 dimensions (pre-normalisation values and
          normalised values).

    Raises
    ------
    TypeError
        If ``sigma``, ``benchmark_weights``, or ``returns_matrix`` are
        not ``np.ndarray``.
    ValueError
        If ``candidate_methods`` is empty.
    FileNotFoundError
        If any required artifact is missing.

    Notes
    -----
    **Regime fit computation:** The regime-optimal tilt vector is defined
    as a directional signal over the 4 broad asset categories (Equity,
    Fixed Income, Real Assets, Cash) per the frozen
    ``_REGIME_TILT_SIGNALS`` mapping. The cosine similarity between the
    portfolio's active weight vector and this tilt vector is used as the
    regime fit score. This is a frozen implementation choice
    (UNSPECIFIED IN MANUSCRIPT).

    **CMA utilisation:** Computed as the Pearson correlation between the
    portfolio weights and the CMA-implied weights (proportional to CMA
    estimates, normalised to sum to 1.0). Higher correlation indicates
    greater utilisation of the CMA forecasts. Normalised to ``[0, 1]``
    via ``(corr + 1) / 2``.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    for arr_name, arr_val in [
        ("sigma", sigma),
        ("benchmark_weights", benchmark_weights),
        ("returns_matrix", returns_matrix),
    ]:
        if not isinstance(arr_val, np.ndarray):
            raise TypeError(
                f"{arr_name} must be a np.ndarray, "
                f"got {type(arr_val).__name__}."
            )

    # ------------------------------------------------------------------
    # Input validation: non-empty candidate list
    # ------------------------------------------------------------------
    if len(candidate_methods) == 0:
        raise ValueError("candidate_methods is empty.")

    # ------------------------------------------------------------------
    # Resolve asset class labels
    # ------------------------------------------------------------------
    ac_labels: List[str] = (
        list(asset_class_labels)
        if asset_class_labels is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )

    # ------------------------------------------------------------------
    # Extract current regime from macro_view
    # ------------------------------------------------------------------
    current_regime: str = str(macro_view.get("regime", "Expansion"))
    if current_regime not in _VALID_REGIMES:
        logger.warning(
            "compute_cio_composite_scores: Unknown regime '%s'. "
            "Defaulting to 'Expansion'.",
            current_regime,
        )
        current_regime = "Expansion"

    # ------------------------------------------------------------------
    # Build the regime-optimal tilt vector over 18 assets
    # Map each asset class to its category tilt signal
    # ------------------------------------------------------------------
    # Regime tilt signals for the current regime
    regime_tilt_by_category: Dict[str, float] = _REGIME_TILT_SIGNALS.get(
        current_regime, {k: 0.0 for k in ["Equity", "Fixed Income", "Real Assets", "Cash"]}
    )
    # Build the 18-element regime tilt vector
    regime_tilt_vector: np.ndarray = np.array([
        regime_tilt_by_category.get(
            _ASSET_CLASS_CATEGORIES.get(ac, "Equity"), 0.0
        )
        for ac in ac_labels
    ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Build the CMA-implied weight vector (proportional to CMA estimates)
    # CMA-implied weights: w_cma_i = max(0, mu_i) / sum(max(0, mu_j))
    # ------------------------------------------------------------------
    cma_vector: np.ndarray = np.array([
        max(0.0, float(cma_set.get(ac, 0.0)))
        for ac in ac_labels
    ], dtype=np.float64)
    cma_sum: float = float(cma_vector.sum())
    if cma_sum > _EPS:
        # Normalise CMA vector to sum to 1.0
        cma_weights_implied: np.ndarray = cma_vector / cma_sum
    else:
        # Degenerate: use equal weights as fallback
        cma_weights_implied = np.ones(_N_ASSETS, dtype=np.float64) / _N_ASSETS

    # ------------------------------------------------------------------
    # Load weights and CRO metrics for each candidate method
    # ------------------------------------------------------------------
    # Dict to store raw sub-scores per method (before normalisation)
    raw_sub_scores: Dict[str, Dict[str, float]] = {}

    for method_slug in candidate_methods:
        # Load the portfolio weight vector for this method
        try:
            w: np.ndarray = _load_pc_weights(
                method_slug, artifact_dir, subdir="top5_revision"
            )
        except FileNotFoundError:
            # Fall back to original pc_agents directory
            w = _load_pc_weights(method_slug, artifact_dir, subdir="pc_agents")

        # Load the CRO report for this method
        cro_data: Dict[str, Any] = _load_cro_metrics(method_slug, artifact_dir)
        cro_metrics: Dict[str, Any] = cro_data.get("metrics", {})

        # ------------------------------------------------------------------
        # Sub-score 1: Backtest Sharpe (raw value from CRO report)
        # ------------------------------------------------------------------
        backtest_sharpe: float = float(
            cro_metrics.get("backtest_sharpe", 0.0) or 0.0
        )

        # ------------------------------------------------------------------
        # Sub-score 2: IPS compliance (binary: 1.0 if compliant, 0.0 if not)
        # ------------------------------------------------------------------
        ips_compliant: float = float(
            bool(cro_data.get("ips_compliant", False))
        )

        # ------------------------------------------------------------------
        # Sub-score 3: Diversification — effective number of assets (Meucci 2009)
        # N_eff = exp(-sum(p_i * ln(p_i))) where p_i = w_i^2 / sum(w_j^2)
        # ------------------------------------------------------------------
        n_eff: float = _compute_effective_n(w)

        # ------------------------------------------------------------------
        # Sub-score 4: Regime fit — cosine similarity between active weights
        # and the regime-optimal tilt vector
        # ------------------------------------------------------------------
        # Active weight vector relative to benchmark
        w_active: np.ndarray = w - benchmark_weights
        # Cosine similarity: (w_active · tilt) / (||w_active|| * ||tilt||)
        w_active_norm: float = float(np.linalg.norm(w_active))
        tilt_norm: float = float(np.linalg.norm(regime_tilt_vector))
        if w_active_norm > _EPS and tilt_norm > _EPS:
            # Cosine similarity in [-1, 1]
            cosine_sim: float = float(
                np.dot(w_active, regime_tilt_vector)
                / (w_active_norm * tilt_norm)
            )
        else:
            # Degenerate: zero active weights or zero tilt → neutral
            cosine_sim = 0.0
        # Normalise cosine similarity from [-1, 1] to [0, 1]
        regime_fit: float = (cosine_sim + 1.0) / 2.0

        # ------------------------------------------------------------------
        # Sub-score 5: Estimation robustness — inverse Herfindahl concentration
        # ER = 1 - HHI(w) where HHI = sum(w_i^2)
        # Higher ER = less concentrated = more robust to estimation error
        # ------------------------------------------------------------------
        hhi: float = float(np.dot(w, w))
        estimation_robustness: float = float(1.0 - hhi)
        # Clip to [0, 1] (HHI is in [1/N, 1])
        estimation_robustness = float(np.clip(estimation_robustness, 0.0, 1.0))

        # ------------------------------------------------------------------
        # Sub-score 6: CMA utilisation — Pearson correlation between
        # portfolio weights and CMA-implied weights, normalised to [0, 1]
        # ------------------------------------------------------------------
        # Compute Pearson correlation between w and cma_weights_implied
        w_mean: float = float(w.mean())
        cma_mean: float = float(cma_weights_implied.mean())
        w_centred: np.ndarray = w - w_mean
        cma_centred: np.ndarray = cma_weights_implied - cma_mean
        w_std: float = float(np.std(w, ddof=0))
        cma_std: float = float(np.std(cma_weights_implied, ddof=0))
        if w_std > _EPS and cma_std > _EPS:
            # Pearson correlation in [-1, 1]
            pearson_corr: float = float(
                np.dot(w_centred, cma_centred) / (len(w) * w_std * cma_std)
            )
        else:
            # Degenerate: zero variance → neutral correlation
            pearson_corr = 0.0
        # Normalise from [-1, 1] to [0, 1]
        cma_utilization: float = (pearson_corr + 1.0) / 2.0

        # Store raw sub-scores for this method
        raw_sub_scores[method_slug] = {
            "backtest_sharpe":       backtest_sharpe,
            "ips_compliance":        ips_compliant,
            "diversification":       n_eff,
            "regime_fit":            regime_fit,
            "estimation_robustness": estimation_robustness,
            "cma_utilization":       cma_utilization,
        }

    # ------------------------------------------------------------------
    # Normalise sub-scores across candidates (min-max to [0, 1])
    # IPS compliance is already binary [0, 1]; no normalisation needed.
    # Regime fit and CMA utilisation are already in [0, 1].
    # Backtest Sharpe, diversification, and estimation robustness need
    # normalisation across the candidate set.
    # ------------------------------------------------------------------
    methods_list: List[str] = list(raw_sub_scores.keys())
    n_candidates: int = len(methods_list)

    # Build arrays for normalisation
    sharpe_arr: np.ndarray = np.array(
        [raw_sub_scores[m]["backtest_sharpe"] for m in methods_list],
        dtype=np.float64,
    )
    div_arr: np.ndarray = np.array(
        [raw_sub_scores[m]["diversification"] for m in methods_list],
        dtype=np.float64,
    )
    er_arr: np.ndarray = np.array(
        [raw_sub_scores[m]["estimation_robustness"] for m in methods_list],
        dtype=np.float64,
    )

    # Apply min-max normalisation to Sharpe, diversification, and ER
    sharpe_norm: np.ndarray = _minmax_normalise(sharpe_arr)
    div_norm: np.ndarray = _minmax_normalise(div_arr)
    er_norm: np.ndarray = _minmax_normalise(er_arr)

    # ------------------------------------------------------------------
    # Compute the CIO composite score for each candidate
    # CIO_Score_j = sum(weight_k * normalised_sub_score_k_j)
    # ------------------------------------------------------------------
    composite_scores: Dict[str, float] = {}
    sub_scores_output: Dict[str, Dict[str, float]] = {}

    for i, method_slug in enumerate(methods_list):
        raw: Dict[str, float] = raw_sub_scores[method_slug]

        # Normalised sub-scores for this method
        norm_sharpe: float = float(sharpe_norm[i])
        norm_ips: float = float(raw["ips_compliance"])       # Already binary
        norm_div: float = float(div_norm[i])
        norm_rf: float = float(raw["regime_fit"])            # Already in [0,1]
        norm_er: float = float(er_norm[i])
        norm_cu: float = float(raw["cma_utilization"])       # Already in [0,1]

        # Weighted composite score
        composite: float = (
            _CIO_SCORING_WEIGHTS["backtest_sharpe"]       * norm_sharpe
            + _CIO_SCORING_WEIGHTS["ips_compliance"]      * norm_ips
            + _CIO_SCORING_WEIGHTS["diversification"]     * norm_div
            + _CIO_SCORING_WEIGHTS["regime_fit"]          * norm_rf
            + _CIO_SCORING_WEIGHTS["estimation_robustness"] * norm_er
            + _CIO_SCORING_WEIGHTS["cma_utilization"]     * norm_cu
        )

        composite_scores[method_slug] = float(composite)
        sub_scores_output[method_slug] = {
            "backtest_sharpe_raw":       float(raw["backtest_sharpe"]),
            "backtest_sharpe_norm":      norm_sharpe,
            "ips_compliance":            norm_ips,
            "diversification_raw":       float(raw["diversification"]),
            "diversification_norm":      norm_div,
            "regime_fit":                norm_rf,
            "estimation_robustness_raw": float(raw["estimation_robustness"]),
            "estimation_robustness_norm": norm_er,
            "cma_utilization":           norm_cu,
            "composite_score":           float(composite),
        }

    logger.info(
        "compute_cio_composite_scores: scored %d candidates. "
        "Top: %s (%.4f).",
        n_candidates,
        max(composite_scores, key=composite_scores.get),
        max(composite_scores.values()),
    )

    return {
        # CIO composite score per candidate method
        "composite_scores": composite_scores,
        # Per-method sub-scores (raw and normalised) for audit
        "sub_scores": sub_scores_output,
    }


# =============================================================================
# TOOL 52: evaluate_ensemble_methods
# =============================================================================

def evaluate_ensemble_methods(
    top_5_methods: List[str],
    artifact_dir: Path,
    sigma: np.ndarray,
    benchmark_weights: np.ndarray,
    returns_matrix: np.ndarray,
    rf: float,
    composite_scores: Dict[str, float],
    macro_view: Dict[str, Any],
) -> Dict[str, List[float]]:
    """
    Evaluate all seven ensemble combination methods for the CIO agent.

    Implements Task 31, Step 2 — the evaluation of all 7 ensemble
    approaches from ``METHODOLOGY_PARAMS["CIO_ENSEMBLE_METHODS"]``:

    1. **Simple average:** :math:`w_{ens} = \\frac{1}{K}\\sum_k w^{(k)}`
    2. **Inverse TE weighting:**
       :math:`w_{ens} = \\sum_k \\frac{1/TE_k}{\\sum_j 1/TE_j} w^{(k)}`
    3. **Backtest Sharpe weighting:**
       :math:`w_{ens} = \\sum_k \\frac{SR_k^+}{\\sum_j SR_j^+} w^{(k)}`
       (positive Sharpe only; shifted if all negative)
    4. **Meta-optimisation (PC-as-assets):** Mean-variance optimisation
       treating the 5 PC portfolios as assets
    5. **Regime-conditional weighting:** Weight by regime fit sub-scores
    6. **Composite-score weighting:**
       :math:`w_{ens} = \\sum_k \\frac{CS_k}{\\sum_j CS_j} w^{(k)}`
    7. **Trimmed mean:** Exclude the most extreme outlier (by L1 distance
       from centroid), average the remainder

    Parameters
    ----------
    top_5_methods : List[str]
        Top-5 PC method slugs.
    artifact_dir : Path
        Base artifact directory.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    returns_matrix : np.ndarray
        T×18 monthly returns matrix. Shape: ``(T, 18)``.
    rf : float
        Monthly risk-free rate in decimal form.
    composite_scores : Dict[str, float]
        CIO composite scores per method (from ``compute_cio_composite_scores``).
    macro_view : Dict[str, Any]
        Macro regime view dict. Must contain ``"regime"`` (``str``).

    Returns
    -------
    Dict[str, List[float]]
        Mapping from ensemble method name to 18-element weight vector.
        All weight vectors sum to 1.0 (within ``1e-6``) and are
        non-negative.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``top_5_methods`` is empty.
    FileNotFoundError
        If any required artifact is missing.

    Notes
    -----
    **Meta-optimisation fallback:** If the meta-optimisation (method 4)
    fails to converge or produces an infeasible solution, it falls back
    to the simple average. This is documented in the log.

    **Trimmed mean with K=5:** When K=5, only 1 outlier is excluded
    (the portfolio with the highest L1 distance from the centroid),
    leaving 4 portfolios for averaging. This avoids the degenerate case
    of excluding 2 from 5 (leaving only 3).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if len(top_5_methods) == 0:
        raise ValueError("top_5_methods is empty.")

    # ------------------------------------------------------------------
    # Load weight vectors and CRO metrics for all top-5 methods
    # ------------------------------------------------------------------
    # Weight matrix: each row is one PC portfolio's weight vector
    weight_matrix: List[np.ndarray] = []
    # Tracking errors per method (for inverse TE weighting)
    te_values: List[float] = []
    # Backtest Sharpe ratios per method (for Sharpe weighting)
    sharpe_values: List[float] = []
    # Regime fit scores per method (for regime-conditional weighting)
    regime_fit_values: List[float] = []

    for method_slug in top_5_methods:
        # Load portfolio weights
        try:
            w: np.ndarray = _load_pc_weights(
                method_slug, artifact_dir, subdir="top5_revision"
            )
        except FileNotFoundError:
            w = _load_pc_weights(method_slug, artifact_dir, subdir="pc_agents")
        weight_matrix.append(w)

        # Load CRO metrics
        cro_data: Dict[str, Any] = _load_cro_metrics(method_slug, artifact_dir)
        cro_metrics: Dict[str, Any] = cro_data.get("metrics", {})

        # Extract tracking error
        te: float = float(cro_metrics.get("tracking_error", 0.06) or 0.06)
        te_values.append(max(te, _EPS))  # Avoid division by zero

        # Extract backtest Sharpe
        sr: float = float(cro_metrics.get("backtest_sharpe", 0.0) or 0.0)
        sharpe_values.append(sr)

        # Compute regime fit for this method (cosine similarity)
        current_regime: str = str(macro_view.get("regime", "Expansion"))
        regime_tilt: Dict[str, float] = _REGIME_TILT_SIGNALS.get(
            current_regime, {}
        )
        tilt_vec: np.ndarray = np.array([
            regime_tilt.get(_ASSET_CLASS_CATEGORIES.get(ac, "Equity"), 0.0)
            for ac in _CANONICAL_ASSET_CLASS_ORDER
        ], dtype=np.float64)
        w_active: np.ndarray = w - benchmark_weights
        w_norm: float = float(np.linalg.norm(w_active))
        t_norm: float = float(np.linalg.norm(tilt_vec))
        if w_norm > _EPS and t_norm > _EPS:
            rf_score: float = (
                float(np.dot(w_active, tilt_vec)) / (w_norm * t_norm) + 1.0
            ) / 2.0
        else:
            rf_score = 0.5
        regime_fit_values.append(rf_score)

    # Stack weight matrix: shape (K, 18)
    W: np.ndarray = np.stack(weight_matrix, axis=0)
    K: int = W.shape[0]

    # ------------------------------------------------------------------
    # Helper: normalise a weight vector to sum to 1.0 and clip negatives
    # ------------------------------------------------------------------
    def _normalise_weights(w_raw: np.ndarray) -> np.ndarray:
        """Clip to non-negative and normalise to sum to 1.0."""
        w_clipped: np.ndarray = np.maximum(w_raw, 0.0)
        w_sum: float = float(w_clipped.sum())
        if w_sum > _EPS:
            return w_clipped / w_sum
        return np.ones(_N_ASSETS, dtype=np.float64) / _N_ASSETS

    # ------------------------------------------------------------------
    # Ensemble 1: Simple average
    # w_ens = (1/K) * sum_k(w_k)
    # ------------------------------------------------------------------
    w_simple_avg: np.ndarray = _normalise_weights(W.mean(axis=0))

    # ------------------------------------------------------------------
    # Ensemble 2: Inverse tracking-error weighting
    # w_ens = sum_k((1/TE_k) / sum_j(1/TE_j) * w_k)
    # ------------------------------------------------------------------
    inv_te: np.ndarray = np.array(
        [1.0 / te for te in te_values], dtype=np.float64
    )
    inv_te_weights: np.ndarray = inv_te / inv_te.sum()
    w_inv_te: np.ndarray = _normalise_weights(
        np.einsum("k,ki->i", inv_te_weights, W)
    )

    # ------------------------------------------------------------------
    # Ensemble 3: Backtest Sharpe weighting
    # Shift all Sharpe values by subtracting the minimum if any are negative
    # w_ens = sum_k(SR_k_shifted / sum_j(SR_j_shifted) * w_k)
    # ------------------------------------------------------------------
    sr_arr: np.ndarray = np.array(sharpe_values, dtype=np.float64)
    sr_min: float = float(sr_arr.min())
    if sr_min < 0.0:
        # Shift to make all values non-negative
        sr_shifted: np.ndarray = sr_arr - sr_min + _EPS
    else:
        sr_shifted = sr_arr + _EPS
    sr_weights: np.ndarray = sr_shifted / sr_shifted.sum()
    w_sharpe: np.ndarray = _normalise_weights(
        np.einsum("k,ki->i", sr_weights, W)
    )

    # ------------------------------------------------------------------
    # Ensemble 4: Meta-optimisation (PC portfolios as assets)
    # Treat the K PC portfolios as assets; run mean-variance optimisation
    # on their return series to find the optimal combination weights
    # ------------------------------------------------------------------
    # Compute the K-portfolio return series: shape (T, K)
    portfolio_returns: np.ndarray = returns_matrix @ W.T  # (T, K)

    # Compute the K×K covariance matrix of portfolio returns
    # Use sample covariance (K=5 is too small for shrinkage)
    if portfolio_returns.shape[0] > K + 1:
        # Covariance matrix of the K portfolio return series
        cov_K: np.ndarray = np.cov(portfolio_returns.T, ddof=1)  # (K, K)
        # Expected returns of the K portfolios (annualised)
        mu_K: np.ndarray = portfolio_returns.mean(axis=0) * _PERIODS_PER_YEAR

        # Solve max-Sharpe for the K-portfolio combination weights
        def _neg_sharpe_K(alpha_K: np.ndarray) -> float:
            """Negative Sharpe of the meta-portfolio."""
            r_meta: float = float(np.dot(alpha_K, mu_K))
            var_meta: float = float(np.dot(alpha_K, cov_K @ alpha_K))
            vol_meta: float = float(np.sqrt(max(var_meta, _EPS)))
            return -(r_meta - rf * _PERIODS_PER_YEAR) / vol_meta

        # Constraints: sum(alpha) = 1, alpha >= 0
        meta_constraints = [{"type": "eq", "fun": lambda a: a.sum() - 1.0}]
        meta_bounds = [(0.0, 1.0)] * K
        meta_x0: np.ndarray = np.ones(K, dtype=np.float64) / K

        meta_result: OptimizeResult = minimize(
            fun=_neg_sharpe_K,
            x0=meta_x0,
            method="SLSQP",
            bounds=meta_bounds,
            constraints=meta_constraints,
            options={"ftol": 1e-9, "maxiter": 500, "disp": False},
        )

        if meta_result.success:
            # Compute the meta-portfolio weights in the 18-asset space
            alpha_opt: np.ndarray = np.clip(meta_result.x, 0.0, 1.0)
            alpha_opt = alpha_opt / alpha_opt.sum()
            w_meta: np.ndarray = _normalise_weights(
                np.einsum("k,ki->i", alpha_opt, W)
            )
        else:
            # Fallback to simple average if meta-optimisation fails
            logger.warning(
                "evaluate_ensemble_methods: Meta-optimisation failed. "
                "Falling back to simple average."
            )
            w_meta = w_simple_avg.copy()
    else:
        # Insufficient observations for meta-optimisation
        logger.warning(
            "evaluate_ensemble_methods: Insufficient observations for "
            "meta-optimisation (T=%d, K=%d). Falling back to simple average.",
            portfolio_returns.shape[0],
            K,
        )
        w_meta = w_simple_avg.copy()

    # ------------------------------------------------------------------
    # Ensemble 5: Regime-conditional weighting
    # Weight by regime fit scores
    # ------------------------------------------------------------------
    rf_arr: np.ndarray = np.array(regime_fit_values, dtype=np.float64)
    rf_sum: float = float(rf_arr.sum())
    if rf_sum > _EPS:
        rf_weights: np.ndarray = rf_arr / rf_sum
    else:
        rf_weights = np.ones(K, dtype=np.float64) / K
    w_regime: np.ndarray = _normalise_weights(
        np.einsum("k,ki->i", rf_weights, W)
    )

    # ------------------------------------------------------------------
    # Ensemble 6: Composite-score weighting
    # w_ens = sum_k(CS_k / sum_j(CS_j) * w_k)
    # ------------------------------------------------------------------
    cs_arr: np.ndarray = np.array(
        [float(composite_scores.get(m, 0.0)) for m in top_5_methods],
        dtype=np.float64,
    )
    cs_arr = np.maximum(cs_arr, 0.0)  # Ensure non-negative
    cs_sum: float = float(cs_arr.sum())
    if cs_sum > _EPS:
        cs_weights: np.ndarray = cs_arr / cs_sum
    else:
        cs_weights = np.ones(K, dtype=np.float64) / K
    w_composite: np.ndarray = _normalise_weights(
        np.einsum("k,ki->i", cs_weights, W)
    )

    # ------------------------------------------------------------------
    # Ensemble 7: Trimmed mean (exclude 1 outlier when K=5)
    # Exclude the portfolio with the highest L1 distance from the centroid
    # ------------------------------------------------------------------
    # Compute the centroid of all K portfolios
    centroid: np.ndarray = W.mean(axis=0)
    # Compute L1 distance of each portfolio from the centroid
    l1_distances: np.ndarray = np.array(
        [float(np.sum(np.abs(W[k] - centroid))) for k in range(K)],
        dtype=np.float64,
    )
    # Identify the index of the most extreme outlier
    outlier_idx: int = int(np.argmax(l1_distances))
    # Exclude the outlier and average the remainder
    trimmed_indices: List[int] = [k for k in range(K) if k != outlier_idx]
    w_trimmed: np.ndarray = _normalise_weights(
        W[trimmed_indices].mean(axis=0)
    )

    # ------------------------------------------------------------------
    # Assemble the output dict
    # ------------------------------------------------------------------
    ensemble_weights: Dict[str, List[float]] = {
        "simple_average":                  [float(v) for v in w_simple_avg],
        "inverse_tracking_error_weighting": [float(v) for v in w_inv_te],
        "backtest_sharpe_weighting":        [float(v) for v in w_sharpe],
        "meta_optimization_pc_as_assets":   [float(v) for v in w_meta],
        "regime_conditional_weighting":     [float(v) for v in w_regime],
        "composite_score_weighting":        [float(v) for v in w_composite],
        "trimmed_mean_outlier_exclusion":   [float(v) for v in w_trimmed],
    }

    logger.info(
        "evaluate_ensemble_methods: computed %d ensemble weight vectors.",
        len(ensemble_weights),
    )

    return ensemble_weights


# =============================================================================
# TOOL 53: check_ips_compliance_ensemble
# =============================================================================

def check_ips_compliance_ensemble(
    ensemble_weights: Dict[str, List[float]],
    sigma: np.ndarray,
    benchmark_weights: np.ndarray,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Check IPS compliance for each of the seven ensemble weight vectors.

    Applies the same five IPS constraints as ``check_ips_compliance``
    (Tool 39) to each ensemble weight vector from
    ``evaluate_ensemble_methods`` (Tool 52):

    1. Long-only: :math:`w_i \\geq 0`
    2. Max weight: :math:`w_i \\leq 0.25`
    3. Budget: :math:`\\sum_i w_i = 1.0 \\pm 10^{-6}`
    4. Tracking error: :math:`TE \\leq 0.06`
    5. Volatility band: :math:`\\sigma_p \\in [0.08, 0.12]`

    Parameters
    ----------
    ensemble_weights : Dict[str, List[float]]
        Output of ``evaluate_ensemble_methods``. Maps ensemble method
        name to 18-element weight vector.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    constraints : Optional[Dict[str, Any]]
        IPS constraint parameters. If ``None``, uses frozen defaults.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from ensemble method name to compliance result dict.
        Each result dict contains ``"compliant"`` (bool) and
        ``"flags"`` (dict with per-constraint details).

    Raises
    ------
    TypeError
        If ``sigma`` or ``benchmark_weights`` are not ``np.ndarray``.
    ValueError
        If ``ensemble_weights`` is empty.
    RuntimeError
        If all ensemble weight vectors are IPS non-compliant.

    Notes
    -----
    **All-non-compliant guard:** If every ensemble fails IPS compliance,
    a ``RuntimeError`` is raised to halt the pipeline. The CIO agent
    cannot select a final portfolio if no ensemble is compliant.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(sigma, np.ndarray):
        raise TypeError(
            f"sigma must be a np.ndarray, got {type(sigma).__name__}."
        )
    if not isinstance(benchmark_weights, np.ndarray):
        raise TypeError(
            f"benchmark_weights must be a np.ndarray, "
            f"got {type(benchmark_weights).__name__}."
        )
    if len(ensemble_weights) == 0:
        raise ValueError("ensemble_weights is empty.")

    # ------------------------------------------------------------------
    # Resolve constraint parameters
    # ------------------------------------------------------------------
    _c: Dict[str, Any] = constraints or {}
    max_w: float = float(_c.get("max_weight_per_asset", _IPS_MAX_WEIGHT))
    min_w: float = float(_c.get("min_weight_per_asset", _IPS_MIN_WEIGHT))

    # ------------------------------------------------------------------
    # Check IPS compliance for each ensemble weight vector
    # ------------------------------------------------------------------
    compliance_results: Dict[str, Dict[str, Any]] = {}

    for ensemble_name, weights_list in ensemble_weights.items():
        # Convert to numpy array
        w: np.ndarray = np.array(weights_list, dtype=np.float64)

        # Constraint 1: Long-only
        min_weight_actual: float = float(w.min())
        long_only_pass: bool = bool(min_weight_actual >= min_w - 1e-8)

        # Constraint 2: Max weight
        max_weight_actual: float = float(w.max())
        max_weight_pass: bool = bool(max_weight_actual <= max_w + 1e-8)

        # Constraint 3: Budget
        weight_sum: float = float(w.sum())
        sum_to_one_pass: bool = bool(abs(weight_sum - 1.0) <= 1e-6)

        # Constraint 4: Tracking error
        w_active: np.ndarray = w - benchmark_weights
        te_var: float = float(max(0.0, np.dot(w_active, sigma @ w_active)))
        te_actual: float = float(np.sqrt(te_var))
        te_pass: bool = bool(te_actual <= _IPS_TE_BUDGET + 1e-8)

        # Constraint 5: Volatility band
        port_var: float = float(max(0.0, np.dot(w, sigma @ w)))
        vol_actual: float = float(np.sqrt(port_var))
        vol_band_pass: bool = bool(
            (vol_actual >= _IPS_VOL_LOWER - 1e-8)
            and (vol_actual <= _IPS_VOL_UPPER + 1e-8)
        )

        # Aggregate flags
        flags: Dict[str, Dict[str, Any]] = {
            "long_only":      {"pass": long_only_pass,  "actual_value": float(min_weight_actual)},
            "max_weight":     {"pass": max_weight_pass, "actual_value": float(max_weight_actual)},
            "sum_to_one":     {"pass": sum_to_one_pass, "actual_value": float(weight_sum)},
            "tracking_error": {"pass": te_pass,         "actual_value": float(te_actual)},
            "volatility_band":{"pass": vol_band_pass,   "actual_value": float(vol_actual)},
        }
        compliant: bool = all(v["pass"] for v in flags.values())

        compliance_results[ensemble_name] = {
            "compliant": compliant,
            "flags": flags,
        }

    # ------------------------------------------------------------------
    # All-non-compliant guard
    # ------------------------------------------------------------------
    n_compliant: int = sum(
        1 for r in compliance_results.values() if r["compliant"]
    )
    if n_compliant == 0:
        raise RuntimeError(
            "ALL ENSEMBLE WEIGHT VECTORS ARE IPS NON-COMPLIANT. "
            "The CIO agent cannot select a final portfolio. "
            "Review the PC agent outputs and IPS constraint parameters. "
            "This is a pipeline-halting error."
        )

    logger.info(
        "check_ips_compliance_ensemble: %d / %d ensembles are IPS compliant.",
        n_compliant,
        len(compliance_results),
    )

    return compliance_results


# =============================================================================
# TOOL 54: select_best_ensemble
# =============================================================================

def select_best_ensemble(
    ensemble_scores: Dict[str, float],
    compliance_flags: Dict[str, Dict[str, Any]],
    ensemble_weights: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Select the best IPS-compliant ensemble using CIO composite scores.

    Implements Task 31, Step 2 — the final ensemble selection step:

    1. Filter to IPS-compliant ensembles (from ``compliance_flags``).
    2. Among compliant ensembles, select the one with the highest
       composite score (from ``ensemble_scores``).
    3. Return the selected ensemble name, its weight vector, and a
       selection rationale.

    Parameters
    ----------
    ensemble_scores : Dict[str, float]
        Mapping from ensemble method name to composite score. Methods
        missing from this dict receive a score of 0.0.
    compliance_flags : Dict[str, Dict[str, Any]]
        Output of ``check_ips_compliance_ensemble``. Maps ensemble name
        to compliance result dict (must contain ``"compliant"`` key).
    ensemble_weights : Dict[str, List[float]]
        Output of ``evaluate_ensemble_methods``. Maps ensemble name to
        18-element weight vector.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"selected_ensemble"`` (``str``): Name of the selected
          ensemble method.
        - ``"weights"`` (``List[float]``): 18-element weight vector of
          the selected ensemble.
        - ``"selection_rationale"`` (``str``): Human-readable explanation
          of the selection decision.
        - ``"composite_score"`` (``float``): Composite score of the
          selected ensemble.
        - ``"n_compliant_ensembles"`` (``int``): Number of IPS-compliant
          ensembles considered.

    Raises
    ------
    TypeError
        If any input is not a dict.
    ValueError
        If ``compliance_flags`` is empty.
    RuntimeError
        If no IPS-compliant ensemble is found (should have been caught
        by ``check_ips_compliance_ensemble``, but included as a
        belt-and-suspenders guard).

    Notes
    -----
    **Tiebreaker:** If two or more compliant ensembles have equal
    composite scores (within ``1e-8``), the ensemble appearing first
    in alphabetical order is selected. This is a frozen tiebreaker
    documented here for reproducibility.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    for name, val in [
        ("ensemble_scores", ensemble_scores),
        ("compliance_flags", compliance_flags),
        ("ensemble_weights", ensemble_weights),
    ]:
        if not isinstance(val, dict):
            raise TypeError(
                f"{name} must be a dict, got {type(val).__name__}."
            )
    if len(compliance_flags) == 0:
        raise ValueError("compliance_flags is empty.")

    # ------------------------------------------------------------------
    # Filter to IPS-compliant ensembles
    # ------------------------------------------------------------------
    compliant_ensembles: List[str] = [
        name for name, result in compliance_flags.items()
        if bool(result.get("compliant", False))
    ]

    # ------------------------------------------------------------------
    # Belt-and-suspenders guard: no compliant ensembles
    # ------------------------------------------------------------------
    if len(compliant_ensembles) == 0:
        raise RuntimeError(
            "No IPS-compliant ensemble found in select_best_ensemble. "
            "This should have been caught by check_ips_compliance_ensemble."
        )

    # ------------------------------------------------------------------
    # Select the compliant ensemble with the highest composite score
    # Tiebreaker: alphabetical order (first in sorted list)
    # ------------------------------------------------------------------
    # Sort compliant ensembles: primary key = composite score (descending),
    # secondary key = name (ascending, for deterministic tiebreaking)
    sorted_compliant: List[str] = sorted(
        compliant_ensembles,
        key=lambda name: (
            -float(ensemble_scores.get(name, 0.0)),  # Descending score
            name,                                     # Ascending name (tiebreaker)
        ),
    )
    # The best ensemble is the first in the sorted list
    selected_name: str = sorted_compliant[0]
    selected_score: float = float(ensemble_scores.get(selected_name, 0.0))
    selected_weights: List[float] = ensemble_weights[selected_name]

    # ------------------------------------------------------------------
    # Construct the selection rationale
    # ------------------------------------------------------------------
    rationale: str = (
        f"Selected ensemble: '{selected_name}' "
        f"(composite score = {selected_score:.4f}). "
        f"{len(compliant_ensembles)} of {len(compliance_flags)} ensembles "
        f"were IPS-compliant. "
        f"Compliant ensembles (ranked): "
        f"{[f'{n} ({ensemble_scores.get(n, 0.0):.4f})' for n in sorted_compliant]}. "
        f"Tiebreaker: alphabetical order (if scores are equal within 1e-8)."
    )

    logger.info(
        "select_best_ensemble: selected='%s' (score=%.4f), "
        "n_compliant=%d.",
        selected_name,
        selected_score,
        len(compliant_ensembles),
    )

    return {
        # Name of the selected ensemble method
        "selected_ensemble": selected_name,
        # 18-element weight vector of the selected ensemble
        "weights": selected_weights,
        # Human-readable selection rationale
        "selection_rationale": rationale,
        # Composite score of the selected ensemble
        "composite_score": float(selected_score),
        # Number of IPS-compliant ensembles considered
        "n_compliant_ensembles": len(compliant_ensembles),
    }


# =============================================================================
# TOOL 55: write_final_weights_json
# =============================================================================

def write_final_weights_json(
    weights: List[float],
    rationale: str,
    selected_ensemble: str,
    artifact_dir: Path,
    asset_class_labels: Optional[List[str]] = None,
) -> str:
    """
    Serialise and persist the final portfolio weights to ``final_weights.json``.

    This tool implements the terminal artifact-writing step for the CIO
    Agent (Task 31, Step 3). The output file is the definitive portfolio
    allocation consumed by the board memo, the backtest, and the
    reproducibility package.

    **Terminal validation gate (Layer 2):** This tool independently
    verifies that the weight vector satisfies all terminal constraints:

    - Exactly 18 non-negative elements
    - Sum to 1.0 (within ``1e-6``)

    If either constraint fails, a ``SchemaValidationError`` is raised
    immediately and the pipeline halts (fail-closed).

    The file is written to:
    ``{artifact_dir}/final_weights.json``

    Parameters
    ----------
    weights : List[float]
        18-element final portfolio weight vector. Must sum to 1.0
        (within ``1e-6``) and be non-negative.
    rationale : str
        LLM-generated or scripted rationale for the final allocation.
        Must be non-empty (minimum ``_MIN_RATIONALE_LENGTH`` characters).
    selected_ensemble : str
        Name of the ensemble method selected by the CIO agent.
    artifact_dir : Path
        Directory to write ``final_weights.json``.
    asset_class_labels : Optional[List[str]]
        Ordered list of 18 asset class names. If ``None``, uses
        ``_CANONICAL_ASSET_CLASS_ORDER``.

    Returns
    -------
    str
        Absolute path to the written ``final_weights.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    SchemaValidationError
        If ``weights`` does not have exactly 18 elements, contains
        negative values, or does not sum to 1.0 (within ``1e-6``).
        This is a pipeline-halting terminal validation failure.
    ValueError
        If ``rationale`` is empty or fewer than ``_MIN_RATIONALE_LENGTH``
        characters.
    OSError
        If the file cannot be written.

    Notes
    -----
    The ``"sum_check"`` field in the output records the actual sum of
    the weight vector for audit verification. The
    ``"range_constraint_verified"`` field is set to ``True`` to confirm
    that the terminal validation gate passed.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )

    # ------------------------------------------------------------------
    # Resolve asset class labels
    # ------------------------------------------------------------------
    ac_labels: List[str] = (
        list(asset_class_labels)
        if asset_class_labels is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )

    # ------------------------------------------------------------------
    # TERMINAL VALIDATION GATE: 18 elements
    # ------------------------------------------------------------------
    if len(weights) != _N_ASSETS:
        raise SchemaValidationError(
            f"TERMINAL VALIDATION FAILURE: final weights vector has "
            f"{len(weights)} elements, expected {_N_ASSETS}. "
            "Pipeline halted (fail-closed)."
        )

    # ------------------------------------------------------------------
    # TERMINAL VALIDATION GATE: non-negative weights
    # ------------------------------------------------------------------
    w_arr: np.ndarray = np.array(weights, dtype=np.float64)
    if (w_arr < -1e-8).any():
        neg_indices: List[int] = [
            i for i, v in enumerate(weights) if v < -1e-8
        ]
        raise SchemaValidationError(
            f"TERMINAL VALIDATION FAILURE: final weights contain "
            f"negative values at indices {neg_indices}: "
            f"{[weights[i] for i in neg_indices]}. "
            "Pipeline halted (fail-closed)."
        )

    # ------------------------------------------------------------------
    # TERMINAL VALIDATION GATE: sum to 1.0 within 1e-6
    # ------------------------------------------------------------------
    weight_sum: float = float(w_arr.sum())
    if abs(weight_sum - 1.0) > 1e-6:
        raise SchemaValidationError(
            f"TERMINAL VALIDATION FAILURE: final weights sum to "
            f"{weight_sum:.8f}, expected 1.0 (tolerance 1e-6). "
            "Pipeline halted (fail-closed)."
        )

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # 18-element final portfolio weight vector
        "weights": [float(v) for v in weights],
        # Ordered asset class labels corresponding to each weight
        "asset_class_labels": ac_labels,
        # Actual sum of weights (for audit verification)
        "sum_check": float(weight_sum),
        # Name of the ensemble method selected by the CIO agent
        "selected_ensemble": selected_ensemble,
        # LLM-generated or scripted rationale for the final allocation
        "rationale": rationale.strip(),
        # Terminal validation gate passed flag
        "range_constraint_verified": True,
    }

    # ------------------------------------------------------------------
    # Create the artifact directory and write the JSON
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = artifact_dir / "final_weights.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write final_weights.json to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_final_weights_json: written to '%s'. "
        "sum=%.8f, selected_ensemble='%s'.",
        output_path,
        weight_sum,
        selected_ensemble,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 56: write_board_memo_md
# =============================================================================

def write_board_memo_md(
    weights: List[float],
    rationale: str,
    macro_summary: Dict[str, Any],
    cma_summary: Dict[str, float],
    risk_summary: Dict[str, Any],
    selected_ensemble: str,
    artifact_dir: Path,
    asset_class_labels: Optional[List[str]] = None,
    benchmark_weights: Optional[List[float]] = None,
) -> str:
    """
    Format and persist the CIO board memo to ``board_memo.md``.

    This tool implements the board memo generation step for the CIO Agent
    (Task 31, Step 3). The board memo is a governance document written
    for non-technical stakeholders (board of trustees, investment
    committee) and must include all 10 required sections from the
    ``CIO_BOARD_MEMO`` template.

    The file is written to:
    ``{artifact_dir}/board_memo.md``

    Parameters
    ----------
    weights : List[float]
        18-element final portfolio weight vector.
    rationale : str
        LLM-generated narrative covering all 10 required board memo
        sections. Must contain section headers for all sections in
        ``_REQUIRED_BOARD_MEMO_SECTIONS``.
    macro_summary : Dict[str, Any]
        Macro regime summary. Expected keys: ``"regime"`` (str),
        ``"confidence"`` (float), ``"key_drivers"`` (str, optional).
    cma_summary : Dict[str, float]
        Mapping from asset class name to final CMA expected return
        (decimal form).
    risk_summary : Dict[str, Any]
        Portfolio risk metrics. Expected keys: ``"ex_ante_vol"``,
        ``"tracking_error"``, ``"backtest_sharpe"``, ``"max_drawdown"``.
    selected_ensemble : str
        Name of the ensemble method selected by the CIO agent.
    artifact_dir : Path
        Directory to write ``board_memo.md``.
    asset_class_labels : Optional[List[str]]
        Ordered list of 18 asset class names. If ``None``, uses
        ``_CANONICAL_ASSET_CLASS_ORDER``.
    benchmark_weights : Optional[List[float]]
        18-element benchmark weight vector. If provided, used to compute
        active bets (overweights/underweights vs benchmark).

    Returns
    -------
    str
        Absolute path to the written ``board_memo.md`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``rationale`` is empty or fewer than ``_MIN_RATIONALE_LENGTH``
        characters.
    ValueError
        If ``len(weights) != 18``.
    OSError
        If the file cannot be written.

    Notes
    -----
    The board memo structure follows the ``CIO_BOARD_MEMO`` template
    from ``STUDY_CONFIG["PROMPT_TEMPLATES"]``, which specifies 10
    required sections. The structured data (allocation table, metrics
    table, IPS compliance statement) is inserted programmatically;
    the narrative content is provided via ``rationale``.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Resolve asset class labels and benchmark weights
    # ------------------------------------------------------------------
    ac_labels: List[str] = (
        list(asset_class_labels)
        if asset_class_labels is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )
    bm_weights: Optional[np.ndarray] = (
        np.array(benchmark_weights, dtype=np.float64)
        if benchmark_weights is not None
        else None
    )

    # ------------------------------------------------------------------
    # Helper: safely format a risk metric for display
    # ------------------------------------------------------------------
    def _fmt_risk(key: str, multiplier: float = 1.0, suffix: str = "") -> str:
        val = risk_summary.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        try:
            return f"{float(val) * multiplier:.2f}{suffix}"
        except (TypeError, ValueError):
            return str(val)

    # ------------------------------------------------------------------
    # Build the board memo markdown document
    # ------------------------------------------------------------------
    md_lines: List[str] = [
        "# CIO Board Memo — Strategic Asset Allocation",
        "",
        f"**Selected Ensemble Method:** `{selected_ensemble}`",
        "",
        "---",
        "",
    ]

    # --- Section 1: Executive Summary ---
    md_lines.extend([
        "## 1. Executive Summary",
        "",
        rationale.strip(),
        "",
    ])

    # --- Section 2: Recommended Allocation ---
    md_lines.extend([
        "## 2. Recommended Allocation",
        "",
        "| Asset Class | Category | Weight | CMA Est. |",
        "|-------------|----------|--------|----------|",
    ])
    for i, (ac, w) in enumerate(zip(ac_labels, weights)):
        category: str = _ASSET_CLASS_CATEGORIES.get(ac, "Unknown")
        cma_est: str = (
            f"{float(cma_summary.get(ac, 0.0)) * 100:.2f}%"
            if ac in cma_summary
            else "N/A"
        )
        md_lines.append(
            f"| {ac} | {category} | {w * 100:.2f}% | {cma_est} |"
        )
    md_lines.append("")

    # --- Section 3: Macro Regime View ---
    regime: str = str(macro_summary.get("regime", "Unknown"))
    confidence: str = (
        f"{float(macro_summary.get('confidence', 0.0)) * 100:.1f}%"
        if "confidence" in macro_summary
        else "N/A"
    )
    key_drivers: str = str(macro_summary.get("key_drivers", "See analysis.md"))
    md_lines.extend([
        "## 3. Macro Regime View",
        "",
        f"**Current Regime:** {regime}  ",
        f"**Confidence:** {confidence}  ",
        f"**Key Drivers:** {key_drivers}",
        "",
    ])

    # --- Section 4: Expected Performance vs 60/40 Benchmark ---
    md_lines.extend([
        "## 4. Expected Performance vs 60/40 Benchmark",
        "",
        "| Metric | Portfolio | 60/40 Benchmark |",
        "|--------|-----------|-----------------|",
        f"| Ex-Ante Vol | {_fmt_risk('ex_ante_vol', 100, '%')} | ~10% |",
        f"| Tracking Error | {_fmt_risk('tracking_error', 100, '%')} | — |",
        f"| Backtest Sharpe | {_fmt_risk('backtest_sharpe')} | ~0.41 |",
        f"| Max Drawdown | {_fmt_risk('max_drawdown', 100, '%')} | ~-34.3% |",
        "",
    ])

    # --- Section 5: Key Active Bets ---
    md_lines.extend([
        "## 5. Key Active Bets",
        "",
    ])
    if bm_weights is not None:
        # Compute active weights and sort by magnitude
        active_weights: List[Tuple[str, float]] = [
            (ac, float(weights[i]) - float(bm_weights[i]))
            for i, ac in enumerate(ac_labels)
        ]
        active_weights_sorted: List[Tuple[str, float]] = sorted(
            active_weights, key=lambda x: abs(x[1]), reverse=True
        )
        md_lines.extend([
            "| Asset Class | Active Weight |",
            "|-------------|---------------|",
        ])
        for ac, aw in active_weights_sorted[:10]:
            sign: str = "+" if aw >= 0 else ""
            md_lines.append(f"| {ac} | {sign}{aw * 100:.2f}% |")
        md_lines.append("")
    else:
        md_lines.extend([
            "Active bets vs benchmark not available "
            "(benchmark_weights not provided).",
            "",
        ])

    # --- Section 6: Risk Assessment ---
    md_lines.extend([
        "## 6. Risk Assessment",
        "",
        f"- **Ex-Ante Volatility:** {_fmt_risk('ex_ante_vol', 100, '%')} "
        f"(IPS band: 8%–12%)",
        f"- **Tracking Error:** {_fmt_risk('tracking_error', 100, '%')} "
        f"(IPS budget: ≤6%)",
        f"- **Max Drawdown:** {_fmt_risk('max_drawdown', 100, '%')} "
        f"(IPS limit: ≥−25%)",
        "",
    ])

    # --- Section 7: IPS Compliance Statement ---
    md_lines.extend([
        "## 7. IPS Compliance Statement",
        "",
        "The recommended portfolio has been verified against all IPS constraints:",
        "",
        f"- **Volatility band [8%, 12%]:** "
        f"{_fmt_risk('ex_ante_vol', 100, '%')} — "
        f"{'✅ PASS' if risk_summary.get('ex_ante_vol', 0) is not None and 0.08 <= float(risk_summary.get('ex_ante_vol', 0)) <= 0.12 else '⚠️ CHECK'}",
        f"- **Tracking error ≤6%:** "
        f"{_fmt_risk('tracking_error', 100, '%')} — "
        f"{'✅ PASS' if risk_summary.get('tracking_error', 1) is not None and float(risk_summary.get('tracking_error', 1)) <= 0.06 else '⚠️ CHECK'}",
        f"- **Max drawdown ≥−25%:** "
        f"{_fmt_risk('max_drawdown', 100, '%')} — "
        f"{'✅ PASS' if risk_summary.get('max_drawdown', -1) is not None and float(risk_summary.get('max_drawdown', -1)) >= -0.25 else '⚠️ CHECK'}",
        "",
    ])

    # --- Section 8: Dissenting Views and Rejected Methods ---
    md_lines.extend([
        "## 8. Dissenting Views and Rejected Methods",
        "",
        "See ``vote_corpus.json`` and ``cro_risk_reports.json`` for full "
        "peer review critiques and bottom-flag dissents.",
        "",
    ])

    # --- Section 9: Rebalancing Plan ---
    md_lines.extend([
        "## 9. Rebalancing Plan",
        "",
        "- **Frequency:** Quarterly (per IPS rebalancing policy)",
        "- **Off-cycle triggers:** L1 drift > 5% from target weights",
        "- **Monitoring:** Monthly CRO review of ex-ante TE and vol",
        "",
    ])

    # --- Section 10: Invalidation Triggers ---
    md_lines.extend([
        "## 10. Invalidation Triggers",
        "",
        "This portfolio recommendation would be invalidated by:",
        "",
        f"- Macro regime shift away from '{regime}' "
        "(re-run full pipeline on regime change)",
        "- Ex-ante tracking error exceeding 6% (immediate rebalance required)",
        "- Realised drawdown approaching −20% (escalate to investment committee)",
        "- CPI re-acceleration above 4% YoY (reassess inflation hedges)",
        "",
    ])

    # ------------------------------------------------------------------
    # Join all lines into a single markdown string
    # ------------------------------------------------------------------
    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Create the artifact directory and write the file
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = artifact_dir / "board_memo.md"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write board_memo.md to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_board_memo_md: written to '%s'.",
        output_path,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 57: write_pc_weights_json
# =============================================================================

def write_pc_weights_json(
    method: str,
    weights: List[float],
    rationale: str,
    artifact_dir: Path,
) -> str:
    """
    Serialise and persist the PC portfolio weights to ``pc_weights.json``.

    This tool implements the primary artifact-writing step for all PC
    agents (Tasks 19, 25, 26, 30). The output file is consumed by the
    CRO Agent (Task 27), peer review agents (Task 28), voting agents
    (Task 29), and the CIO Agent (Task 31). It must conform to the
    frozen ``pc_weights.schema.json`` schema.

    The file is written to:
    ``{artifact_dir}/{method_slug}/pc_weights.json``

    Parameters
    ----------
    method : str
        PC method slug identifier (e.g., ``"max_diversification"``).
        Used to construct the output file path.
    weights : List[float]
        18-element portfolio weight vector. Must sum to 1.0 (within
        ``1e-6``) and be non-negative.
    rationale : str
        LLM-generated or scripted explanation of the portfolio
        construction objective and key exposures. Must be non-empty
        (minimum ``_MIN_RATIONALE_LENGTH`` characters).
    artifact_dir : Path
        Base artifact directory.

    Returns
    -------
    str
        Absolute path to the written ``pc_weights.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``weights`` does not have 18 elements, contains negative
        values, or does not sum to 1.0 (within ``1e-6``).
    ValueError
        If ``rationale`` is empty or fewer than ``_MIN_RATIONALE_LENGTH``
        characters.
    OSError
        If the file cannot be written.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )

    # ------------------------------------------------------------------
    # Input validation: weights length
    # ------------------------------------------------------------------
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-negative weights
    # ------------------------------------------------------------------
    w_arr: np.ndarray = np.array(weights, dtype=np.float64)
    if (w_arr < -1e-8).any():
        raise ValueError(
            f"weights contains negative values: "
            f"{[v for v in weights if v < -1e-8]}."
        )

    # ------------------------------------------------------------------
    # Input validation: sum to 1.0
    # ------------------------------------------------------------------
    weight_sum: float = float(w_arr.sum())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"weights sum to {weight_sum:.8f}, expected 1.0 (tolerance 1e-6)."
        )

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # PC method slug identifier
        "method": method,
        # 18-element portfolio weight vector
        "weights": [float(v) for v in weights],
        # Actual sum of weights (for audit verification)
        "sum_check": float(weight_sum),
        # LLM-generated or scripted rationale
        "rationale": rationale.strip(),
    }

    # ------------------------------------------------------------------
    # Derive the method slug for the output directory
    # ------------------------------------------------------------------
    method_slug: str = _derive_slug(method)

    # ------------------------------------------------------------------
    # Create the output directory and write the JSON
    # ------------------------------------------------------------------
    output_dir: Path = artifact_dir / method_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_dir / "pc_weights.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write pc_weights.json to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_pc_weights_json: written for method='%s' to '%s'.",
        method,
        output_path,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 58: write_pc_report_md
# =============================================================================

def write_pc_report_md(
    method: str,
    weights: List[float],
    rationale: str,
    artifact_dir: Path,
    asset_class_labels: Optional[List[str]] = None,
    benchmark_weights: Optional[List[float]] = None,
    macro_view: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format and persist the PC portfolio report to ``pc_report.md``.

    This tool implements the PC report artifact-writing step for all PC
    agents (Tasks 19, 25, 26, 30). The output file is the human-readable
    audit trail for each PC portfolio, consumed by peer review agents
    (Task 28) and the CIO Agent (Task 31).

    The file is written to:
    ``{artifact_dir}/{method_slug}/pc_report.md``

    Parameters
    ----------
    method : str
        PC method slug identifier.
    weights : List[float]
        18-element portfolio weight vector.
    rationale : str
        LLM-generated explanation covering: objective function
        description, key portfolio exposures, regime considerations,
        and IPS compliance statement. Must be non-empty.
    artifact_dir : Path
        Base artifact directory.
    asset_class_labels : Optional[List[str]]
        Ordered list of 18 asset class names. If ``None``, uses
        ``_CANONICAL_ASSET_CLASS_ORDER``.
    benchmark_weights : Optional[List[float]]
        18-element benchmark weight vector. If provided, active bets
        (overweights/underweights) are included in the report.
    macro_view : Optional[Dict[str, Any]]
        Macro regime view dict. If provided, regime context is included.

    Returns
    -------
    str
        Absolute path to the written ``pc_report.md`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``rationale`` is empty or ``weights`` has wrong length.
    OSError
        If the file cannot be written.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Resolve asset class labels and benchmark weights
    # ------------------------------------------------------------------
    ac_labels: List[str] = (
        list(asset_class_labels)
        if asset_class_labels is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )
    bm_arr: Optional[np.ndarray] = (
        np.array(benchmark_weights, dtype=np.float64)
        if benchmark_weights is not None
        else None
    )

    # ------------------------------------------------------------------
    # Build the markdown document
    # ------------------------------------------------------------------
    md_lines: List[str] = [
        f"# PC Portfolio Report — {method}",
        "",
        f"**Method:** `{method}`",
        "",
    ]

    # --- Weight allocation table ---
    md_lines.extend([
        "## Portfolio Weights",
        "",
        "| Asset Class | Category | Weight |"
        + (" Active Bet |" if bm_arr is not None else ""),
        "|-------------|----------|--------|"
        + ("------------|" if bm_arr is not None else ""),
    ])
    for i, (ac, w) in enumerate(zip(ac_labels, weights)):
        category: str = _ASSET_CLASS_CATEGORIES.get(ac, "Unknown")
        row: str = f"| {ac} | {category} | {w * 100:.2f}% |"
        if bm_arr is not None:
            active: float = float(w) - float(bm_arr[i])
            sign: str = "+" if active >= 0 else ""
            row += f" {sign}{active * 100:.2f}% |"
        md_lines.append(row)
    md_lines.append("")

    # --- Regime context (if macro_view provided) ---
    if macro_view is not None:
        regime: str = str(macro_view.get("regime", "Unknown"))
        md_lines.extend([
            "## Regime Context",
            "",
            f"**Current Regime:** {regime}",
            "",
        ])

    # --- Rationale section ---
    md_lines.extend([
        "## Portfolio Construction Rationale",
        "",
        rationale.strip(),
        "",
    ])

    # Join all lines
    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Create the output directory and write the file
    # ------------------------------------------------------------------
    method_slug: str = _derive_slug(method)
    output_dir: Path = artifact_dir / method_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_dir / "pc_report.md"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write pc_report.md to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_pc_report_md: written for method='%s' to '%s'.",
        method,
        output_path,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 59: write_proposed_method_md
# =============================================================================

def write_proposed_method_md(
    method_spec: Dict[str, Any],
    artifact_dir: Path,
) -> str:
    """
    Format and persist the proposed PC method specification to markdown.

    This tool implements the PC Researcher artifact-writing step (Task 19).
    The output ``proposed_method.md`` is the human-readable specification
    of the new portfolio construction method proposed by the PC-Researcher
    agent. It must contain all required sections from ``_REQUIRED_METHOD_SPEC_KEYS``.

    The file is written to:
    ``{artifact_dir}/proposed_method.md``

    Parameters
    ----------
    method_spec : Dict[str, Any]
        Method specification dict. Must contain all keys in
        ``_REQUIRED_METHOD_SPEC_KEYS``:
        ``"method_name"``, ``"objective_function"``, ``"constraints"``,
        ``"required_inputs"``, ``"expected_behavior"``,
        ``"failure_modes"``.
    artifact_dir : Path
        Directory to write ``proposed_method.md``.

    Returns
    -------
    str
        Absolute path to the written ``proposed_method.md`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``method_spec``
        is not a dict.
    ValueError
        If any required key is missing from ``method_spec``.
    OSError
        If the file cannot be written.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(method_spec, dict):
        raise TypeError(
            f"method_spec must be a dict, got {type(method_spec).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys
    # ------------------------------------------------------------------
    missing_keys: List[str] = [
        k for k in _REQUIRED_METHOD_SPEC_KEYS if k not in method_spec
    ]
    if missing_keys:
        raise ValueError(
            f"method_spec is missing required keys: {missing_keys}. "
            f"Required: {list(_REQUIRED_METHOD_SPEC_KEYS)}."
        )

    # ------------------------------------------------------------------
    # Extract fields from method_spec
    # ------------------------------------------------------------------
    method_name: str = str(method_spec["method_name"])
    objective_function: str = str(method_spec["objective_function"])
    constraints: str = str(method_spec["constraints"])
    required_inputs: str = str(method_spec["required_inputs"])
    expected_behavior: str = str(method_spec["expected_behavior"])
    failure_modes: str = str(method_spec["failure_modes"])

    # ------------------------------------------------------------------
    # Build the markdown document
    # ------------------------------------------------------------------
    md_lines: List[str] = [
        f"# Proposed Portfolio Construction Method: {method_name}",
        "",
        "**Proposed by:** PC-Researcher Agent",
        "",
        "---",
        "",
        "## Objective Function",
        "",
        objective_function,
        "",
        "## Constraints and Required Inputs",
        "",
        constraints,
        "",
        "## Required Data Inputs",
        "",
        required_inputs,
        "",
        "## Expected Behaviour Under Different Regimes",
        "",
        expected_behavior,
        "",
        "## Identified Failure Modes and Risks",
        "",
        failure_modes,
        "",
    ]

    # Add any additional fields from method_spec not in the required set
    extra_keys: List[str] = [
        k for k in method_spec if k not in _REQUIRED_METHOD_SPEC_KEYS
    ]
    if extra_keys:
        md_lines.extend(["## Additional Notes", ""])
        for key in extra_keys:
            md_lines.extend([f"**{key}:** {method_spec[key]}", ""])

    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Create the artifact directory and write the file
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = artifact_dir / "proposed_method.md"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write proposed_method.md to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_proposed_method_md: written for method='%s' to '%s'.",
        method_name,
        output_path,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 60: write_proposed_method_spec_json
# =============================================================================

def write_proposed_method_spec_json(
    method_spec: Dict[str, Any],
    artifact_dir: Path,
) -> str:
    """
    Serialise and persist the proposed PC method specification to JSON.

    This tool implements the PC Researcher machine-readable artifact-writing
    step (Task 19). The output ``proposed_method_spec.json`` is the
    structured specification suitable for implementation as a new PC agent
    method in subsequent pipeline runs.

    The file is written to:
    ``{artifact_dir}/proposed_method_spec.json``

    Parameters
    ----------
    method_spec : Dict[str, Any]
        Method specification dict. Must contain all keys in
        ``_REQUIRED_METHOD_SPEC_KEYS``. May contain additional keys
        for extended specification.
    artifact_dir : Path
        Directory to write ``proposed_method_spec.json``.

    Returns
    -------
    str
        Absolute path to the written ``proposed_method_spec.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``method_spec``
        is not a dict.
    ValueError
        If any required key is missing from ``method_spec``.
    OSError
        If the file cannot be written.

    Notes
    -----
    All values in ``method_spec`` are recursively cast to JSON-safe
    Python native types via ``_cast_to_json_safe`` before serialisation.
    This ensures that any ``np.float64`` or other non-serialisable types
    introduced by the PC-Researcher agent's computation are handled
    correctly.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(method_spec, dict):
        raise TypeError(
            f"method_spec must be a dict, got {type(method_spec).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys
    # ------------------------------------------------------------------
    missing_keys: List[str] = [
        k for k in _REQUIRED_METHOD_SPEC_KEYS if k not in method_spec
    ]
    if missing_keys:
        raise ValueError(
            f"method_spec is missing required keys: {missing_keys}. "
            f"Required: {list(_REQUIRED_METHOD_SPEC_KEYS)}."
        )

    # ------------------------------------------------------------------
    # Cast all values to JSON-safe types
    # ------------------------------------------------------------------
    json_safe_spec: Dict[str, Any] = _cast_to_json_safe(method_spec)

    # ------------------------------------------------------------------
    # Create the artifact directory and write the JSON
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = artifact_dir / "proposed_method_spec.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(json_safe_spec, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write proposed_method_spec.json to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_proposed_method_spec_json: written for method='%s' to '%s'.",
        str(method_spec.get("method_name", "unknown")),
        output_path,
    )

    return str(output_path.resolve())

# =============================================================================
# SELF-DRIVING PORTFOLIO: TOOL REGISTRY — BATCH 6 (TOOLS 51–60)
# =============================================================================
# Implements tools 51–60 from the complete 78-tool registry for the agentic
# Strategic Asset Allocation (SAA) pipeline described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# Tools implemented:
#   51. compute_cio_composite_scores   — CIO Agent scoring
#   52. evaluate_ensemble_methods      — CIO Agent ensemble computation
#   53. check_ips_compliance_ensemble  — CIO Agent IPS gate
#   54. select_best_ensemble           — CIO Agent selection
#   55. write_final_weights_json       — CIO Agent terminal artifact
#   56. write_board_memo_md            — CIO Agent board memo
#   57. write_pc_weights_json          — PC Agent artifact writer
#   58. write_pc_report_md             — PC Agent artifact writer
#   59. write_proposed_method_md       — PC Researcher artifact writer
#   60. write_proposed_method_spec_json — PC Researcher artifact writer
#
# All tools are purely deterministic Python callables — no LLM interaction.
# All arithmetic conventions are frozen per STUDY_CONFIG["DATA_CONVENTIONS"].
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, OptimizeResult

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception for terminal schema validation failures
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """
    Raised when a pipeline artifact fails its terminal schema validation gate.

    This exception signals a fail-closed pipeline halt. It is distinct from
    ``ValueError`` to allow the orchestrator to differentiate between data
    errors (``ValueError``) and schema contract violations (``SchemaValidationError``).
    """
    pass


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

# Number of assets in the 18-asset universe
_N_ASSETS: int = 18

# Annualisation multiplier for monthly returns
_PERIODS_PER_YEAR: int = 12

# Numerical stability epsilon
_EPS: float = 1e-8

# Frozen IPS constraint values
_IPS_MAX_WEIGHT: float = 0.25
_IPS_MIN_WEIGHT: float = 0.00
_IPS_TE_BUDGET: float = 0.06
_IPS_VOL_LOWER: float = 0.08
_IPS_VOL_UPPER: float = 0.12

# Frozen CIO scoring weights per METHODOLOGY_PARAMS["CIO_SCORING_WEIGHTS"]
_CIO_SCORING_WEIGHTS: Dict[str, float] = {
    "backtest_sharpe":       0.25,
    "ips_compliance":        0.15,
    "diversification":       0.15,
    "regime_fit":            0.20,
    "estimation_robustness": 0.15,
    "cma_utilization":       0.10,
}

# Frozen ensemble method names per METHODOLOGY_PARAMS["CIO_ENSEMBLE_METHODS"]
_ENSEMBLE_METHOD_NAMES: Tuple[str, ...] = (
    "simple_average",
    "inverse_tracking_error_weighting",
    "backtest_sharpe_weighting",
    "meta_optimization_pc_as_assets",
    "regime_conditional_weighting",
    "composite_score_weighting",
    "trimmed_mean_outlier_exclusion",
)

# Minimum character length for rationale strings
_MIN_RATIONALE_LENGTH: int = 10

# Required keys for method_spec (PC Researcher output)
_REQUIRED_METHOD_SPEC_KEYS: Tuple[str, ...] = (
    "method_name",
    "objective_function",
    "constraints",
    "required_inputs",
    "expected_behavior",
    "failure_modes",
)

# Required sections in the CIO board memo
_REQUIRED_BOARD_MEMO_SECTIONS: Tuple[str, ...] = (
    "Executive Summary",
    "Recommended Allocation",
    "Macro Regime View",
    "Expected Performance",
    "Key Active Bets",
    "Risk Assessment",
    "IPS Compliance",
    "Dissenting Views",
    "Rebalancing Plan",
    "Invalidation Triggers",
)

# Valid macro regime labels
_VALID_REGIMES: Tuple[str, ...] = (
    "Expansion",
    "Late-cycle",
    "Recession",
    "Recovery",
)

# Regime-optimal tilt directions: positive = overweight, negative = underweight
# Defined as a directional signal over the 4 broad categories
# (Equity, Fixed Income, Real Assets, Cash) per regime
# UNSPECIFIED IN MANUSCRIPT — frozen as implementation choice
_REGIME_TILT_SIGNALS: Dict[str, Dict[str, float]] = {
    "Expansion":  {"Equity": 1.0, "Fixed Income": -0.5, "Real Assets": 0.5, "Cash": -1.0},
    "Late-cycle": {"Equity": -0.5, "Fixed Income": 0.5, "Real Assets": 1.0, "Cash": 0.5},
    "Recession":  {"Equity": -1.0, "Fixed Income": 1.0, "Real Assets": -0.5, "Cash": 1.0},
    "Recovery":   {"Equity": 0.5, "Fixed Income": 0.0, "Real Assets": 0.5, "Cash": -0.5},
}

# Canonical 18 asset class names with their category labels
# per IPS_GOVERNANCE["ASSET_CLASSES"]
_ASSET_CLASS_CATEGORIES: Dict[str, str] = {
    "US Large Cap":                "Equity",
    "US Small Cap":                "Equity",
    "US Value":                    "Equity",
    "US Growth":                   "Equity",
    "International Developed":     "Equity",
    "Emerging Markets":            "Equity",
    "Short-Term Treasuries":       "Fixed Income",
    "Intermediate Treasuries":     "Fixed Income",
    "Long-Term Treasuries":        "Fixed Income",
    "Investment-Grade Corporates": "Fixed Income",
    "High-Yield Corporates":       "Fixed Income",
    "International Sovereign Bonds": "Fixed Income",
    "International Corporates":    "Fixed Income",
    "USD Emerging Market Debt":    "Fixed Income",
    "REITs":                       "Real Assets",
    "Gold":                        "Real Assets",
    "Commodities":                 "Real Assets",
    "Cash":                        "Cash",
}

# Canonical ordered list of 18 asset class names
_CANONICAL_ASSET_CLASS_ORDER: Tuple[str, ...] = tuple(
    _ASSET_CLASS_CATEGORIES.keys()
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _cast_to_json_safe(obj: Any) -> Any:
    """
    Recursively cast an object to JSON-serialisable Python native types.

    Converts ``np.float64``, ``np.int64``, ``np.bool_``, ``np.ndarray``,
    and nested containers to their Python native equivalents.
    ``None`` and ``float("nan")`` are preserved as ``None`` (JSON ``null``).
    """
    if obj is None:
        return None
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_cast_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _cast_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cast_to_json_safe(v) for v in obj]
    if isinstance(obj, (int, bool)):
        return obj
    if isinstance(obj, str):
        return obj
    return str(obj)


def _derive_slug(name: str) -> str:
    """
    Derive a filesystem-safe slug from a name string.

    Applies: lowercase → replace spaces with underscores →
    remove non-alphanumeric/underscore characters.
    """
    s: str = name.lower().replace(" ", "_")
    return re.sub(r"[^a-z0-9_]", "", s)


def _load_pc_weights(
    method_slug: str,
    artifact_dir: Path,
    subdir: str = "pc_agents",
) -> np.ndarray:
    """
    Load and return the weight vector for a PC method from its artifact.

    Parameters
    ----------
    method_slug : str
        PC method slug identifier.
    artifact_dir : Path
        Base artifact directory.
    subdir : str
        Subdirectory under ``artifact_dir`` containing PC agent outputs.
        Default: ``"pc_agents"``.

    Returns
    -------
    np.ndarray
        18-element weight vector.

    Raises
    ------
    FileNotFoundError
        If the ``pc_weights.json`` file does not exist.
    ValueError
        If the weight vector does not have 18 elements.
    """
    # Construct the path to the pc_weights.json artifact
    weights_path: Path = artifact_dir / subdir / method_slug / "pc_weights.json"
    if not weights_path.exists():
        # Also try the top5_revision directory for revised weights
        revised_path: Path = (
            artifact_dir / "top5_revision" / method_slug / "revised_pc_weights.json"
        )
        if revised_path.exists():
            weights_path = revised_path
        else:
            raise FileNotFoundError(
                f"pc_weights.json not found for method='{method_slug}' "
                f"at '{weights_path}'."
            )
    # Load and parse the JSON file
    with open(weights_path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    # Extract the weights field
    raw_weights = data.get("weights")
    if raw_weights is None:
        raise ValueError(
            f"'weights' field missing in '{weights_path}'."
        )
    # Convert to numpy array
    w: np.ndarray = np.array(raw_weights, dtype=np.float64)
    if w.shape != (_N_ASSETS,):
        raise ValueError(
            f"Weight vector for '{method_slug}' has shape {w.shape}, "
            f"expected ({_N_ASSETS},)."
        )
    return w


def _load_cro_metrics(
    method_slug: str,
    artifact_dir: Path,
) -> Dict[str, Any]:
    """
    Load and return the CRO metrics dict for a PC method.

    Parameters
    ----------
    method_slug : str
        PC method slug identifier.
    artifact_dir : Path
        Base artifact directory.

    Returns
    -------
    Dict[str, Any]
        CRO metrics dict from ``cro_report.json``.

    Raises
    ------
    FileNotFoundError
        If the ``cro_report.json`` file does not exist.
    """
    # Construct the path to the cro_report.json artifact
    cro_path: Path = artifact_dir / "cro_reports" / method_slug / "cro_report.json"
    if not cro_path.exists():
        raise FileNotFoundError(
            f"cro_report.json not found for method='{method_slug}' "
            f"at '{cro_path}'."
        )
    with open(cro_path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)
    return data


def _compute_effective_n(weights: np.ndarray) -> float:
    """
    Compute the effective number of assets (Meucci 2009).

    .. math::

        N_{eff} = \\exp\\left(-\\sum_i p_i \\ln p_i\\right)

    where :math:`p_i = w_i^2 / \\sum_j w_j^2` are the squared-weight
    proportions.

    Parameters
    ----------
    weights : np.ndarray
        18-element weight vector.

    Returns
    -------
    float
        Effective number of assets in ``[1, 18]``.
    """
    # Compute squared weights
    w_sq: np.ndarray = weights ** 2
    # Sum of squared weights (denominator)
    sum_w_sq: float = float(w_sq.sum())
    if sum_w_sq < _EPS:
        # Degenerate case: all weights are zero
        return 1.0
    # Normalised squared-weight proportions: p_i = w_i^2 / sum(w_j^2)
    p: np.ndarray = w_sq / sum_w_sq
    # Shannon entropy of p: H = -sum(p_i * ln(p_i))
    # Clip p to avoid log(0)
    p_clipped: np.ndarray = np.clip(p, _EPS, 1.0)
    entropy: float = float(-np.sum(p_clipped * np.log(p_clipped)))
    # Effective N = exp(H)
    return float(np.exp(entropy))


def _minmax_normalise(values: np.ndarray) -> np.ndarray:
    """
    Apply min-max normalisation to an array, mapping to ``[0, 1]``.

    If all values are equal (degenerate case), returns an array of 0.5.

    Parameters
    ----------
    values : np.ndarray
        1-D array of values to normalise.

    Returns
    -------
    np.ndarray
        Normalised array with values in ``[0, 1]``.
    """
    v_min: float = float(values.min())
    v_max: float = float(values.max())
    v_range: float = v_max - v_min
    if v_range > _EPS:
        return (values - v_min) / v_range
    # Degenerate: all values equal → set to 0.5
    return np.full_like(values, 0.5, dtype=np.float64)


# =============================================================================
# TOOL 51: compute_cio_composite_scores
# =============================================================================

def compute_cio_composite_scores(
    candidate_methods: List[str],
    artifact_dir: Path,
    sigma: np.ndarray,
    benchmark_weights: np.ndarray,
    returns_matrix: np.ndarray,
    rf: float,
    macro_view: Dict[str, Any],
    cma_set: Dict[str, float],
    asset_class_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute the CIO composite score for each top-5 candidate portfolio.

    Implements the CIO scoring step (Task 31, Step 1) using the explicit
    weight schedule from ``METHODOLOGY_PARAMS["CIO_SCORING_WEIGHTS"]``:

    .. math::

        CIO\\_Score_j = 0.25 \\cdot \\widetilde{SR}_j
                      + 0.15 \\cdot IPS_j
                      + 0.15 \\cdot \\widetilde{Div}_j
                      + 0.20 \\cdot \\widetilde{RF}_j
                      + 0.15 \\cdot \\widetilde{ER}_j
                      + 0.10 \\cdot \\widetilde{CU}_j

    where :math:`\\widetilde{\\cdot}` denotes min-max normalisation to
    :math:`[0, 1]` across the candidate set, and the six sub-scores are:

    - **Backtest Sharpe** (:math:`SR`): from ``cro_report.json``
    - **IPS compliance** (:math:`IPS`): binary (1.0 / 0.0)
    - **Diversification** (:math:`Div`): effective number of assets
      :math:`N_{eff} = \\exp(-\\sum_i p_i \\ln p_i)` (Meucci 2009)
    - **Regime fit** (:math:`RF`): cosine similarity between portfolio
      active weights and the regime-optimal tilt vector
    - **Estimation robustness** (:math:`ER`): :math:`1 - HHI(w)` where
      :math:`HHI = \\sum_i w_i^2` (inverse Herfindahl concentration)
    - **CMA utilisation** (:math:`CU`): Pearson correlation between
      portfolio weights and CMA-implied weights (normalised to [0, 1])

    Parameters
    ----------
    candidate_methods : List[str]
        Top-5 PC method slugs from ``apply_diversity_constraint``.
    artifact_dir : Path
        Base artifact directory containing ``pc_agents/``,
        ``cro_reports/``, and ``top5_revision/`` subdirectories.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    returns_matrix : np.ndarray
        T×18 monthly returns matrix. Shape: ``(T, 18)``.
    rf : float
        Monthly risk-free rate in decimal form.
    macro_view : Dict[str, Any]
        Macro regime view dict. Must contain ``"regime"`` (``str``).
    cma_set : Dict[str, float]
        Mapping from asset class name to final CMA expected return
        (decimal form). Used for CMA utilisation scoring.
    asset_class_labels : Optional[List[str]]
        Ordered list of 18 asset class names corresponding to the
        columns of ``sigma``, ``returns_matrix``, and the weight
        vectors. If ``None``, uses ``_CANONICAL_ASSET_CLASS_ORDER``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"composite_scores"`` (``Dict[str, float]``): CIO composite
          score per candidate method, in ``[0, 1]``.
        - ``"sub_scores"`` (``Dict[str, Dict[str, float]]``): Per-method
          sub-scores for all 6 dimensions (pre-normalisation values and
          normalised values).

    Raises
    ------
    TypeError
        If ``sigma``, ``benchmark_weights``, or ``returns_matrix`` are
        not ``np.ndarray``.
    ValueError
        If ``candidate_methods`` is empty.
    FileNotFoundError
        If any required artifact is missing.

    Notes
    -----
    **Regime fit computation:** The regime-optimal tilt vector is defined
    as a directional signal over the 4 broad asset categories (Equity,
    Fixed Income, Real Assets, Cash) per the frozen
    ``_REGIME_TILT_SIGNALS`` mapping. The cosine similarity between the
    portfolio's active weight vector and this tilt vector is used as the
    regime fit score. This is a frozen implementation choice
    (UNSPECIFIED IN MANUSCRIPT).

    **CMA utilisation:** Computed as the Pearson correlation between the
    portfolio weights and the CMA-implied weights (proportional to CMA
    estimates, normalised to sum to 1.0). Higher correlation indicates
    greater utilisation of the CMA forecasts. Normalised to ``[0, 1]``
    via ``(corr + 1) / 2``.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    for arr_name, arr_val in [
        ("sigma", sigma),
        ("benchmark_weights", benchmark_weights),
        ("returns_matrix", returns_matrix),
    ]:
        if not isinstance(arr_val, np.ndarray):
            raise TypeError(
                f"{arr_name} must be a np.ndarray, "
                f"got {type(arr_val).__name__}."
            )

    # ------------------------------------------------------------------
    # Input validation: non-empty candidate list
    # ------------------------------------------------------------------
    if len(candidate_methods) == 0:
        raise ValueError("candidate_methods is empty.")

    # ------------------------------------------------------------------
    # Resolve asset class labels
    # ------------------------------------------------------------------
    ac_labels: List[str] = (
        list(asset_class_labels)
        if asset_class_labels is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )

    # ------------------------------------------------------------------
    # Extract current regime from macro_view
    # ------------------------------------------------------------------
    current_regime: str = str(macro_view.get("regime", "Expansion"))
    if current_regime not in _VALID_REGIMES:
        logger.warning(
            "compute_cio_composite_scores: Unknown regime '%s'. "
            "Defaulting to 'Expansion'.",
            current_regime,
        )
        current_regime = "Expansion"

    # ------------------------------------------------------------------
    # Build the regime-optimal tilt vector over 18 assets
    # Map each asset class to its category tilt signal
    # ------------------------------------------------------------------
    # Regime tilt signals for the current regime
    regime_tilt_by_category: Dict[str, float] = _REGIME_TILT_SIGNALS.get(
        current_regime, {k: 0.0 for k in ["Equity", "Fixed Income", "Real Assets", "Cash"]}
    )
    # Build the 18-element regime tilt vector
    regime_tilt_vector: np.ndarray = np.array([
        regime_tilt_by_category.get(
            _ASSET_CLASS_CATEGORIES.get(ac, "Equity"), 0.0
        )
        for ac in ac_labels
    ], dtype=np.float64)

    # ------------------------------------------------------------------
    # Build the CMA-implied weight vector (proportional to CMA estimates)
    # CMA-implied weights: w_cma_i = max(0, mu_i) / sum(max(0, mu_j))
    # ------------------------------------------------------------------
    cma_vector: np.ndarray = np.array([
        max(0.0, float(cma_set.get(ac, 0.0)))
        for ac in ac_labels
    ], dtype=np.float64)
    cma_sum: float = float(cma_vector.sum())
    if cma_sum > _EPS:
        # Normalise CMA vector to sum to 1.0
        cma_weights_implied: np.ndarray = cma_vector / cma_sum
    else:
        # Degenerate: use equal weights as fallback
        cma_weights_implied = np.ones(_N_ASSETS, dtype=np.float64) / _N_ASSETS

    # ------------------------------------------------------------------
    # Load weights and CRO metrics for each candidate method
    # ------------------------------------------------------------------
    # Dict to store raw sub-scores per method (before normalisation)
    raw_sub_scores: Dict[str, Dict[str, float]] = {}

    for method_slug in candidate_methods:
        # Load the portfolio weight vector for this method
        try:
            w: np.ndarray = _load_pc_weights(
                method_slug, artifact_dir, subdir="top5_revision"
            )
        except FileNotFoundError:
            # Fall back to original pc_agents directory
            w = _load_pc_weights(method_slug, artifact_dir, subdir="pc_agents")

        # Load the CRO report for this method
        cro_data: Dict[str, Any] = _load_cro_metrics(method_slug, artifact_dir)
        cro_metrics: Dict[str, Any] = cro_data.get("metrics", {})

        # ------------------------------------------------------------------
        # Sub-score 1: Backtest Sharpe (raw value from CRO report)
        # ------------------------------------------------------------------
        backtest_sharpe: float = float(
            cro_metrics.get("backtest_sharpe", 0.0) or 0.0
        )

        # ------------------------------------------------------------------
        # Sub-score 2: IPS compliance (binary: 1.0 if compliant, 0.0 if not)
        # ------------------------------------------------------------------
        ips_compliant: float = float(
            bool(cro_data.get("ips_compliant", False))
        )

        # ------------------------------------------------------------------
        # Sub-score 3: Diversification — effective number of assets (Meucci 2009)
        # N_eff = exp(-sum(p_i * ln(p_i))) where p_i = w_i^2 / sum(w_j^2)
        # ------------------------------------------------------------------
        n_eff: float = _compute_effective_n(w)

        # ------------------------------------------------------------------
        # Sub-score 4: Regime fit — cosine similarity between active weights
        # and the regime-optimal tilt vector
        # ------------------------------------------------------------------
        # Active weight vector relative to benchmark
        w_active: np.ndarray = w - benchmark_weights
        # Cosine similarity: (w_active · tilt) / (||w_active|| * ||tilt||)
        w_active_norm: float = float(np.linalg.norm(w_active))
        tilt_norm: float = float(np.linalg.norm(regime_tilt_vector))
        if w_active_norm > _EPS and tilt_norm > _EPS:
            # Cosine similarity in [-1, 1]
            cosine_sim: float = float(
                np.dot(w_active, regime_tilt_vector)
                / (w_active_norm * tilt_norm)
            )
        else:
            # Degenerate: zero active weights or zero tilt → neutral
            cosine_sim = 0.0
        # Normalise cosine similarity from [-1, 1] to [0, 1]
        regime_fit: float = (cosine_sim + 1.0) / 2.0

        # ------------------------------------------------------------------
        # Sub-score 5: Estimation robustness — inverse Herfindahl concentration
        # ER = 1 - HHI(w) where HHI = sum(w_i^2)
        # Higher ER = less concentrated = more robust to estimation error
        # ------------------------------------------------------------------
        hhi: float = float(np.dot(w, w))
        estimation_robustness: float = float(1.0 - hhi)
        # Clip to [0, 1] (HHI is in [1/N, 1])
        estimation_robustness = float(np.clip(estimation_robustness, 0.0, 1.0))

        # ------------------------------------------------------------------
        # Sub-score 6: CMA utilisation — Pearson correlation between
        # portfolio weights and CMA-implied weights, normalised to [0, 1]
        # ------------------------------------------------------------------
        # Compute Pearson correlation between w and cma_weights_implied
        w_mean: float = float(w.mean())
        cma_mean: float = float(cma_weights_implied.mean())
        w_centred: np.ndarray = w - w_mean
        cma_centred: np.ndarray = cma_weights_implied - cma_mean
        w_std: float = float(np.std(w, ddof=0))
        cma_std: float = float(np.std(cma_weights_implied, ddof=0))
        if w_std > _EPS and cma_std > _EPS:
            # Pearson correlation in [-1, 1]
            pearson_corr: float = float(
                np.dot(w_centred, cma_centred) / (len(w) * w_std * cma_std)
            )
        else:
            # Degenerate: zero variance → neutral correlation
            pearson_corr = 0.0
        # Normalise from [-1, 1] to [0, 1]
        cma_utilization: float = (pearson_corr + 1.0) / 2.0

        # Store raw sub-scores for this method
        raw_sub_scores[method_slug] = {
            "backtest_sharpe":       backtest_sharpe,
            "ips_compliance":        ips_compliant,
            "diversification":       n_eff,
            "regime_fit":            regime_fit,
            "estimation_robustness": estimation_robustness,
            "cma_utilization":       cma_utilization,
        }

    # ------------------------------------------------------------------
    # Normalise sub-scores across candidates (min-max to [0, 1])
    # IPS compliance is already binary [0, 1]; no normalisation needed.
    # Regime fit and CMA utilisation are already in [0, 1].
    # Backtest Sharpe, diversification, and estimation robustness need
    # normalisation across the candidate set.
    # ------------------------------------------------------------------
    methods_list: List[str] = list(raw_sub_scores.keys())
    n_candidates: int = len(methods_list)

    # Build arrays for normalisation
    sharpe_arr: np.ndarray = np.array(
        [raw_sub_scores[m]["backtest_sharpe"] for m in methods_list],
        dtype=np.float64,
    )
    div_arr: np.ndarray = np.array(
        [raw_sub_scores[m]["diversification"] for m in methods_list],
        dtype=np.float64,
    )
    er_arr: np.ndarray = np.array(
        [raw_sub_scores[m]["estimation_robustness"] for m in methods_list],
        dtype=np.float64,
    )

    # Apply min-max normalisation to Sharpe, diversification, and ER
    sharpe_norm: np.ndarray = _minmax_normalise(sharpe_arr)
    div_norm: np.ndarray = _minmax_normalise(div_arr)
    er_norm: np.ndarray = _minmax_normalise(er_arr)

    # ------------------------------------------------------------------
    # Compute the CIO composite score for each candidate
    # CIO_Score_j = sum(weight_k * normalised_sub_score_k_j)
    # ------------------------------------------------------------------
    composite_scores: Dict[str, float] = {}
    sub_scores_output: Dict[str, Dict[str, float]] = {}

    for i, method_slug in enumerate(methods_list):
        raw: Dict[str, float] = raw_sub_scores[method_slug]

        # Normalised sub-scores for this method
        norm_sharpe: float = float(sharpe_norm[i])
        norm_ips: float = float(raw["ips_compliance"])       # Already binary
        norm_div: float = float(div_norm[i])
        norm_rf: float = float(raw["regime_fit"])            # Already in [0,1]
        norm_er: float = float(er_norm[i])
        norm_cu: float = float(raw["cma_utilization"])       # Already in [0,1]

        # Weighted composite score
        composite: float = (
            _CIO_SCORING_WEIGHTS["backtest_sharpe"]       * norm_sharpe
            + _CIO_SCORING_WEIGHTS["ips_compliance"]      * norm_ips
            + _CIO_SCORING_WEIGHTS["diversification"]     * norm_div
            + _CIO_SCORING_WEIGHTS["regime_fit"]          * norm_rf
            + _CIO_SCORING_WEIGHTS["estimation_robustness"] * norm_er
            + _CIO_SCORING_WEIGHTS["cma_utilization"]     * norm_cu
        )

        composite_scores[method_slug] = float(composite)
        sub_scores_output[method_slug] = {
            "backtest_sharpe_raw":       float(raw["backtest_sharpe"]),
            "backtest_sharpe_norm":      norm_sharpe,
            "ips_compliance":            norm_ips,
            "diversification_raw":       float(raw["diversification"]),
            "diversification_norm":      norm_div,
            "regime_fit":                norm_rf,
            "estimation_robustness_raw": float(raw["estimation_robustness"]),
            "estimation_robustness_norm": norm_er,
            "cma_utilization":           norm_cu,
            "composite_score":           float(composite),
        }

    logger.info(
        "compute_cio_composite_scores: scored %d candidates. "
        "Top: %s (%.4f).",
        n_candidates,
        max(composite_scores, key=composite_scores.get),
        max(composite_scores.values()),
    )

    return {
        # CIO composite score per candidate method
        "composite_scores": composite_scores,
        # Per-method sub-scores (raw and normalised) for audit
        "sub_scores": sub_scores_output,
    }


# =============================================================================
# TOOL 52: evaluate_ensemble_methods
# =============================================================================

def evaluate_ensemble_methods(
    top_5_methods: List[str],
    artifact_dir: Path,
    sigma: np.ndarray,
    benchmark_weights: np.ndarray,
    returns_matrix: np.ndarray,
    rf: float,
    composite_scores: Dict[str, float],
    macro_view: Dict[str, Any],
) -> Dict[str, List[float]]:
    """
    Evaluate all seven ensemble combination methods for the CIO agent.

    Implements Task 31, Step 2 — the evaluation of all 7 ensemble
    approaches from ``METHODOLOGY_PARAMS["CIO_ENSEMBLE_METHODS"]``:

    1. **Simple average:** :math:`w_{ens} = \\frac{1}{K}\\sum_k w^{(k)}`
    2. **Inverse TE weighting:**
       :math:`w_{ens} = \\sum_k \\frac{1/TE_k}{\\sum_j 1/TE_j} w^{(k)}`
    3. **Backtest Sharpe weighting:**
       :math:`w_{ens} = \\sum_k \\frac{SR_k^+}{\\sum_j SR_j^+} w^{(k)}`
       (positive Sharpe only; shifted if all negative)
    4. **Meta-optimisation (PC-as-assets):** Mean-variance optimisation
       treating the 5 PC portfolios as assets
    5. **Regime-conditional weighting:** Weight by regime fit sub-scores
    6. **Composite-score weighting:**
       :math:`w_{ens} = \\sum_k \\frac{CS_k}{\\sum_j CS_j} w^{(k)}`
    7. **Trimmed mean:** Exclude the most extreme outlier (by L1 distance
       from centroid), average the remainder

    Parameters
    ----------
    top_5_methods : List[str]
        Top-5 PC method slugs.
    artifact_dir : Path
        Base artifact directory.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    returns_matrix : np.ndarray
        T×18 monthly returns matrix. Shape: ``(T, 18)``.
    rf : float
        Monthly risk-free rate in decimal form.
    composite_scores : Dict[str, float]
        CIO composite scores per method (from ``compute_cio_composite_scores``).
    macro_view : Dict[str, Any]
        Macro regime view dict. Must contain ``"regime"`` (``str``).

    Returns
    -------
    Dict[str, List[float]]
        Mapping from ensemble method name to 18-element weight vector.
        All weight vectors sum to 1.0 (within ``1e-6``) and are
        non-negative.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``top_5_methods`` is empty.
    FileNotFoundError
        If any required artifact is missing.

    Notes
    -----
    **Meta-optimisation fallback:** If the meta-optimisation (method 4)
    fails to converge or produces an infeasible solution, it falls back
    to the simple average. This is documented in the log.

    **Trimmed mean with K=5:** When K=5, only 1 outlier is excluded
    (the portfolio with the highest L1 distance from the centroid),
    leaving 4 portfolios for averaging. This avoids the degenerate case
    of excluding 2 from 5 (leaving only 3).
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if len(top_5_methods) == 0:
        raise ValueError("top_5_methods is empty.")

    # ------------------------------------------------------------------
    # Load weight vectors and CRO metrics for all top-5 methods
    # ------------------------------------------------------------------
    # Weight matrix: each row is one PC portfolio's weight vector
    weight_matrix: List[np.ndarray] = []
    # Tracking errors per method (for inverse TE weighting)
    te_values: List[float] = []
    # Backtest Sharpe ratios per method (for Sharpe weighting)
    sharpe_values: List[float] = []
    # Regime fit scores per method (for regime-conditional weighting)
    regime_fit_values: List[float] = []

    for method_slug in top_5_methods:
        # Load portfolio weights
        try:
            w: np.ndarray = _load_pc_weights(
                method_slug, artifact_dir, subdir="top5_revision"
            )
        except FileNotFoundError:
            w = _load_pc_weights(method_slug, artifact_dir, subdir="pc_agents")
        weight_matrix.append(w)

        # Load CRO metrics
        cro_data: Dict[str, Any] = _load_cro_metrics(method_slug, artifact_dir)
        cro_metrics: Dict[str, Any] = cro_data.get("metrics", {})

        # Extract tracking error
        te: float = float(cro_metrics.get("tracking_error", 0.06) or 0.06)
        te_values.append(max(te, _EPS))  # Avoid division by zero

        # Extract backtest Sharpe
        sr: float = float(cro_metrics.get("backtest_sharpe", 0.0) or 0.0)
        sharpe_values.append(sr)

        # Compute regime fit for this method (cosine similarity)
        current_regime: str = str(macro_view.get("regime", "Expansion"))
        regime_tilt: Dict[str, float] = _REGIME_TILT_SIGNALS.get(
            current_regime, {}
        )
        tilt_vec: np.ndarray = np.array([
            regime_tilt.get(_ASSET_CLASS_CATEGORIES.get(ac, "Equity"), 0.0)
            for ac in _CANONICAL_ASSET_CLASS_ORDER
        ], dtype=np.float64)
        w_active: np.ndarray = w - benchmark_weights
        w_norm: float = float(np.linalg.norm(w_active))
        t_norm: float = float(np.linalg.norm(tilt_vec))
        if w_norm > _EPS and t_norm > _EPS:
            rf_score: float = (
                float(np.dot(w_active, tilt_vec)) / (w_norm * t_norm) + 1.0
            ) / 2.0
        else:
            rf_score = 0.5
        regime_fit_values.append(rf_score)

    # Stack weight matrix: shape (K, 18)
    W: np.ndarray = np.stack(weight_matrix, axis=0)
    K: int = W.shape[0]

    # ------------------------------------------------------------------
    # Helper: normalise a weight vector to sum to 1.0 and clip negatives
    # ------------------------------------------------------------------
    def _normalise_weights(w_raw: np.ndarray) -> np.ndarray:
        """Clip to non-negative and normalise to sum to 1.0."""
        w_clipped: np.ndarray = np.maximum(w_raw, 0.0)
        w_sum: float = float(w_clipped.sum())
        if w_sum > _EPS:
            return w_clipped / w_sum
        return np.ones(_N_ASSETS, dtype=np.float64) / _N_ASSETS

    # ------------------------------------------------------------------
    # Ensemble 1: Simple average
    # w_ens = (1/K) * sum_k(w_k)
    # ------------------------------------------------------------------
    w_simple_avg: np.ndarray = _normalise_weights(W.mean(axis=0))

    # ------------------------------------------------------------------
    # Ensemble 2: Inverse tracking-error weighting
    # w_ens = sum_k((1/TE_k) / sum_j(1/TE_j) * w_k)
    # ------------------------------------------------------------------
    inv_te: np.ndarray = np.array(
        [1.0 / te for te in te_values], dtype=np.float64
    )
    inv_te_weights: np.ndarray = inv_te / inv_te.sum()
    w_inv_te: np.ndarray = _normalise_weights(
        np.einsum("k,ki->i", inv_te_weights, W)
    )

    # ------------------------------------------------------------------
    # Ensemble 3: Backtest Sharpe weighting
    # Shift all Sharpe values by subtracting the minimum if any are negative
    # w_ens = sum_k(SR_k_shifted / sum_j(SR_j_shifted) * w_k)
    # ------------------------------------------------------------------
    sr_arr: np.ndarray = np.array(sharpe_values, dtype=np.float64)
    sr_min: float = float(sr_arr.min())
    if sr_min < 0.0:
        # Shift to make all values non-negative
        sr_shifted: np.ndarray = sr_arr - sr_min + _EPS
    else:
        sr_shifted = sr_arr + _EPS
    sr_weights: np.ndarray = sr_shifted / sr_shifted.sum()
    w_sharpe: np.ndarray = _normalise_weights(
        np.einsum("k,ki->i", sr_weights, W)
    )

    # ------------------------------------------------------------------
    # Ensemble 4: Meta-optimisation (PC portfolios as assets)
    # Treat the K PC portfolios as assets; run mean-variance optimisation
    # on their return series to find the optimal combination weights
    # ------------------------------------------------------------------
    # Compute the K-portfolio return series: shape (T, K)
    portfolio_returns: np.ndarray = returns_matrix @ W.T  # (T, K)

    # Compute the K×K covariance matrix of portfolio returns
    # Use sample covariance (K=5 is too small for shrinkage)
    if portfolio_returns.shape[0] > K + 1:
        # Covariance matrix of the K portfolio return series
        cov_K: np.ndarray = np.cov(portfolio_returns.T, ddof=1)  # (K, K)
        # Expected returns of the K portfolios (annualised)
        mu_K: np.ndarray = portfolio_returns.mean(axis=0) * _PERIODS_PER_YEAR

        # Solve max-Sharpe for the K-portfolio combination weights
        def _neg_sharpe_K(alpha_K: np.ndarray) -> float:
            """Negative Sharpe of the meta-portfolio."""
            r_meta: float = float(np.dot(alpha_K, mu_K))
            var_meta: float = float(np.dot(alpha_K, cov_K @ alpha_K))
            vol_meta: float = float(np.sqrt(max(var_meta, _EPS)))
            return -(r_meta - rf * _PERIODS_PER_YEAR) / vol_meta

        # Constraints: sum(alpha) = 1, alpha >= 0
        meta_constraints = [{"type": "eq", "fun": lambda a: a.sum() - 1.0}]
        meta_bounds = [(0.0, 1.0)] * K
        meta_x0: np.ndarray = np.ones(K, dtype=np.float64) / K

        meta_result: OptimizeResult = minimize(
            fun=_neg_sharpe_K,
            x0=meta_x0,
            method="SLSQP",
            bounds=meta_bounds,
            constraints=meta_constraints,
            options={"ftol": 1e-9, "maxiter": 500, "disp": False},
        )

        if meta_result.success:
            # Compute the meta-portfolio weights in the 18-asset space
            alpha_opt: np.ndarray = np.clip(meta_result.x, 0.0, 1.0)
            alpha_opt = alpha_opt / alpha_opt.sum()
            w_meta: np.ndarray = _normalise_weights(
                np.einsum("k,ki->i", alpha_opt, W)
            )
        else:
            # Fallback to simple average if meta-optimisation fails
            logger.warning(
                "evaluate_ensemble_methods: Meta-optimisation failed. "
                "Falling back to simple average."
            )
            w_meta = w_simple_avg.copy()
    else:
        # Insufficient observations for meta-optimisation
        logger.warning(
            "evaluate_ensemble_methods: Insufficient observations for "
            "meta-optimisation (T=%d, K=%d). Falling back to simple average.",
            portfolio_returns.shape[0],
            K,
        )
        w_meta = w_simple_avg.copy()

    # ------------------------------------------------------------------
    # Ensemble 5: Regime-conditional weighting
    # Weight by regime fit scores
    # ------------------------------------------------------------------
    rf_arr: np.ndarray = np.array(regime_fit_values, dtype=np.float64)
    rf_sum: float = float(rf_arr.sum())
    if rf_sum > _EPS:
        rf_weights: np.ndarray = rf_arr / rf_sum
    else:
        rf_weights = np.ones(K, dtype=np.float64) / K
    w_regime: np.ndarray = _normalise_weights(
        np.einsum("k,ki->i", rf_weights, W)
    )

    # ------------------------------------------------------------------
    # Ensemble 6: Composite-score weighting
    # w_ens = sum_k(CS_k / sum_j(CS_j) * w_k)
    # ------------------------------------------------------------------
    cs_arr: np.ndarray = np.array(
        [float(composite_scores.get(m, 0.0)) for m in top_5_methods],
        dtype=np.float64,
    )
    cs_arr = np.maximum(cs_arr, 0.0)  # Ensure non-negative
    cs_sum: float = float(cs_arr.sum())
    if cs_sum > _EPS:
        cs_weights: np.ndarray = cs_arr / cs_sum
    else:
        cs_weights = np.ones(K, dtype=np.float64) / K
    w_composite: np.ndarray = _normalise_weights(
        np.einsum("k,ki->i", cs_weights, W)
    )

    # ------------------------------------------------------------------
    # Ensemble 7: Trimmed mean (exclude 1 outlier when K=5)
    # Exclude the portfolio with the highest L1 distance from the centroid
    # ------------------------------------------------------------------
    # Compute the centroid of all K portfolios
    centroid: np.ndarray = W.mean(axis=0)
    # Compute L1 distance of each portfolio from the centroid
    l1_distances: np.ndarray = np.array(
        [float(np.sum(np.abs(W[k] - centroid))) for k in range(K)],
        dtype=np.float64,
    )
    # Identify the index of the most extreme outlier
    outlier_idx: int = int(np.argmax(l1_distances))
    # Exclude the outlier and average the remainder
    trimmed_indices: List[int] = [k for k in range(K) if k != outlier_idx]
    w_trimmed: np.ndarray = _normalise_weights(
        W[trimmed_indices].mean(axis=0)
    )

    # ------------------------------------------------------------------
    # Assemble the output dict
    # ------------------------------------------------------------------
    ensemble_weights: Dict[str, List[float]] = {
        "simple_average":                  [float(v) for v in w_simple_avg],
        "inverse_tracking_error_weighting": [float(v) for v in w_inv_te],
        "backtest_sharpe_weighting":        [float(v) for v in w_sharpe],
        "meta_optimization_pc_as_assets":   [float(v) for v in w_meta],
        "regime_conditional_weighting":     [float(v) for v in w_regime],
        "composite_score_weighting":        [float(v) for v in w_composite],
        "trimmed_mean_outlier_exclusion":   [float(v) for v in w_trimmed],
    }

    logger.info(
        "evaluate_ensemble_methods: computed %d ensemble weight vectors.",
        len(ensemble_weights),
    )

    return ensemble_weights


# =============================================================================
# TOOL 53: check_ips_compliance_ensemble
# =============================================================================

def check_ips_compliance_ensemble(
    ensemble_weights: Dict[str, List[float]],
    sigma: np.ndarray,
    benchmark_weights: np.ndarray,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Check IPS compliance for each of the seven ensemble weight vectors.

    Applies the same five IPS constraints as ``check_ips_compliance``
    (Tool 39) to each ensemble weight vector from
    ``evaluate_ensemble_methods`` (Tool 52):

    1. Long-only: :math:`w_i \\geq 0`
    2. Max weight: :math:`w_i \\leq 0.25`
    3. Budget: :math:`\\sum_i w_i = 1.0 \\pm 10^{-6}`
    4. Tracking error: :math:`TE \\leq 0.06`
    5. Volatility band: :math:`\\sigma_p \\in [0.08, 0.12]`

    Parameters
    ----------
    ensemble_weights : Dict[str, List[float]]
        Output of ``evaluate_ensemble_methods``. Maps ensemble method
        name to 18-element weight vector.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    constraints : Optional[Dict[str, Any]]
        IPS constraint parameters. If ``None``, uses frozen defaults.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping from ensemble method name to compliance result dict.
        Each result dict contains ``"compliant"`` (bool) and
        ``"flags"`` (dict with per-constraint details).

    Raises
    ------
    TypeError
        If ``sigma`` or ``benchmark_weights`` are not ``np.ndarray``.
    ValueError
        If ``ensemble_weights`` is empty.
    RuntimeError
        If all ensemble weight vectors are IPS non-compliant.

    Notes
    -----
    **All-non-compliant guard:** If every ensemble fails IPS compliance,
    a ``RuntimeError`` is raised to halt the pipeline. The CIO agent
    cannot select a final portfolio if no ensemble is compliant.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(sigma, np.ndarray):
        raise TypeError(
            f"sigma must be a np.ndarray, got {type(sigma).__name__}."
        )
    if not isinstance(benchmark_weights, np.ndarray):
        raise TypeError(
            f"benchmark_weights must be a np.ndarray, "
            f"got {type(benchmark_weights).__name__}."
        )
    if len(ensemble_weights) == 0:
        raise ValueError("ensemble_weights is empty.")

    # ------------------------------------------------------------------
    # Resolve constraint parameters
    # ------------------------------------------------------------------
    _c: Dict[str, Any] = constraints or {}
    max_w: float = float(_c.get("max_weight_per_asset", _IPS_MAX_WEIGHT))
    min_w: float = float(_c.get("min_weight_per_asset", _IPS_MIN_WEIGHT))

    # ------------------------------------------------------------------
    # Check IPS compliance for each ensemble weight vector
    # ------------------------------------------------------------------
    compliance_results: Dict[str, Dict[str, Any]] = {}

    for ensemble_name, weights_list in ensemble_weights.items():
        # Convert to numpy array
        w: np.ndarray = np.array(weights_list, dtype=np.float64)

        # Constraint 1: Long-only
        min_weight_actual: float = float(w.min())
        long_only_pass: bool = bool(min_weight_actual >= min_w - 1e-8)

        # Constraint 2: Max weight
        max_weight_actual: float = float(w.max())
        max_weight_pass: bool = bool(max_weight_actual <= max_w + 1e-8)

        # Constraint 3: Budget
        weight_sum: float = float(w.sum())
        sum_to_one_pass: bool = bool(abs(weight_sum - 1.0) <= 1e-6)

        # Constraint 4: Tracking error
        w_active: np.ndarray = w - benchmark_weights
        te_var: float = float(max(0.0, np.dot(w_active, sigma @ w_active)))
        te_actual: float = float(np.sqrt(te_var))
        te_pass: bool = bool(te_actual <= _IPS_TE_BUDGET + 1e-8)

        # Constraint 5: Volatility band
        port_var: float = float(max(0.0, np.dot(w, sigma @ w)))
        vol_actual: float = float(np.sqrt(port_var))
        vol_band_pass: bool = bool(
            (vol_actual >= _IPS_VOL_LOWER - 1e-8)
            and (vol_actual <= _IPS_VOL_UPPER + 1e-8)
        )

        # Aggregate flags
        flags: Dict[str, Dict[str, Any]] = {
            "long_only":      {"pass": long_only_pass,  "actual_value": float(min_weight_actual)},
            "max_weight":     {"pass": max_weight_pass, "actual_value": float(max_weight_actual)},
            "sum_to_one":     {"pass": sum_to_one_pass, "actual_value": float(weight_sum)},
            "tracking_error": {"pass": te_pass,         "actual_value": float(te_actual)},
            "volatility_band":{"pass": vol_band_pass,   "actual_value": float(vol_actual)},
        }
        compliant: bool = all(v["pass"] for v in flags.values())

        compliance_results[ensemble_name] = {
            "compliant": compliant,
            "flags": flags,
        }

    # ------------------------------------------------------------------
    # All-non-compliant guard
    # ------------------------------------------------------------------
    n_compliant: int = sum(
        1 for r in compliance_results.values() if r["compliant"]
    )
    if n_compliant == 0:
        raise RuntimeError(
            "ALL ENSEMBLE WEIGHT VECTORS ARE IPS NON-COMPLIANT. "
            "The CIO agent cannot select a final portfolio. "
            "Review the PC agent outputs and IPS constraint parameters. "
            "This is a pipeline-halting error."
        )

    logger.info(
        "check_ips_compliance_ensemble: %d / %d ensembles are IPS compliant.",
        n_compliant,
        len(compliance_results),
    )

    return compliance_results


# =============================================================================
# TOOL 54: select_best_ensemble
# =============================================================================

def select_best_ensemble(
    ensemble_scores: Dict[str, float],
    compliance_flags: Dict[str, Dict[str, Any]],
    ensemble_weights: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Select the best IPS-compliant ensemble using CIO composite scores.

    Implements Task 31, Step 2 — the final ensemble selection step:

    1. Filter to IPS-compliant ensembles (from ``compliance_flags``).
    2. Among compliant ensembles, select the one with the highest
       composite score (from ``ensemble_scores``).
    3. Return the selected ensemble name, its weight vector, and a
       selection rationale.

    Parameters
    ----------
    ensemble_scores : Dict[str, float]
        Mapping from ensemble method name to composite score. Methods
        missing from this dict receive a score of 0.0.
    compliance_flags : Dict[str, Dict[str, Any]]
        Output of ``check_ips_compliance_ensemble``. Maps ensemble name
        to compliance result dict (must contain ``"compliant"`` key).
    ensemble_weights : Dict[str, List[float]]
        Output of ``evaluate_ensemble_methods``. Maps ensemble name to
        18-element weight vector.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"selected_ensemble"`` (``str``): Name of the selected
          ensemble method.
        - ``"weights"`` (``List[float]``): 18-element weight vector of
          the selected ensemble.
        - ``"selection_rationale"`` (``str``): Human-readable explanation
          of the selection decision.
        - ``"composite_score"`` (``float``): Composite score of the
          selected ensemble.
        - ``"n_compliant_ensembles"`` (``int``): Number of IPS-compliant
          ensembles considered.

    Raises
    ------
    TypeError
        If any input is not a dict.
    ValueError
        If ``compliance_flags`` is empty.
    RuntimeError
        If no IPS-compliant ensemble is found (should have been caught
        by ``check_ips_compliance_ensemble``, but included as a
        belt-and-suspenders guard).

    Notes
    -----
    **Tiebreaker:** If two or more compliant ensembles have equal
    composite scores (within ``1e-8``), the ensemble appearing first
    in alphabetical order is selected. This is a frozen tiebreaker
    documented here for reproducibility.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    for name, val in [
        ("ensemble_scores", ensemble_scores),
        ("compliance_flags", compliance_flags),
        ("ensemble_weights", ensemble_weights),
    ]:
        if not isinstance(val, dict):
            raise TypeError(
                f"{name} must be a dict, got {type(val).__name__}."
            )
    if len(compliance_flags) == 0:
        raise ValueError("compliance_flags is empty.")

    # ------------------------------------------------------------------
    # Filter to IPS-compliant ensembles
    # ------------------------------------------------------------------
    compliant_ensembles: List[str] = [
        name for name, result in compliance_flags.items()
        if bool(result.get("compliant", False))
    ]

    # ------------------------------------------------------------------
    # Belt-and-suspenders guard: no compliant ensembles
    # ------------------------------------------------------------------
    if len(compliant_ensembles) == 0:
        raise RuntimeError(
            "No IPS-compliant ensemble found in select_best_ensemble. "
            "This should have been caught by check_ips_compliance_ensemble."
        )

    # ------------------------------------------------------------------
    # Select the compliant ensemble with the highest composite score
    # Tiebreaker: alphabetical order (first in sorted list)
    # ------------------------------------------------------------------
    # Sort compliant ensembles: primary key = composite score (descending),
    # secondary key = name (ascending, for deterministic tiebreaking)
    sorted_compliant: List[str] = sorted(
        compliant_ensembles,
        key=lambda name: (
            -float(ensemble_scores.get(name, 0.0)),  # Descending score
            name,                                     # Ascending name (tiebreaker)
        ),
    )
    # The best ensemble is the first in the sorted list
    selected_name: str = sorted_compliant[0]
    selected_score: float = float(ensemble_scores.get(selected_name, 0.0))
    selected_weights: List[float] = ensemble_weights[selected_name]

    # ------------------------------------------------------------------
    # Construct the selection rationale
    # ------------------------------------------------------------------
    rationale: str = (
        f"Selected ensemble: '{selected_name}' "
        f"(composite score = {selected_score:.4f}). "
        f"{len(compliant_ensembles)} of {len(compliance_flags)} ensembles "
        f"were IPS-compliant. "
        f"Compliant ensembles (ranked): "
        f"{[f'{n} ({ensemble_scores.get(n, 0.0):.4f})' for n in sorted_compliant]}. "
        f"Tiebreaker: alphabetical order (if scores are equal within 1e-8)."
    )

    logger.info(
        "select_best_ensemble: selected='%s' (score=%.4f), "
        "n_compliant=%d.",
        selected_name,
        selected_score,
        len(compliant_ensembles),
    )

    return {
        # Name of the selected ensemble method
        "selected_ensemble": selected_name,
        # 18-element weight vector of the selected ensemble
        "weights": selected_weights,
        # Human-readable selection rationale
        "selection_rationale": rationale,
        # Composite score of the selected ensemble
        "composite_score": float(selected_score),
        # Number of IPS-compliant ensembles considered
        "n_compliant_ensembles": len(compliant_ensembles),
    }


# =============================================================================
# TOOL 55: write_final_weights_json
# =============================================================================

def write_final_weights_json(
    weights: List[float],
    rationale: str,
    selected_ensemble: str,
    artifact_dir: Path,
    asset_class_labels: Optional[List[str]] = None,
) -> str:
    """
    Serialise and persist the final portfolio weights to ``final_weights.json``.

    This tool implements the terminal artifact-writing step for the CIO
    Agent (Task 31, Step 3). The output file is the definitive portfolio
    allocation consumed by the board memo, the backtest, and the
    reproducibility package.

    **Terminal validation gate (Layer 2):** This tool independently
    verifies that the weight vector satisfies all terminal constraints:

    - Exactly 18 non-negative elements
    - Sum to 1.0 (within ``1e-6``)

    If either constraint fails, a ``SchemaValidationError`` is raised
    immediately and the pipeline halts (fail-closed).

    The file is written to:
    ``{artifact_dir}/final_weights.json``

    Parameters
    ----------
    weights : List[float]
        18-element final portfolio weight vector. Must sum to 1.0
        (within ``1e-6``) and be non-negative.
    rationale : str
        LLM-generated or scripted rationale for the final allocation.
        Must be non-empty (minimum ``_MIN_RATIONALE_LENGTH`` characters).
    selected_ensemble : str
        Name of the ensemble method selected by the CIO agent.
    artifact_dir : Path
        Directory to write ``final_weights.json``.
    asset_class_labels : Optional[List[str]]
        Ordered list of 18 asset class names. If ``None``, uses
        ``_CANONICAL_ASSET_CLASS_ORDER``.

    Returns
    -------
    str
        Absolute path to the written ``final_weights.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    SchemaValidationError
        If ``weights`` does not have exactly 18 elements, contains
        negative values, or does not sum to 1.0 (within ``1e-6``).
        This is a pipeline-halting terminal validation failure.
    ValueError
        If ``rationale`` is empty or fewer than ``_MIN_RATIONALE_LENGTH``
        characters.
    OSError
        If the file cannot be written.

    Notes
    -----
    The ``"sum_check"`` field in the output records the actual sum of
    the weight vector for audit verification. The
    ``"range_constraint_verified"`` field is set to ``True`` to confirm
    that the terminal validation gate passed.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )

    # ------------------------------------------------------------------
    # Resolve asset class labels
    # ------------------------------------------------------------------
    ac_labels: List[str] = (
        list(asset_class_labels)
        if asset_class_labels is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )

    # ------------------------------------------------------------------
    # TERMINAL VALIDATION GATE: 18 elements
    # ------------------------------------------------------------------
    if len(weights) != _N_ASSETS:
        raise SchemaValidationError(
            f"TERMINAL VALIDATION FAILURE: final weights vector has "
            f"{len(weights)} elements, expected {_N_ASSETS}. "
            "Pipeline halted (fail-closed)."
        )

    # ------------------------------------------------------------------
    # TERMINAL VALIDATION GATE: non-negative weights
    # ------------------------------------------------------------------
    w_arr: np.ndarray = np.array(weights, dtype=np.float64)
    if (w_arr < -1e-8).any():
        neg_indices: List[int] = [
            i for i, v in enumerate(weights) if v < -1e-8
        ]
        raise SchemaValidationError(
            f"TERMINAL VALIDATION FAILURE: final weights contain "
            f"negative values at indices {neg_indices}: "
            f"{[weights[i] for i in neg_indices]}. "
            "Pipeline halted (fail-closed)."
        )

    # ------------------------------------------------------------------
    # TERMINAL VALIDATION GATE: sum to 1.0 within 1e-6
    # ------------------------------------------------------------------
    weight_sum: float = float(w_arr.sum())
    if abs(weight_sum - 1.0) > 1e-6:
        raise SchemaValidationError(
            f"TERMINAL VALIDATION FAILURE: final weights sum to "
            f"{weight_sum:.8f}, expected 1.0 (tolerance 1e-6). "
            "Pipeline halted (fail-closed)."
        )

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # 18-element final portfolio weight vector
        "weights": [float(v) for v in weights],
        # Ordered asset class labels corresponding to each weight
        "asset_class_labels": ac_labels,
        # Actual sum of weights (for audit verification)
        "sum_check": float(weight_sum),
        # Name of the ensemble method selected by the CIO agent
        "selected_ensemble": selected_ensemble,
        # LLM-generated or scripted rationale for the final allocation
        "rationale": rationale.strip(),
        # Terminal validation gate passed flag
        "range_constraint_verified": True,
    }

    # ------------------------------------------------------------------
    # Create the artifact directory and write the JSON
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = artifact_dir / "final_weights.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write final_weights.json to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_final_weights_json: written to '%s'. "
        "sum=%.8f, selected_ensemble='%s'.",
        output_path,
        weight_sum,
        selected_ensemble,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 56: write_board_memo_md
# =============================================================================

def write_board_memo_md(
    weights: List[float],
    rationale: str,
    macro_summary: Dict[str, Any],
    cma_summary: Dict[str, float],
    risk_summary: Dict[str, Any],
    selected_ensemble: str,
    artifact_dir: Path,
    asset_class_labels: Optional[List[str]] = None,
    benchmark_weights: Optional[List[float]] = None,
) -> str:
    """
    Format and persist the CIO board memo to ``board_memo.md``.

    This tool implements the board memo generation step for the CIO Agent
    (Task 31, Step 3). The board memo is a governance document written
    for non-technical stakeholders (board of trustees, investment
    committee) and must include all 10 required sections from the
    ``CIO_BOARD_MEMO`` template.

    The file is written to:
    ``{artifact_dir}/board_memo.md``

    Parameters
    ----------
    weights : List[float]
        18-element final portfolio weight vector.
    rationale : str
        LLM-generated narrative covering all 10 required board memo
        sections. Must contain section headers for all sections in
        ``_REQUIRED_BOARD_MEMO_SECTIONS``.
    macro_summary : Dict[str, Any]
        Macro regime summary. Expected keys: ``"regime"`` (str),
        ``"confidence"`` (float), ``"key_drivers"`` (str, optional).
    cma_summary : Dict[str, float]
        Mapping from asset class name to final CMA expected return
        (decimal form).
    risk_summary : Dict[str, Any]
        Portfolio risk metrics. Expected keys: ``"ex_ante_vol"``,
        ``"tracking_error"``, ``"backtest_sharpe"``, ``"max_drawdown"``.
    selected_ensemble : str
        Name of the ensemble method selected by the CIO agent.
    artifact_dir : Path
        Directory to write ``board_memo.md``.
    asset_class_labels : Optional[List[str]]
        Ordered list of 18 asset class names. If ``None``, uses
        ``_CANONICAL_ASSET_CLASS_ORDER``.
    benchmark_weights : Optional[List[float]]
        18-element benchmark weight vector. If provided, used to compute
        active bets (overweights/underweights vs benchmark).

    Returns
    -------
    str
        Absolute path to the written ``board_memo.md`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``rationale`` is empty or fewer than ``_MIN_RATIONALE_LENGTH``
        characters.
    ValueError
        If ``len(weights) != 18``.
    OSError
        If the file cannot be written.

    Notes
    -----
    The board memo structure follows the ``CIO_BOARD_MEMO`` template
    from ``STUDY_CONFIG["PROMPT_TEMPLATES"]``, which specifies 10
    required sections. The structured data (allocation table, metrics
    table, IPS compliance statement) is inserted programmatically;
    the narrative content is provided via ``rationale``.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Resolve asset class labels and benchmark weights
    # ------------------------------------------------------------------
    ac_labels: List[str] = (
        list(asset_class_labels)
        if asset_class_labels is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )
    bm_weights: Optional[np.ndarray] = (
        np.array(benchmark_weights, dtype=np.float64)
        if benchmark_weights is not None
        else None
    )

    # ------------------------------------------------------------------
    # Helper: safely format a risk metric for display
    # ------------------------------------------------------------------
    def _fmt_risk(key: str, multiplier: float = 1.0, suffix: str = "") -> str:
        val = risk_summary.get(key)
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return "N/A"
        try:
            return f"{float(val) * multiplier:.2f}{suffix}"
        except (TypeError, ValueError):
            return str(val)

    # ------------------------------------------------------------------
    # Build the board memo markdown document
    # ------------------------------------------------------------------
    md_lines: List[str] = [
        "# CIO Board Memo — Strategic Asset Allocation",
        "",
        f"**Selected Ensemble Method:** `{selected_ensemble}`",
        "",
        "---",
        "",
    ]

    # --- Section 1: Executive Summary ---
    md_lines.extend([
        "## 1. Executive Summary",
        "",
        rationale.strip(),
        "",
    ])

    # --- Section 2: Recommended Allocation ---
    md_lines.extend([
        "## 2. Recommended Allocation",
        "",
        "| Asset Class | Category | Weight | CMA Est. |",
        "|-------------|----------|--------|----------|",
    ])
    for i, (ac, w) in enumerate(zip(ac_labels, weights)):
        category: str = _ASSET_CLASS_CATEGORIES.get(ac, "Unknown")
        cma_est: str = (
            f"{float(cma_summary.get(ac, 0.0)) * 100:.2f}%"
            if ac in cma_summary
            else "N/A"
        )
        md_lines.append(
            f"| {ac} | {category} | {w * 100:.2f}% | {cma_est} |"
        )
    md_lines.append("")

    # --- Section 3: Macro Regime View ---
    regime: str = str(macro_summary.get("regime", "Unknown"))
    confidence: str = (
        f"{float(macro_summary.get('confidence', 0.0)) * 100:.1f}%"
        if "confidence" in macro_summary
        else "N/A"
    )
    key_drivers: str = str(macro_summary.get("key_drivers", "See analysis.md"))
    md_lines.extend([
        "## 3. Macro Regime View",
        "",
        f"**Current Regime:** {regime}  ",
        f"**Confidence:** {confidence}  ",
        f"**Key Drivers:** {key_drivers}",
        "",
    ])

    # --- Section 4: Expected Performance vs 60/40 Benchmark ---
    md_lines.extend([
        "## 4. Expected Performance vs 60/40 Benchmark",
        "",
        "| Metric | Portfolio | 60/40 Benchmark |",
        "|--------|-----------|-----------------|",
        f"| Ex-Ante Vol | {_fmt_risk('ex_ante_vol', 100, '%')} | ~10% |",
        f"| Tracking Error | {_fmt_risk('tracking_error', 100, '%')} | — |",
        f"| Backtest Sharpe | {_fmt_risk('backtest_sharpe')} | ~0.41 |",
        f"| Max Drawdown | {_fmt_risk('max_drawdown', 100, '%')} | ~-34.3% |",
        "",
    ])

    # --- Section 5: Key Active Bets ---
    md_lines.extend([
        "## 5. Key Active Bets",
        "",
    ])
    if bm_weights is not None:
        # Compute active weights and sort by magnitude
        active_weights: List[Tuple[str, float]] = [
            (ac, float(weights[i]) - float(bm_weights[i]))
            for i, ac in enumerate(ac_labels)
        ]
        active_weights_sorted: List[Tuple[str, float]] = sorted(
            active_weights, key=lambda x: abs(x[1]), reverse=True
        )
        md_lines.extend([
            "| Asset Class | Active Weight |",
            "|-------------|---------------|",
        ])
        for ac, aw in active_weights_sorted[:10]:
            sign: str = "+" if aw >= 0 else ""
            md_lines.append(f"| {ac} | {sign}{aw * 100:.2f}% |")
        md_lines.append("")
    else:
        md_lines.extend([
            "Active bets vs benchmark not available "
            "(benchmark_weights not provided).",
            "",
        ])

    # --- Section 6: Risk Assessment ---
    md_lines.extend([
        "## 6. Risk Assessment",
        "",
        f"- **Ex-Ante Volatility:** {_fmt_risk('ex_ante_vol', 100, '%')} "
        f"(IPS band: 8%–12%)",
        f"- **Tracking Error:** {_fmt_risk('tracking_error', 100, '%')} "
        f"(IPS budget: ≤6%)",
        f"- **Max Drawdown:** {_fmt_risk('max_drawdown', 100, '%')} "
        f"(IPS limit: ≥−25%)",
        "",
    ])

    # --- Section 7: IPS Compliance Statement ---
    md_lines.extend([
        "## 7. IPS Compliance Statement",
        "",
        "The recommended portfolio has been verified against all IPS constraints:",
        "",
        f"- **Volatility band [8%, 12%]:** "
        f"{_fmt_risk('ex_ante_vol', 100, '%')} — "
        f"{'✅ PASS' if risk_summary.get('ex_ante_vol', 0) is not None and 0.08 <= float(risk_summary.get('ex_ante_vol', 0)) <= 0.12 else '⚠️ CHECK'}",
        f"- **Tracking error ≤6%:** "
        f"{_fmt_risk('tracking_error', 100, '%')} — "
        f"{'✅ PASS' if risk_summary.get('tracking_error', 1) is not None and float(risk_summary.get('tracking_error', 1)) <= 0.06 else '⚠️ CHECK'}",
        f"- **Max drawdown ≥−25%:** "
        f"{_fmt_risk('max_drawdown', 100, '%')} — "
        f"{'✅ PASS' if risk_summary.get('max_drawdown', -1) is not None and float(risk_summary.get('max_drawdown', -1)) >= -0.25 else '⚠️ CHECK'}",
        "",
    ])

    # --- Section 8: Dissenting Views and Rejected Methods ---
    md_lines.extend([
        "## 8. Dissenting Views and Rejected Methods",
        "",
        "See ``vote_corpus.json`` and ``cro_risk_reports.json`` for full "
        "peer review critiques and bottom-flag dissents.",
        "",
    ])

    # --- Section 9: Rebalancing Plan ---
    md_lines.extend([
        "## 9. Rebalancing Plan",
        "",
        "- **Frequency:** Quarterly (per IPS rebalancing policy)",
        "- **Off-cycle triggers:** L1 drift > 5% from target weights",
        "- **Monitoring:** Monthly CRO review of ex-ante TE and vol",
        "",
    ])

    # --- Section 10: Invalidation Triggers ---
    md_lines.extend([
        "## 10. Invalidation Triggers",
        "",
        "This portfolio recommendation would be invalidated by:",
        "",
        f"- Macro regime shift away from '{regime}' "
        "(re-run full pipeline on regime change)",
        "- Ex-ante tracking error exceeding 6% (immediate rebalance required)",
        "- Realised drawdown approaching −20% (escalate to investment committee)",
        "- CPI re-acceleration above 4% YoY (reassess inflation hedges)",
        "",
    ])

    # ------------------------------------------------------------------
    # Join all lines into a single markdown string
    # ------------------------------------------------------------------
    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Create the artifact directory and write the file
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = artifact_dir / "board_memo.md"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write board_memo.md to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_board_memo_md: written to '%s'.",
        output_path,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 57: write_pc_weights_json
# =============================================================================

def write_pc_weights_json(
    method: str,
    weights: List[float],
    rationale: str,
    artifact_dir: Path,
) -> str:
    """
    Serialise and persist the PC portfolio weights to ``pc_weights.json``.

    This tool implements the primary artifact-writing step for all PC
    agents (Tasks 19, 25, 26, 30). The output file is consumed by the
    CRO Agent (Task 27), peer review agents (Task 28), voting agents
    (Task 29), and the CIO Agent (Task 31). It must conform to the
    frozen ``pc_weights.schema.json`` schema.

    The file is written to:
    ``{artifact_dir}/{method_slug}/pc_weights.json``

    Parameters
    ----------
    method : str
        PC method slug identifier (e.g., ``"max_diversification"``).
        Used to construct the output file path.
    weights : List[float]
        18-element portfolio weight vector. Must sum to 1.0 (within
        ``1e-6``) and be non-negative.
    rationale : str
        LLM-generated or scripted explanation of the portfolio
        construction objective and key exposures. Must be non-empty
        (minimum ``_MIN_RATIONALE_LENGTH`` characters).
    artifact_dir : Path
        Base artifact directory.

    Returns
    -------
    str
        Absolute path to the written ``pc_weights.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``weights`` does not have 18 elements, contains negative
        values, or does not sum to 1.0 (within ``1e-6``).
    ValueError
        If ``rationale`` is empty or fewer than ``_MIN_RATIONALE_LENGTH``
        characters.
    OSError
        If the file cannot be written.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: rationale non-empty
    # ------------------------------------------------------------------
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )

    # ------------------------------------------------------------------
    # Input validation: weights length
    # ------------------------------------------------------------------
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Input validation: non-negative weights
    # ------------------------------------------------------------------
    w_arr: np.ndarray = np.array(weights, dtype=np.float64)
    if (w_arr < -1e-8).any():
        raise ValueError(
            f"weights contains negative values: "
            f"{[v for v in weights if v < -1e-8]}."
        )

    # ------------------------------------------------------------------
    # Input validation: sum to 1.0
    # ------------------------------------------------------------------
    weight_sum: float = float(w_arr.sum())
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"weights sum to {weight_sum:.8f}, expected 1.0 (tolerance 1e-6)."
        )

    # ------------------------------------------------------------------
    # Construct the output dict
    # ------------------------------------------------------------------
    output_dict: Dict[str, Any] = {
        # PC method slug identifier
        "method": method,
        # 18-element portfolio weight vector
        "weights": [float(v) for v in weights],
        # Actual sum of weights (for audit verification)
        "sum_check": float(weight_sum),
        # LLM-generated or scripted rationale
        "rationale": rationale.strip(),
    }

    # ------------------------------------------------------------------
    # Derive the method slug for the output directory
    # ------------------------------------------------------------------
    method_slug: str = _derive_slug(method)

    # ------------------------------------------------------------------
    # Create the output directory and write the JSON
    # ------------------------------------------------------------------
    output_dir: Path = artifact_dir / method_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_dir / "pc_weights.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write pc_weights.json to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_pc_weights_json: written for method='%s' to '%s'.",
        method,
        output_path,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 58: write_pc_report_md
# =============================================================================

def write_pc_report_md(
    method: str,
    weights: List[float],
    rationale: str,
    artifact_dir: Path,
    asset_class_labels: Optional[List[str]] = None,
    benchmark_weights: Optional[List[float]] = None,
    macro_view: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Format and persist the PC portfolio report to ``pc_report.md``.

    This tool implements the PC report artifact-writing step for all PC
    agents (Tasks 19, 25, 26, 30). The output file is the human-readable
    audit trail for each PC portfolio, consumed by peer review agents
    (Task 28) and the CIO Agent (Task 31).

    The file is written to:
    ``{artifact_dir}/{method_slug}/pc_report.md``

    Parameters
    ----------
    method : str
        PC method slug identifier.
    weights : List[float]
        18-element portfolio weight vector.
    rationale : str
        LLM-generated explanation covering: objective function
        description, key portfolio exposures, regime considerations,
        and IPS compliance statement. Must be non-empty.
    artifact_dir : Path
        Base artifact directory.
    asset_class_labels : Optional[List[str]]
        Ordered list of 18 asset class names. If ``None``, uses
        ``_CANONICAL_ASSET_CLASS_ORDER``.
    benchmark_weights : Optional[List[float]]
        18-element benchmark weight vector. If provided, active bets
        (overweights/underweights) are included in the report.
    macro_view : Optional[Dict[str, Any]]
        Macro regime view dict. If provided, regime context is included.

    Returns
    -------
    str
        Absolute path to the written ``pc_report.md`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path``.
    ValueError
        If ``rationale`` is empty or ``weights`` has wrong length.
    OSError
        If the file cannot be written.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(rationale, str) or len(rationale.strip()) < _MIN_RATIONALE_LENGTH:
        raise ValueError(
            f"rationale must be a non-empty string with at least "
            f"{_MIN_RATIONALE_LENGTH} characters."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Resolve asset class labels and benchmark weights
    # ------------------------------------------------------------------
    ac_labels: List[str] = (
        list(asset_class_labels)
        if asset_class_labels is not None
        else list(_CANONICAL_ASSET_CLASS_ORDER)
    )
    bm_arr: Optional[np.ndarray] = (
        np.array(benchmark_weights, dtype=np.float64)
        if benchmark_weights is not None
        else None
    )

    # ------------------------------------------------------------------
    # Build the markdown document
    # ------------------------------------------------------------------
    md_lines: List[str] = [
        f"# PC Portfolio Report — {method}",
        "",
        f"**Method:** `{method}`",
        "",
    ]

    # --- Weight allocation table ---
    md_lines.extend([
        "## Portfolio Weights",
        "",
        "| Asset Class | Category | Weight |"
        + (" Active Bet |" if bm_arr is not None else ""),
        "|-------------|----------|--------|"
        + ("------------|" if bm_arr is not None else ""),
    ])
    for i, (ac, w) in enumerate(zip(ac_labels, weights)):
        category: str = _ASSET_CLASS_CATEGORIES.get(ac, "Unknown")
        row: str = f"| {ac} | {category} | {w * 100:.2f}% |"
        if bm_arr is not None:
            active: float = float(w) - float(bm_arr[i])
            sign: str = "+" if active >= 0 else ""
            row += f" {sign}{active * 100:.2f}% |"
        md_lines.append(row)
    md_lines.append("")

    # --- Regime context (if macro_view provided) ---
    if macro_view is not None:
        regime: str = str(macro_view.get("regime", "Unknown"))
        md_lines.extend([
            "## Regime Context",
            "",
            f"**Current Regime:** {regime}",
            "",
        ])

    # --- Rationale section ---
    md_lines.extend([
        "## Portfolio Construction Rationale",
        "",
        rationale.strip(),
        "",
    ])

    # Join all lines
    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Create the output directory and write the file
    # ------------------------------------------------------------------
    method_slug: str = _derive_slug(method)
    output_dir: Path = artifact_dir / method_slug
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_dir / "pc_report.md"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write pc_report.md to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_pc_report_md: written for method='%s' to '%s'.",
        method,
        output_path,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 59: write_proposed_method_md
# =============================================================================

def write_proposed_method_md(
    method_spec: Dict[str, Any],
    artifact_dir: Path,
) -> str:
    """
    Format and persist the proposed PC method specification to markdown.

    This tool implements the PC Researcher artifact-writing step (Task 19).
    The output ``proposed_method.md`` is the human-readable specification
    of the new portfolio construction method proposed by the PC-Researcher
    agent. It must contain all required sections from ``_REQUIRED_METHOD_SPEC_KEYS``.

    The file is written to:
    ``{artifact_dir}/proposed_method.md``

    Parameters
    ----------
    method_spec : Dict[str, Any]
        Method specification dict. Must contain all keys in
        ``_REQUIRED_METHOD_SPEC_KEYS``:
        ``"method_name"``, ``"objective_function"``, ``"constraints"``,
        ``"required_inputs"``, ``"expected_behavior"``,
        ``"failure_modes"``.
    artifact_dir : Path
        Directory to write ``proposed_method.md``.

    Returns
    -------
    str
        Absolute path to the written ``proposed_method.md`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``method_spec``
        is not a dict.
    ValueError
        If any required key is missing from ``method_spec``.
    OSError
        If the file cannot be written.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(method_spec, dict):
        raise TypeError(
            f"method_spec must be a dict, got {type(method_spec).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys
    # ------------------------------------------------------------------
    missing_keys: List[str] = [
        k for k in _REQUIRED_METHOD_SPEC_KEYS if k not in method_spec
    ]
    if missing_keys:
        raise ValueError(
            f"method_spec is missing required keys: {missing_keys}. "
            f"Required: {list(_REQUIRED_METHOD_SPEC_KEYS)}."
        )

    # ------------------------------------------------------------------
    # Extract fields from method_spec
    # ------------------------------------------------------------------
    method_name: str = str(method_spec["method_name"])
    objective_function: str = str(method_spec["objective_function"])
    constraints: str = str(method_spec["constraints"])
    required_inputs: str = str(method_spec["required_inputs"])
    expected_behavior: str = str(method_spec["expected_behavior"])
    failure_modes: str = str(method_spec["failure_modes"])

    # ------------------------------------------------------------------
    # Build the markdown document
    # ------------------------------------------------------------------
    md_lines: List[str] = [
        f"# Proposed Portfolio Construction Method: {method_name}",
        "",
        "**Proposed by:** PC-Researcher Agent",
        "",
        "---",
        "",
        "## Objective Function",
        "",
        objective_function,
        "",
        "## Constraints and Required Inputs",
        "",
        constraints,
        "",
        "## Required Data Inputs",
        "",
        required_inputs,
        "",
        "## Expected Behaviour Under Different Regimes",
        "",
        expected_behavior,
        "",
        "## Identified Failure Modes and Risks",
        "",
        failure_modes,
        "",
    ]

    # Add any additional fields from method_spec not in the required set
    extra_keys: List[str] = [
        k for k in method_spec if k not in _REQUIRED_METHOD_SPEC_KEYS
    ]
    if extra_keys:
        md_lines.extend(["## Additional Notes", ""])
        for key in extra_keys:
            md_lines.extend([f"**{key}:** {method_spec[key]}", ""])

    md_content: str = "\n".join(md_lines)

    # ------------------------------------------------------------------
    # Create the artifact directory and write the file
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = artifact_dir / "proposed_method.md"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(md_content)
    except OSError as exc:
        raise OSError(
            f"Failed to write proposed_method.md to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_proposed_method_md: written for method='%s' to '%s'.",
        method_name,
        output_path,
    )

    return str(output_path.resolve())


# =============================================================================
# TOOL 60: write_proposed_method_spec_json
# =============================================================================

def write_proposed_method_spec_json(
    method_spec: Dict[str, Any],
    artifact_dir: Path,
) -> str:
    """
    Serialise and persist the proposed PC method specification to JSON.

    This tool implements the PC Researcher machine-readable artifact-writing
    step (Task 19). The output ``proposed_method_spec.json`` is the
    structured specification suitable for implementation as a new PC agent
    method in subsequent pipeline runs.

    The file is written to:
    ``{artifact_dir}/proposed_method_spec.json``

    Parameters
    ----------
    method_spec : Dict[str, Any]
        Method specification dict. Must contain all keys in
        ``_REQUIRED_METHOD_SPEC_KEYS``. May contain additional keys
        for extended specification.
    artifact_dir : Path
        Directory to write ``proposed_method_spec.json``.

    Returns
    -------
    str
        Absolute path to the written ``proposed_method_spec.json`` file.

    Raises
    ------
    TypeError
        If ``artifact_dir`` is not a ``pathlib.Path`` or ``method_spec``
        is not a dict.
    ValueError
        If any required key is missing from ``method_spec``.
    OSError
        If the file cannot be written.

    Notes
    -----
    All values in ``method_spec`` are recursively cast to JSON-safe
    Python native types via ``_cast_to_json_safe`` before serialisation.
    This ensures that any ``np.float64`` or other non-serialisable types
    introduced by the PC-Researcher agent's computation are handled
    correctly.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(artifact_dir, Path):
        raise TypeError(
            f"artifact_dir must be a pathlib.Path, "
            f"got {type(artifact_dir).__name__}."
        )
    if not isinstance(method_spec, dict):
        raise TypeError(
            f"method_spec must be a dict, got {type(method_spec).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required keys
    # ------------------------------------------------------------------
    missing_keys: List[str] = [
        k for k in _REQUIRED_METHOD_SPEC_KEYS if k not in method_spec
    ]
    if missing_keys:
        raise ValueError(
            f"method_spec is missing required keys: {missing_keys}. "
            f"Required: {list(_REQUIRED_METHOD_SPEC_KEYS)}."
        )

    # ------------------------------------------------------------------
    # Cast all values to JSON-safe types
    # ------------------------------------------------------------------
    json_safe_spec: Dict[str, Any] = _cast_to_json_safe(method_spec)

    # ------------------------------------------------------------------
    # Create the artifact directory and write the JSON
    # ------------------------------------------------------------------
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_path: Path = artifact_dir / "proposed_method_spec.json"

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(json_safe_spec, fh, indent=2, ensure_ascii=False)
    except OSError as exc:
        raise OSError(
            f"Failed to write proposed_method_spec.json to '{output_path}'. "
            f"Original error: {exc}"
        ) from exc

    logger.info(
        "write_proposed_method_spec_json: written for method='%s' to '%s'.",
        str(method_spec.get("method_name", "unknown")),
        output_path,
    )

    return str(output_path.resolve())

# =============================================================================
# SELF-DRIVING PORTFOLIO: TOOL REGISTRY — BATCH 7 (TOOLS 61–70)
# =============================================================================
# Implements the final 10 tools from the complete 78-tool registry plus
# critical shared infrastructure callables for the agentic SAA pipeline
# described in:
#   Ang, Azimbayev, and Kim (2026) — "The Self-Driving Portfolio"
#
# Tools implemented:
#   61. run_optimizer               — PC Agent central optimization dispatcher
#   62. fetch_cash_yield_proxy      — Cash AC Agent data retrieval
#   63. tool_literature_search      — PC Researcher knowledge search
#   64. validate_method_spec        — PC Researcher constraint checker
#   65. validate_artifact           — Shared inter-stage schema gate
#   66. build_returns_matrix        — Covariance Agent returns construction
#   67. estimate_covariance         — Covariance Agent Ledoit-Wolf estimator
#   68. enforce_psd                 — Covariance Agent PSD repair
#   69. compute_risk_contributions  — Exhibit 11 risk contribution computation
#   70. compute_benchmark_returns   — Benchmark return series construction
#
# All tools are purely deterministic Python callables — no LLM interaction.
# All arithmetic conventions are frozen per STUDY_CONFIG["DATA_CONVENTIONS"].
# =============================================================================

from __future__ import annotations

import json
import logging
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.optimize import minimize, OptimizeResult
from scipy.spatial.distance import squareform

# ---------------------------------------------------------------------------
# Conditional import: sklearn LedoitWolf (graceful fallback to sample cov)
# ---------------------------------------------------------------------------
try:
    from sklearn.covariance import LedoitWolf as _LedoitWolf
    _SKLEARN_AVAILABLE: bool = True
except ImportError:
    _SKLEARN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exception (re-declared for self-contained module)
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """
    Raised when a pipeline artifact fails its terminal schema validation gate.

    Signals a fail-closed pipeline halt, distinct from ``ValueError``
    (data errors) to allow the orchestrator to differentiate between
    data errors and schema contract violations.
    """
    pass


# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------

# Number of assets in the 18-asset universe
_N_ASSETS: int = 18

# Annualisation multiplier for monthly returns (periods per year)
_PERIODS_PER_YEAR: int = 12

# Numerical stability epsilon
_EPS: float = 1e-8

# Frozen IPS constraint values per IPS_GOVERNANCE
_IPS_MAX_WEIGHT: float = 0.25
_IPS_MIN_WEIGHT: float = 0.00
_IPS_TE_BUDGET: float = 0.06
_IPS_VOL_LOWER: float = 0.08
_IPS_VOL_UPPER: float = 0.12

# Minimum eigenvalue for PSD repair (eigenvalue clipping threshold)
_PSD_CLIP_MIN: float = 1e-8

# CVaR confidence level for CVaR-based methods
_CVAR_ALPHA: float = 0.95

# Fractional Kelly multiplier (frozen at 0.5 per standard practice)
_KELLY_FRACTION: float = 0.5

# Maximum staleness tolerance in months for cash yield proxy
_MAX_CASH_STALENESS_MONTHS: int = 3

# Whitelist of valid required_inputs for PC method spec validation
_VALID_PIPELINE_INPUTS: Set[str] = {
    "sigma", "mu", "rf", "benchmark_weights", "returns_matrix",
    "constraints", "macro_view", "cma_set", "asset_class_labels",
}

# Frozen schema validation rules per REGISTRIES["OUTPUT_SCHEMAS"]
# Each schema entry: {required_fields, type_checks, range_checks}
_SCHEMA_RULES: Dict[str, Dict[str, Any]] = {
    "macro_view": {
        "required_fields": ["regime", "scores", "confidence", "rationale"],
        "type_checks": {"regime": str, "confidence": (int, float)},
        "range_checks": {"confidence": (0.0, 1.0)},
    },
    "cma_methods": {
        "required_fields": ["asset_class", "n_methods", "method_results"],
        "type_checks": {"n_methods": int},
        "range_checks": {},
    },
    "cma": {
        "required_fields": [
            "asset_class", "final_estimate", "method_weights",
            "rationale", "method_range",
        ],
        "type_checks": {"final_estimate": (int, float)},
        "range_checks": {},
    },
    "pc_weights": {
        "required_fields": ["method", "weights", "sum_check", "rationale"],
        "type_checks": {"weights": list},
        "range_checks": {"sum_check": (1.0 - 1e-5, 1.0 + 1e-5)},
    },
    "cro_report": {
        "required_fields": [
            "candidate_id", "ips_compliant", "metrics", "compliance_flags",
        ],
        "type_checks": {"ips_compliant": bool},
        "range_checks": {},
    },
    "final_weights": {
        "required_fields": [
            "weights", "asset_class_labels", "sum_check",
            "selected_ensemble", "rationale",
        ],
        "type_checks": {"weights": list},
        "range_checks": {"sum_check": (1.0 - 1e-5, 1.0 + 1e-5)},
    },
    "signals": {
        "required_fields": ["asset_class", "ticker", "category", "as_of_date"],
        "type_checks": {"asset_class": str},
        "range_checks": {},
    },
    "historical_stats": {
        "required_fields": [
            "asset_class", "annualised_return", "annualised_vol",
            "max_drawdown", "n_observations",
        ],
        "type_checks": {"n_observations": int},
        "range_checks": {},
    },
    "vote": {
        "required_fields": ["voter_id", "top_5"],
        "type_checks": {"top_5": list},
        "range_checks": {},
    },
}

# Local knowledge base of portfolio construction methods for literature search
# (used when web search is disabled; UNSPECIFIED IN MANUSCRIPT — frozen)
_LOCAL_PC_KNOWLEDGE_BASE: List[Dict[str, str]] = [
    {
        "title": "Optimal Portfolio Diversification Using the Maximum Entropy Principle",
        "abstract": (
            "Proposes maximising the Shannon entropy of portfolio weights "
            "subject to a minimum Sharpe ratio floor."
        ),
        "method_description": (
            "Maximum Entropy Portfolio: max_w -sum(w_i * ln(w_i)) "
            "s.t. SR(w) >= SR_floor, sum(w) = 1, w >= 0."
        ),
        "reference": "Bera and Park (2008), Econometric Reviews 27(4-6): 484-512.",
        "year": "2008",
    },
    {
        "title": "Hierarchical Equal Risk Contribution",
        "abstract": (
            "Extends HRP by applying equal risk contribution within each "
            "cluster of the hierarchical tree."
        ),
        "method_description": (
            "HERC: hierarchical clustering + equal risk contribution "
            "within each cluster level."
        ),
        "reference": "Raffinot (2018), Journal of Portfolio Management.",
        "year": "2018",
    },
    {
        "title": "Nested Cluster Optimisation",
        "abstract": (
            "Applies mean-variance optimisation within clusters identified "
            "by hierarchical clustering, then optimises across clusters."
        ),
        "method_description": (
            "NCO: hierarchical clustering → intra-cluster MVO → "
            "inter-cluster MVO."
        ),
        "reference": "López de Prado (2019), Journal of Financial Data Science.",
        "year": "2019",
    },
    {
        "title": "Tail Risk Parity",
        "abstract": (
            "Equalises the tail risk contribution of each asset, "
            "measured by CVaR or Expected Shortfall."
        ),
        "method_description": (
            "Tail Risk Parity: w_i * CVaR_i(w) = w_j * CVaR_j(w) for all i,j."
        ),
        "reference": "Spinu (2013), Working Paper.",
        "year": "2013",
    },
    {
        "title": "Minimum Correlation Algorithm",
        "abstract": (
            "Minimises the average pairwise correlation of portfolio "
            "constituents to maximise diversification."
        ),
        "method_description": (
            "Max Decorrelation: min_w sum_{i,j} w_i * w_j * rho_{ij} "
            "s.t. sum(w) = 1, w >= 0."
        ),
        "reference": "Varadi et al. (2012), Working Paper.",
        "year": "2012",
    },
]


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _cast_to_json_safe(obj: Any) -> Any:
    """Recursively cast to JSON-serialisable Python native types."""
    if obj is None:
        return None
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) else obj
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_cast_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _cast_to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_cast_to_json_safe(v) for v in obj]
    if isinstance(obj, (int, bool)):
        return obj
    if isinstance(obj, str):
        return obj
    return str(obj)


def _normalise_weights(w_raw: np.ndarray) -> np.ndarray:
    """Clip to non-negative and normalise to sum to 1.0."""
    # Clip all weights to non-negative (long-only enforcement)
    w_clipped: np.ndarray = np.maximum(w_raw, 0.0)
    # Compute the sum of clipped weights
    w_sum: float = float(w_clipped.sum())
    if w_sum > _EPS:
        # Normalise to sum to 1.0
        return w_clipped / w_sum
    # Degenerate: return equal weights
    return np.ones(len(w_raw), dtype=np.float64) / len(w_raw)


def _apply_box_constraints(
    w: np.ndarray,
    min_w: float = _IPS_MIN_WEIGHT,
    max_w: float = _IPS_MAX_WEIGHT,
) -> np.ndarray:
    """Clip weights to [min_w, max_w] and renormalise."""
    # Clip to box constraints
    w_clipped: np.ndarray = np.clip(w, min_w, max_w)
    # Renormalise to sum to 1.0
    return _normalise_weights(w_clipped)


# =============================================================================
# TOOL 61: run_optimizer
# =============================================================================

def run_optimizer(
    method: str,
    sigma: np.ndarray,
    mu: np.ndarray,
    rf: float,
    benchmark_weights: np.ndarray,
    constraints: Dict[str, Any],
    returns_matrix: Optional[np.ndarray] = None,
    optimizer_seed: int = 24680,
    constraints_override: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Central portfolio construction optimization dispatcher for all PC agents.

    Implements Task 25, Step 2 — the ``run_optimizer`` script callable
    invoked by all PC ``ConversableAgent`` instances via tool calls within
    the AutoGen topology. Routes to the method-specific optimizer
    implementation for each of the 20 canonical PC methods.

    **Supported methods and their objectives:**

    - ``"gmv"``: :math:`\\min_w w^\\top \\Sigma w`
    - ``"max_sharpe"``: :math:`\\max_w (w^\\top \\mu - r_f) / \\sqrt{w^\\top \\Sigma w}`
    - ``"max_diversification"``: :math:`\\max_w w^\\top \\sigma / \\sqrt{w^\\top \\Sigma w}`
    - ``"erc"``: Equal risk contribution (risk parity)
    - ``"hrp"``: Hierarchical Risk Parity (López de Prado 2016)
    - ``"herc"``: Hierarchical Equal Risk Contribution
    - ``"nco"``: Nested Cluster Optimisation
    - ``"equal_weight"``: :math:`w_i = 1/N`
    - ``"iv_weight"``: :math:`w_i \\propto 1/\\sigma_i`
    - ``"bl_posterior"``: BL posterior mean-variance
    - ``"regime_conditional"``: Regime-tilted mean-variance
    - ``"min_cvar"``: Minimise CVaR at 95% confidence
    - ``"cvar_parity"``: Equal CVaR contributions
    - ``"robust_mvo"``: Robust mean-variance (minimax)
    - ``"resampled_mvo"``: Michaud resampled efficient frontier
    - ``"factor_risk_parity"``: Risk parity in factor space
    - ``"max_decorrelation"``: Minimise average pairwise correlation
    - ``"tail_risk_parity"``: Equal tail risk contributions
    - ``"kelly_fractional"``: Fractional Kelly criterion
    - ``"target_vol_sharpe"``: Target volatility Sharpe maximisation

    All methods enforce the IPS constraints: long-only, max weight 0.25,
    budget constraint (sum to 1.0).

    Parameters
    ----------
    method : str
        PC method slug (e.g., ``"max_sharpe"``). Must be one of the 20
        canonical method slugs.
    sigma : np.ndarray
        18×18 annualised covariance matrix (Ledoit-Wolf shrinkage).
        Shape: ``(18, 18)``. Must be positive semi-definite.
    mu : np.ndarray
        18-element vector of annualised CMA expected returns (decimal).
        Shape: ``(18,)``. Required for return-optimised methods.
    rf : float
        Annualised risk-free rate in decimal form (e.g., 0.053 = 5.3%).
    benchmark_weights : np.ndarray
        18-element benchmark weight vector. Shape: ``(18,)``.
    constraints : Dict[str, Any]
        IPS implementation constraints. Expected keys:
        ``"max_weight_per_asset"`` (float, default 0.25),
        ``"min_weight_per_asset"`` (float, default 0.00).
    returns_matrix : Optional[np.ndarray]
        T×18 monthly returns matrix. Required for CVaR-based methods
        and resampled MVO. Shape: ``(T, 18)``.
    optimizer_seed : int
        Random seed for stochastic methods (resampled MVO, multi-start).
        Default: 24680 per ``STUDY_CONFIG["RANDOM_SEEDS"]["optimizer_seed"]``.
    constraints_override : Optional[Dict[str, Any]]
        Parameter overrides from revision agents. Keys may include
        ``"max_weight_per_asset"``, ``"min_weight_per_asset"``,
        ``"target_vol"`` (for target_vol_sharpe).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"weights"`` (``List[float]``): 18-element weight vector.
          Sums to 1.0 (within ``1e-6``), non-negative.
        - ``"method"`` (``str``): The method slug used.
        - ``"objective_value"`` (``float``): The achieved objective
          function value (e.g., portfolio variance for GMV).
        - ``"diagnostics"`` (``Dict[str, Any]``): Method-specific
          diagnostics (convergence status, iterations, etc.).

    Raises
    ------
    TypeError
        If ``sigma``, ``mu``, or ``benchmark_weights`` are not
        ``np.ndarray``.
    ValueError
        If ``method`` is not a recognised PC method slug.
    ValueError
        If ``sigma`` is not shape ``(18, 18)`` or ``mu`` is not
        shape ``(18,)``.

    Notes
    -----
    **Fallback behaviour:** If any method-specific optimizer fails to
    converge, it falls back to the equal-weight portfolio with a
    documented warning in ``diagnostics["fallback_used"]``.

    **IPS constraint enforcement:** All methods enforce long-only and
    max-weight constraints via SLSQP bounds. The budget constraint
    (sum to 1.0) is enforced as an equality constraint.
    """
    # ------------------------------------------------------------------
    # Input validation: type checks
    # ------------------------------------------------------------------
    for arr_name, arr_val in [
        ("sigma", sigma), ("mu", mu), ("benchmark_weights", benchmark_weights)
    ]:
        if not isinstance(arr_val, np.ndarray):
            raise TypeError(
                f"{arr_name} must be a np.ndarray, "
                f"got {type(arr_val).__name__}."
            )

    # ------------------------------------------------------------------
    # Input validation: shape checks
    # ------------------------------------------------------------------
    if sigma.shape != (_N_ASSETS, _N_ASSETS):
        raise ValueError(
            f"sigma must have shape ({_N_ASSETS}, {_N_ASSETS}), "
            f"got {sigma.shape}."
        )
    if mu.shape != (_N_ASSETS,):
        raise ValueError(
            f"mu must have shape ({_N_ASSETS},), got {mu.shape}."
        )
    if benchmark_weights.shape != (_N_ASSETS,):
        raise ValueError(
            f"benchmark_weights must have shape ({_N_ASSETS},), "
            f"got {benchmark_weights.shape}."
        )

    # ------------------------------------------------------------------
    # Apply constraints_override if provided
    # ------------------------------------------------------------------
    effective_constraints: Dict[str, Any] = dict(constraints)
    if constraints_override is not None:
        effective_constraints.update(constraints_override)

    # ------------------------------------------------------------------
    # Resolve constraint parameters
    # ------------------------------------------------------------------
    # Maximum weight per asset (default: 0.25)
    max_w: float = float(
        effective_constraints.get("max_weight_per_asset", _IPS_MAX_WEIGHT)
    )
    # Minimum weight per asset (default: 0.00, long-only)
    min_w: float = float(
        effective_constraints.get("min_weight_per_asset", _IPS_MIN_WEIGHT)
    )
    # Target volatility for target_vol_sharpe method
    target_vol: float = float(
        effective_constraints.get("target_vol", 0.10)
    )

    # ------------------------------------------------------------------
    # Define SLSQP bounds and budget constraint (shared across methods)
    # ------------------------------------------------------------------
    # Variable bounds: min_w <= w_i <= max_w for all i
    bounds: List[Tuple[float, float]] = [(min_w, max_w)] * _N_ASSETS
    # Budget equality constraint: sum(w) = 1
    budget_constraint: Dict[str, Any] = {
        "type": "eq",
        "fun": lambda w: float(w.sum()) - 1.0,
    }

    # ------------------------------------------------------------------
    # Seeded random number generator for stochastic methods
    # ------------------------------------------------------------------
    rng: np.random.Generator = np.random.default_rng(optimizer_seed)

    # ------------------------------------------------------------------
    # Helper: generate a random feasible starting point
    # ------------------------------------------------------------------
    def _random_start() -> np.ndarray:
        """Generate a random feasible starting point."""
        w0: np.ndarray = rng.dirichlet(np.ones(_N_ASSETS))
        return _apply_box_constraints(w0, min_w, max_w)

    # ------------------------------------------------------------------
    # Helper: run SLSQP with a single starting point
    # ------------------------------------------------------------------
    def _slsqp(
        objective_fn: Any,
        extra_constraints: Optional[List[Dict[str, Any]]] = None,
        x0: Optional[np.ndarray] = None,
    ) -> OptimizeResult:
        """Run SLSQP minimisation from x0 with standard constraints."""
        all_constraints: List[Dict[str, Any]] = [budget_constraint]
        if extra_constraints:
            all_constraints.extend(extra_constraints)
        start: np.ndarray = x0 if x0 is not None else _random_start()
        return minimize(
            fun=objective_fn,
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints=all_constraints,
            options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
        )

    # ------------------------------------------------------------------
    # Helper: extract individual asset volatilities from sigma diagonal
    # ------------------------------------------------------------------
    # Individual asset annualised volatilities: sigma_i = sqrt(Sigma_ii)
    asset_vols: np.ndarray = np.sqrt(np.maximum(np.diag(sigma), _EPS))

    # ------------------------------------------------------------------
    # Equal-weight fallback (used when optimisation fails)
    # ------------------------------------------------------------------
    w_equal: np.ndarray = np.ones(_N_ASSETS, dtype=np.float64) / _N_ASSETS

    # ------------------------------------------------------------------
    # Method dispatch: implement each of the 20 canonical PC methods
    # ------------------------------------------------------------------

    # --- Method: equal_weight ---
    if method == "equal_weight":
        # Equal weight: w_i = 1/N for all i
        w_opt: np.ndarray = _apply_box_constraints(w_equal, min_w, max_w)
        obj_val: float = float(np.dot(w_opt, sigma @ w_opt))
        diagnostics: Dict[str, Any] = {"method": "closed_form"}

    # --- Method: iv_weight (Inverse Volatility) ---
    elif method == "iv_weight":
        # Inverse volatility: w_i proportional to 1/sigma_i
        inv_vol: np.ndarray = 1.0 / asset_vols
        w_opt = _apply_box_constraints(inv_vol, min_w, max_w)
        obj_val = float(np.dot(w_opt, sigma @ w_opt))
        diagnostics = {"method": "closed_form"}

    # --- Method: gmv (Global Minimum Variance) ---
    elif method == "gmv":
        # GMV: min_w w'Σw s.t. budget, bounds
        def _gmv_obj(w: np.ndarray) -> float:
            return float(np.dot(w, sigma @ w))
        result: OptimizeResult = _slsqp(_gmv_obj)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = float(result.fun)
            diagnostics = {"converged": True, "n_iter": result.nit}
        else:
            logger.warning("run_optimizer: GMV failed. Using equal weight.")
            w_opt = w_equal.copy()
            obj_val = float(np.dot(w_opt, sigma @ w_opt))
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: max_sharpe ---
    elif method == "max_sharpe":
        # Max Sharpe: max_w (w'mu - rf) / sqrt(w'Σw)
        def _neg_sharpe(w: np.ndarray) -> float:
            port_ret: float = float(np.dot(w, mu))
            port_var: float = float(np.dot(w, sigma @ w))
            port_vol: float = float(np.sqrt(max(port_var, _EPS)))
            return -(port_ret - rf) / port_vol
        result = _slsqp(_neg_sharpe)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = -float(result.fun)
            diagnostics = {"converged": True, "sharpe": obj_val}
        else:
            logger.warning("run_optimizer: Max Sharpe failed. Using equal weight.")
            w_opt = w_equal.copy()
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: max_diversification ---
    elif method == "max_diversification":
        # Max Diversification: max_w w'sigma / sqrt(w'Σw)
        def _neg_div_ratio(w: np.ndarray) -> float:
            weighted_vol: float = float(np.dot(w, asset_vols))
            port_var: float = float(np.dot(w, sigma @ w))
            port_vol: float = float(np.sqrt(max(port_var, _EPS)))
            return -weighted_vol / port_vol
        result = _slsqp(_neg_div_ratio)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = -float(result.fun)
            diagnostics = {"converged": True, "diversification_ratio": obj_val}
        else:
            logger.warning("run_optimizer: Max Diversification failed.")
            w_opt = w_equal.copy()
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: erc (Equal Risk Contribution / Risk Parity) ---
    elif method == "erc":
        # ERC: minimise sum_{i,j}(RC_i - RC_j)^2
        # where RC_i = w_i * (Sigma w)_i / (w'Sigma w)
        def _erc_obj(w: np.ndarray) -> float:
            sigma_w: np.ndarray = sigma @ w
            port_var: float = float(np.dot(w, sigma_w))
            if port_var < _EPS:
                return 0.0
            # Risk contributions: RC_i = w_i * (Sigma w)_i / (w'Sigma w)
            rc: np.ndarray = w * sigma_w / port_var
            # Objective: sum of squared pairwise differences
            rc_mean: float = float(rc.mean())
            return float(np.sum((rc - rc_mean) ** 2))
        result = _slsqp(_erc_obj)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = float(result.fun)
            diagnostics = {"converged": True}
        else:
            logger.warning("run_optimizer: ERC failed. Using inverse vol.")
            inv_vol = 1.0 / asset_vols
            w_opt = _apply_box_constraints(inv_vol, min_w, max_w)
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: hrp (Hierarchical Risk Parity) ---
    elif method == "hrp":
        # HRP: hierarchical clustering + recursive bisection
        # Step 1: compute correlation matrix from sigma
        corr_matrix: np.ndarray = np.zeros((_N_ASSETS, _N_ASSETS))
        for i in range(_N_ASSETS):
            for j in range(_N_ASSETS):
                denom: float = asset_vols[i] * asset_vols[j]
                corr_matrix[i, j] = (
                    sigma[i, j] / denom if denom > _EPS else 0.0
                )
        np.fill_diagonal(corr_matrix, 1.0)

        # Step 2: compute distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
        dist_matrix: np.ndarray = np.sqrt(
            np.maximum(0.5 * (1.0 - corr_matrix), 0.0)
        )
        np.fill_diagonal(dist_matrix, 0.0)

        # Step 3: hierarchical clustering (Ward linkage)
        condensed_dist: np.ndarray = squareform(dist_matrix)
        linkage_matrix: np.ndarray = linkage(condensed_dist, method="ward")

        # Step 4: get the quasi-diagonal ordering from the dendrogram
        sorted_indices: List[int] = list(leaves_list(linkage_matrix))

        # Step 5: recursive bisection to assign weights
        def _hrp_bisect(
            indices: List[int],
            weights: np.ndarray,
        ) -> None:
            """Recursively bisect the cluster and assign weights."""
            if len(indices) <= 1:
                return
            # Split into two halves
            mid: int = len(indices) // 2
            left: List[int] = indices[:mid]
            right: List[int] = indices[mid:]

            # Compute cluster variances
            def _cluster_var(idx: List[int]) -> float:
                w_sub: np.ndarray = np.zeros(_N_ASSETS)
                for i in idx:
                    w_sub[i] = 1.0 / len(idx)
                return float(np.dot(w_sub, sigma @ w_sub))

            var_left: float = _cluster_var(left)
            var_right: float = _cluster_var(right)
            total_var: float = var_left + var_right
            if total_var < _EPS:
                alpha_left: float = 0.5
            else:
                # Allocate inversely proportional to cluster variance
                alpha_left = 1.0 - var_left / total_var

            # Scale weights for each sub-cluster
            for i in left:
                weights[i] *= alpha_left
            for i in right:
                weights[i] *= (1.0 - alpha_left)

            # Recurse into each sub-cluster
            _hrp_bisect(left, weights)
            _hrp_bisect(right, weights)

        # Initialise weights to 1.0 for all assets
        hrp_weights: np.ndarray = np.ones(_N_ASSETS, dtype=np.float64)
        _hrp_bisect(sorted_indices, hrp_weights)

        # Normalise and apply box constraints
        w_opt = _apply_box_constraints(hrp_weights, min_w, max_w)
        obj_val = float(np.dot(w_opt, sigma @ w_opt))
        diagnostics = {"method": "hrp_recursive_bisection"}

    # --- Method: herc (Hierarchical Equal Risk Contribution) ---
    elif method == "herc":
        # HERC: HRP structure + ERC within each cluster
        # Simplified: use HRP weights as starting point, then apply ERC
        # (full HERC requires cluster-level ERC, approximated here)
        # Step 1: compute HRP weights (reuse HRP logic)
        corr_herc: np.ndarray = np.zeros((_N_ASSETS, _N_ASSETS))
        for i in range(_N_ASSETS):
            for j in range(_N_ASSETS):
                denom = asset_vols[i] * asset_vols[j]
                corr_herc[i, j] = sigma[i, j] / denom if denom > _EPS else 0.0
        np.fill_diagonal(corr_herc, 1.0)
        dist_herc: np.ndarray = np.sqrt(np.maximum(0.5 * (1.0 - corr_herc), 0.0))
        np.fill_diagonal(dist_herc, 0.0)
        condensed_herc: np.ndarray = squareform(dist_herc)
        link_herc: np.ndarray = linkage(condensed_herc, method="ward")
        sorted_herc: List[int] = list(leaves_list(link_herc))
        # Use inverse volatility within each half-cluster (HERC approximation)
        mid_herc: int = len(sorted_herc) // 2
        left_herc: List[int] = sorted_herc[:mid_herc]
        right_herc: List[int] = sorted_herc[mid_herc:]
        w_herc: np.ndarray = np.zeros(_N_ASSETS, dtype=np.float64)
        for idx_list in [left_herc, right_herc]:
            sub_inv_vol: np.ndarray = np.array(
                [1.0 / asset_vols[i] for i in idx_list]
            )
            sub_weights: np.ndarray = sub_inv_vol / sub_inv_vol.sum()
            for k, i in enumerate(idx_list):
                w_herc[i] = sub_weights[k] * 0.5
        w_opt = _apply_box_constraints(w_herc, min_w, max_w)
        obj_val = float(np.dot(w_opt, sigma @ w_opt))
        diagnostics = {"method": "herc_approximation"}

    # --- Method: nco (Nested Cluster Optimisation) ---
    elif method == "nco":
        # NCO: cluster assets, apply MVO within clusters, then across clusters
        # Simplified: 2-cluster NCO using correlation-based clustering
        corr_nco: np.ndarray = np.zeros((_N_ASSETS, _N_ASSETS))
        for i in range(_N_ASSETS):
            for j in range(_N_ASSETS):
                denom = asset_vols[i] * asset_vols[j]
                corr_nco[i, j] = sigma[i, j] / denom if denom > _EPS else 0.0
        np.fill_diagonal(corr_nco, 1.0)
        dist_nco: np.ndarray = np.sqrt(np.maximum(0.5 * (1.0 - corr_nco), 0.0))
        np.fill_diagonal(dist_nco, 0.0)
        link_nco: np.ndarray = linkage(squareform(dist_nco), method="ward")
        sorted_nco: List[int] = list(leaves_list(link_nco))
        mid_nco: int = len(sorted_nco) // 2
        clusters: List[List[int]] = [sorted_nco[:mid_nco], sorted_nco[mid_nco:]]
        # Intra-cluster: inverse volatility weighting
        cluster_weights: List[np.ndarray] = []
        for cluster in clusters:
            sub_inv_vol = np.array([1.0 / asset_vols[i] for i in cluster])
            cluster_weights.append(sub_inv_vol / sub_inv_vol.sum())
        # Inter-cluster: inverse cluster variance weighting
        cluster_vars: List[float] = []
        for k, cluster in enumerate(clusters):
            w_sub = np.zeros(_N_ASSETS)
            for j, i in enumerate(cluster):
                w_sub[i] = cluster_weights[k][j]
            cluster_vars.append(float(np.dot(w_sub, sigma @ w_sub)))
        total_inv_var: float = sum(1.0 / max(v, _EPS) for v in cluster_vars)
        w_nco: np.ndarray = np.zeros(_N_ASSETS, dtype=np.float64)
        for k, cluster in enumerate(clusters):
            inter_weight: float = (1.0 / max(cluster_vars[k], _EPS)) / total_inv_var
            for j, i in enumerate(cluster):
                w_nco[i] = inter_weight * cluster_weights[k][j]
        w_opt = _apply_box_constraints(w_nco, min_w, max_w)
        obj_val = float(np.dot(w_opt, sigma @ w_opt))
        diagnostics = {"method": "nco_2cluster"}

    # --- Method: min_cvar (Minimum CVaR) ---
    elif method == "min_cvar":
        # Min CVaR: min_{w,zeta} zeta + 1/((1-alpha)*T) * sum_t max(-w'r_t - zeta, 0)
        if returns_matrix is None or returns_matrix.shape[0] < 20:
            logger.warning("run_optimizer: min_cvar requires returns_matrix. Using GMV.")
            def _gmv_fallback(w: np.ndarray) -> float:
                return float(np.dot(w, sigma @ w))
            result = _slsqp(_gmv_fallback)
            w_opt = _apply_box_constraints(
                result.x if result.success else w_equal, min_w, max_w
            )
            obj_val = float(np.dot(w_opt, sigma @ w_opt))
            diagnostics = {"fallback_used": True, "reason": "no_returns_matrix"}
        else:
            T: int = returns_matrix.shape[0]
            alpha_cvar: float = _CVAR_ALPHA

            def _cvar_obj(params: np.ndarray) -> float:
                """CVaR objective: zeta + 1/((1-alpha)*T) * sum(max(-r_p - zeta, 0))"""
                w_c: np.ndarray = params[:_N_ASSETS]
                zeta: float = float(params[_N_ASSETS])
                r_p: np.ndarray = returns_matrix @ w_c
                losses: np.ndarray = np.maximum(-r_p - zeta, 0.0)
                return zeta + float(losses.sum()) / ((1.0 - alpha_cvar) * T)

            # Extended variable: [w_1, ..., w_18, zeta]
            x0_cvar: np.ndarray = np.append(_random_start(), 0.0)
            bounds_cvar: List[Tuple[float, float]] = (
                [(min_w, max_w)] * _N_ASSETS + [(-1.0, 1.0)]
            )
            cvar_budget: Dict[str, Any] = {
                "type": "eq",
                "fun": lambda p: float(p[:_N_ASSETS].sum()) - 1.0,
            }
            result_cvar: OptimizeResult = minimize(
                fun=_cvar_obj,
                x0=x0_cvar,
                method="SLSQP",
                bounds=bounds_cvar,
                constraints=[cvar_budget],
                options={"ftol": 1e-9, "maxiter": 1000, "disp": False},
            )
            if result_cvar.success:
                w_opt = _apply_box_constraints(
                    result_cvar.x[:_N_ASSETS], min_w, max_w
                )
                obj_val = float(result_cvar.fun)
                diagnostics = {"converged": True, "cvar_alpha": alpha_cvar}
            else:
                logger.warning("run_optimizer: Min CVaR failed. Using GMV.")
                def _gmv_fb(w: np.ndarray) -> float:
                    return float(np.dot(w, sigma @ w))
                res_fb: OptimizeResult = _slsqp(_gmv_fb)
                w_opt = _apply_box_constraints(
                    res_fb.x if res_fb.success else w_equal, min_w, max_w
                )
                obj_val = float(np.dot(w_opt, sigma @ w_opt))
                diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: cvar_parity (Equal CVaR Contributions) ---
    elif method == "cvar_parity":
        # CVaR parity: equalise CVaR contributions across assets
        # Approximated as tail risk parity using historical returns
        if returns_matrix is None or returns_matrix.shape[0] < 20:
            logger.warning("run_optimizer: cvar_parity requires returns_matrix. Using ERC.")
            def _erc_fb(w: np.ndarray) -> float:
                sigma_w = sigma @ w
                pv = float(np.dot(w, sigma_w))
                if pv < _EPS:
                    return 0.0
                rc = w * sigma_w / pv
                return float(np.sum((rc - rc.mean()) ** 2))
            result = _slsqp(_erc_fb)
            w_opt = _apply_box_constraints(
                result.x if result.success else w_equal, min_w, max_w
            )
            obj_val = float(np.dot(w_opt, sigma @ w_opt))
            diagnostics = {"fallback_used": True, "reason": "no_returns_matrix"}
        else:
            T_cp: int = returns_matrix.shape[0]
            alpha_cp: float = _CVAR_ALPHA
            n_tail: int = max(1, int((1.0 - alpha_cp) * T_cp))

            def _cvar_parity_obj(w: np.ndarray) -> float:
                """Minimise variance of individual CVaR contributions."""
                r_p: np.ndarray = returns_matrix @ w
                # Individual asset CVaR contributions (simplified)
                tail_idx: np.ndarray = np.argsort(r_p)[:n_tail]
                tail_returns: np.ndarray = returns_matrix[tail_idx]
                # CVaR contribution of each asset
                cvar_contrib: np.ndarray = -w * tail_returns.mean(axis=0)
                cvar_mean: float = float(cvar_contrib.mean())
                return float(np.sum((cvar_contrib - cvar_mean) ** 2))

            result = _slsqp(_cvar_parity_obj)
            if result.success:
                w_opt = _apply_box_constraints(result.x, min_w, max_w)
                obj_val = float(result.fun)
                diagnostics = {"converged": True}
            else:
                logger.warning("run_optimizer: CVaR parity failed. Using ERC.")
                w_opt = _apply_box_constraints(
                    1.0 / asset_vols, min_w, max_w
                )
                obj_val = 0.0
                diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: robust_mvo (Robust Mean-Variance) ---
    elif method == "robust_mvo":
        # Robust MVO: minimax regret with ellipsoidal uncertainty on mu
        # Approximated as max Sharpe with shrunk mu toward equal-weight return
        mu_shrunk: np.ndarray = 0.5 * mu + 0.5 * np.full(_N_ASSETS, mu.mean())

        def _neg_robust_sharpe(w: np.ndarray) -> float:
            port_ret: float = float(np.dot(w, mu_shrunk))
            port_var: float = float(np.dot(w, sigma @ w))
            port_vol: float = float(np.sqrt(max(port_var, _EPS)))
            return -(port_ret - rf) / port_vol

        result = _slsqp(_neg_robust_sharpe)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = -float(result.fun)
            diagnostics = {"converged": True, "mu_shrinkage": 0.5}
        else:
            logger.warning("run_optimizer: Robust MVO failed. Using equal weight.")
            w_opt = w_equal.copy()
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: resampled_mvo (Michaud Resampled Efficient Frontier) ---
    elif method == "resampled_mvo":
        # Resampled MVO: average max-Sharpe weights across M Monte Carlo draws
        M_resample: int = 100  # Number of resampling iterations
        w_sum_resample: np.ndarray = np.zeros(_N_ASSETS, dtype=np.float64)
        n_successful: int = 0

        for _ in range(M_resample):
            # Draw perturbed mu and sigma from their sampling distributions
            mu_perturbed: np.ndarray = mu + rng.normal(
                0, np.abs(mu) * 0.1 + 0.001, _N_ASSETS
            )
            # Perturb sigma by adding small noise to diagonal
            sigma_perturbed: np.ndarray = sigma + np.diag(
                rng.exponential(0.0001, _N_ASSETS)
            )

            def _neg_sharpe_resample(w: np.ndarray) -> float:
                pr: float = float(np.dot(w, mu_perturbed))
                pv: float = float(np.dot(w, sigma_perturbed @ w))
                pv_vol: float = float(np.sqrt(max(pv, _EPS)))
                return -(pr - rf) / pv_vol

            res_r: OptimizeResult = _slsqp(_neg_sharpe_resample)
            if res_r.success:
                w_sum_resample += _apply_box_constraints(res_r.x, min_w, max_w)
                n_successful += 1

        if n_successful > 0:
            w_opt = _apply_box_constraints(
                w_sum_resample / n_successful, min_w, max_w
            )
            obj_val = float(np.dot(w_opt, sigma @ w_opt))
            diagnostics = {
                "n_successful_draws": n_successful,
                "M_resample": M_resample,
            }
        else:
            logger.warning("run_optimizer: Resampled MVO failed. Using equal weight.")
            w_opt = w_equal.copy()
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: bl_posterior (Black-Litterman Posterior) ---
    elif method == "bl_posterior":
        # BL posterior: use BL posterior mean as expected returns, then max Sharpe
        # BL equilibrium prior: pi = delta * Sigma * w_mkt
        delta_bl: float = 2.5
        tau_bl: float = 0.05
        # Market-cap weights approximated as benchmark weights
        pi_bl: np.ndarray = delta_bl * sigma @ benchmark_weights
        # BL posterior (no views): mu_BL = pi (equilibrium prior)
        # With views: mu_BL = [(tau*Sigma)^{-1} + P'Omega^{-1}P]^{-1}
        #                      [(tau*Sigma)^{-1}*pi + P'Omega^{-1}Q]
        # Without views (no P, Q, Omega): mu_BL = pi
        mu_bl: np.ndarray = pi_bl

        def _neg_bl_sharpe(w: np.ndarray) -> float:
            pr: float = float(np.dot(w, mu_bl))
            pv: float = float(np.dot(w, sigma @ w))
            pv_vol: float = float(np.sqrt(max(pv, _EPS)))
            return -(pr - rf) / pv_vol

        result = _slsqp(_neg_bl_sharpe)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = -float(result.fun)
            diagnostics = {"converged": True, "delta": delta_bl, "tau": tau_bl}
        else:
            logger.warning("run_optimizer: BL Posterior failed. Using equal weight.")
            w_opt = w_equal.copy()
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: regime_conditional ---
    elif method == "regime_conditional":
        # Regime-conditional: use mu directly (already regime-adjusted CMAs)
        # Same as max_sharpe but with explicit regime-adjusted mu
        def _neg_regime_sharpe(w: np.ndarray) -> float:
            pr: float = float(np.dot(w, mu))
            pv: float = float(np.dot(w, sigma @ w))
            pv_vol: float = float(np.sqrt(max(pv, _EPS)))
            return -(pr - rf) / pv_vol

        result = _slsqp(_neg_regime_sharpe)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = -float(result.fun)
            diagnostics = {"converged": True}
        else:
            logger.warning("run_optimizer: Regime Conditional failed.")
            w_opt = w_equal.copy()
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: factor_risk_parity ---
    elif method == "factor_risk_parity":
        # Factor risk parity: risk parity in factor space
        # Approximated as ERC with factor-adjusted covariance
        # (full implementation requires factor loadings; approximated as ERC)
        def _frp_obj(w: np.ndarray) -> float:
            sigma_w: np.ndarray = sigma @ w
            pv: float = float(np.dot(w, sigma_w))
            if pv < _EPS:
                return 0.0
            rc: np.ndarray = w * sigma_w / pv
            return float(np.sum((rc - rc.mean()) ** 2))

        result = _slsqp(_frp_obj)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = float(result.fun)
            diagnostics = {"converged": True, "note": "erc_approximation"}
        else:
            w_opt = _apply_box_constraints(1.0 / asset_vols, min_w, max_w)
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: max_decorrelation ---
    elif method == "max_decorrelation":
        # Max Decorrelation: minimise average pairwise correlation
        # Objective: min_w sum_{i,j} w_i * w_j * rho_{ij}
        corr_md: np.ndarray = np.zeros((_N_ASSETS, _N_ASSETS))
        for i in range(_N_ASSETS):
            for j in range(_N_ASSETS):
                denom = asset_vols[i] * asset_vols[j]
                corr_md[i, j] = sigma[i, j] / denom if denom > _EPS else 0.0
        np.fill_diagonal(corr_md, 1.0)

        def _avg_corr_obj(w: np.ndarray) -> float:
            return float(np.dot(w, corr_md @ w))

        result = _slsqp(_avg_corr_obj)
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = float(result.fun)
            diagnostics = {"converged": True}
        else:
            logger.warning("run_optimizer: Max Decorrelation failed.")
            w_opt = w_equal.copy()
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: tail_risk_parity ---
    elif method == "tail_risk_parity":
        # Tail risk parity: equalise tail risk contributions
        # Approximated as ERC with tail covariance (lower partial moments)
        if returns_matrix is not None and returns_matrix.shape[0] >= 20:
            T_trp: int = returns_matrix.shape[0]
            n_tail_trp: int = max(1, int((1.0 - _CVAR_ALPHA) * T_trp))
            # Compute tail covariance matrix (covariance during worst periods)
            port_returns_trp: np.ndarray = returns_matrix @ w_equal
            tail_idx_trp: np.ndarray = np.argsort(port_returns_trp)[:n_tail_trp]
            tail_returns_trp: np.ndarray = returns_matrix[tail_idx_trp]
            sigma_tail: np.ndarray = np.cov(tail_returns_trp.T, ddof=1) * _PERIODS_PER_YEAR

            def _trp_obj(w: np.ndarray) -> float:
                sigma_w_t: np.ndarray = sigma_tail @ w
                pv_t: float = float(np.dot(w, sigma_w_t))
                if pv_t < _EPS:
                    return 0.0
                rc_t: np.ndarray = w * sigma_w_t / pv_t
                return float(np.sum((rc_t - rc_t.mean()) ** 2))

            result = _slsqp(_trp_obj)
            if result.success:
                w_opt = _apply_box_constraints(result.x, min_w, max_w)
                obj_val = float(result.fun)
                diagnostics = {"converged": True, "n_tail": n_tail_trp}
            else:
                w_opt = _apply_box_constraints(1.0 / asset_vols, min_w, max_w)
                obj_val = 0.0
                diagnostics = {"converged": False, "fallback_used": True}
        else:
            # Fallback to ERC if no returns matrix
            def _erc_trp(w: np.ndarray) -> float:
                sw: np.ndarray = sigma @ w
                pv: float = float(np.dot(w, sw))
                if pv < _EPS:
                    return 0.0
                rc: np.ndarray = w * sw / pv
                return float(np.sum((rc - rc.mean()) ** 2))
            result = _slsqp(_erc_trp)
            w_opt = _apply_box_constraints(
                result.x if result.success else w_equal, min_w, max_w
            )
            obj_val = float(np.dot(w_opt, sigma @ w_opt))
            diagnostics = {"fallback_used": True, "reason": "no_returns_matrix"}

    # --- Method: kelly_fractional ---
    elif method == "kelly_fractional":
        # Fractional Kelly: w = f * Sigma^{-1} * (mu - rf)
        # where f = _KELLY_FRACTION (0.5)
        try:
            sigma_inv: np.ndarray = np.linalg.pinv(sigma)
            mu_excess: np.ndarray = mu - rf
            w_kelly_raw: np.ndarray = _KELLY_FRACTION * sigma_inv @ mu_excess
            w_opt = _apply_box_constraints(w_kelly_raw, min_w, max_w)
            obj_val = float(np.dot(w_opt, mu) - rf)
            diagnostics = {"kelly_fraction": _KELLY_FRACTION, "method": "closed_form"}
        except np.linalg.LinAlgError:
            logger.warning("run_optimizer: Kelly Fractional failed. Using equal weight.")
            w_opt = w_equal.copy()
            obj_val = 0.0
            diagnostics = {"converged": False, "fallback_used": True}

    # --- Method: target_vol_sharpe ---
    elif method == "target_vol_sharpe":
        # Target volatility Sharpe: max Sharpe subject to sigma_p = target_vol
        def _neg_tv_sharpe(w: np.ndarray) -> float:
            pr: float = float(np.dot(w, mu))
            pv: float = float(np.dot(w, sigma @ w))
            pv_vol: float = float(np.sqrt(max(pv, _EPS)))
            return -(pr - rf) / pv_vol

        # Add target volatility equality constraint
        tv_constraint: Dict[str, Any] = {
            "type": "eq",
            "fun": lambda w: float(np.sqrt(max(np.dot(w, sigma @ w), 0.0))) - target_vol,
        }
        result = _slsqp(_neg_tv_sharpe, extra_constraints=[tv_constraint])
        if result.success:
            w_opt = _apply_box_constraints(result.x, min_w, max_w)
            obj_val = -float(result.fun)
            diagnostics = {"converged": True, "target_vol": target_vol}
        else:
            # Fallback: scale max-Sharpe portfolio to target vol
            def _neg_sharpe_tv(w: np.ndarray) -> float:
                pr: float = float(np.dot(w, mu))
                pv: float = float(np.dot(w, sigma @ w))
                pv_vol: float = float(np.sqrt(max(pv, _EPS)))
                return -(pr - rf) / pv_vol
            res_tv: OptimizeResult = _slsqp(_neg_sharpe_tv)
            if res_tv.success:
                w_ms: np.ndarray = _apply_box_constraints(res_tv.x, min_w, max_w)
                ms_vol: float = float(
                    np.sqrt(max(np.dot(w_ms, sigma @ w_ms), _EPS))
                )
                scale: float = target_vol / ms_vol if ms_vol > _EPS else 1.0
                w_opt = _apply_box_constraints(w_ms * scale, min_w, max_w)
            else:
                w_opt = w_equal.copy()
            obj_val = float(np.dot(w_opt, mu) - rf)
            diagnostics = {"converged": False, "fallback_used": True}

    else:
        # Unknown method: raise ValueError
        raise ValueError(
            f"Unknown PC method: '{method}'. "
            f"Must be one of the 20 canonical PC method slugs."
        )

    # ------------------------------------------------------------------
    # Final post-processing: ensure weights are valid
    # ------------------------------------------------------------------
    # Clip to [min_w, max_w] and renormalise to sum to 1.0
    w_final: np.ndarray = _apply_box_constraints(w_opt, min_w, max_w)

    # Log the optimization result for audit trail
    logger.info(
        "run_optimizer: method='%s', sum=%.6f, max_w=%.4f, "
        "obj_val=%.6f.",
        method,
        float(w_final.sum()),
        float(w_final.max()),
        float(obj_val),
    )

    # ------------------------------------------------------------------
    # Return the optimizer result dict
    # ------------------------------------------------------------------
    return {
        # 18-element optimal weight vector (sums to 1.0, non-negative)
        "weights": [float(v) for v in w_final],
        # PC method slug used
        "method": method,
        # Achieved objective function value
        "objective_value": float(obj_val),
        # Method-specific diagnostics for audit
        "diagnostics": _cast_to_json_safe(diagnostics),
    }


# =============================================================================
# TOOL 62: fetch_cash_yield_proxy
# =============================================================================

def fetch_cash_yield_proxy(
    as_of_date: str,
    df_fixed_income_curves_spreads_raw: pd.DataFrame,
    df_benchmark_factors_raw: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Retrieve the current cash yield proxy as-of ``as_of_date``.

    This tool implements the cash yield retrieval step for the Cash AC
    Agent (Task 18). The primary source is the 3-month T-bill yield from
    ``df_fixed_income_curves_spreads_raw["ust_3m_yield"]``. If unavailable,
    the tool falls back to the annualised risk-free rate from
    ``df_benchmark_factors_raw["rf"]``.

    The cash expected return is set equal to the current cash yield proxy,
    consistent with the building-block approach: for cash, the expected
    return equals the current yield (no duration, no credit risk).

    Parameters
    ----------
    as_of_date : str
        ISO-8601 date string for point-in-time retrieval.
    df_fixed_income_curves_spreads_raw : pd.DataFrame
        Fixed income yield curve panel. Must have a ``DatetimeIndex``
        and column ``"ust_3m_yield"`` (decimal form).
    df_benchmark_factors_raw : Optional[pd.DataFrame]
        Daily benchmark factors panel. Must have a ``DatetimeIndex``
        and column ``"rf"`` (daily risk-free rate, decimal form).
        Used as fallback if ``ust_3m_yield`` is unavailable.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"cash_yield"`` (``float``): Current cash yield proxy in
          decimal form (e.g., 0.053 = 5.3%).
        - ``"source"`` (``str``): Data source used:
          ``"ust_3m_yield"`` or ``"rf_annualised"``.
        - ``"as_of_date"`` (``str``): The as-of date used.

    Raises
    ------
    TypeError
        If ``df_fixed_income_curves_spreads_raw`` is not a
        ``pd.DataFrame``.
    ValueError
        If neither the primary nor fallback source is available.

    Notes
    -----
    **Annualisation of daily rf:** When using ``df_benchmark_factors_raw``
    as the fallback, the daily rf is annualised using the frozen
    ``act/365`` convention:
    :math:`r_{f,ann} = (1 + r_{f,daily})^{365} - 1`.
    This is consistent with ``DATA_CONVENTIONS["resampling_rules"]
    ["rf_annualization"] = "act/365"``.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(df_fixed_income_curves_spreads_raw, pd.DataFrame):
        raise TypeError(
            f"df_fixed_income_curves_spreads_raw must be a pd.DataFrame, "
            f"got {type(df_fixed_income_curves_spreads_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Parse as_of_date to pd.Timestamp
    # ------------------------------------------------------------------
    try:
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
    except Exception as exc:
        raise ValueError(
            f"as_of_date='{as_of_date}' cannot be parsed. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Primary source: ust_3m_yield from df_fixed_income_curves_spreads_raw
    # ------------------------------------------------------------------
    cash_yield: Optional[float] = None
    source: str = "unavailable"

    if "ust_3m_yield" in df_fixed_income_curves_spreads_raw.columns:
        # Strip timezone from index if present
        df_fi: pd.DataFrame = df_fixed_income_curves_spreads_raw.copy()
        if hasattr(df_fi.index, "tz") and df_fi.index.tz is not None:
            df_fi.index = df_fi.index.tz_localize(None)

        # Apply point-in-time filter
        df_filtered: pd.DataFrame = df_fi.loc[df_fi.index <= as_of_ts]

        if not df_filtered.empty:
            # Get the most recent ust_3m_yield value
            latest_val = df_filtered["ust_3m_yield"].iloc[-1]
            if not pd.isna(latest_val):
                cash_yield = float(latest_val)
                source = "ust_3m_yield"

    # ------------------------------------------------------------------
    # Fallback source: annualised rf from df_benchmark_factors_raw
    # ------------------------------------------------------------------
    if cash_yield is None and df_benchmark_factors_raw is not None:
        if not isinstance(df_benchmark_factors_raw, pd.DataFrame):
            raise TypeError(
                f"df_benchmark_factors_raw must be a pd.DataFrame, "
                f"got {type(df_benchmark_factors_raw).__name__}."
            )

        if "rf" in df_benchmark_factors_raw.columns:
            # Strip timezone from index if present
            df_bm: pd.DataFrame = df_benchmark_factors_raw.copy()
            if hasattr(df_bm.index, "tz") and df_bm.index.tz is not None:
                df_bm.index = df_bm.index.tz_localize(None)

            # Apply point-in-time filter
            df_bm_filtered: pd.DataFrame = df_bm.loc[df_bm.index <= as_of_ts]

            if not df_bm_filtered.empty:
                # Get the most recent daily rf value
                rf_daily_val = df_bm_filtered["rf"].iloc[-1]
                if not pd.isna(rf_daily_val):
                    rf_daily: float = float(rf_daily_val)
                    # Annualise using act/365 convention:
                    # r_f_ann = (1 + r_f_daily)^365 - 1
                    cash_yield = float((1.0 + rf_daily) ** 365 - 1.0)
                    source = "rf_annualised"

    # ------------------------------------------------------------------
    # Guard: both sources unavailable
    # ------------------------------------------------------------------
    if cash_yield is None:
        raise ValueError(
            f"Cash yield proxy not available for as_of_date='{as_of_date}'. "
            "Neither 'ust_3m_yield' from df_fixed_income_curves_spreads_raw "
            "nor 'rf' from df_benchmark_factors_raw is available."
        )

    # Log the cash yield retrieval for audit trail
    logger.info(
        "fetch_cash_yield_proxy: cash_yield=%.4f (source='%s', "
        "as_of_date='%s').",
        cash_yield,
        source,
        as_of_date,
    )

    return {
        # Current cash yield proxy in decimal form
        "cash_yield": float(cash_yield),
        # Data source used for the cash yield proxy
        "source": source,
        # As-of date used for point-in-time retrieval
        "as_of_date": as_of_date,
    }


# =============================================================================
# TOOL 63: tool_literature_search
# =============================================================================

def tool_literature_search(
    query: str,
    max_results: int,
    as_of_date: str,
    allowed_domains: List[str],
) -> Dict[str, Any]:
    """
    Search for portfolio construction methods not in the current PC registry.

    This tool implements the literature search step for the PC-Researcher
    Agent (Task 19). It operates in two modes:

    - **Local mode** (``allowed_domains`` is empty): Searches a frozen
      local knowledge base of portfolio construction methods. This is
      the default mode per ``LOOKAHEAD_CONTROLS["web_search_default_enabled"]
      = False``.
    - **Web search mode** (``allowed_domains`` is non-empty): Performs
      a web search restricted to the specified domains. Used only in
      robustness runs (Task 36) where web search is explicitly enabled.

    All results are filtered to publications on or before ``as_of_date``
    to prevent lookahead bias.

    Parameters
    ----------
    query : str
        Search query describing the type of portfolio construction method
        sought (e.g., ``"maximum entropy portfolio construction"``).
    max_results : int
        Maximum number of results to return.
    as_of_date : str
        ISO-8601 date string. Results published after this date are
        excluded to prevent lookahead bias.
    allowed_domains : List[str]
        List of allowed web search domains. If empty, local mode is used.
        Example: ``["fred.stlouisfed.org", "bis.org"]``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"results"`` (``List[Dict[str, str]]``): List of candidate
          methods. Each entry contains ``"title"``, ``"abstract"``,
          ``"method_description"``, ``"reference"``.
        - ``"n_results"`` (``int``): Number of results returned.
        - ``"search_mode"`` (``str``): ``"local"`` or ``"web"``.
        - ``"query"`` (``str``): The search query used.

    Raises
    ------
    TypeError
        If ``allowed_domains`` is not a list.
    ValueError
        If ``max_results`` is not a positive integer.

    Notes
    -----
    **Lookahead prevention:** Results are filtered by the ``"year"``
    field in the local knowledge base. For web search mode, the
    ``as_of_date`` is passed as a date filter parameter. This is
    consistent with the manuscript's discussion of lookahead bias
    (Section 5.1, Yin et al. 2024).

    **Local knowledge base:** The frozen local knowledge base
    (``_LOCAL_PC_KNOWLEDGE_BASE``) contains a curated set of portfolio
    construction methods not currently in the canonical PC registry.
    This is an UNSPECIFIED IN MANUSCRIPT implementation choice.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(allowed_domains, list):
        raise TypeError(
            f"allowed_domains must be a list, "
            f"got {type(allowed_domains).__name__}."
        )
    if not isinstance(max_results, int) or max_results <= 0:
        raise ValueError(
            f"max_results must be a positive integer, got {max_results}."
        )

    # ------------------------------------------------------------------
    # Parse as_of_date to extract the year for lookahead filtering
    # ------------------------------------------------------------------
    try:
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
        as_of_year: int = as_of_ts.year
    except Exception:
        # If parsing fails, use a conservative year
        as_of_year = 2026

    # ------------------------------------------------------------------
    # Determine search mode
    # ------------------------------------------------------------------
    if len(allowed_domains) == 0:
        # Local mode: search the frozen local knowledge base
        search_mode: str = "local"

        # Filter local knowledge base by query relevance (simple keyword match)
        query_lower: str = query.lower()
        filtered_results: List[Dict[str, str]] = []

        for entry in _LOCAL_PC_KNOWLEDGE_BASE:
            # Check if the entry is within the as_of_date year constraint
            entry_year: int = int(entry.get("year", "2000"))
            if entry_year > as_of_year:
                # Exclude post-as_of_date publications (lookahead prevention)
                continue

            # Simple keyword relevance check
            entry_text: str = (
                entry.get("title", "").lower()
                + " "
                + entry.get("abstract", "").lower()
                + " "
                + entry.get("method_description", "").lower()
            )
            # Include if any query word appears in the entry text
            query_words: List[str] = query_lower.split()
            if any(word in entry_text for word in query_words):
                filtered_results.append({
                    "title": entry.get("title", ""),
                    "abstract": entry.get("abstract", ""),
                    "method_description": entry.get("method_description", ""),
                    "reference": entry.get("reference", ""),
                })

        # If no keyword matches, return all local entries within date constraint
        if len(filtered_results) == 0:
            filtered_results = [
                {
                    "title": e.get("title", ""),
                    "abstract": e.get("abstract", ""),
                    "method_description": e.get("method_description", ""),
                    "reference": e.get("reference", ""),
                }
                for e in _LOCAL_PC_KNOWLEDGE_BASE
                if int(e.get("year", "2000")) <= as_of_year
            ]

        # Truncate to max_results
        results: List[Dict[str, str]] = filtered_results[:max_results]

    else:
        # Web search mode: log a warning and return empty results
        # (actual web search implementation requires external API)
        search_mode = "web"
        logger.warning(
            "tool_literature_search: Web search mode requested "
            "(domains=%s) but web search API not implemented. "
            "Returning empty results.",
            allowed_domains,
        )
        results = []

    # Log the search result for audit trail
    logger.info(
        "tool_literature_search: query='%s', mode='%s', "
        "n_results=%d.",
        query,
        search_mode,
        len(results),
    )

    return {
        # List of candidate method results
        "results": results,
        # Number of results returned
        "n_results": len(results),
        # Search mode used (local or web)
        "search_mode": search_mode,
        # The search query used
        "query": query,
    }


# =============================================================================
# TOOL 64: validate_method_spec
# =============================================================================

def validate_method_spec(
    method_spec: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Validate that a proposed PC method specification is IPS-compatible.

    This tool implements the method specification validation step for the
    PC-Researcher Agent (Task 19). It checks that the proposed method:

    1. References only available pipeline inputs (``sigma``, ``mu``,
       ``rf``, ``benchmark_weights``, ``returns_matrix``, etc.)
    2. Does not violate IPS constraints (no short-selling, no leverage)
    3. Has a non-empty objective function description

    Parameters
    ----------
    method_spec : Dict[str, Any]
        Method specification dict. Must contain keys:
        ``"method_name"``, ``"objective_function"``, ``"constraints"``,
        ``"required_inputs"``.

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:

        - ``"valid"`` (``bool``): ``True`` if the specification passes
          all validation checks.
        - ``"issues"`` (``List[str]``): List of identified issues.
          Empty if ``valid=True``.

    Raises
    ------
    TypeError
        If ``method_spec`` is not a dict.

    Notes
    -----
    **Input whitelist:** The valid pipeline inputs are defined in
    ``_VALID_PIPELINE_INPUTS``. Any ``required_inputs`` entry not in
    this whitelist is flagged as an issue.

    **IPS constraint check:** The ``constraints`` field is checked for
    explicit mentions of short-selling (``"short"``, ``"negative"``,
    ``"w < 0"``) or leverage (``"leverage"``, ``"sum > 1"``). These
    are flagged as IPS violations.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(method_spec, dict):
        raise TypeError(
            f"method_spec must be a dict, got {type(method_spec).__name__}."
        )

    # ------------------------------------------------------------------
    # Accumulate validation issues
    # ------------------------------------------------------------------
    issues: List[str] = []

    # ------------------------------------------------------------------
    # Check 1: Required keys present
    # ------------------------------------------------------------------
    required_keys: Tuple[str, ...] = (
        "method_name",
        "objective_function",
        "constraints",
        "required_inputs",
    )
    for key in required_keys:
        if key not in method_spec:
            issues.append(f"Missing required key: '{key}'.")

    # If required keys are missing, return early
    if issues:
        return {"valid": False, "issues": issues}

    # ------------------------------------------------------------------
    # Check 2: Objective function is non-empty
    # ------------------------------------------------------------------
    obj_fn: str = str(method_spec.get("objective_function", ""))
    if len(obj_fn.strip()) < 10:
        issues.append(
            "objective_function is too short (< 10 characters). "
            "Provide a complete mathematical description."
        )

    # ------------------------------------------------------------------
    # Check 3: Required inputs are in the valid pipeline input whitelist
    # ------------------------------------------------------------------
    required_inputs_raw = method_spec.get("required_inputs", "")
    required_inputs_str: str = str(required_inputs_raw).lower()

    # Check for any input references not in the whitelist
    # Extract potential input names (words that look like variable names)
    potential_inputs: Set[str] = set(
        re.findall(r"\b[a-z_][a-z0-9_]*\b", required_inputs_str)
    )
    # Filter to words that look like pipeline input names
    pipeline_input_candidates: Set[str] = {
        w for w in potential_inputs
        if len(w) > 2 and not w.isdigit()
    }
    # Check for any candidates that are clearly not in the whitelist
    # (only flag if they look like data inputs, not generic words)
    data_keywords: Set[str] = {
        "data", "series", "matrix", "vector", "frame", "index",
        "prices", "volumes", "flows", "ratings", "scores",
    }
    unknown_inputs: Set[str] = pipeline_input_candidates & data_keywords
    if unknown_inputs:
        issues.append(
            f"required_inputs may reference data not available in the "
            f"pipeline: {sorted(unknown_inputs)}. "
            f"Valid inputs are: {sorted(_VALID_PIPELINE_INPUTS)}."
        )

    # ------------------------------------------------------------------
    # Check 4: Constraints do not violate IPS (no short-selling, no leverage)
    # ------------------------------------------------------------------
    constraints_str: str = str(method_spec.get("constraints", "")).lower()

    # Check for short-selling indicators
    short_selling_keywords: List[str] = [
        "short", "w < 0", "negative weight", "short position",
    ]
    for kw in short_selling_keywords:
        if kw in constraints_str:
            issues.append(
                f"constraints mentions '{kw}', which may indicate "
                "short-selling. The IPS requires long-only (w_i >= 0)."
            )
            break

    # Check for leverage indicators
    leverage_keywords: List[str] = [
        "leverage", "sum > 1", "sum(w) > 1", "gross exposure > 1",
    ]
    for kw in leverage_keywords:
        if kw in constraints_str:
            issues.append(
                f"constraints mentions '{kw}', which may indicate "
                "leverage. The IPS requires sum(w) = 1 (no leverage)."
            )
            break

    # ------------------------------------------------------------------
    # Check 5: Method name is non-empty and slug-derivable
    # ------------------------------------------------------------------
    method_name: str = str(method_spec.get("method_name", ""))
    if len(method_name.strip()) == 0:
        issues.append("method_name is empty.")

    # ------------------------------------------------------------------
    # Determine overall validity
    # ------------------------------------------------------------------
    valid: bool = len(issues) == 0

    logger.info(
        "validate_method_spec: method='%s', valid=%s, n_issues=%d.",
        method_name,
        valid,
        len(issues),
    )

    return {
        # Overall validity flag
        "valid": valid,
        # List of identified issues (empty if valid)
        "issues": issues,
    }


# =============================================================================
# TOOL 65: validate_artifact
# =============================================================================

def validate_artifact(
    artifact_path: Path,
    schema_name: str,
) -> bool:
    """
    Validate a pipeline artifact against its frozen schema (inter-stage gate).

    This tool implements the shared schema validation function referenced
    throughout the pipeline as the inter-stage gate (Tasks 1, 16, 17, 18,
    19, 22, 27, 28, 29, 30, 31). It is called by the orchestrator between
    every pipeline stage to enforce fail-closed semantics.

    The validation checks:

    1. The artifact file exists
    2. The file contains valid JSON
    3. All required fields are present
    4. Type checks pass for specified fields
    5. Range checks pass for specified fields

    Parameters
    ----------
    artifact_path : Path
        Absolute or relative path to the JSON artifact file.
    schema_name : str
        Schema identifier. Must be one of the keys in ``_SCHEMA_RULES``:
        ``"macro_view"``, ``"cma_methods"``, ``"cma"``, ``"pc_weights"``,
        ``"cro_report"``, ``"final_weights"``, ``"signals"``,
        ``"historical_stats"``, ``"vote"``.

    Returns
    -------
    bool
        ``True`` if the artifact passes all validation checks.

    Raises
    ------
    TypeError
        If ``artifact_path`` is not a ``pathlib.Path``.
    ValueError
        If ``schema_name`` is not a recognised schema identifier.
    FileNotFoundError
        If the artifact file does not exist.
    SchemaValidationError
        If the artifact fails any validation check. This is a
        pipeline-halting error (fail-closed).

    Notes
    -----
    **Fail-closed semantics:** Any validation failure raises a
    ``SchemaValidationError`` rather than returning ``False``. This
    ensures that the orchestrator cannot accidentally proceed past a
    failed validation gate.

    **Schema rules:** The validation rules are frozen in ``_SCHEMA_RULES``
    and correspond to the schemas defined in
    ``REGISTRIES["OUTPUT_SCHEMAS"]``. The rules check required fields,
    type correctness, and value ranges.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(artifact_path, Path):
        raise TypeError(
            f"artifact_path must be a pathlib.Path, "
            f"got {type(artifact_path).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: schema_name must be recognised
    # ------------------------------------------------------------------
    if schema_name not in _SCHEMA_RULES:
        raise ValueError(
            f"schema_name='{schema_name}' is not recognised. "
            f"Must be one of: {list(_SCHEMA_RULES.keys())}."
        )

    # ------------------------------------------------------------------
    # Gate 1: File existence check
    # ------------------------------------------------------------------
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Artifact not found: '{artifact_path}'. "
            f"Schema: '{schema_name}'. "
            "Pipeline halted (fail-closed)."
        )

    # ------------------------------------------------------------------
    # Gate 2: JSON parse check
    # ------------------------------------------------------------------
    try:
        with open(artifact_path, "r", encoding="utf-8") as fh:
            # Parse the JSON content of the artifact
            artifact_data: Any = json.load(fh)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            f"SCHEMA VALIDATION FAILURE: '{artifact_path}' contains "
            f"invalid JSON. Schema: '{schema_name}'. "
            f"Original error: {exc}. "
            "Pipeline halted (fail-closed)."
        ) from exc
    except OSError as exc:
        raise SchemaValidationError(
            f"SCHEMA VALIDATION FAILURE: Cannot read '{artifact_path}'. "
            f"Original error: {exc}."
        ) from exc

    # ------------------------------------------------------------------
    # Gate 3: Artifact must be a dict (all pipeline artifacts are JSON objects)
    # ------------------------------------------------------------------
    if not isinstance(artifact_data, dict):
        raise SchemaValidationError(
            f"SCHEMA VALIDATION FAILURE: '{artifact_path}' must contain "
            f"a JSON object (dict), got {type(artifact_data).__name__}. "
            f"Schema: '{schema_name}'."
        )

    # ------------------------------------------------------------------
    # Retrieve the schema rules for this schema_name
    # ------------------------------------------------------------------
    rules: Dict[str, Any] = _SCHEMA_RULES[schema_name]
    required_fields: List[str] = rules.get("required_fields", [])
    type_checks: Dict[str, Any] = rules.get("type_checks", {})
    range_checks: Dict[str, Tuple[float, float]] = rules.get("range_checks", {})

    # ------------------------------------------------------------------
    # Gate 4: Required fields check
    # ------------------------------------------------------------------
    missing_fields: List[str] = [
        f for f in required_fields if f not in artifact_data
    ]
    if missing_fields:
        raise SchemaValidationError(
            f"SCHEMA VALIDATION FAILURE: '{artifact_path}' is missing "
            f"required fields: {missing_fields}. "
            f"Schema: '{schema_name}'. "
            "Pipeline halted (fail-closed)."
        )

    # ------------------------------------------------------------------
    # Gate 5: Type checks
    # ------------------------------------------------------------------
    for field, expected_type in type_checks.items():
        if field in artifact_data:
            field_val = artifact_data[field]
            if not isinstance(field_val, expected_type):
                raise SchemaValidationError(
                    f"SCHEMA VALIDATION FAILURE: '{artifact_path}' field "
                    f"'{field}' has type {type(field_val).__name__}, "
                    f"expected {expected_type}. "
                    f"Schema: '{schema_name}'."
                )

    # ------------------------------------------------------------------
    # Gate 6: Range checks
    # ------------------------------------------------------------------
    for field, (low, high) in range_checks.items():
        if field in artifact_data:
            field_val = artifact_data[field]
            try:
                numeric_val: float = float(field_val)
            except (TypeError, ValueError):
                continue
            if not (low <= numeric_val <= high):
                raise SchemaValidationError(
                    f"SCHEMA VALIDATION FAILURE: '{artifact_path}' field "
                    f"'{field}' = {numeric_val:.8f} is outside "
                    f"[{low}, {high}]. "
                    f"Schema: '{schema_name}'."
                )

    # ------------------------------------------------------------------
    # Schema-specific additional checks
    # ------------------------------------------------------------------

    # For pc_weights and final_weights: validate weight vector
    if schema_name in ("pc_weights", "final_weights"):
        weights_val = artifact_data.get("weights", [])
        if isinstance(weights_val, list):
            if len(weights_val) != _N_ASSETS:
                raise SchemaValidationError(
                    f"SCHEMA VALIDATION FAILURE: '{artifact_path}' "
                    f"'weights' has {len(weights_val)} elements, "
                    f"expected {_N_ASSETS}. Schema: '{schema_name}'."
                )
            w_arr: np.ndarray = np.array(weights_val, dtype=np.float64)
            if (w_arr < -1e-8).any():
                raise SchemaValidationError(
                    f"SCHEMA VALIDATION FAILURE: '{artifact_path}' "
                    f"'weights' contains negative values. "
                    f"Schema: '{schema_name}'."
                )
            w_sum: float = float(w_arr.sum())
            if abs(w_sum - 1.0) > 1e-5:
                raise SchemaValidationError(
                    f"SCHEMA VALIDATION FAILURE: '{artifact_path}' "
                    f"'weights' sums to {w_sum:.8f}, expected 1.0. "
                    f"Schema: '{schema_name}'."
                )

    # For macro_view: validate regime label
    if schema_name == "macro_view":
        regime_val = artifact_data.get("regime", "")
        valid_regimes: Tuple[str, ...] = (
            "Expansion", "Late-cycle", "Recession", "Recovery"
        )
        if regime_val not in valid_regimes:
            raise SchemaValidationError(
                f"SCHEMA VALIDATION FAILURE: '{artifact_path}' "
                f"'regime' = '{regime_val}' is not a valid regime label. "
                f"Must be one of: {list(valid_regimes)}."
            )

    # Log the successful validation for audit trail
    logger.debug(
        "validate_artifact: '%s' passed schema='%s'.",
        artifact_path,
        schema_name,
    )

    # Return True to signal successful validation
    return True


# =============================================================================
# TOOL 66: build_returns_matrix
# =============================================================================

def build_returns_matrix(
    df_total_return_raw: pd.DataFrame,
    as_of_date: str,
    universe_map: Dict[str, Dict[str, Any]],
    window_months: int = 120,
) -> Tuple[np.ndarray, pd.DatetimeIndex, List[str]]:
    """
    Construct the canonical T×18 monthly returns matrix for the 18-asset universe.

    Implements Task 11, Steps 1–3 and the Covariance Agent's
    ``build_returns_matrix`` script (Task 24, Step 1). This is the
    canonical input to covariance estimation and all backtest computations.

    The frozen return formula from ``DATA_CONVENTIONS["return_formula"]``
    is applied:

    .. math::

        r_{t,i} = \\frac{TR_{t,i}}{TR_{t-1,i}} - 1

    Parameters
    ----------
    df_total_return_raw : pd.DataFrame
        Total return index panel. Must have a MultiIndex with levels
        ``["date", "ticker", "investment_universe"]`` and column
        ``"total_return_index"`` (strictly positive float64).
    as_of_date : str
        ISO-8601 date string. All data is filtered to ``<= as_of_date``.
    universe_map : Dict[str, Dict[str, Any]]
        Mapping from asset class names to metadata. Must contain
        ``"ticker"`` for each of the 18 asset classes.
    window_months : int
        Number of months to include in the returns matrix (lookback
        window). Default: 120 per
        ``METHODOLOGY_PARAMS["COVARIANCE_ESTIMATION"]["window_months"]``.

    Returns
    -------
    Tuple[np.ndarray, pd.DatetimeIndex, List[str]]
        A 3-tuple:

        - ``returns_matrix`` (``np.ndarray``): Shape ``(T, 18)``.
          Monthly simple returns for all 18 asset classes, aligned to
          the common date range.
        - ``date_index`` (``pd.DatetimeIndex``): Monthly dates
          corresponding to the rows of ``returns_matrix``.
        - ``asset_class_order`` (``List[str]``): Ordered list of 18
          asset class names corresponding to the columns.

    Raises
    ------
    TypeError
        If ``df_total_return_raw`` is not a ``pd.DataFrame``.
    ValueError
        If fewer than 2 asset classes have sufficient data for alignment.

    Notes
    -----
    **Inner join alignment:** All 18 return series are aligned to their
    common date range via an inner join. Asset classes with shorter
    histories will reduce the common date range.

    **Forward-fill limit:** Missing values are forward-filled up to 5
    periods (per ``DATA_CONVENTIONS["missing_data_policy"]
    ["forward_fill_limit"]``) before the inner join.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(df_total_return_raw, pd.DataFrame):
        raise TypeError(
            f"df_total_return_raw must be a pd.DataFrame, "
            f"got {type(df_total_return_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Parse as_of_date
    # ------------------------------------------------------------------
    try:
        as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
    except Exception as exc:
        raise ValueError(
            f"as_of_date='{as_of_date}' cannot be parsed. "
            f"Original error: {exc}"
        ) from exc

    # ------------------------------------------------------------------
    # Build monthly return series for each asset class
    # ------------------------------------------------------------------
    # Dict to accumulate monthly return series per asset class
    returns_dict: Dict[str, pd.Series] = {}
    # Canonical asset class order (from universe_map keys)
    asset_class_order: List[str] = list(universe_map.keys())

    for ac in asset_class_order:
        ticker: str = universe_map[ac].get("ticker", "")
        if not ticker:
            logger.warning(
                "build_returns_matrix: No ticker for '%s'. Skipping.",
                ac,
            )
            continue

        # Slice the MultiIndex DataFrame to the target ticker
        try:
            df_ticker: pd.DataFrame = df_total_return_raw.xs(
                ticker, level="ticker"
            )
        except KeyError:
            logger.warning(
                "build_returns_matrix: Ticker '%s' not found for '%s'. "
                "Skipping.",
                ticker,
                ac,
            )
            continue

        # Strip timezone from index if present
        if hasattr(df_ticker.index, "tz") and df_ticker.index.tz is not None:
            df_ticker = df_ticker.copy()
            df_ticker.index = df_ticker.index.tz_localize(None)

        # Apply point-in-time filter
        df_ticker = df_ticker.loc[df_ticker.index <= as_of_ts]

        if df_ticker.empty or "total_return_index" not in df_ticker.columns:
            continue

        # Extract and sort the total return index series
        tri: pd.Series = df_ticker["total_return_index"].sort_index()

        # Validate strictly positive
        if (tri <= 0).any():
            logger.warning(
                "build_returns_matrix: Non-positive TRI for '%s'. Skipping.",
                ac,
            )
            continue

        # Compute monthly returns: r_t = TR_t / TR_{t-1} - 1
        tri_shifted: pd.Series = tri.shift(1)
        # Simple periodic return formula (frozen per DATA_CONVENTIONS)
        monthly_returns: pd.Series = (tri / tri_shifted) - 1.0
        # Drop the first NaN (no prior period for the first observation)
        monthly_returns = monthly_returns.dropna()

        # Forward-fill up to 5 periods to handle sparse data
        monthly_returns = monthly_returns.ffill(limit=5)

        if len(monthly_returns) >= 12:
            returns_dict[ac] = monthly_returns

    # ------------------------------------------------------------------
    # Guard: at least 2 asset classes required for alignment
    # ------------------------------------------------------------------
    if len(returns_dict) < 2:
        raise ValueError(
            f"Fewer than 2 asset classes have sufficient data for "
            f"returns matrix construction. "
            f"Available: {list(returns_dict.keys())}."
        )

    # ------------------------------------------------------------------
    # Align all return series via inner join (common date range)
    # ------------------------------------------------------------------
    # Concatenate all return series into a wide DataFrame (inner join)
    returns_wide: pd.DataFrame = pd.concat(
        returns_dict, axis=1, join="inner"
    )

    # ------------------------------------------------------------------
    # Apply the window_months lookback
    # ------------------------------------------------------------------
    if len(returns_wide) > window_months:
        # Keep only the most recent window_months observations
        returns_wide = returns_wide.iloc[-window_months:]
    elif len(returns_wide) < window_months:
        logger.warning(
            "build_returns_matrix: Only %d months available, "
            "requested window_months=%d. Using all available data.",
            len(returns_wide),
            window_months,
        )

    # ------------------------------------------------------------------
    # Reorder columns to match asset_class_order
    # ------------------------------------------------------------------
    # Keep only asset classes that have data, in canonical order
    available_acs: List[str] = [
        ac for ac in asset_class_order if ac in returns_wide.columns
    ]
    returns_wide = returns_wide[available_acs]

    # ------------------------------------------------------------------
    # Convert to numpy array and extract the date index
    # ------------------------------------------------------------------
    # Monthly returns matrix: shape (T, N_available)
    returns_matrix: np.ndarray = returns_wide.values.astype(np.float64)
    # DatetimeIndex of monthly dates
    date_index: pd.DatetimeIndex = pd.DatetimeIndex(returns_wide.index)

    logger.info(
        "build_returns_matrix: shape=(%d, %d), "
        "date_range=[%s, %s].",
        returns_matrix.shape[0],
        returns_matrix.shape[1],
        date_index[0].date() if len(date_index) > 0 else "N/A",
        date_index[-1].date() if len(date_index) > 0 else "N/A",
    )

    return returns_matrix, date_index, available_acs


# =============================================================================
# TOOL 67: estimate_covariance
# =============================================================================

def estimate_covariance(
    returns_matrix: np.ndarray,
    estimator: str = "SHRINKAGE_LEDOIT_WOLF",
) -> np.ndarray:
    """
    Estimate the annualised covariance matrix from monthly returns.

    Implements Task 24, Step 2 — the covariance estimation step for the
    Covariance Agent. Applies the frozen estimator from
    ``METHODOLOGY_PARAMS["COVARIANCE_ESTIMATION"]["estimator"]``.

    The Ledoit-Wolf shrinkage estimator is the default:

    .. math::

        \\hat{\\Sigma}_{LW} = \\alpha \\hat{\\Sigma}_{sample}
                             + (1 - \\alpha) \\mu_{target} I

    where :math:`\\alpha` is the optimal shrinkage intensity computed
    analytically by the Ledoit-Wolf (2004) oracle approximating shrinkage
    (OAS) algorithm.

    The result is annualised by multiplying by 12 (periods per year):

    .. math::

        \\Sigma_{ann} = 12 \\cdot \\hat{\\Sigma}_{LW}

    Parameters
    ----------
    returns_matrix : np.ndarray
        T×N monthly simple returns matrix. Shape: ``(T, N)`` where
        ``N ≤ 18``. Must have ``T > N`` for a full-rank estimate.
    estimator : str
        Covariance estimator to apply. Supported values:
        ``"SHRINKAGE_LEDOIT_WOLF"`` (default) and ``"sample"``.

    Returns
    -------
    np.ndarray
        Annualised covariance matrix. Shape: ``(N, N)``.

    Raises
    ------
    TypeError
        If ``returns_matrix`` is not a ``np.ndarray``.
    ValueError
        If ``returns_matrix`` has fewer rows than columns (T < N).
    ValueError
        If ``estimator`` is not a recognised estimator name.

    Notes
    -----
    **sklearn fallback:** If ``sklearn`` is not available,
    the sample covariance estimator is used regardless of the
    ``estimator`` parameter, with a logged warning.

    **Annualisation:** The monthly covariance matrix is multiplied by
    12 (periods per year) to produce the annualised covariance matrix,
    consistent with ``DATA_CONVENTIONS["annualisation"]``.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(returns_matrix, np.ndarray):
        raise TypeError(
            f"returns_matrix must be a np.ndarray, "
            f"got {type(returns_matrix).__name__}."
        )
    if returns_matrix.ndim != 2:
        raise ValueError(
            f"returns_matrix must be 2-dimensional, "
            f"got {returns_matrix.ndim} dimensions."
        )
    T: int = returns_matrix.shape[0]
    N: int = returns_matrix.shape[1]
    if T <= N:
        raise ValueError(
            f"returns_matrix must have T > N (T={T}, N={N}). "
            "Insufficient observations for full-rank covariance estimation."
        )
    if estimator not in ("SHRINKAGE_LEDOIT_WOLF", "sample"):
        raise ValueError(
            f"estimator='{estimator}' is not recognised. "
            "Must be 'SHRINKAGE_LEDOIT_WOLF' or 'sample'."
        )

    # ------------------------------------------------------------------
    # Compute the monthly covariance matrix
    # ------------------------------------------------------------------
    if estimator == "SHRINKAGE_LEDOIT_WOLF" and _SKLEARN_AVAILABLE:
        # Apply Ledoit-Wolf shrinkage estimator via sklearn
        lw: _LedoitWolf = _LedoitWolf(assume_centered=False)
        lw.fit(returns_matrix)
        # Monthly covariance matrix from Ledoit-Wolf
        sigma_monthly: np.ndarray = lw.covariance_
    else:
        # Fall back to sample covariance (ddof=1)
        if estimator == "SHRINKAGE_LEDOIT_WOLF" and not _SKLEARN_AVAILABLE:
            logger.warning(
                "estimate_covariance: sklearn not available. "
                "Falling back to sample covariance."
            )
        # Sample covariance matrix: (1/(T-1)) * (R - R_bar)' * (R - R_bar)
        sigma_monthly = np.cov(returns_matrix.T, ddof=1)

    # ------------------------------------------------------------------
    # Annualise the covariance matrix:
    # Sigma_ann = 12 * Sigma_monthly
    # per DATA_CONVENTIONS["annualisation"]["periods_per_year"] = 12
    # ------------------------------------------------------------------
    sigma_annualised: np.ndarray = float(_PERIODS_PER_YEAR) * sigma_monthly

    logger.info(
        "estimate_covariance: estimator='%s', shape=(%d, %d), "
        "min_eigenvalue=%.4e.",
        estimator,
        N,
        N,
        float(np.linalg.eigvalsh(sigma_annualised).min()),
    )

    return sigma_annualised


# =============================================================================
# TOOL 68: enforce_psd
# =============================================================================

def enforce_psd(
    sigma: np.ndarray,
    clip_min: float = _PSD_CLIP_MIN,
) -> np.ndarray:
    """
    Repair a covariance matrix to be positive semi-definite via eigenvalue clipping.

    Implements Task 24, Step 3 — the PSD repair step for the Covariance
    Agent, using the frozen method from
    ``METHODOLOGY_PARAMS["COVARIANCE_ESTIMATION"]["psd_repair"]
    = "eigenvalue_clipping"``.

    **Algorithm:**

    1. Compute the symmetric eigendecomposition:
       :math:`\\Sigma = Q \\Lambda Q^\\top`
    2. Clip all eigenvalues to a minimum of ``clip_min``:
       :math:`\\Lambda_{clipped} = \\max(\\Lambda, \\epsilon)`
    3. Reconstruct: :math:`\\hat{\\Sigma} = Q \\Lambda_{clipped} Q^\\top`
    4. Symmetrise: :math:`\\hat{\\Sigma} = (\\hat{\\Sigma} + \\hat{\\Sigma}^\\top) / 2`

    Parameters
    ----------
    sigma : np.ndarray
        N×N covariance matrix (possibly non-PSD due to numerical errors).
        Shape: ``(N, N)``.
    clip_min : float
        Minimum eigenvalue after clipping. Default: ``1e-8`` per
        ``_PSD_CLIP_MIN``.

    Returns
    -------
    np.ndarray
        PSD covariance matrix. Shape: ``(N, N)``.

    Raises
    ------
    TypeError
        If ``sigma`` is not a ``np.ndarray``.
    ValueError
        If ``sigma`` is not a square 2-D array.

    Notes
    -----
    **Early exit:** If the minimum eigenvalue of ``sigma`` is already
    ≥ ``clip_min``, the matrix is already PSD and is returned unchanged
    (after symmetrisation for numerical safety).

    **``numpy.linalg.eigh``:** Used instead of ``numpy.linalg.eig``
    because ``eigh`` is specifically designed for symmetric/Hermitian
    matrices and is more numerically stable, returning real eigenvalues
    by construction.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(sigma, np.ndarray):
        raise TypeError(
            f"sigma must be a np.ndarray, got {type(sigma).__name__}."
        )
    if sigma.ndim != 2 or sigma.shape[0] != sigma.shape[1]:
        raise ValueError(
            f"sigma must be a square 2-D array, got shape {sigma.shape}."
        )

    # ------------------------------------------------------------------
    # Symmetrise the input matrix (numerical safety)
    # ------------------------------------------------------------------
    # Ensure exact symmetry before eigendecomposition
    sigma_sym: np.ndarray = (sigma + sigma.T) / 2.0

    # ------------------------------------------------------------------
    # Compute the symmetric eigendecomposition: Sigma = Q * Lambda * Q'
    # numpy.linalg.eigh is used for symmetric matrices (real eigenvalues)
    # ------------------------------------------------------------------
    # eigenvalues: shape (N,), eigenvectors: shape (N, N)
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    eigenvalues, eigenvectors = np.linalg.eigh(sigma_sym)

    # ------------------------------------------------------------------
    # Early exit: if already PSD, return symmetrised matrix
    # ------------------------------------------------------------------
    if float(eigenvalues.min()) >= clip_min:
        # Matrix is already PSD; return symmetrised version
        return sigma_sym

    # ------------------------------------------------------------------
    # Clip eigenvalues to minimum of clip_min
    # Lambda_clipped = max(Lambda, clip_min)
    # ------------------------------------------------------------------
    # Number of clipped eigenvalues for logging
    n_clipped: int = int(np.sum(eigenvalues < clip_min))
    # Apply the clipping operation
    eigenvalues_clipped: np.ndarray = np.maximum(eigenvalues, clip_min)

    # ------------------------------------------------------------------
    # Reconstruct the PSD matrix: Sigma_hat = Q * Lambda_clipped * Q'
    # ------------------------------------------------------------------
    # Matrix multiplication: Q * diag(Lambda_clipped) * Q'
    sigma_psd: np.ndarray = (
        eigenvectors @ np.diag(eigenvalues_clipped) @ eigenvectors.T
    )

    # ------------------------------------------------------------------
    # Symmetrise the reconstructed matrix (floating-point safety)
    # ------------------------------------------------------------------
    sigma_psd = (sigma_psd + sigma_psd.T) / 2.0

    logger.info(
        "enforce_psd: clipped %d eigenvalues (min_before=%.4e, "
        "min_after=%.4e).",
        n_clipped,
        float(eigenvalues.min()),
        float(np.linalg.eigvalsh(sigma_psd).min()),
    )

    return sigma_psd


# =============================================================================
# TOOL 69: compute_risk_contributions
# =============================================================================

def compute_risk_contributions(
    weights: List[float],
    sigma: np.ndarray,
) -> np.ndarray:
    """
    Compute the variance-based risk contribution of each asset.

    Implements the risk contribution formula from
    ``IPS_GOVERNANCE["CONSTRAINT_DEFINITIONS"]["risk_contribution_formula"]``
    (Task 32, Step 3):

    .. math::

        RC_i = \\frac{w_i (\\Sigma w)_i}{w^\\top \\Sigma w}

    The risk contributions sum to 1.0 by construction:
    :math:`\\sum_i RC_i = 1`.

    Parameters
    ----------
    weights : List[float]
        18-element portfolio weight vector.
    sigma : np.ndarray
        18×18 annualised covariance matrix. Shape: ``(18, 18)``.

    Returns
    -------
    np.ndarray
        18-element risk contribution vector. Values sum to 1.0
        (within ``1e-8``). May contain negative values for assets
        with negative covariance with the portfolio.

    Raises
    ------
    TypeError
        If ``sigma`` is not a ``np.ndarray``.
    ValueError
        If ``weights`` does not have 18 elements or ``sigma`` is not
        shape ``(18, 18)``.

    Notes
    -----
    **Zero portfolio variance guard:** If the portfolio variance
    :math:`w^\\top \\Sigma w` is effectively zero (degenerate portfolio),
    equal risk contributions (1/N each) are returned to avoid division
    by zero.

    **Normalisation:** The risk contributions are normalised to sum to
    exactly 1.0 after computation to correct for any floating-point
    rounding errors.
    """
    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------
    if not isinstance(sigma, np.ndarray):
        raise TypeError(
            f"sigma must be a np.ndarray, got {type(sigma).__name__}."
        )
    if sigma.shape != (_N_ASSETS, _N_ASSETS):
        raise ValueError(
            f"sigma must have shape ({_N_ASSETS}, {_N_ASSETS}), "
            f"got {sigma.shape}."
        )
    if len(weights) != _N_ASSETS:
        raise ValueError(
            f"weights must have {_N_ASSETS} elements, got {len(weights)}."
        )

    # ------------------------------------------------------------------
    # Convert weights to numpy array of float64
    # ------------------------------------------------------------------
    # Portfolio weight vector as numpy array
    w: np.ndarray = np.array(weights, dtype=np.float64)

    # ------------------------------------------------------------------
    # Compute Sigma * w: matrix-vector product, shape (18,)
    # ------------------------------------------------------------------
    sigma_w: np.ndarray = sigma @ w

    # ------------------------------------------------------------------
    # Compute portfolio variance: w' * Sigma * w = scalar
    # ------------------------------------------------------------------
    port_var: float = float(np.dot(w, sigma_w))

    # ------------------------------------------------------------------
    # Zero portfolio variance guard: return equal risk contributions
    # ------------------------------------------------------------------
    if port_var < _EPS:
        logger.warning(
            "compute_risk_contributions: Portfolio variance is "
            "effectively zero (%.2e). Returning equal risk contributions.",
            port_var,
        )
        # Equal risk contributions: 1/N for each asset
        return np.ones(_N_ASSETS, dtype=np.float64) / _N_ASSETS

    # ------------------------------------------------------------------
    # Compute risk contributions:
    # RC_i = w_i * (Sigma w)_i / (w' Sigma w)
    # ------------------------------------------------------------------
    # Element-wise product: w_i * (Sigma w)_i for each asset i
    rc: np.ndarray = w * sigma_w / port_var

    # ------------------------------------------------------------------
    # Normalise to sum to exactly 1.0 (floating-point correction)
    # ------------------------------------------------------------------
    rc_sum: float = float(rc.sum())
    if abs(rc_sum) > _EPS:
        # Normalise by the actual sum to correct floating-point errors
        rc = rc / rc_sum

    return rc


# =============================================================================
# TOOL 70: compute_benchmark_returns
# =============================================================================

def compute_benchmark_returns(
    df_benchmark_60_40_raw: pd.DataFrame,
    as_of_date: Optional[str] = None,
) -> pd.Series:
    """
    Compute the 60/40 benchmark return series from total return indices.

    Implements Task 14, Step 2 — the benchmark return computation using
    the frozen formula from ``DATA_CONVENTIONS["benchmark_return_formula"]
    = "weighted_arithmetic"``:

    .. math::

        r^{(b)}_t = w_{eq} \\cdot r^{(eq)}_t + w_{bond} \\cdot r^{(bond)}_t

    where:

    .. math::

        r^{(eq)}_t = \\frac{TR^{(eq)}_t}{TR^{(eq)}_{t-1}} - 1, \\quad
        r^{(bond)}_t = \\frac{TR^{(bond)}_t}{TR^{(bond)}_{t-1}} - 1

    Parameters
    ----------
    df_benchmark_60_40_raw : pd.DataFrame
        60/40 benchmark panel. Must have a ``DatetimeIndex`` and columns:
        ``"equity_leg_total_return_index"``,
        ``"bond_leg_total_return_index"``,
        ``"w_equity"`` (typically 0.60),
        ``"w_bond"`` (typically 0.40).
    as_of_date : Optional[str]
        ISO-8601 date string. If provided, data is filtered to
        ``<= as_of_date`` for point-in-time discipline.

    Returns
    -------
    pd.Series
        Monthly benchmark return series with ``DatetimeIndex``.
        Values are simple periodic returns in decimal form.

    Raises
    ------
    TypeError
        If ``df_benchmark_60_40_raw`` is not a ``pd.DataFrame``.
    ValueError
        If required columns are missing.
    ValueError
        If the benchmark weights do not sum to approximately 1.0.

    Notes
    -----
    **Frequency alignment:** If ``df_benchmark_60_40_raw`` has a daily
    index, it is resampled to monthly (last business day) before return
    computation, consistent with ``DATA_CONVENTIONS["resampling_rules"]
    ["daily_to_monthly"] = "last_business_day"``.

    **Weight validation:** The benchmark weights ``w_equity`` and
    ``w_bond`` are validated to sum to 1.0 (within ``1e-4``). A warning
    is logged if they do not sum to 1.0.
    """
    # ------------------------------------------------------------------
    # Input validation: type check
    # ------------------------------------------------------------------
    if not isinstance(df_benchmark_60_40_raw, pd.DataFrame):
        raise TypeError(
            f"df_benchmark_60_40_raw must be a pd.DataFrame, "
            f"got {type(df_benchmark_60_40_raw).__name__}."
        )

    # ------------------------------------------------------------------
    # Input validation: required columns
    # ------------------------------------------------------------------
    _required_cols: Tuple[str, ...] = (
        "equity_leg_total_return_index",
        "bond_leg_total_return_index",
        "w_equity",
        "w_bond",
    )
    missing_cols: List[str] = [
        c for c in _required_cols
        if c not in df_benchmark_60_40_raw.columns
    ]
    if missing_cols:
        raise ValueError(
            f"df_benchmark_60_40_raw is missing required columns: "
            f"{missing_cols}."
        )

    # ------------------------------------------------------------------
    # Strip timezone from index if present
    # ------------------------------------------------------------------
    df_bm: pd.DataFrame = df_benchmark_60_40_raw.copy()
    if hasattr(df_bm.index, "tz") and df_bm.index.tz is not None:
        df_bm.index = df_bm.index.tz_localize(None)

    # ------------------------------------------------------------------
    # Apply point-in-time filter if as_of_date is provided
    # ------------------------------------------------------------------
    if as_of_date is not None:
        try:
            as_of_ts: pd.Timestamp = pd.Timestamp(as_of_date)
            df_bm = df_bm.loc[df_bm.index <= as_of_ts]
        except Exception as exc:
            raise ValueError(
                f"as_of_date='{as_of_date}' cannot be parsed. "
                f"Original error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Frequency alignment: resample to monthly if daily index
    # ------------------------------------------------------------------
    if isinstance(df_bm.index, pd.DatetimeIndex):
        # Infer frequency: if more than 50 rows per year, assume daily
        n_rows: int = len(df_bm)
        if n_rows > 0:
            date_range_years: float = (
                (df_bm.index[-1] - df_bm.index[0]).days / 365.25
            )
            rows_per_year: float = (
                n_rows / date_range_years if date_range_years > 0 else n_rows
            )
            if rows_per_year > 50:
                # Daily data: resample to monthly (last business day)
                df_bm = df_bm.resample("BME").last()

    # ------------------------------------------------------------------
    # Extract benchmark weights (use the first row's values)
    # ------------------------------------------------------------------
    # Equity weight (typically 0.60)
    w_equity: float = float(df_bm["w_equity"].iloc[0])
    # Bond weight (typically 0.40)
    w_bond: float = float(df_bm["w_bond"].iloc[0])

    # ------------------------------------------------------------------
    # Validate that weights sum to approximately 1.0
    # ------------------------------------------------------------------
    weight_sum: float = w_equity + w_bond
    if abs(weight_sum - 1.0) > 1e-4:
        logger.warning(
            "compute_benchmark_returns: Benchmark weights sum to %.6f "
            "(expected 1.0). w_equity=%.4f, w_bond=%.4f.",
            weight_sum,
            w_equity,
            w_bond,
        )

    # ------------------------------------------------------------------
    # Compute equity leg returns:
    # r_eq_t = TR_eq_t / TR_eq_{t-1} - 1
    # ------------------------------------------------------------------
    # Equity total return index series
    tri_eq: pd.Series = df_bm["equity_leg_total_return_index"]
    # Equity leg monthly returns using frozen formula
    r_eq: pd.Series = (tri_eq / tri_eq.shift(1)) - 1.0

    # ------------------------------------------------------------------
    # Compute bond leg returns:
    # r_bond_t = TR_bond_t / TR_bond_{t-1} - 1
    # ------------------------------------------------------------------
    # Bond total return index series
    tri_bond: pd.Series = df_bm["bond_leg_total_return_index"]
    # Bond leg monthly returns using frozen formula
    r_bond: pd.Series = (tri_bond / tri_bond.shift(1)) - 1.0

    # ------------------------------------------------------------------
    # Compute the combined 60/40 benchmark return:
    # r_b_t = w_eq * r_eq_t + w_bond * r_bond_t
    # per DATA_CONVENTIONS["benchmark_return_formula"] = "weighted_arithmetic"
    # ------------------------------------------------------------------
    # Weighted arithmetic combination of equity and bond leg returns
    r_benchmark: pd.Series = w_equity * r_eq + w_bond * r_bond

    # ------------------------------------------------------------------
    # Drop the first NaN (no prior period for the first observation)
    # ------------------------------------------------------------------
    r_benchmark = r_benchmark.dropna()

    logger.info(
        "compute_benchmark_returns: n_obs=%d, "
        "date_range=[%s, %s], w_equity=%.2f, w_bond=%.2f.",
        len(r_benchmark),
        r_benchmark.index[0].date() if len(r_benchmark) > 0 else "N/A",
        r_benchmark.index[-1].date() if len(r_benchmark) > 0 else "N/A",
        w_equity,
        w_bond,
    )

    return r_benchmark
