"""
gdex.py — Greedy Design of Experiments (GDEX)
==============================================
A two-stage pipeline for synthetic-control experimental design,
following the framework of Abadie & Zhao (2025).

Stage 1  Greedy unit assignment
         For each candidate ceiling on treated units (1 … N-1), runs a
         fully vectorised greedy algorithm that jointly assigns all N
         units to treated or control by maximising incremental R² against
         the national (column) average at each step.  The split with the
         lowest L2 score is retained.

Stage 2  Convex weight optimisation
         Given the selected unit partition, solves a convex quadratic
         programme to find simplex weights w, v such that the weighted
         treated and control pre-period means best approximate the
         national average.

Stage 3  Conformal inference  (GDEX_infer)
         Partitions the pre-period into fitting and blank sub-periods,
         runs Stages 1-2 on the fitting sub-period only, then uses the
         held-out blank periods to construct a permutation p-value and
         split-conformal confidence intervals for each post-intervention
         period, following Theorems 2 and 3 of Abadie & Zhao (2025).

Public API
----------
GDEX(X_pre, solver)          -> GDEXResult
GDEX_infer(X_pre, X_post, …) -> GDEXInferenceResult
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb

import numpy as np
import cvxpy as cp
from joblib import Parallel, delayed


# ═══════════════════════════════════════════════════════════════════════════════
# RESULT CONTAINERS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GDEXResult:
    """
    Output of the two-stage GDEX pipeline.

    Attributes
    ----------
    treated_units   : column indices of X_pre assigned to treatment
    control_units   : column indices of X_pre assigned to control
    treated_weights : optimal simplex weights over treated units  (Stage 2)
    control_weights : optimal simplex weights over control units  (Stage 2)
    treated_mean    : weighted pre-period mean for treated group
    control_mean    : weighted pre-period mean for control group
    score           : L2 score used to select the optimal max_treated
    max_treated     : winning treated-unit ceiling from Stage 1 sweep
    """
    treated_units:   np.ndarray
    control_units:   np.ndarray
    treated_weights: np.ndarray
    control_weights: np.ndarray
    treated_mean:    np.ndarray
    control_mean:    np.ndarray
    score:           float
    max_treated:     int


@dataclass
class GDEXInferenceResult:
    """
    Output of the GDEX inference layer.

    Attributes
    ----------
    gdex            : the underlying GDEXResult
    gaps_blank      : û_t for each blank period  (reference distribution)
    gaps_post       : û_t for each post-intervention period (test statistic)
    p_value         : permutation p-value for H0: no treatment effect
    ci_lower        : lower bound of (1-alpha) conformal CI, per post period
    ci_upper        : upper bound of (1-alpha) conformal CI, per post period
    alpha           : significance level used
    fitting_periods : row indices of X_pre used to fit GDEX
    blank_periods   : row indices of X_pre held out for inference
    """
    gdex:            GDEXResult
    gaps_blank:      np.ndarray
    gaps_post:       np.ndarray
    p_value:         float
    ci_lower:        np.ndarray
    ci_upper:        np.ndarray
    alpha:           float
    fitting_periods: np.ndarray
    blank_periods:   np.ndarray


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 HELPERS — GREEDY ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _initialize_state(X_pre: np.ndarray, x_bar: np.ndarray) -> dict:
    """
    Precompute everything that is constant across greedy iterations and
    initialise all mutable loop state.

    Parameters
    ----------
    X_pre : (T0, N) pre-period data matrix
    x_bar : (T0,)  national average series  (column mean of X_pre)

    Returns
    -------
    state : dict holding masks, running sums, counters, and precomputed
            quantities (y_c, ss_tot) needed by the inner loop helpers.
    """
    T0, N  = X_pre.shape
    y_c    = x_bar - x_bar.mean()
    ss_tot = float(np.sum(y_c ** 2))

    return dict(
        remaining_mask = np.ones(N, dtype=bool),
        treated_sum    = np.zeros(T0, dtype=float),
        control_sum    = np.zeros(T0, dtype=float),
        treated_idx    = [],
        control_idx    = [],
        k_t    = 0,
        k_c    = 0,
        y_c    = y_c,
        ss_tot = ss_tot,
    )


def _compute_r2_candidates(
    X_rem:       np.ndarray,
    current_sum: np.ndarray,
    k:           int,
    ss_tot:      float,
    y_c:         np.ndarray,
) -> np.ndarray:
    """
    Vectorised incremental R² for every remaining unit, given the current
    running sum and count of the group it would join.

    Parameters
    ----------
    X_rem        : (T0, R) columns of X_pre for remaining units
    current_sum  : (T0,)  running sum of the group (treated or control)
    k            : current unit count in that group
    ss_tot       : total sum of squares of y_c  (constant)
    y_c          : (T0,) national deviation series

    Returns
    -------
    r2 : (R,) incremental R² values, one per remaining candidate unit
    """
    current_mean = current_sum / max(1, k)
    delta        = X_rem - current_mean[:, None]       # (T0, R)
    ss_res_new   = (
        ss_tot
        - 2.0 * (y_c @ delta)
        + np.sum(delta ** 2, axis=0)
    )
    return 1.0 - ss_res_new / ss_tot


def _check_control_eligible(R: int, k_t: int, max_treated: int) -> bool:
    """
    Guard condition: only try adding a unit to control when enough
    remaining units exist to still satisfy the treated quota.

    Parameters
    ----------
    R           : number of remaining units
    k_t         : treated units assigned so far
    max_treated : ceiling on treated units
    """
    return (R - 1) >= max(1, max_treated - k_t)


def _select_best_unit(
    rem_idx:    np.ndarray,
    r2_treated: np.ndarray | None,
    r2_control: np.ndarray | None,
) -> tuple[int, bool]:
    """
    Pick the global winner across treated and control candidate lists.

    Parameters
    ----------
    rem_idx    : (R,) original column indices of remaining units
    r2_treated : (R,) R² values if added to treated  — None if ineligible
    r2_control : (R,) R² values if added to control  — None if ineligible

    Returns
    -------
    best_unit      : integer column index into X_pre
    best_is_treated: True if the winner is assigned to treated
    """
    best_r2, best_unit, best_is_treated = -np.inf, -1, False

    if r2_treated is not None:
        t_max = float(r2_treated.max())
        if t_max > best_r2:
            best_r2         = t_max
            best_unit       = int(rem_idx[r2_treated.argmax()])
            best_is_treated = True

    if r2_control is not None:
        c_max = float(r2_control.max())
        if c_max > best_r2:
            best_unit       = int(rem_idx[r2_control.argmax()])
            best_is_treated = False

    return best_unit, best_is_treated


def _assign_unit(
    state:     dict,
    X_pre:     np.ndarray,
    best_unit: int,
    is_treated: bool,
) -> None:
    """
    Mutate state in-place: append the winner, update the running sum,
    increment the counter, and clear the mask bit.
    """
    if is_treated:
        state["treated_idx"].append(best_unit)
        state["treated_sum"] += X_pre[:, best_unit]
        state["k_t"] += 1
    else:
        state["control_idx"].append(best_unit)
        state["control_sum"] += X_pre[:, best_unit]
        state["k_c"] += 1

    state["remaining_mask"][best_unit] = False


def _compute_final_means(
    treated_sum: np.ndarray, k_t: int,
    control_sum: np.ndarray, k_c: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Divide running sums by counts, guarding against zero division.
    """
    return treated_sum / max(1, k_t), control_sum / max(1, k_c)


def _greedy_assignment(
    X_pre:       np.ndarray,
    x_bar:       np.ndarray,
    max_treated: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the greedy unit-assignment loop for a single value of max_treated.

    Returns
    -------
    treated_idx  : selected treated column indices
    control_idx  : selected control column indices
    treated_mean : uniform mean over treated units
    control_mean : uniform mean over control units
    """
    state        = _initialize_state(X_pre, x_bar)
    y_c, ss_tot  = state["y_c"], state["ss_tot"]

    while state["remaining_mask"].any():
        rem_idx = np.nonzero(state["remaining_mask"])[0]
        X_rem   = X_pre[:, rem_idx]
        R       = X_rem.shape[1]

        r2_treated = (
            _compute_r2_candidates(
                X_rem, state["treated_sum"], state["k_t"], ss_tot, y_c,
            )
            if state["k_t"] < max_treated else None
        )
        r2_control = (
            _compute_r2_candidates(
                X_rem, state["control_sum"], state["k_c"], ss_tot, y_c,
            )
            if _check_control_eligible(R, state["k_t"], max_treated) else None
        )

        best_unit, is_treated = _select_best_unit(rem_idx, r2_treated, r2_control)
        _assign_unit(state, X_pre, best_unit, is_treated)

    treated_mean, control_mean = _compute_final_means(
        state["treated_sum"], state["k_t"],
        state["control_sum"], state["k_c"],
    )
    return (
        np.array(state["treated_idx"]),
        np.array(state["control_idx"]),
        treated_mean,
        control_mean,
    )


def _evaluate_split(
    max_treated: int,
    X_pre:       np.ndarray,
    x_bar:       np.ndarray,
) -> dict:
    """
    Run greedy assignment for one max_treated value and compute its L2 score.
    Called in parallel by GDEX.
    """
    treated, control, t_mean, c_mean = _greedy_assignment(X_pre, x_bar, max_treated)
    score = float(
        np.linalg.norm(t_mean - x_bar) + np.linalg.norm(c_mean - x_bar)
    )
    return dict(
        max_treated   = max_treated,
        treated_units = treated,
        control_units = control,
        treated_mean  = t_mean,
        control_mean  = c_mean,
        score         = score,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 HELPERS — CONVEX WEIGHT OPTIMISATION
# ═══════════════════════════════════════════════════════════════════════════════

def _build_scm_problem(
    X_treated: np.ndarray,
    X_control: np.ndarray,
    target:    np.ndarray,
) -> tuple[cp.Problem, cp.Variable, cp.Variable]:
    """
    Declare CVXPY variables, objective, and simplex constraints.
    Does not solve — purely symbolic construction.

    Solves:
        min  ‖X_treated w − target‖² + ‖X_control v − target‖²
        s.t. w ≥ 0,  1ᵀw = 1
             v ≥ 0,  1ᵀv = 1

    Returns
    -------
    prob : unsolved cp.Problem
    w    : (n_treated,) cp.Variable
    v    : (n_control,) cp.Variable
    """
    w = cp.Variable(X_treated.shape[1])
    v = cp.Variable(X_control.shape[1])

    objective = cp.Minimize(
        cp.sum_squares(X_treated @ w - target)
        + cp.sum_squares(X_control @ v - target)
    )
    constraints = [
        w >= 0, cp.sum(w) == 1,
        v >= 0, cp.sum(v) == 1,
    ]
    return cp.Problem(objective, constraints), w, v


def _solve_problem(prob: cp.Problem, solver: str = "OSQP") -> str:
    """
    Solve a pre-built CVXPY problem and return the solver status string.
    Separating solve from build lets tests mock either half independently.

    Returns
    -------
    status : e.g. "optimal", "optimal_inaccurate", "infeasible"
    """
    prob.solve(solver=getattr(cp, solver), verbose=False)
    return prob.status


def _extract_weights(raw: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Post-process a raw CVXPY weight vector into a valid probability simplex:
      1. clip negatives to zero  (numerical solver noise)
      2. renormalise so weights sum to 1

    Parameters
    ----------
    raw : raw .value array from a cp.Variable
    eps : small constant guarding against zero-sum edge case
    """
    clipped = np.clip(raw, 0.0, 1.0)
    return clipped / (clipped.sum() + eps)


def _optimize_weights(
    X_treated: np.ndarray,
    X_control: np.ndarray,
    target:    np.ndarray,
    solver:    str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build, solve, and extract weights for a single treated/control partition.

    Raises RuntimeError if the solver does not reach an acceptable status.
    """
    prob, w, v = _build_scm_problem(X_treated, X_control, target)
    status     = _solve_problem(prob, solver=solver)

    if status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(
            f"SCM solver did not converge — status: '{status}'. "
            "Consider inspecting your data or trying a different solver."
        )

    return _extract_weights(w.value), _extract_weights(v.value)


# ═══════════════════════════════════════════════════════════════════════════════
# GDEX — GREEDY DESIGN OF EXPERIMENTS
# ═══════════════════════════════════════════════════════════════════════════════

def GDEX(
    X_pre:  np.ndarray,
    solver: str = "OSQP",
) -> GDEXResult:
    """
    Greedy Design of Experiments (GDEX).

    Runs a fully parallelised sweep over all possible treated-unit ceilings
    (Stage 1), retains the split with the best L2 score, then solves a
    convex quadratic programme to find optimal simplex weights for both
    groups (Stage 2).

    Parameters
    ----------
    X_pre  : (T0, N) array — each column is a unit, each row a time period
    solver : CVXPY solver name for Stage 2  (default "OSQP")

    Returns
    -------
    GDEXResult dataclass with units, weights, weighted means, score,
    and the winning max_treated value.
    """
    if X_pre.ndim != 2:
        raise ValueError(
            f"X_pre must be 2-D (T0 × N), got shape {X_pre.shape}."
        )
    if X_pre.shape[1] < 2:
        raise ValueError(
            "Need at least 2 units to form a treated / control split."
        )

    T0, N = X_pre.shape
    x_bar = X_pre.mean(axis=1)            # (T0,) national average

    # ── Stage 1: sweep all possible treated ceilings in parallel ──────────────
    all_splits = Parallel(n_jobs=-1)(
        delayed(_evaluate_split)(max_treated, X_pre, x_bar)
        for max_treated in range(1, N)
    )
    best = min(all_splits, key=lambda r: r["score"])

    # ── Stage 2: optimise weights for the winning split ────────────────────────
    X_treated = X_pre[:, best["treated_units"]]
    X_control = X_pre[:, best["control_units"]]

    treated_weights, control_weights = _optimize_weights(
        X_treated, X_control, x_bar, solver=solver,
    )

    return GDEXResult(
        treated_units   = best["treated_units"],
        control_units   = best["control_units"],
        treated_weights = treated_weights,
        control_weights = control_weights,
        treated_mean    = X_treated @ treated_weights,
        control_mean    = X_control @ control_weights,
        score           = best["score"],
        max_treated     = best["max_treated"],
    )


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 HELPERS — CONFORMAL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def _split_periods(
    T0:           int,
    fitting_frac: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Partition the T0 pre-intervention rows into fitting (E) and blank (B)
    periods.  Fitting periods are taken from the front; blank from the tail,
    mirroring the Walmart illustration in Abadie & Zhao (2025).

    Parameters
    ----------
    T0           : total number of pre-intervention rows
    fitting_frac : fraction of rows used for weight estimation (default 0.75)

    Returns
    -------
    fitting_idx : row indices passed to GDEX
    blank_idx   : row indices held out for inference
    """
    if not 0.0 < fitting_frac < 1.0:
        raise ValueError(f"fitting_frac must be in (0, 1), got {fitting_frac}.")

    n_fit = max(1, int(np.floor(T0 * fitting_frac)))
    if n_fit >= T0:
        raise ValueError(
            f"fitting_frac={fitting_frac} leaves no blank periods. "
            "Reduce fitting_frac or increase T0."
        )

    all_idx     = np.arange(T0)
    fitting_idx = all_idx[:n_fit]
    blank_idx   = all_idx[n_fit:]
    return fitting_idx, blank_idx


def _compute_gaps(
    X_full:          np.ndarray,
    treated_units:   np.ndarray,
    control_units:   np.ndarray,
    treated_weights: np.ndarray,
    control_weights: np.ndarray,
    period_idx:      np.ndarray,
) -> np.ndarray:
    """
    Compute the synthetic gap
        û_t = Σ_j w*_j Y_jt  −  Σ_j v*_j Y_jt
    for a specified set of row indices into X_full.

    Parameters
    ----------
    X_full          : (T_total, N) full data matrix (pre + post stacked)
    treated_units   : column indices for treated group
    control_units   : column indices for control group
    treated_weights : (n_treated,) simplex weights
    control_weights : (n_control,) simplex weights
    period_idx      : row indices to evaluate gaps over

    Returns
    -------
    gaps : (len(period_idx),) array of û_t values
    """
    Y                 = X_full[period_idx, :]
    synthetic_treated = Y[:, treated_units] @ treated_weights
    synthetic_control = Y[:, control_units] @ control_weights
    return synthetic_treated - synthetic_control


def _test_statistic(gaps: np.ndarray) -> float:
    """
    S(e) = mean |û_t|  — equation (18) in Abadie & Zhao (2025).
    """
    return float(np.mean(np.abs(gaps)))


def _permutation_pvalue(
    gaps_blank: np.ndarray,
    gaps_post:  np.ndarray,
    max_combos: int = 10_000,
) -> float:
    """
    Permutation p-value by drawing (T_post)-combinations from the union
    of blank and post-intervention gaps — equation (19) in Abadie & Zhao.

    When |Π| exceeds max_combos, a seeded Monte Carlo approximation is
    used instead of exhaustive enumeration.

    Parameters
    ----------
    gaps_blank : û_t for blank periods     (T_B,)
    gaps_post  : û_t for post periods      (T_post,)
    max_combos : cap before switching to Monte Carlo

    Returns
    -------
    p_value : fraction of permutations whose S >= S(observed)
    """
    T_post   = len(gaps_post)
    all_gaps = np.concatenate([gaps_blank, gaps_post])
    T_all    = len(all_gaps)
    S_obs    = _test_statistic(gaps_post)
    n_combos = comb(T_all, T_post)

    if n_combos <= max_combos:
        # Exact enumeration
        count = sum(
            1
            for chosen in combinations(range(T_all), T_post)
            if _test_statistic(all_gaps[list(chosen)]) >= S_obs
        )
        return count / n_combos

    # Monte Carlo approximation
    rng   = np.random.default_rng(seed=0)
    count = sum(
        1
        for _ in range(max_combos)
        if _test_statistic(
            all_gaps[rng.choice(T_all, size=T_post, replace=False)]
        ) >= S_obs
    )
    return count / max_combos


def _conformal_ci(
    gaps_blank: np.ndarray,
    gaps_post:  np.ndarray,
    alpha:      float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split-conformal confidence interval — equations (20)-(21) in
    Abadie & Zhao (2025).

    The (1-alpha)-quantile of |û_t| over blank periods is the half-width,
    applied symmetrically around each post-period gap estimate.

    Parameters
    ----------
    gaps_blank : û_t for blank periods  (T_B,)
    gaps_post  : û_t for post periods   (T_post,)
    alpha      : significance level, e.g. 0.05

    Returns
    -------
    ci_lower : (T_post,) lower bounds
    ci_upper : (T_post,) upper bounds
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if len(gaps_blank) == 0:
        raise ValueError("No blank periods available — cannot construct CI.")

    q = float(np.quantile(np.abs(gaps_blank), 1.0 - alpha))
    return gaps_post - q, gaps_post + q


# ═══════════════════════════════════════════════════════════════════════════════
# GDEX_infer — GREEDY DESIGN OF EXPERIMENTS WITH CONFORMAL INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def GDEX_infer(
    X_pre:        np.ndarray,
    X_post:       np.ndarray,
    alpha:        float = 0.05,
    fitting_frac: float = 0.75,
    solver:       str   = "OSQP",
    max_combos:   int   = 10_000,
) -> GDEXInferenceResult:
    """
    GDEX with full Abadie-Zhao conformal inference.

    Runs the two-stage GDEX pipeline on the fitting portion of the
    pre-period, then uses the held-out blank periods to construct a
    permutation p-value and split-conformal confidence intervals for
    each post-intervention period.

    Parameters
    ----------
    X_pre        : (T0, N)     pre-intervention matrix
    X_post       : (T_post, N) post-intervention matrix
    alpha        : significance level for CI and p-value (default 0.05)
    fitting_frac : fraction of T0 rows used to fit GDEX (default 0.75)
    solver       : CVXPY solver name for Stage 2 (default "OSQP")
    max_combos   : max exact permutations before switching to Monte Carlo

    Returns
    -------
    GDEXInferenceResult with gaps, p-value, CI bands, and period indices
    """
    if X_pre.ndim != 2 or X_post.ndim != 2:
        raise ValueError("X_pre and X_post must both be 2-D arrays.")
    if X_pre.shape[1] != X_post.shape[1]:
        raise ValueError(
            f"X_pre has {X_pre.shape[1]} units but X_post has "
            f"{X_post.shape[1]}. Column counts must match."
        )

    T0     = X_pre.shape[0]
    T_post = X_post.shape[0]
    X_full = np.vstack([X_pre, X_post])            # (T0 + T_post, N)

    # ── Period split ──────────────────────────────────────────────────────────
    fitting_idx, blank_idx = _split_periods(T0, fitting_frac)

    # ── Stages 1 + 2: GDEX on fitting periods only ────────────────────────────
    result = GDEX(X_pre[fitting_idx, :], solver=solver)

    # ── Compute gaps over blank and post-intervention periods ──────────────────
    gaps_blank = _compute_gaps(
        X_full,
        result.treated_units, result.control_units,
        result.treated_weights, result.control_weights,
        blank_idx,                                 # blank rows index into X_full directly
    )

    post_idx  = np.arange(T0, T0 + T_post)        # post rows sit after all T0 pre rows
    gaps_post = _compute_gaps(
        X_full,
        result.treated_units, result.control_units,
        result.treated_weights, result.control_weights,
        post_idx,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    p_value            = _permutation_pvalue(gaps_blank, gaps_post, max_combos)
    ci_lower, ci_upper = _conformal_ci(gaps_blank, gaps_post, alpha)

    return GDEXInferenceResult(
        gdex            = result,
        gaps_blank      = gaps_blank,
        gaps_post       = gaps_post,
        p_value         = p_value,
        ci_lower        = ci_lower,
        ci_upper        = ci_upper,
        alpha           = alpha,
        fitting_periods = fitting_idx,
        blank_periods   = blank_idx,
    )
