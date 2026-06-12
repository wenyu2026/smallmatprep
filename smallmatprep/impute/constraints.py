"""Material-domain constraints for physically plausible imputation.

All constraint functions are designed as **post-processing** steps that can be
applied after **any** imputation algorithm (median, KNN, ExtraTrees, etc.)
without modifying the original method's internals.

Four constraint families
-----------------------
1. **Composition constraint** — Normalize alloy/electrolyte composition columns
   so they sum to a target (e.g., ``A_frac + B_frac + C_frac = 1``).
2. **Range constraint** — Clip imputed values to physically plausible intervals
   (e.g., density ≥ 0, band gap ≥ 0, melting point within bounds).
3. **Similarity constraint** — Soft-pull imputed values toward the mean of
   their nearest neighbors in feature space (KNN smoothing).
4. **Uncertainty output** — Run multiple imputations (bootstrap / different
   random seeds) and return mean ± std for each missing entry.
"""
from __future__ import annotations

import warnings
from copy import deepcopy
from functools import partial

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
# 1.  Composition constraint
# ──────────────────────────────────────────────────────────────────────


def apply_composition_constraint(
    df: pd.DataFrame,
    group_columns: list[str],
    sum_target: float = 1.0,
    rtol: float = 0.02,
    missing_mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Normalise composition columns so they sum to *sum_target*.

    For rows where **all** group columns are observed before imputation,
    no adjustment is applied.  For rows where **at least one** column was
    imputed, the entire group is scaled to ``sum_target``.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed data (may already contain values).
    group_columns : list[str]
        Columns that form a composition set (e.g. ``["A_frac", "B_frac"]``).
    sum_target : float, default=1.0
        The value the group should sum to (e.g. ``1.0`` for fractions,
        ``100.0`` for weight percentages).
    rtol : float, default=0.02
        Relative tolerance.  If the observed sum deviates by less than
        ``rtol``, it is left untouched.
    missing_mask : np.ndarray, optional
        Boolean mask of shape ``(n_rows, n_cols_in_group)`` indicating which
        entries were originally missing.  If ``None``, all rows are adjusted.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with composition columns normalised.
    """
    out = df.copy()
    present = [c for c in group_columns if c in out.columns]
    if len(present) < 2:
        return out

    cols = out[present].to_numpy(dtype=float, copy=True)

    if missing_mask is not None:
        mask = missing_mask.any(axis=1)
    else:
        row_sums = cols.sum(axis=1)
        mask = ~np.isclose(row_sums, sum_target, rtol=rtol)

    if not mask.any():
        return out

    current = cols[mask].sum(axis=1, keepdims=True)
    # avoid division by zero
    current = np.where(np.abs(current) < 1e-15, sum_target, current)
    cols[mask] = cols[mask] * (sum_target / current)

    out[present] = cols
    return out


# ──────────────────────────────────────────────────────────────────────
# 2.  Physical range constraint
# ──────────────────────────────────────────────────────────────────────


def apply_range_constraint(
    df: pd.DataFrame,
    constraints: dict[str, tuple[float | None, float | None]],
    missing_mask: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Clip imputed values to physically plausible ranges.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed data.
    constraints : dict of {str: (min, max)}
        Mapping from column name to ``(lower_bound, upper_bound)``.
        Use ``None`` for a bound that should not be enforced.
        Example::

            {
                "density_g_ml": (0.5, 25.0),
                "band_gap_eV": (0.0, None),   # non-negative only
            }

    missing_mask : dict of {str: np.ndarray}, optional
        Mapping from column name to boolean array indicating originally
        missing entries.  If provided, only originally-missing values are
        clipped; observed values are left untouched.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with values clipped to bounds.
    """
    out = df.copy()

    for col, (lo, hi) in constraints.items():
        if col not in out.columns:
            warnings.warn(f"Constraint column '{col}' not found, skipping.")
            continue

        values = out[col].to_numpy(dtype=float, copy=True)

        if missing_mask is not None and col in missing_mask:
            idx = missing_mask[col]
        else:
            idx = slice(None)  # all rows

        if lo is not None:
            values[idx] = np.maximum(values[idx], lo)
        if hi is not None:
            values[idx] = np.minimum(values[idx], hi)

        out[col] = values

    return out


# ──────────────────────────────────────────────────────────────────────
# 3.  Material similarity constraint (KNN soft-pull)
# ──────────────────────────────────────────────────────────────────────


def apply_similarity_constraint(
    df: pd.DataFrame,
    feature_cols: list[str],
    missing_mask: np.ndarray,
    n_neighbors: int = 3,
    alpha: float = 0.3,
    **kwargs,
) -> pd.DataFrame:
    """Pull imputed values toward the mean of similar complete materials.

    For each row that had missing values, find its *n_neighbors* nearest
    neighbours (using **only complete** rows as references) and blend the
    imputed value with the neighbour mean:

    .. math::

        x_{\\text{final}} = (1 - \\alpha) \\cdot x_{\\text{imputed}}
                          + \\alpha \\cdot x_{\\text{neighbour\\_mean}}

    Parameters
    ----------
    df : pd.DataFrame
        Imputed data with all *feature_cols* complete (no NaNs).
    feature_cols : list[str]
        Feature columns used for neighbour distance computation.
    missing_mask : np.ndarray
        Boolean mask of shape ``(n_rows, n_features)`` indicating originally
        missing entries.
    n_neighbors : int, default=3
        Number of neighbours.
    alpha : float, default=0.3
        Blending weight in ``[0, 1]``.  ``0`` = no pull, ``1`` = full
        replacement with neighbour mean.
    **kwargs
        Passed to scikit-learn's ``NearestNeighbors`` (e.g. ``metric``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with similarity-constrained values.
    """
    from sklearn.neighbors import NearestNeighbors

    if alpha <= 0.0 or n_neighbors < 1:
        return df.copy()

    out = df.copy()
    present_cols = [c for c in feature_cols if c in out.columns]
    if len(present_cols) < 2:
        return out

    values = out[present_cols].to_numpy(dtype=float, copy=True)
    complete_rows = ~missing_mask.any(axis=1)
    incomplete_rows = missing_mask.any(axis=1)

    n_complete = complete_rows.sum()
    if n_complete < 1 or incomplete_rows.sum() < 1:
        return out

    effective_k = min(n_neighbors, n_complete)

    nn = NearestNeighbors(n_neighbors=effective_k, **kwargs)
    nn.fit(values[complete_rows])

    incomplete_idx = np.where(incomplete_rows)[0]
    dists, neigh_idx = nn.kneighbors(values[incomplete_rows])

    for local_i, row_idx in enumerate(incomplete_idx):
        neighbours = values[complete_rows][neigh_idx[local_i]]
        neighbour_mean = neighbours.mean(axis=0)

        row_missing = missing_mask[row_idx]
        imputed_vals = values[row_idx, row_missing]
        neighbour_vals = neighbour_mean[row_missing]

        corrected = (1.0 - alpha) * imputed_vals + alpha * neighbour_vals
        values[row_idx, row_missing] = corrected

    out[present_cols] = values
    return out


# ──────────────────────────────────────────────────────────────────────
# 4.  Uncertainty output via multiple imputation
# ──────────────────────────────────────────────────────────────────────


def impute_with_uncertainty(
    df: pd.DataFrame,
    feature_cols: list[str],
    imputer_fn,
    n_runs: int = 30,
    random_state: int = 42,
    constraints: list | None = None,
    missing_mask: np.ndarray | None = None,
    return_all: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Run multiple imputations and return mean ± std uncertainty.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with missing values.
    feature_cols : list[str]
        Columns to impute.
    imputer_fn : callable
        A function with signature ``imputer_fn(df, feature_cols) -> pd.DataFrame``
        that returns an imputed DataFrame.  The function is called
        ``n_runs`` times.
    n_runs : int, default=30
        Number of imputation runs.
    random_state : int, default=42
        Base random state.  Each run uses a different seed:
        ``random_state + run_idx * 7``.
    constraints : list of callables, optional
        List of constraint functions to apply after each run.  Each callable
        should have signature ``fn(df, feature_cols, missing_mask, **_) ->
        pd.DataFrame``.
    missing_mask : np.ndarray, optional
        Pre-computed missing mask.  If ``None``, it is computed from *df*.
    return_all : bool, default=False
        If ``True``, also return the full array of all imputed versions
        and the missing mask.

    Returns
    -------
    mean_df : pd.DataFrame
        Mean values across imputation runs.
    std_df : pd.DataFrame
        Standard deviation across imputation runs (uncertainty).
    (optional) all_imputations : np.ndarray, shape ``(n_runs, n_rows, n_features)``
        Only returned if *return_all* is ``True``.
    (optional) missing_mask : np.ndarray
        Only returned if *return_all* is ``True``.
    """
    present_cols = [c for c in feature_cols if c in df.columns]

    if missing_mask is None:
        mask = np.isnan(df[present_cols].to_numpy(dtype=float))
    else:
        mask = missing_mask

    if not mask.any():
        return df.copy(), df.copy()[present_cols].mul(0.0)

    # Storage: (n_runs, n_rows, n_features)
    all_results = np.zeros((n_runs, len(df), len(present_cols)))

    for run in range(n_runs):
        seed = random_state + run * 7
        # Set a deterministic seed in the imputer if possible
        result = imputer_fn(df, present_cols)
        result_cols = result[present_cols].to_numpy(dtype=float)

        if constraints:
            for constraint_fn in constraints:
                result = constraint_fn(
                    result,
                    present_cols,
                    mask,
                    random_state=seed,
                )
                result_cols = result[present_cols].to_numpy(dtype=float)

        all_results[run] = result_cols

    mean_vals = all_results.mean(axis=0)
    std_vals = all_results.std(axis=0, ddof=1)

    mean_df = df.copy()
    mean_df[present_cols] = mean_vals

    std_df = pd.DataFrame(
        std_vals,
        columns=present_cols,
        index=df.index,
    )

    if not return_all:
        return mean_df, std_df

    return mean_df, std_df, all_results, mask


# ──────────────────────────────────────────────────────────────────────
# 5.  Convenience: apply all constraints at once
# ──────────────────────────────────────────────────────────────────────


def apply_all_constraints(
    df: pd.DataFrame,
    feature_cols: list[str],
    missing_mask: np.ndarray,
    composition_groups: list[list[str]] | None = None,
    range_constraints: dict[str, tuple[float | None, float | None]] | None = None,
    similarity_kw: dict | None = None,
) -> pd.DataFrame:
    """Chain multiple constraint types in a single call.

    Parameters
    ----------
    df : pd.DataFrame
        Imputed data.
    feature_cols : list[str]
        All feature columns.
    missing_mask : np.ndarray
        Boolean mask indicating originally missing entries.
    composition_groups : list of lists, optional
        Each inner list is a group of columns that should sum to 1.
        Example: ``[["A_frac", "B_frac", "C_frac"]]``.
    range_constraints : dict, optional
        Passed to :func:`apply_range_constraint`.
    similarity_kw : dict, optional
        Keyword arguments for :func:`apply_similarity_constraint`.
        Must include at least ``n_neighbors`` and ``alpha`` if used.

    Returns
    -------
    pd.DataFrame
        Constrained data.
    """
    out = df.copy()

    if composition_groups:
        for group in composition_groups:
            present = [c for c in group if c in feature_cols]
            if len(present) >= 2:
                col_idx = [feature_cols.index(c) for c in present]
                group_mask = missing_mask[:, col_idx] if missing_mask is not None else None
                out = apply_composition_constraint(
                    out, present, missing_mask=group_mask,
                )

    if range_constraints:
        col_masks = {
            col: missing_mask[:, feature_cols.index(col)]
            for col in range_constraints
            if col in feature_cols
        }
        out = apply_range_constraint(out, range_constraints, missing_mask=col_masks)

    if similarity_kw:
        out = apply_similarity_constraint(
            out, feature_cols, missing_mask, **similarity_kw,
        )

    return out


# ──────────────────────────────────────────────────────────────────────
# 6.  Wrapper: impute + constraints in one step
# ──────────────────────────────────────────────────────────────────────


def constrained_impute(
    df: pd.DataFrame,
    feature_cols: list[str],
    imputer_fn,
    composition_groups: list[list[str]] | None = None,
    range_constraints: dict[str, tuple[float | None, float | None]] | None = None,
    similarity_kw: dict | None = None,
    uncertainty_kw: dict | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Run imputation, apply material constraints, optionally estimate uncertainty.

    This is the **top-level entry point** for constrained imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Input data with missing values.
    feature_cols : list[str]
        Columns to impute.
    imputer_fn : callable
        Imputation function with signature
        ``imputer_fn(df, feature_cols) -> pd.DataFrame``.
    composition_groups, range_constraints, similarity_kw :
        Passed to :func:`apply_all_constraints`.
    uncertainty_kw : dict, optional
        If provided, run uncertainty estimation.
        Keys: ``n_runs``, ``random_state``.

    Returns
    -------
    pd.DataFrame or (pd.DataFrame, pd.DataFrame)
        If *uncertainty_kw* is ``None``, returns a single constrained
        DataFrame.  Otherwise returns ``(mean_df, std_df)``.
    """
    present_cols = [c for c in feature_cols if c in df.columns]
    missing_mask = np.isnan(df[present_cols].to_numpy(dtype=float))

    if uncertainty_kw:
        # Build constraint callables for the uncertainty loop
        constraint_fns = []
        if composition_groups or range_constraints or similarity_kw:
            constraint_fns.append(
                partial(
                    apply_constraints_in_loop,
                    composition_groups=composition_groups,
                    range_constraints=range_constraints,
                    similarity_kw=similarity_kw,
                    feature_cols=present_cols,
                )
            )
        return impute_with_uncertainty(
            df,
            present_cols,
            imputer_fn,
            missing_mask=missing_mask,
            constraints=constraint_fns or None,
            **uncertainty_kw,
        )

    # Single-run constrained imputation
    out = imputer_fn(df, present_cols)
    out = apply_all_constraints(
        out,
        present_cols,
        missing_mask,
        composition_groups=composition_groups,
        range_constraints=range_constraints,
        similarity_kw=similarity_kw,
    )
    return out


def apply_constraints_in_loop(
    df: pd.DataFrame,
    feature_cols: list[str],
    missing_mask: np.ndarray,
    composition_groups=None,
    range_constraints=None,
    similarity_kw=None,
    **_,  # absorb extra kwargs like random_state
) -> pd.DataFrame:
    """Helper for applying constraints inside the uncertainty loop."""
    return apply_all_constraints(
        df,
        feature_cols,
        missing_mask,
        composition_groups=composition_groups,
        range_constraints=range_constraints,
        similarity_kw=similarity_kw,
    )


# ──────────────────────────────────────────────────────────────────────
# 7.  Utility: build a standard constraints config for common materials
# ──────────────────────────────────────────────────────────────────────


def electrolyte_default_constraints() -> dict:
    """Return default constraint config for electrolyte datasets.

    Includes physical ranges for common electrolyte properties.

    Returns
    -------
    dict
        Keys: ``range_constraints``, ``similarity_kw``.
    """
    return {
        "range_constraints": {
            "temperature_K": (200.0, 500.0),
            "concentration_mol_L": (0.0, 10.0),
            "solvent_A": (0.0, 100.0),
            "solvent_B": (0.0, 100.0),
            "salt_S": (0.0, 5.0),
            "additive_C": (0.0, 1.0),
            "density_g_ml": (0.5, 3.0),
            "viscosity_index": (0.0, 20.0),
        },
        "similarity_kw": {
            "n_neighbors": 3,
            "alpha": 0.25,
        },
    }


def alloy_default_constraints() -> dict:
    """Return default constraint config for alloy composition datasets.

    Returns
    -------
    dict
        Keys: ``composition_groups``, ``range_constraints``.
    """
    return {
        "composition_groups": [
            # Example: if columns exist, they will be normalised
        ],
        "range_constraints": {
            "density_g_ml": (0.5, 25.0),
            "melting_point_K": (300.0, 4000.0),
            "elastic_modulus_GPa": (0.0, 1000.0),
            "band_gap_eV": (0.0, 15.0),
            "conductivity_S_cm": (0.0, 1e8),
        },
    }
