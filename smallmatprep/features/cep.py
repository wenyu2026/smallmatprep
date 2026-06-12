"""CEP-style physics-prior feature helpers."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def add_cep_feature(
    df: pd.DataFrame,
    formula: Callable[[pd.DataFrame], object],
    col_name: str = "cep_estimate",
) -> pd.DataFrame:
    """Add a CEP/theory estimate column computed from a user formula.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    formula : callable
        Function that receives *df* and returns one numeric value per row.
    col_name : str
        Name of the new feature column.
    """
    out = df.copy()
    values = formula(out)
    values = np.asarray(values, dtype=float)

    if values.ndim != 1:
        raise ValueError("CEP formula must return a 1D array-like value")
    if len(values) != len(out):
        raise ValueError(
            f"CEP formula returned {len(values)} values for {len(out)} rows"
        )
    if np.isnan(values).any() or np.isinf(values).any():
        raise ValueError("CEP formula returned NaN or Inf values")

    out[col_name] = values
    return out


def simple_electrolyte_cep(
    df: pd.DataFrame,
    temperature_col: str = "temperature_K",
    concentration_col: str = "concentration_mol_L",
    lambda_0: float = 12.0,
    debye_coeff: float = 1.0,
    activation_energy_j_mol: float = 9000.0,
    reference_temperature_K: float = 298.15,
) -> np.ndarray:
    """Return a simplified electrolyte conductivity prior.

    The prior combines a concentration-dependent molar conductivity term with
    Arrhenius-like temperature scaling. It is intended as a compact physics
    trend feature, not a full electrochemical simulator.
    """
    missing_cols = [
        col for col in [temperature_col, concentration_col]
        if col not in df.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing columns for CEP formula: {missing_cols}")

    temperature = df[temperature_col].astype(float).to_numpy()
    concentration = df[concentration_col].astype(float).to_numpy()

    if np.isnan(temperature).any() or np.isnan(concentration).any():
        raise ValueError("CEP input columns contain NaN values")
    if (temperature <= 0).any():
        raise ValueError("Temperature values must be > 0 K")
    if (concentration < 0).any():
        raise ValueError("Concentration values must be >= 0")

    gas_constant = 8.314462618
    c_sqrt = np.sqrt(np.maximum(concentration, 0.0))
    molar_conductivity = np.maximum(lambda_0 - debye_coeff * c_sqrt, 0.0)
    thermal_scale = np.exp(
        (-activation_energy_j_mol / gas_constant)
        * ((1.0 / temperature) - (1.0 / reference_temperature_K))
    )
    return concentration * molar_conductivity * thermal_scale
