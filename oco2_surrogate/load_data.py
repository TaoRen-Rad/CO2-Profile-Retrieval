"""
Data loading and preprocessing utilities for retrieval training.
"""

import numpy as np
import os
import pandas as pd
from typing import List
from . import RAND_SEED
from .absc import sh2vmr

BASE_DATA_DIR = os.environ.get("OCO2_DATA_DIR", "data/LOS_OCO2_DATA")
STATE_FILE_TEMPLATE = os.environ.get("OCO2_STATE_FILE_TEMPLATE", "df_state_{yymm}.parquet")
LOS_FILE_TEMPLATE = os.environ.get("OCO2_LOS_FILE_TEMPLATE", "df_los_{yymm}.parquet")

RADIANCE_SCALER = 1e20

def add_suffix(prefix: str, n: int) -> List[str]:
    """Generate column names with numeric suffixes."""
    return [f"{prefix}_{i}" for i in range(n)]

def state2retrieved(df_state: pd.DataFrame) -> pd.DataFrame:
    """Convert state DataFrame to retrieved format by applying scaling and transformations."""
    df = df_state.copy()

    # Drop outcome_flag if present
    # if 'outcome_flag' in df.columns:
        # df.drop(columns=['outcome_flag'], inplace=True)

    # Drop columns that include 'apriori' or 'xco2'
    drop_columns: List[str] = []
    for name in df.columns:
        lower = name.lower()
        if ('apriori' in lower) or ('xco2' in lower) or ('outcome_flag' in lower):
            drop_columns.append(name)
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True)

    return df

def state2apriori(df_state: pd.DataFrame) -> pd.DataFrame:
    """Extract apriori columns from state DataFrame."""
    df = df_state.copy()
    apriori_columns = []
    for name in df.columns:
        lower = name.lower()
        if ('apriori' in lower) or ('met' in lower):
            apriori_columns.append(name)
    
    # to_be_removed = ["temperature_offset_apriori_fph", "h2o_scale_factor_apriori"]
    # for name in to_be_removed:
    #     if name in apriori_columns:
    #         apriori_columns.remove(name)
    if apriori_columns:
        df = df[apriori_columns]
    return df

def state2wf(df_state: pd.DataFrame) -> pd.DataFrame:
    """Extract weighting function columns from state DataFrame."""
    wf_columns = add_suffix("xco2_pressure_weighting_function", 20)
    return df_state[wf_columns].copy()

def between(x, x_min, x_max):
    return (x >= x_min) & (x <= x_max)

def screen_asia(df_screen: pd.DataFrame) -> pd.DataFrame:
    latitude = df_screen["latitude"]
    longitude = df_screen["longitude"]
    idx = between(latitude, 20, 45) & between(longitude, 110, 145)
    return idx

def load_retrieved_data(yymms, fraction=0.1, load_modeled=True, 
                        filter_outcome_flag=False, filter_asia=False,
                        load_screen = False):
    """
    Load and preprocess data for retrieval training.
    
    Args:
        yymms: List of year-month strings (e.g., ["1701"])
        fraction: Fraction of data to load (for sampling)
        load_modeled: Whether to load modeled or measured radiance data
        
    Returns:
        tuple: (df_geometry, df_retrieved, df_apriori, df_wf, df_radiance)
    """
    df_screen = pd.read_parquet([f"{BASE_DATA_DIR}/df_screen_{yymm}.parquet" for yymm in yymms]).sample(frac=fraction, random_state=RAND_SEED)
    df_geometry = pd.read_parquet([f"{BASE_DATA_DIR}/df_geometry_{yymm}.parquet" for yymm in yymms]).sample(frac=fraction, random_state=RAND_SEED)
    df_state = pd.read_parquet([f"{BASE_DATA_DIR}/{STATE_FILE_TEMPLATE.format(yymm=yymm)}" for yymm in yymms]).sample(frac=fraction, random_state=RAND_SEED)
    df_los = pd.read_parquet([f"{BASE_DATA_DIR}/{LOS_FILE_TEMPLATE.format(yymm=yymm)}" for yymm in yymms]).sample(frac=fraction, random_state=RAND_SEED)
    
    if load_modeled:
        df_radiance = pd.read_parquet([f"{BASE_DATA_DIR}/df_modeled_{yymm}.parquet" for yymm in yymms]).sample(frac=fraction, random_state=RAND_SEED)
    #     df_radiance = df_radiance.iloc[:, 3:]  # Skip first 3 columns
    else:
        df_radiance = pd.read_parquet([f"{BASE_DATA_DIR}/df_measured_{yymm}.parquet" for yymm in yymms]).sample(frac=fraction, random_state=RAND_SEED)
    
    # create a full index as initial index
    idx = pd.Series(True, index=df_state.index)
    if filter_outcome_flag:
        idx = idx & (df_state["outcome_flag"] == 1)
    if filter_asia:
        idx = idx & screen_asia(df_screen)

    df_screen = df_screen.loc[idx]
    df_geometry = df_geometry.loc[idx]
    df_state = df_state.loc[idx]
    df_los = df_los.loc[idx]
    
    df_radiance = df_radiance.loc[idx]

    df_retrieved = state2retrieved(df_state)
    df_apriori = state2apriori(df_state)
    df_wf = state2wf(df_state)
    if not load_screen:
        return df_geometry, df_retrieved, df_apriori, df_wf, df_los, df_radiance / RADIANCE_SCALER
    else:
        return df_screen, df_geometry, df_retrieved, df_apriori, df_wf, df_los, df_radiance / RADIANCE_SCALER

def get_wf_input_indices(df_retrieved):
    """Get indices for weighting function input variables in retrieved DataFrame."""
    wf_input_idx = []
    for i, name in enumerate(df_retrieved.columns):
        start_candidate = ["temperature_profile", "specific_humidity_profile", "surface_pressure"]
        if any(name.startswith(candidate) for candidate in start_candidate):
            wf_input_idx.append(i)
    
    return wf_input_idx

def get_co2_profile_indices(df_retrieved):
    """Get indices for CO2 profile variables in retrieved DataFrame."""
    co2_profile_idx = []
    for i, name in enumerate(df_retrieved.columns):
        if name.startswith("co2_profile"):
            co2_profile_idx.append(i)
    
    return co2_profile_idx

def get_state_indices(retrieved_columns, band):
    """Get indices for band-specific variables in retrieved DataFrame."""
    band_indices = []
    
    band_filters = {
        "o2": lambda x: 'co2' not in x.lower(),
        "weak_co2": lambda x: 'strong_co2' not in x.lower() and 'fluorescence' not in x.lower(),
        "strong_co2": lambda x: 'weak_co2' not in x.lower() and 'fluorescence' not in x.lower(),
    }
    
    for i, col in enumerate(retrieved_columns):
        if band_filters[band](col):
            band_indices.append(i)
    
    return band_indices
