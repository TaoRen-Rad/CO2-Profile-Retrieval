import json
import os
import h5py
import numpy as np
from scipy.constants import g, c
import torch

ABSCO_DIR = os.environ.get("OCO2_ABSCO_DIR", "data/absco")
OCO2_CONSTANT_DIR = os.environ.get("OCO2_CONSTANT_DIR", "data/OCO2")
M_dry = 28.96e-3
M_h2o = 18.015e-3
g = g
c = c
ABSCO_PATH = {
    "CO2": f"{ABSCO_DIR}/co2_v51.hdf",
    "H2O": f"{ABSCO_DIR}/h2o_v51.hdf",
    "O2": f"{ABSCO_DIR}/o2_v51.hdf"
}
BAND_NAMES = ["o2", "weak_co2", "strong_co2"]
BASELINE_SOLAR_DISTANCE = 149597871.0e3
NAME2ID = {
    "H2O": "01",
    "CO2": "02",
    "O2": "07",
}
ID2NAME = {v: k for k, v in NAME2ID.items()}
BRDF_NU0S = np.array([0.77, 1.615, 2.06])

def wl2wn(wl):
    return 1e4 / wl

def wn2wl(wn):
    return 1e4 / wn

def byte2string(array: np.ndarray) -> str:
    return b''.join(array).decode('utf-8')


def concatenated_string(array: np.ndarray) -> np.ndarray:
    return np.array([byte2string(i) for i in array])

def vmr2sh(vmr):  # vmr = mole fraction
    w = M_h2o / M_dry
    return (w * vmr) / (1 - vmr + w * vmr)

def sh2vmr(sh):
    w = M_h2o / M_dry
    return sh / (sh + w * (1 - sh))


def load_constant(channel_index: int, sounding_index: int,
                  dtype: torch.dtype = torch.float64):
    with open(os.path.join(OCO2_CONSTANT_DIR, "index.json")) as file:
        indexs = json.load(file)

    orbit = list(indexs.keys())[0]

    year, l1b_name, l2d_name, _ = indexs[orbit]
    l1b_name = os.path.join(OCO2_CONSTANT_DIR, "L1BScND", str(year), l1b_name)
    l2d_name = os.path.join(OCO2_CONSTANT_DIR, "L2DiaND", str(year), l2d_name)

    with h5py.File(l2d_name) as l2d, h5py.File(l1b_name) as l1b:
        full_idx = l2d["RetrievalHeader/sounding_id"][:]
        rows = np.arange(len(full_idx))
        rows = rows[full_idx%10==1]
        full_idx = full_idx[full_idx%10==1]
        row_idx = 0
        row = rows[row_idx]
        
        dispersion_coef_samp = l1b["/InstrumentHeader/dispersion_coef_samp"][channel_index, sounding_index, :].flatten()
        ils_delta_lambda = l1b["InstrumentHeader/ils_delta_lambda"][channel_index, sounding_index, :, :]
        ils_relative_response = l1b["InstrumentHeader/ils_relative_response"][channel_index, sounding_index, :, :]

        brdf_factor_names = ["rahman_factor", "hotspot_parameter", "asymmetry_parameter", "anisotropy_parameter", "breon_factor"]
        brdf_factors = np.zeros([3, len(brdf_factor_names)])
        for band_idx, band in enumerate(BAND_NAMES):
            for term_idx, term in enumerate(brdf_factor_names):
                factor_name = f"brdf_{term}_{band}"
                brdf_factors[band_idx, term_idx] = l2d[f"BRDFResults/{factor_name}"][row]

    DISPERSION_COEF_SAMP = torch.tensor(dispersion_coef_samp, dtype=dtype)
    BRDF_FACTORS = torch.tensor(brdf_factors[channel_index, :], dtype=dtype)
    ILS_DELTA_LAMBDA = torch.tensor(ils_delta_lambda, dtype=dtype)
    ILS_RELATIVE_RESPONSE = torch.tensor(ils_relative_response, dtype=dtype)

    return DISPERSION_COEF_SAMP, BRDF_FACTORS, ILS_DELTA_LAMBDA, ILS_RELATIVE_RESPONSE
