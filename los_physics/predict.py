import torch
import numpy as np
import pandas as pd
import tqdm
from .constant import sh2vmr
from .los import LOS


def predict_los(df_geometry, df_retrieved, model_los, channel_index, device='cuda', batch_size=128):
    """Predict line-of-sight radiance using LOS model
    
    Args:
        df_geometry: Geometry data (DataFrame)
        df_retrieved: Retrieved state data (DataFrame)
        model_los: LOS model instance
        channel_index: Band index (0=O2, 1=weak_CO2, 2=strong_CO2)
        device: Device to run inference on
        batch_size: Batch size for inference
    
    Returns:
        los_values: Predicted LOS radiance (numpy array of shape [N, 1016])
    """
    band_names = ["o2", "weak_co2", "strong_co2"]
    band = band_names[channel_index]
    dtype = torch.float64
    
    # Extract atmospheric profiles
    N = len(df_geometry)
    co2_profile = torch.tensor(
        df_retrieved[[f"co2_profile_{i}" for i in range(20)]].values, 
        dtype=dtype
    )
    vector_pressure_levels = torch.tensor(
        df_retrieved[[f"vector_pressure_levels_{i}" for i in range(20)]].values,
        dtype=dtype
    )
    temperature_profile = torch.tensor(
        df_retrieved[[f"temperature_profile_{i}" for i in range(20)]].values,
        dtype=dtype
    )
    h2o_profile = torch.tensor(
        df_retrieved[[f"h2o_profile_{i}" for i in range(20)]].values,
        dtype=dtype
    )
    # h2o_profile = torch.tensor(
    #     sh2vmr(df_retrieved[[f"specific_humidity_profile_{i}" for i in range(20)]].values),
    #     dtype=dtype
    # )
    o2_profile = torch.ones_like(vector_pressure_levels) * 0.20935
    
    # Extract dispersion and BRDF parameters
    dispersion_offset = torch.tensor(df_retrieved[f"offset_{band}"].values, dtype=dtype)
    dispersion_spacing = torch.tensor(df_retrieved[f"spacing_{band}"].values, dtype=dtype)
    brdf_weight = torch.tensor(df_retrieved[f"brdf_weight_{band}"].values, dtype=dtype)
    brdf_weight_slope = torch.tensor(df_retrieved[f"brdf_weight_slope_{band}"].values, dtype=dtype)
    brdf_weight_quadratic = torch.tensor(df_retrieved[f"brdf_weight_quadratic_{band}"].values, dtype=dtype)
    
    # Extract geometry parameters
    cos_phi = torch.tensor(df_geometry["cos_relative_azimuth"].values, dtype=dtype)
    sin_phi = torch.tensor(df_geometry["sin_relative_azimuth"].values, dtype=dtype)
    cos_theta_i = torch.tensor(df_geometry["cos_solar_zenith"].values, dtype=dtype)
    sin_theta_i = torch.tensor(df_geometry["sin_solar_zenith"].values, dtype=dtype)
    cos_theta_r = torch.tensor(df_geometry["cos_zenith"].values, dtype=dtype)
    sin_theta_r = torch.tensor(df_geometry["sin_zenith"].values, dtype=dtype)
    solar_info = torch.tensor(
        df_geometry[["solar_distance", "solar_relative_velocity"]].values,
        dtype=dtype
    )
    
    # Stack inputs
    geo_angles = torch.stack([cos_theta_i, cos_theta_r, sin_theta_i, sin_theta_r, cos_phi], dim=1)
    brdf_weights = torch.stack([brdf_weight, brdf_weight_slope, brdf_weight_quadratic], dim=1)
    disp_offset_space = torch.stack([dispersion_offset, dispersion_spacing], dim=1)
    
    # Select appropriate VMR based on channel
    if channel_index == 0:
        specie_vmrs = o2_profile
    else:
        specie_vmrs = co2_profile
    
    # Batch inference
    n_wn = 1016
    wls_all = torch.zeros((N, n_wn), dtype=dtype)
    I_all = torch.zeros((N, n_wn), dtype=dtype)
    
    model_los = model_los.to(device, dtype=dtype)
    model_los.eval()
    
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, N, batch_size), desc="Predicting LOS"):
            i_end = min(i + batch_size, N)
            
            # Move batch to device
            p_batch = vector_pressure_levels[i:i_end].to(device=device, dtype=dtype)
            T_batch = temperature_profile[i:i_end].to(device=device, dtype=dtype)
            broad_batch = h2o_profile[i:i_end].to(device=device, dtype=dtype)
            specie_batch = specie_vmrs[i:i_end].to(device=device, dtype=dtype)
            geo_batch = geo_angles[i:i_end].to(device=device, dtype=dtype)
            brdf_batch = brdf_weights[i:i_end].to(device=device, dtype=dtype)
            disp_batch = disp_offset_space[i:i_end].to(device=device, dtype=dtype)
            solar_info_batch = solar_info[i:i_end].to(device=device, dtype=dtype)
            
            # Forward pass
            wls_batch, I_batch = model_los(
                pressures=p_batch,
                temperatures=T_batch,
                broadener_vmrs=broad_batch,
                specie_vmrs=specie_batch,
                geo_angles=geo_batch,
                brdf_weights=brdf_batch,
                disp_offset_space=disp_batch,
                solar_info=solar_info_batch,
            )
            
            # Save results
            wls_all[i:i_end] = wls_batch.detach().cpu()
            I_all[i:i_end] = I_batch.detach().cpu()
    
    return I_all.numpy()

def predict(df_geometry, df_retrieved, device='cuda', batch_size=64):
    ans = np.zeros((len(df_geometry), 3048), dtype=np.float64)
    for channel_index in range(3):
        print("Creating LOS model for channel", channel_index)
        los_model = LOS(channel_index, 0).to(device, dtype=torch.float64)
        I_all = predict_los(df_geometry, df_retrieved, los_model, channel_index, device=device, batch_size=batch_size)
        slice_idx = slice(channel_index * 1016, (channel_index + 1) * 1016)
        ans[:, slice_idx] = I_all
    df_los = pd.DataFrame(ans, index=df_geometry.index, columns=range(3048))
    return df_los