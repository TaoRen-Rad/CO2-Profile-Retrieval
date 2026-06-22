import pickle
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from los_physics.los import LOS
from train_forward import ForwardModel, load_scalers


BAND_NAME_TO_INDEX = {"o2": 0, "weak_co2": 1, "strong_co2": 2}


def load_forward_model(
    band: str,
    base_name: str,
    data_dir: str,
    device: str = "cuda",
    checkpoint: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    base_path = f"status/{base_name}_{band}"
    data_path = data_dir

    scalers_path = f"{data_path}/scalers.pth"
    scalers = load_scalers(scalers_path)
    geometry_scaler = scalers["geometry"]
    retrieved_scaler = scalers["retrieved"]
    radiance_scaler = scalers["radiance"]

    indices_path = f"{data_path}/indices.pth"
    indices = torch.load(indices_path, map_location="cpu")
    band_indices = indices["all_state_indices"][band]
    nnan_indices = indices["all_nnan_indices"][band]

    bandidx = BAND_NAME_TO_INDEX[band]
    rad_indices = [bandidx * 1016 + i for i in range(1016)]
    radiance_scaler_sliced_mean = radiance_scaler.mean_[rad_indices]
    radiance_scaler_sliced_scale = radiance_scaler.scale_[rad_indices]

    with open(f"{base_path}/model_config.pkl", "rb") as f:
        model_config = pickle.load(f)

    model = ForwardModel(**model_config)
    model.geometry_scaler.register_buffer("mean_", geometry_scaler.mean_)
    model.geometry_scaler.register_buffer("scale_", geometry_scaler.scale_)
    model.retrieved_scaler.register_buffer("mean_", retrieved_scaler.mean_)
    model.retrieved_scaler.register_buffer("scale_", retrieved_scaler.scale_)
    model.radiance_scaler.register_buffer("mean_", radiance_scaler_sliced_mean)
    model.radiance_scaler.register_buffer("scale_", radiance_scaler_sliced_scale)

    if checkpoint is not None:
        model_path = f"{base_path}/mlp_model_{checkpoint}.pth"
    else:
        model_path = f"{base_path}/mlp_model_best.pth"

    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model = model.to(device, dtype=torch.float32)
    model.eval()

    return {
        "model": model,
        "geometry_scaler": geometry_scaler,
        "retrieved_scaler": retrieved_scaler,
        "radiance_scaler": radiance_scaler,
        "band": band,
        "band_indices": band_indices,
        "nnan_indices": nnan_indices,
    }


class OCO2SingleBandForwardModel(nn.Module):
    def __init__(
        self,
        band: str,
        base_name: str,
        data_dir: str,
        checkpoints: List[int],
        geometry_column_names: List[str],
        retrieved_column_names: List[str],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.band = band
        self.dtype = dtype
        self.channel_index = BAND_NAME_TO_INDEX[band]
        self.rad_indices = [self.channel_index * 1016 + i for i in range(1016)]
        self.los_model = LOS(self.channel_index, 0, dtype=dtype)
        self.model_infos = [
            load_forward_model(band, base_name, data_dir, checkpoint=checkpoint)
            for checkpoint in checkpoints
        ]
        self.forward_models = nn.ModuleList(
            [model_info["model"] for model_info in self.model_infos]
        )
        self._create_los_indices(geometry_column_names, retrieved_column_names)
        self.geometry_scalers = self.model_infos[0]["geometry_scaler"]
        self.retrieved_scalers = self.model_infos[0]["retrieved_scaler"]
        self.radiance_scalers = self.model_infos[0]["radiance_scaler"]
        rad_mean = self.radiance_scalers.mean_[self.rad_indices]
        rad_std = self.radiance_scalers.scale_[self.rad_indices]
        self.register_buffer("rad_mean", rad_mean)
        self.register_buffer("rad_std", rad_std)

    def _create_los_indices(
        self, geometry_column_names: List[str], retrieved_column_names: List[str]
    ) -> None:
        self.co2_profile_indices = [
            retrieved_column_names.index(f"co2_profile_{i}") for i in range(20)
        ]
        self.vector_pressure_levels_indices = [
            retrieved_column_names.index(f"vector_pressure_levels_{i}")
            for i in range(20)
        ]
        self.temperature_profile_indices = [
            retrieved_column_names.index(f"temperature_profile_{i}")
            for i in range(20)
        ]
        self.h2o_profile_indices = [
            retrieved_column_names.index(f"h2o_profile_{i}") for i in range(20)
        ]
        disp_offset_space_names = [f"offset_{self.band}", f"spacing_{self.band}"]
        self.disp_offset_space_indices = [
            retrieved_column_names.index(column_name)
            for column_name in disp_offset_space_names
        ]
        brdf_weight_names = [
            f"brdf_weight_{self.band}",
            f"brdf_weight_slope_{self.band}",
            f"brdf_weight_quadratic_{self.band}",
        ]
        self.brdf_weights_indices = [
            retrieved_column_names.index(column_name)
            for column_name in brdf_weight_names
        ]
        geo_angle_names = [
            "cos_solar_zenith",
            "cos_zenith",
            "sin_solar_zenith",
            "sin_zenith",
            "cos_relative_azimuth",
        ]
        self.geo_angles_indices = [
            geometry_column_names.index(column_name) for column_name in geo_angle_names
        ]
        solar_info_names = ["solar_distance", "solar_relative_velocity"]
        self.solar_info_indices = [
            geometry_column_names.index(column_name) for column_name in solar_info_names
        ]

    def forward_los(self, geometry: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        co2_profile = retrieved[:, self.co2_profile_indices]
        vector_pressure_levels = retrieved[:, self.vector_pressure_levels_indices]
        temperature_profile = retrieved[:, self.temperature_profile_indices]
        h2o_profile = retrieved[:, self.h2o_profile_indices]
        o2_profile = (
            torch.ones_like(vector_pressure_levels) * 0.20935
        ).to(geometry.device, dtype=geometry.dtype)
        if self.band == "o2":
            specie_vmrs = o2_profile
        else:
            specie_vmrs = co2_profile

        disp_offset_space = retrieved[:, self.disp_offset_space_indices]
        brdf_weights = retrieved[:, self.brdf_weights_indices]
        geo_angles = geometry[:, self.geo_angles_indices]
        solar_info = geometry[:, self.solar_info_indices]
        _, los = self.los_model(
            pressures=vector_pressure_levels,
            temperatures=temperature_profile,
            broadener_vmrs=h2o_profile,
            specie_vmrs=specie_vmrs,
            geo_angles=geo_angles,
            brdf_weights=brdf_weights,
            disp_offset_space=disp_offset_space,
            solar_info=solar_info,
        )

        return los

    def forward(self, geometry: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        los = self.forward_los(geometry, retrieved)
        rad = torch.zeros_like(los)

        for forward_model in self.forward_models:
            rad += forward_model.predict(geometry, retrieved, los)
        return rad / len(self.forward_models)


class OCO2ForwardModel(nn.Module):
    def __init__(
        self,
        base_name: str,
        data_dir: str,
        checkpoints: List[int],
        geometry_column_names: List[str],
        retrieved_column_names: List[str],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.forward_models = nn.ModuleList(
            [
                OCO2SingleBandForwardModel(
                    band,
                    base_name,
                    data_dir,
                    checkpoints,
                    geometry_column_names,
                    retrieved_column_names,
                    dtype=dtype,
                )
                for band in ["o2", "weak_co2", "strong_co2"]
            ]
        )

    def forward(self, geometry: torch.Tensor, retrieved: torch.Tensor) -> torch.Tensor:
        rad = torch.zeros((geometry.shape[0], 1016 * 3)).to(
            geometry.device, dtype=geometry.dtype
        )
        for forward_model in self.forward_models:
            rad[:, forward_model.rad_indices] = forward_model(geometry, retrieved)
        return rad
