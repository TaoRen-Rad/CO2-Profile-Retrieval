import torch
from torch import nn
from .constant import BAND_NAMES, load_constant, M_dry, g, c, wl2wn, wn2wl, vmr2sh
from .absc import AbsorptionCoefficientModule# , sh2vmr
from .solar import SolarModule
from .brdf import BRDFModeule

class LOS(nn.Module):
    def __init__(self, channel_index: int, sounding_index: int, 
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self._channel_index = channel_index
        self._sounding_index = sounding_index
        self._dtype = dtype
        (
            dispersion_coef_samp,
            brdf_factors,
            ils_delta_lambda,
            ils_relative_response
        ) = load_constant(channel_index, sounding_index, dtype)
        self.register_buffer("_dispersion_coef_samp", dispersion_coef_samp)
        self.register_buffer("_ils_delta_lambda", ils_delta_lambda)
        self.register_buffer("_ils_relative_response", ils_relative_response)
        self._band_name = BAND_NAMES[channel_index]
        if channel_index == 0:
            self._specie_name = "O2"
        else:
            self._specie_name = "CO2"
        self._broadener_name = "H2O"
        
        wn_range = self._get_wn_range()
        self._specie_absc_mod = AbsorptionCoefficientModule(
            gas_name=self._specie_name,
            wavenumber_range=wn_range,
            dtype=self._dtype,
        )
        self._broadener_absc_mod = AbsorptionCoefficientModule(
            gas_name = self._broadener_name,
            wavenumber_range=wn_range,
            dtype=self._dtype,
        )
        lbl_wns = self._specie_absc_mod.wavenumbers       # (W,)
        lbl_wls = wn2wl(lbl_wns)                    # (W,)
        self.register_buffer("_lbl_wns", lbl_wns)
        self.register_buffer("_lbl_wls", lbl_wls)
        self._solar_mod = SolarModule(
            lbl_wns=self._lbl_wns,
            dtype=self._dtype,
        )

        self._brdf_mod = BRDFModeule(
            lbl_wns=self._lbl_wns,
            brdf_factor=brdf_factors,
            channel_index=self._channel_index,
            dtype=self._dtype,
        )
        
    def _disp2wns(self, dispersion_coef):
        samples = (torch.arange(1016) + 1).reshape(1, -1).to(
            dtype = dispersion_coef.dtype, device = dispersion_coef.device)
        wls = torch.zeros((dispersion_coef.shape[0], 1016), 
                dtype=dispersion_coef.dtype, device = dispersion_coef.device)
        for i in range(6):
            wls += dispersion_coef[:, i:i+1] * samples ** i
        return wls, samples

    def _get_wn_range(self):
        wls, samples = self._disp2wns(self._dispersion_coef_samp.reshape(1, -1))
        wl_min = wls[:, 0].min() + self._ils_delta_lambda[0, 0].item()
        wl_max = wls[:, -1].max() + self._ils_delta_lambda[-1, -1].item()
        wn_min = wl2wn(wl_max) - 1.0
        wn_max = wl2wn(wl_min) + 1.0
        return [float(wn_min), float(wn_max)]

    # def _compute_optical_thickness(self, pressures, temperatures, broadener_vmrs, 
    #         specie_vmrs):
    #     """
    #     Compute optical thickness (tau) for species and H2O from unscaled retrieved data.
        
    #     Args:
    #         pressures: Pressures (B, L)
    #         temperatures: Temperatures (B, L)
    #         broadener_vmrs: Broadener VMRs (B, L)
    #         specie_vmrs: Specie VMRs (B, L)
            
    #     Returns:
    #         tau_specie: Flattened optical thickness for species (B, (L-1)*C)
    #         tau_h2o: Flattened optical thickness for H2O (B, (L-1)*C)
    #     """
    #     # Extract atmospheric profiles

    #     B, L = pressures.shape

    #     # Flatten (B, L) → (B*L,) for interpolation
    #     P_flat  = pressures.reshape(-1)       # (B*L,)
    #     T_flat  = temperatures.reshape(-1)    # (B*L,)
    #     Br_flat = broadener_vmrs.reshape(-1)      # (B*L,)

    #     # AbsorptionCoefficientModule returns (B*L, W)
    #     kappa_specie = self._specie_absc_mod(P_flat, T_flat, Br_flat)   # (B*L, W)
    #     kappa_broadener    = self._broadener_absc_mod(P_flat, T_flat, Br_flat)      # (B*L, W)

    #     # Reshape to (B, L, W) and apply volume mixing ratios
    #     W = kappa_specie.shape[-1]

    #     kappa_specie = kappa_specie.contiguous().view(B, L, W)   # (B, L, W)
    #     kappa_broadener    = kappa_broadener.contiguous().view(B, L, W)      # (B, L, W)

    #     specie_vmrs_3d = specie_vmrs.unsqueeze(-1)               # (B, L, 1)
    #     broadener_vmrs_3d    = broadener_vmrs.unsqueeze(-1)                # (B, L, 1)

    #     kappa_specie_if = kappa_specie * specie_vmrs_3d   # (B, L, W)
    #     kappa_broadener_if    = kappa_broadener * broadener_vmrs_3d        # (B, L, W)

    #     # Convert VMR to specific humidity
    #     q_if = vmr2sh(broadener_vmrs)   # (B, L)

    #     # Compute scaling factor: factor_if = (1 - q)/(g * M_dry)
    #     factor_if = (1.0 - q_if) / (g * M_dry)     # (B, L)
    #     factor_if_3d = factor_if.unsqueeze(-1)               # (B, L, 1)

    #     # Interface values: values_if = κ_if * factor_if
    #     values_specie_if = kappa_specie_if * factor_if_3d    # (B, L, W)
    #     values_broadener_if    = kappa_broadener_if * factor_if_3d       # (B, L, W)

    #     # Average from interfaces to layer centers: L_interfaces → L_layers = L-1
    #     values_specie_layer = 0.5 * (
    #         values_specie_if[:, 1:, :] + values_specie_if[:, :-1, :]
    #     )   # (B, L-1, W)

    #     values_broadener_layer = 0.5 * (
    #         values_broadener_if[:, 1:, :] + values_broadener_if[:, :-1, :]
    #     )   # (B, L-1, W)

    #     # Compute layer thickness from interface pressures: L-1 layers from L interfaces
    #     dp = pressures[:, 1:] - pressures[:, :-1]    # (B, L-1)
    #     # dp = dp.abs()  # Ensure positive values
    #     dp_3d = dp.unsqueeze(-1)                     # (B, L-1, 1)

    #     # Compute per-species per-layer optical thickness τ
    #     tau_specie = values_specie_layer * dp_3d     # (B, L-1, W)
    #     tau_broadener    = values_broadener_layer * dp_3d        # (B, L-1, W)

    #     # Flatten layer dimension for MLP input
    #     tau_specie = tau_specie.contiguous().view(B, -1, W)   # (B, (L-1)*W)
    #     tau_broadener    = tau_broadener.contiguous().view(B, -1, W)      # (B, (L-1)*W)
        
    #     return torch.sum(tau_specie + tau_broadener, dim=1)

    def _compute_optical_thickness(self, pressures, temperatures, broadener_vmrs, 
                                   specie_vmrs):
        """
        Compute optical thickness (tau) for specie + H2O from unscaled retrieved data.

        Args:
            pressures:       (B, L)  interface pressures
            temperatures:    (B, L)
            broadener_vmrs:  (B, L)  e.g. H2O VMR
            specie_vmrs:     (B, L)  e.g. CO2 / O2 VMR

        Returns:
            taus: (B, W) total optical thickness integrated over layers
        """
        B, L = pressures.shape

        # Flatten (B, L) → (B*L,) for kappa look-up
        P_flat  = pressures.reshape(-1)        # (B*L,)
        T_flat  = temperatures.reshape(-1)     # (B*L,)
        Br_flat = broadener_vmrs.reshape(-1)   # (B*L,)

        # AbsorptionCoefficientModule returns (B*L, W)
        kappa_specie    = self._specie_absc_mod(P_flat, T_flat, Br_flat)     # (B*L, W)
        kappa_broadener = self._broadener_absc_mod(P_flat, T_flat, Br_flat)  # (B*L, W)

        W = kappa_specie.shape[-1]

        # Reshape to (B, L, W)
        kappa_specie    = kappa_specie.view(B, L, W)        # (B, L, W)
        kappa_broadener = kappa_broadener.view(B, L, W)     # (B, L, W)

        # VMRs to broadcast over W
        specie_vmrs_3d    = specie_vmrs.unsqueeze(-1)       # (B, L, 1)
        broadener_vmrs_3d = broadener_vmrs.unsqueeze(-1)    # (B, L, 1)

        # Convert VMR(H2O) to specific humidity q
        q_if = vmr2sh(broadener_vmrs)                       # (B, L)

        # Scaling factor at interfaces: (1 - q)/(g * M_dry)
        factor_if    = (1.0 - q_if) / (g * M_dry)           # (B, L)
        factor_if_3d = factor_if.unsqueeze(-1)              # (B, L, 1)

        # ---- 关键：从 kappa 开始就合并 specie + broadener ----
        # kappa_total_if = κ_specie * VMR_specie + κ_broadener * VMR_broadener
        kappa_total_if = (
            kappa_specie * specie_vmrs_3d
            + kappa_broadener * broadener_vmrs_3d
        )                                                   # (B, L, W)

        # Apply mass-scaling factor (hydrostatic, dry-air mass)
        values_total_if = kappa_total_if * factor_if_3d     # (B, L, W)

        # Interface → layer centers: L_interfaces → (L-1) layers
        values_total_layer = 0.5 * (
            values_total_if[:, 1:, :] + values_total_if[:, :-1, :]
        )                                                   # (B, L-1, W)

        # Layer pressure thickness
        dp    = pressures[:, 1:] - pressures[:, :-1]        # (B, L-1)
        dp_3d = dp.unsqueeze(-1)                            # (B, L-1, 1)

        # Layer optical thickness τ_layer
        tau_layer = values_total_layer * dp_3d              # (B, L-1, W)

        # Integrate over layers → total τ (B, W)
        taus = tau_layer.sum(dim=1)                         # (B, W)

        return taus

    def _convolving(self, grid_wls: torch.Tensor,
                    unconvolved_rad: torch.Tensor, 
                    doppler_scaler: torch.Tensor):
        """
        Batched PyTorch version of ILS convolution with Doppler correction.
        
        Args:
            measured_wavelengths: Target wavelength grid, shape (B, G)
            unconvolved_rad: High-resolution radiance, shape (B, W)
            lbl_wls: High-resolution wavelength grid, shape (W,)
            ils_delta_lambda: ILS wavelength offsets, shape (G, N_ILS)
            ils_relative_response: ILS response weights, shape (G, N_ILS)
            doppler_scaler: Doppler scaling factor per sounding, shape (B,) or None
        
        Returns:
            convolved_rad: Convolved radiance, shape (B, G)
        """
        B, G = grid_wls.shape
        W = self._lbl_wls.shape[0]
        N_ILS = self._ils_delta_lambda.shape[1]
        
        lbl_wls_doppler = self._lbl_wls.unsqueeze(0) * doppler_scaler.reshape(-1, 1)  # (B, W)
        
        # Flip wavelength grid for descending order (like the numpy version)
        wl_cal_flipped = torch.flip(lbl_wls_doppler, dims=[1])  # (B, W)
        unconvolved_rad_flipped = torch.flip(unconvolved_rad, dims=[1])  # (B, W)
        
        # Expand measured wavelengths to include ILS dimension: (B, G, 1) + (G, N_ILS) -> (B, G, N_ILS)
        convolving_wl = grid_wls.unsqueeze(-1) + self._ils_delta_lambda.unsqueeze(0)  # (B, G, N_ILS)
        
        # Flatten for batch interpolation
        convolving_wl_flat = convolving_wl.reshape(B, -1)  # (B, G*N_ILS)
        
        # Interpolate for all batch samples at once
        # For each batch, interpolate at G*N_ILS wavelength points
        convolving_rad_flat = torch.zeros_like(convolving_wl_flat)  # (B, G*N_ILS)
        
        # 1. batched searchsorted：每一行用自己的网格
        indices = torch.searchsorted(wl_cal_flipped, convolving_wl_flat)   # (B, L)
        indices = indices.clamp(1, W - 1)                                  # 保证左右都有点

        idx_left = indices - 1
        idx_right = indices

        # 2. 用 gather 取左右点
        wl_cal_left  = wl_cal_flipped.gather(1, idx_left)   # (B, L)
        wl_cal_right = wl_cal_flipped.gather(1, idx_right)  # (B, L)

        rad_left  = unconvolved_rad_flipped.gather(1, idx_left)   # (B, L)
        rad_right = unconvolved_rad_flipped.gather(1, idx_right)  # (B, L)

        # 3. 线性插值
        t = (convolving_wl_flat - wl_cal_left) / (wl_cal_right - wl_cal_left + 1e-10)

        convolving_rad_flat[:] = rad_left + t * (rad_right - rad_left)
        
        # Reshape back to (B, G, N_ILS)
        convolving_rad = convolving_rad_flat.reshape(B, G, N_ILS)
        
        # Apply ILS response: (B, G, N_ILS) * (G, N_ILS) -> (B, G, N_ILS)
        responsed = convolving_rad * self._ils_relative_response.unsqueeze(0)
        
        # Integrate using trapezoidal rule along ILS dimension
        # For each (b, g), integrate over N_ILS points
        delta_wl = convolving_wl[:, :, 1:] - convolving_wl[:, :, :-1]  # (B, G, N_ILS-1)
        avg_response = (responsed[:, :, 1:] + responsed[:, :, :-1]) / 2  # (B, G, N_ILS-1)
        convolved_rad = torch.sum(avg_response * delta_wl, dim=-1)  # (B, G)
        
        return convolved_rad
                
    def forward(self, pressures, temperatures, broadener_vmrs, specie_vmrs,
                geo_angles, brdf_weights, disp_offset_space, solar_info):
        """
        pressures: (B, L)
        temperatures: (B, L)
        broadener_vmrs: (B, L)
        specie_vmrs: (B, L)
        geo_angles: (B, 5)
        brdf_weights: (B, 3)
        disp_offset_space: (B, 2) offset, spacing
        solar_info: (B, 2) solar_distance, solar_velocity
        """
        taus = self._compute_optical_thickness(
            pressures, temperatures, broadener_vmrs, specie_vmrs)   # (B, W)
        srfa = self._brdf_mod(geo_angles, brdf_weights)   # (B, W)
        
        solar_distance, solar_velocity = solar_info[:, 0], solar_info[:, 1]
        I_solar = self._solar_mod(solar_distance)
        cos_ti, cos_tr, _, _, _ = geo_angles[:, 0], geo_angles[:, 1], geo_angles[:, 2], geo_angles[:, 3], geo_angles[:, 4]
        angle_term = ((cos_ti + cos_tr) / (cos_ti * cos_tr)).reshape(-1, 1)
        I_absco = I_solar * torch.exp(-taus * angle_term) * srfa * cos_ti.reshape(-1, 1) / (2 * torch.pi)
        doppler_scaler = 1.0 + solar_velocity / c
        
        dispersion_coef = self._dispersion_coef_samp.clone().reshape(1, -1).repeat(pressures.shape[0], 1)
        dispersion_coef[:, 0] = disp_offset_space[:, 0]
        dispersion_coef[:, 1] = disp_offset_space[:, 1]
        grid_wls, _ = self._disp2wns(dispersion_coef)
        I_convolved = self._convolving(grid_wls, I_absco, doppler_scaler)
        return grid_wls, I_convolved