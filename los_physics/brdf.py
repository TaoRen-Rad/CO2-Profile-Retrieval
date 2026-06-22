import torch
from torch import nn
from .constant import BRDF_NU0S, wl2wn

class BRDFModeule(nn.Module):
    def __init__(self, lbl_wns: torch.Tensor,
                 brdf_factor: torch.Tensor, channel_index: int, 
                 dtype: torch.dtype = torch.float64):
        super().__init__()
        self._dtype = dtype
        self.register_buffer("_lambda", lbl_wns.reshape(1, -1))
        self.register_buffer("r_s", brdf_factor[0].to(self._dtype))
        self.register_buffer("rho_0", brdf_factor[1].to(self._dtype))
        self.register_buffer("Theta", brdf_factor[2].to(self._dtype))
        self.register_buffer("k", brdf_factor[3].to(self._dtype))
        self.register_buffer("_lambda_0", torch.tensor(wl2wn(BRDF_NU0S[channel_index]), dtype=self._dtype))

    def rahman(self, cos_ti, cos_tr, sin_ti, sin_tr, cos_phi):
        cos_phi = -cos_phi

        # term1
        term1 = (cos_ti * cos_tr) ** (self.k - 1) / (cos_ti + cos_tr) ** (1 - self.k)

        # cos(g)
        cos_g = cos_ti * cos_tr + sin_ti * sin_tr * cos_phi

        # F(g)
        F_g = (1 - self.Theta ** 2) / (1 + self.Theta ** 2 + 2 * self.Theta * cos_g) ** 1.5

        tan_ti = sin_ti / cos_ti
        tan_tr = sin_tr / cos_tr
        G_sq = tan_ti**2 + tan_tr**2 - 2 * tan_ti * tan_tr * cos_phi
        G_sq = torch.clamp(G_sq, min=0.0)
        G = torch.sqrt(G_sq)

        R_G = (1 - self.rho_0) / (1 + G)

        return self.rho_0 * term1 * F_g * (1 + R_G)

    def forward(self, geo_angles: torch.Tensor, brdf_weights: torch.Tensor):
        """
        lbl_wls: (W,)
        geo_angles: (B, 5), (cos_theta_i, cos_theta_r, sin_theta_i, sin_theta_r, cos_phi)
        brdf_weights: (B, 3), (w, s, q) / (brdf_weight, brdf_weight_slope, brdf_weight_quadratic)
        """
        rahman_factors = self.rahman(geo_angles[:, 0], geo_angles[:, 1], geo_angles[:, 2], geo_angles[:, 3], geo_angles[:, 4])
        w, s, q = brdf_weights[:, 0:1], brdf_weights[:, 1:2], brdf_weights[:, 2:3]
        # nu = lbl_wls.reshape(1, -1)
        # nu, nu_0 = wl2wn(nu), wl2wn(self.nu_0)
        # spectral_factor = (w + s*(nu - nu_0) + q*(nu - nu_0)**2)
        spectral_factor = (w + s*(self._lambda - self._lambda_0) + q*(self._lambda - self._lambda_0)**2)
        return self.r_s.reshape(-1, 1) * rahman_factors.reshape(-1, 1) * spectral_factor