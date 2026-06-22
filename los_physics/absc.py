from typing import List, Optional, Tuple
import h5py
import numpy as np
import torch
from torch import nn
from scipy.constants import N_A
# from sklearn.decomposition import PCA
import numpy as np
from .constant import M_dry, M_h2o, ABSCO_PATH, ID2NAME, byte2string


class AbsorptionCoefficientModule(nn.Module):
    """
    PyTorch version of AbsorptionCoefficient.
    """

    def __init__(
        self,
        gas_name: str,
        # file_path: str,
        wavenumber_range: Optional[Tuple[float, float]],
        # use_pca: bool = True,
        # n_components: int = 100,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self._gas_name = gas_name
        self._file_path = ABSCO_PATH[gas_name]
        self._dtype = dtype
        self._device = device if device is not None else torch.device("cpu")

        # 对应 numpy 的 dtype
        if self._dtype == torch.float64:
            self._np_dtype = np.float64
        elif self._dtype == torch.float32:
            self._np_dtype = np.float32
        else:
            # 其他情况就退回 float64
            self._np_dtype = np.float64

        self._read_database()
        
        # 直接进行 wavenumber 裁剪（在 numpy 层面）
        if wavenumber_range is not None:
            wn_min, wn_max = wavenumber_range
            mask = (self._wavenumbers_raw >= wn_min) & (self._wavenumbers_raw <= wn_max)
            wavenumbers_clipped = self._wavenumbers_raw[mask]
            absco_clipped = self._absorption_coefficients_raw[..., mask]
        else:
            # 不裁剪时，使用全部数据
            wavenumbers_clipped = self._wavenumbers_raw
            absco_clipped = self._absorption_coefficients_raw
            
        # if use_pca:
        #     self._pca = PCA(n_components=n_components)
        #     shape = list(absco_clipped.shape)
        #     absco_clipped = self._pca.fit_transform(absco_clipped.reshape(-1, shape[-1]))
        #     shape[-1] = n_components
        #     absco_clipped = absco_clipped.reshape(*shape)

        # 只保存裁剪后的数据作为 buffer
        self.register_buffer(
            "wavenumbers",
            torch.from_numpy(wavenumbers_clipped).to(self._device, self._dtype),
        )
        self.register_buffer(
            "absco",
            torch.from_numpy(absco_clipped).to(self._device, self._dtype),
        )

        # 网格轴（不裁剪）
        self.register_buffer(
            "pressures",
            torch.from_numpy(self._pressures).to(self._device, self._dtype),
        )
        self.register_buffer(
            "temperatures",
            torch.from_numpy(self._temperatures).to(self._device, self._dtype),
        )  # (nP, nT)
        self.register_buffer(
            "broadener_vmrs",
            torch.from_numpy(self._broadener_vmrs).to(self._device, self._dtype),
        )

    # ---------- 读取 HDF5 ----------
    def _read_database(self):
        with h5py.File(self._file_path, "r") as f:
            self._gas_index = byte2string(f["Gas_Index"][:])
            self._gas_name = ID2NAME[self._gas_index]
            self._broadener_index = byte2string(f["Broadener_Index"][:])
            self._broadener_name = ID2NAME[self._broadener_index]

            self._broadener_vmrs = np.array(
                f[f"Broadener_{self._broadener_index}_VMR"][:],
                dtype=self._np_dtype,
            )
            self._pressures = np.array(
                f["Pressure"][:],
                dtype=self._np_dtype,
            )
            self._temperatures = np.array(
                f["Temperature"][:],
                dtype=self._np_dtype,
            )
            self._wavenumbers_raw = np.array(
                f["Wavenumber"][:],
                dtype=self._np_dtype,
            )

            # 单位转换保持和你原来一致：* N_A * 1e-4
            self._absorption_coefficients_raw = np.array(
                f[f"Gas_{self._gas_index}_Absorption"][:, :, :, :] * N_A * 1e-4,
                dtype=self._np_dtype,
            )

    # ---------- 主插值逻辑：forward ----------
    def forward(
        self,
        pressures_in: torch.Tensor,
        temperatures_in: torch.Tensor,
        broadener_vmrs_in: torch.Tensor,
    ) -> torch.Tensor:
        """
        pressures_in, temperatures_in, broadener_vmrs_in:
            - shape 必须一致，可为任意形状（例如 (N,), (batch, n_layers) 等）
        返回：
            abscoef: (*input_shape, nW_clip)
        """

        # 确保都是 tensor，且在同一 dtype/device
        P_in = pressures_in #.to(device=self._device, dtype=self._dtype)
        T_in = temperatures_in #.to(device=self._device, dtype=self._dtype)
        V_in = broadener_vmrs_in #.to(device=self._device, dtype=self._dtype)

        in_shape = P_in.shape
        P_flat = P_in.reshape(-1)  # (N,)
        T_flat = T_in.reshape(-1)  # (N,)
        V_flat = V_in.reshape(-1)  # (N,)
        N = P_flat.numel()

        # 网格
        P_grid = self.pressures             # (nP,)
        T_grid = self.temperatures          # (nP, nT)
        V_grid = self.broadener_vmrs        # (nV,)
        absco = self.absco                  # (nP, nT, nV, nWc)

        nP = P_grid.shape[0]
        nT = T_grid.shape[1]
        nV = V_grid.shape[0]
        nW = absco.shape[-1]

        # ---------- 1. pressure 方向 ----------
        p_idx = torch.searchsorted(P_grid, P_flat)
        p_idx = p_idx.clamp(1, nP - 1)
        p0 = p_idx - 1
        p1 = p_idx

        P0 = P_grid[p0]
        P1 = P_grid[p1]
        wp = (P_flat - P0) / (P1 - P0 + 1e-14)  # (N,)

        # ---------- 2. vmr 方向 ----------
        v_idx = torch.searchsorted(V_grid, V_flat)
        v_idx = v_idx.clamp(1, nV - 1)
        v0 = v_idx - 1
        v1 = v_idx

        V0 = V_grid[v0]
        V1 = V_grid[v1]
        wv = (V_flat - V0) / (V1 - V0 + 1e-14)  # (N,)

        # ---------- 3. temperature 方向 ----------
        T_row0 = T_grid[p0]  # (N, nT)
        T_row1 = T_grid[p1]  # (N, nT)
        T_vals = T_flat.unsqueeze(-1)  # (N, 1)

        t_idx0 = torch.searchsorted(T_row0, T_vals).squeeze(-1)
        t_idx1 = torch.searchsorted(T_row1, T_vals).squeeze(-1)

        t_idx0 = t_idx0.clamp(1, nT - 1)
        t_idx1 = t_idx1.clamp(1, nT - 1)

        t0_l = t_idx0 - 1
        t0_r = t_idx0
        t1_l = t_idx1 - 1
        t1_r = t_idx1

        t0_l_val = torch.gather(T_row0, 1, t0_l.unsqueeze(-1)).squeeze(-1)
        t0_r_val = torch.gather(T_row0, 1, t0_r.unsqueeze(-1)).squeeze(-1)
        t1_l_val = torch.gather(T_row1, 1, t1_l.unsqueeze(-1)).squeeze(-1)
        t1_r_val = torch.gather(T_row1, 1, t1_r.unsqueeze(-1)).squeeze(-1)

        wt0 = (T_flat - t0_l_val) / (t0_r_val - t0_l_val + 1e-14)
        wt1 = (T_flat - t1_l_val) / (t1_r_val - t1_l_val + 1e-14)

        # ---------- 4. 从 absco 网格中取 8 个角点 ----------
        c000 = absco[p0, t0_l, v0]  # (N, nW)
        c001 = absco[p0, t0_l, v1]
        c010 = absco[p0, t0_r, v0]
        c011 = absco[p0, t0_r, v1]

        c100 = absco[p1, t1_l, v0]
        c101 = absco[p1, t1_l, v1]
        c110 = absco[p1, t1_r, v0]
        c111 = absco[p1, t1_r, v1]

        # vmr 方向插值
        wv_ = wv.unsqueeze(-1)
        c00 = c000 + (c001 - c000) * wv_
        c01 = c010 + (c011 - c010) * wv_
        c10 = c100 + (c101 - c100) * wv_
        c11 = c110 + (c111 - c110) * wv_

        # temperature 方向
        wt0_ = wt0.unsqueeze(-1)
        wt1_ = wt1.unsqueeze(-1)
        c0 = c00 + (c01 - c00) * wt0_
        c1 = c10 + (c11 - c10) * wt1_

        # pressure 方向
        wp_ = wp.unsqueeze(-1)
        c = c0 + (c1 - c0) * wp_  # (N, nW)

        out = c.reshape(*in_shape, nW)
        return out

    # ---------- 属性 ----------
    @property
    def gas_name(self) -> str:
        return self._gas_name

    @property
    def broadener_name(self) -> str:
        return self._broadener_name