import os
import torch
import h5py
import numpy as np
from torch import nn
from typing import Optional, Tuple
from .constant import ABSCO_DIR, BASELINE_SOLAR_DISTANCE


class SolarModule(nn.Module):
    """
    PyTorch Module for solar spectrum interpolation.
    Supports .to() for device/dtype conversion.
    """
    
    def __init__(
        self,
        lbl_wns: torch.Tensor,
        dtype: torch.dtype = torch.float64,
        device: Optional[torch.device] = None,
    ):
        """
        filename:  HDF5 文件路径
        dtype:     浮点精度，默认 float64，方便做数值优化
        device:    torch.device，例如 'cuda' 或 'cpu'，默认为 cpu
        """
        super().__init__()
        self._filename = os.path.join(ABSCO_DIR, "oco_solar_model.h5")
        self._dtype = dtype
        self._device = device if device is not None else torch.device('cpu')
        
        # 对应 numpy 的 dtype
        if self._dtype == torch.float64:
            self._np_dtype = np.float64
        elif self._dtype == torch.float32:
            self._np_dtype = np.float32
        else:
            self._np_dtype = np.float64
        
        self._load_data(lbl_wns)

    def _load_data(self, lbl_wns: torch.Tensor):
        """从 HDF5 文件加载太阳光谱数据"""
        ans_wn = []
        ans_spectrum = []

        with h5py.File(self._filename, 'r') as f:
            for i in range(3, 0, -1):
                wn_absc = f[f"Solar/Absorption/Absorption_{i}/wavenumber"][:]
                absc = f[f"Solar/Absorption/Absorption_{i}/spectrum"][:]

                wn = f[f"Solar/Continuum/Continuum_{i}/wavenumber"][:]
                spectrum_continum = f[f"Solar/Continuum/Continuum_{i}/spectrum"][:]

                # 在 continuum 上插值
                spectrum = absc * np.interp(wn_absc, wn, spectrum_continum)

                ans_wn.append(wn_absc)
                ans_spectrum.append(spectrum)

        ans_wn = np.hstack(ans_wn).astype(self._np_dtype)
        ans_spectrum = (np.hstack(ans_spectrum) / 1e20).astype(self._np_dtype)

        # 转成 torch tensor
        wn_grid = torch.from_numpy(ans_wn)
        spectrum_grid = torch.from_numpy(ans_spectrum)

        # 确保波数是升序
        sort_idx = torch.argsort(wn_grid)
        wn_grid = wn_grid[sort_idx]
        spectrum_grid = spectrum_grid[sort_idx]
        

        # 保存原始形状
        in_shape = lbl_wns.shape
        
        # 展平成一维，方便插值运算
        wn_flat = lbl_wns.reshape(-1)

        idx = torch.searchsorted(wn_grid, wn_flat, right=False)

        # 边界处理：限制在 [1, len-1]，这样 idx-1 和 idx 都合法
        idx = torch.clamp(idx, 1, wn_grid.numel() - 1)

        x0 = wn_grid[idx - 1]
        x1 = wn_grid[idx]
        y0 = spectrum_grid[idx - 1]
        y1 = spectrum_grid[idx]

        # 线性插值权重
        t = (wn_flat - x0) / (x1 - x0 + 1e-14)
        y = y0 + t * (y1 - y0)

        # reshape 回原来的形状
        y = y.reshape(in_shape)


        self.register_buffer("lbl_wns", lbl_wns.to(self._device, self._dtype))
        self.register_buffer("I_solars", y.to(self._device, self._dtype))

    def forward(self, solar_distance: torch.Tensor) -> torch.Tensor:
        solar_power_scale_factor = (BASELINE_SOLAR_DISTANCE / solar_distance) ** 2
        return self.I_solars.reshape(1, -1) * solar_power_scale_factor.reshape(-1, 1)

    