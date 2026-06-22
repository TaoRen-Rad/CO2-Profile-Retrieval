"""
Optimized inference utilities for diffusion model
Includes torch.jit optimization and fast DDIM sampling
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import time


class OptimizedDDIMSampler:
    """
    Optimized DDIM sampler with pre-computed schedules and JIT compilation
    Target: <500ms inference for batch of samples
    """
    
    def __init__(self, model, num_inference_steps=20, eta=0.0, device='cpu', dtype=torch.float32):
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.eta = eta
        self.device = device
        self.dtype = dtype
        
        # Pre-compute timestep schedule
        self.timesteps = self._compute_timesteps()
        
        # Pre-compute alpha values for all timesteps
        self.alpha_schedule = self._precompute_alphas()
        
    def _compute_timesteps(self):
        """Pre-compute timestep schedule"""
        num_train_timesteps = self.model.diffusion.num_timesteps
        step_size = num_train_timesteps // self.num_inference_steps
        timesteps = torch.arange(0, num_train_timesteps, step_size, device=self.device)
        return torch.flip(timesteps, dims=[0])
    
    def _precompute_alphas(self):
        """Pre-compute alpha values for faster sampling"""
        alphas = []
        for i, t in enumerate(self.timesteps):
            alpha_cumprod_t = self.model.diffusion.noise_scheduler.alphas_cumprod[t]
            
            if i < len(self.timesteps) - 1:
                t_prev = self.timesteps[i + 1]
                alpha_cumprod_t_prev = self.model.diffusion.noise_scheduler.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            
            alphas.append({
                'alpha_t': alpha_cumprod_t,
                'alpha_t_prev': alpha_cumprod_t_prev,
                'sqrt_alpha_t': torch.sqrt(alpha_cumprod_t),
                'sqrt_one_minus_alpha_t': torch.sqrt(1 - alpha_cumprod_t),
                'sqrt_alpha_t_prev': torch.sqrt(alpha_cumprod_t_prev)
            })
        
        return alphas
    
    @torch.no_grad()
    def sample(self, geo, ret, num_samples=1):
        """
        Fast DDIM sampling with pre-computed schedules
        
        Args:
            geo: (B, geo_dim) - geometry features
            ret: (B, ret_dim) - retrieved state features
            num_samples: number of samples per input
        
        Returns:
            (B*num_samples, rad_dim) - predicted radiance
        """
        batch_size = geo.shape[0]
        
        # Extract band-specific retrieved state
        ret_band = ret[:, self.model.band_indices]
        
        # Prepare conditioning
        condition = torch.cat([geo, ret_band], dim=1)
        
        # Expand condition for multiple samples
        if num_samples > 1:
            condition = condition.repeat_interleave(num_samples, dim=0)
        
        # Start from pure noise
        x = torch.randn(batch_size * num_samples, self.model.latent_dim, 
                       device=self.device, dtype=self.dtype)
        
        # Denoise step by step using pre-computed schedule
        for i, t in enumerate(self.timesteps):
            t_batch = torch.full((batch_size * num_samples,), t, 
                                device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.model.diffusion.denoising_net(x, t_batch, condition)
            
            # Get pre-computed alpha values
            alphas = self.alpha_schedule[i]
            
            # Predict x_0
            x_0_pred = (x - alphas['sqrt_one_minus_alpha_t'] * noise_pred) / alphas['sqrt_alpha_t']
            
            # Compute variance for stochastic sampling
            if self.eta > 0 and i < len(self.timesteps) - 1:
                variance = self.eta**2 * (1 - alphas['alpha_t_prev']) / (1 - alphas['alpha_t']) * \
                          (1 - alphas['alpha_t'] / alphas['alpha_t_prev'])
                variance = torch.sqrt(variance)
            else:
                variance = 0.0
            
            # Direction pointing to x_t
            dir_xt_coef = torch.sqrt(1 - alphas['alpha_t_prev'] - variance**2)
            dir_xt = dir_xt_coef * noise_pred
            
            # Compute x_{t-1}
            x = alphas['sqrt_alpha_t_prev'] * x_0_pred + dir_xt
            
            # Add stochastic noise
            if self.eta > 0 and i < len(self.timesteps) - 1:
                noise = torch.randn_like(x)
                x = x + variance * noise
        
        # Decode to radiance
        radiance_pred = self.model.decoder(x)
        
        return radiance_pred
    
    @torch.no_grad()
    def benchmark(self, geo, ret, num_runs=100):
        """
        Benchmark inference speed
        
        Returns:
            avg_time_ms, std_time_ms
        """
        # Warmup
        for _ in range(10):
            _ = self.sample(geo, ret)
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.sample(geo, ret)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        avg_time = sum(times) / len(times)
        std_time = (sum((t - avg_time)**2 for t in times) / len(times))**0.5
        
        return avg_time, std_time


def create_optimized_sampler(model, num_inference_steps=20, eta=0.0):
    """
    Factory function to create an optimized sampler
    
    Args:
        model: ConditionalDiffusionModel instance
        num_inference_steps: number of denoising steps (10-20 recommended)
        eta: stochasticity parameter (0=deterministic, 1=stochastic)
    
    Returns:
        OptimizedDDIMSampler instance
    """
    sampler = OptimizedDDIMSampler(
        model=model,
        num_inference_steps=num_inference_steps,
        eta=eta,
        device=model.device,
        dtype=model.dtype
    )
    
    return sampler


@torch.jit.script
def ddim_step(x: torch.Tensor, noise_pred: torch.Tensor, 
              sqrt_alpha_t: torch.Tensor, sqrt_one_minus_alpha_t: torch.Tensor,
              sqrt_alpha_t_prev: torch.Tensor, dir_coef: torch.Tensor) -> torch.Tensor:
    """
    JIT-compiled DDIM step for maximum speed
    
    Args:
        x: current latent
        noise_pred: predicted noise
        sqrt_alpha_t: sqrt of alpha at timestep t
        sqrt_one_minus_alpha_t: sqrt of (1 - alpha) at timestep t
        sqrt_alpha_t_prev: sqrt of alpha at previous timestep
        dir_coef: direction coefficient
    
    Returns:
        x at previous timestep
    """
    # Predict x_0
    x_0_pred = (x - sqrt_one_minus_alpha_t * noise_pred) / sqrt_alpha_t
    
    # Direction pointing to x_t
    dir_xt = dir_coef * noise_pred
    
    # Compute x_{t-1}
    x_prev = sqrt_alpha_t_prev * x_0_pred + dir_xt
    
    return x_prev


class JITOptimizedSampler:
    """
    Further optimized sampler using JIT compilation for critical paths
    """
    
    def __init__(self, model, num_inference_steps=20, device='cpu', dtype=torch.float32):
        self.model = model
        self.num_inference_steps = num_inference_steps
        self.device = device
        self.dtype = dtype
        
        # Pre-compute schedules
        self.timesteps = self._compute_timesteps()
        self.coefficients = self._precompute_coefficients()
        
    def _compute_timesteps(self):
        num_train_timesteps = self.model.diffusion.num_timesteps
        step_size = num_train_timesteps // self.num_inference_steps
        timesteps = torch.arange(0, num_train_timesteps, step_size, device=self.device)
        return torch.flip(timesteps, dims=[0])
    
    def _precompute_coefficients(self):
        """Pre-compute all coefficients needed for DDIM steps"""
        coeffs = []
        
        for i, t in enumerate(self.timesteps):
            alpha_t = self.model.diffusion.noise_scheduler.alphas_cumprod[t]
            
            if i < len(self.timesteps) - 1:
                t_prev = self.timesteps[i + 1]
                alpha_t_prev = self.model.diffusion.noise_scheduler.alphas_cumprod[t_prev]
            else:
                alpha_t_prev = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            sqrt_alpha_t_prev = torch.sqrt(alpha_t_prev)
            dir_coef = torch.sqrt(1 - alpha_t_prev)
            
            coeffs.append({
                't': t,
                'sqrt_alpha_t': sqrt_alpha_t,
                'sqrt_one_minus_alpha_t': sqrt_one_minus_alpha_t,
                'sqrt_alpha_t_prev': sqrt_alpha_t_prev,
                'dir_coef': dir_coef
            })
        
        return coeffs
    
    @torch.no_grad()
    def sample(self, geo, ret):
        """
        Ultra-fast sampling using JIT-compiled steps
        """
        batch_size = geo.shape[0]
        
        # Prepare conditioning
        ret_band = ret[:, self.model.band_indices]
        condition = torch.cat([geo, ret_band], dim=1)
        
        # Start from noise
        x = torch.randn(batch_size, self.model.latent_dim, 
                       device=self.device, dtype=self.dtype)
        
        # Denoise using JIT-compiled steps
        for i, coeff in enumerate(self.coefficients):
            t_batch = torch.full((batch_size,), coeff['t'], 
                                device=self.device, dtype=torch.long)
            
            # Predict noise (this is the bottleneck - consider TorchScript for denoising_net too)
            noise_pred = self.model.diffusion.denoising_net(x, t_batch, condition)
            
            # JIT-compiled denoising step
            x = ddim_step(
                x, noise_pred,
                coeff['sqrt_alpha_t'],
                coeff['sqrt_one_minus_alpha_t'],
                coeff['sqrt_alpha_t_prev'],
                coeff['dir_coef']
            )
        
        # Decode
        radiance_pred = self.model.decoder(x)
        
        return radiance_pred

