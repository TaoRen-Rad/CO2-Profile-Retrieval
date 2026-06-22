import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class NoiseScheduler:
    """Noise scheduler for diffusion models with support for both linear and cosine schedules"""
    
    def __init__(self, num_timesteps=200, beta_start=1e-4, beta_end=0.02, 
                 schedule_type='linear', device='cpu', dtype=torch.float32):
        self.num_timesteps = num_timesteps
        self.device = device
        self.dtype = dtype
        
        # Create beta schedule
        if schedule_type == 'linear':
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps, 
                                       device=device, dtype=dtype)
        elif schedule_type == 'cosine':
            self.betas = self._cosine_beta_schedule(num_timesteps, device, dtype)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device, dtype=dtype), 
                                              self.alphas_cumprod[:-1]])
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # Clip for numerical stability
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def _cosine_beta_schedule(self, timesteps, device, dtype, s=0.008):
        """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, device=device, dtype=dtype)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def add_noise(self, x_start: torch.Tensor, noise: torch.Tensor, 
                  timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to x_start at given timesteps: q(x_t | x_0)"""
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[timesteps]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[timesteps]
        
        # Reshape for broadcasting
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def get_posterior_mean_variance(self, x_t: torch.Tensor, x_0_pred: torch.Tensor, 
                                    timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute posterior mean and variance"""
        posterior_mean = (
            self.posterior_mean_coef1[timesteps].unsqueeze(-1) * x_0_pred +
            self.posterior_mean_coef2[timesteps].unsqueeze(-1) * x_t
        )
        posterior_variance = self.posterior_variance[timesteps].unsqueeze(-1)
        return posterior_mean, posterior_variance


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal positional embeddings for timestep encoding"""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class DenoisingNetwork(nn.Module):
    """
    Denoising network for diffusion model
    Takes noisy latent and timestep, outputs predicted noise
    """
    
    def __init__(self, latent_dim, condition_dim, time_emb_dim=128, 
                 hidden_dims=None, activation=nn.SiLU(), dropout_rate=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_emb_dim = time_emb_dim
        
        if hidden_dims is None:
            hidden_dims = [latent_dim*2, latent_dim*4, latent_dim*4, latent_dim*2]
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            activation,
        )
        
        # Network layers with time and condition concatenation
        layers = []
        prev_dim = latent_dim + time_emb_dim + condition_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            
            # Add skip connection dimension for next layer
            if i < len(hidden_dims) - 1:
                prev_dim = hidden_dim + condition_dim + time_emb_dim
            else:
                prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.layers = nn.ModuleList(layers)
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.dropout_rate = dropout_rate
    
    def forward(self, x_noisy, timesteps, condition):
        """
        Args:
            x_noisy: (B, latent_dim) - noisy latent
            timesteps: (B,) - timestep indices
            condition: (B, condition_dim) - conditioning features
        Returns:
            (B, latent_dim) - predicted noise
        """
        # Get time embedding
        t_emb = self.time_mlp(timesteps)
        
        # Initial concatenation
        h = torch.cat([x_noisy, t_emb, condition], dim=1)
        
        # Forward through layers with skip connections
        layer_idx = 0
        for i, hidden_dim in enumerate(self.hidden_dims):
            # Linear + activation + dropout
            h = self.layers[layer_idx](h)  # Linear
            h = self.layers[layer_idx + 1](h)  # Activation
            h = self.layers[layer_idx + 2](h)  # Dropout
            layer_idx += 3
            
            # Add skip connection (except for last layer)
            if i < len(self.hidden_dims) - 1:
                h = torch.cat([h, t_emb, condition], dim=1)
        
        # Output layer
        noise_pred = self.layers[layer_idx](h)
        
        return noise_pred


class LatentDiffusion(nn.Module):
    """
    Latent Diffusion Model for spectral emulation
    Operates in compressed latent space for efficiency
    """
    
    def __init__(self, latent_dim, condition_dim, num_timesteps=200, 
                 beta_start=1e-4, beta_end=0.02, schedule_type='linear',
                 denoising_hidden_dims=None, time_emb_dim=128, 
                 dropout_rate=0.1, device='cpu', dtype=torch.float32):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.num_timesteps = num_timesteps
        self.device = device
        self.dtype = dtype
        
        # Noise scheduler
        self.noise_scheduler = NoiseScheduler(
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type,
            device=device,
            dtype=dtype
        )
        
        # Denoising network
        self.denoising_net = DenoisingNetwork(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            time_emb_dim=time_emb_dim,
            hidden_dims=denoising_hidden_dims,
            dropout_rate=dropout_rate
        )
    
    def forward(self, x_start, condition):
        """
        Training forward pass
        Args:
            x_start: (B, latent_dim) - clean latent vectors
            condition: (B, condition_dim) - conditioning features
        Returns:
            noise_pred, noise, timesteps for loss computation
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_timesteps, (batch_size,), 
            device=x_start.device, dtype=torch.long
        )
        
        # Sample noise
        noise = torch.randn_like(x_start)
        
        # Add noise to latent
        x_noisy = self.noise_scheduler.add_noise(x_start, noise, timesteps)
        
        # Predict noise
        noise_pred = self.denoising_net(x_noisy, timesteps, condition)
        
        return noise_pred, noise, timesteps
    
    @torch.no_grad()
    def sample_ddpm(self, condition, num_samples=1):
        """
        DDPM sampling (slower but more accurate)
        Args:
            condition: (B, condition_dim) - conditioning features
            num_samples: number of samples to generate per condition
        Returns:
            (B*num_samples, latent_dim) - sampled latent vectors
        """
        batch_size = condition.shape[0]
        
        # Expand condition for multiple samples
        if num_samples > 1:
            condition = condition.repeat_interleave(num_samples, dim=0)
        
        # Start from pure noise
        x = torch.randn(batch_size * num_samples, self.latent_dim, 
                       device=self.device, dtype=self.dtype)
        
        # Denoise step by step
        for t in reversed(range(self.num_timesteps)):
            timesteps = torch.full((batch_size * num_samples,), t, 
                                  device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.denoising_net(x, timesteps, condition)
            
            # Compute x_{t-1}
            alpha = self.noise_scheduler.alphas[t]
            alpha_cumprod = self.noise_scheduler.alphas_cumprod[t]
            beta = self.noise_scheduler.betas[t]
            
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_cumprod) * noise_pred) / torch.sqrt(alpha_cumprod)
            
            if t > 0:
                # Get posterior mean and variance
                posterior_mean, posterior_variance = self.noise_scheduler.get_posterior_mean_variance(
                    x, x_0_pred, timesteps
                )
                
                # Add noise
                noise = torch.randn_like(x)
                x = posterior_mean + torch.sqrt(posterior_variance) * noise
            else:
                x = x_0_pred
        
        return x
    
    @torch.no_grad()
    def sample_ddim(self, condition, num_inference_steps=20, eta=0.0, num_samples=1):
        """
        DDIM sampling (faster inference)
        Args:
            condition: (B, condition_dim) - conditioning features
            num_inference_steps: number of denoising steps (fewer = faster)
            eta: stochasticity parameter (0 = deterministic, 1 = DDPM)
            num_samples: number of samples to generate per condition
        Returns:
            (B*num_samples, latent_dim) - sampled latent vectors
        """
        batch_size = condition.shape[0]
        
        # Expand condition for multiple samples
        if num_samples > 1:
            condition = condition.repeat_interleave(num_samples, dim=0)
        
        # Create inference timestep schedule
        step_size = self.num_timesteps // num_inference_steps
        timesteps = torch.arange(0, self.num_timesteps, step_size, device=self.device)
        timesteps = torch.flip(timesteps, dims=[0])
        
        # Start from pure noise
        x = torch.randn(batch_size * num_samples, self.latent_dim, 
                       device=self.device, dtype=self.dtype)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size * num_samples,), t, 
                                device=self.device, dtype=torch.long)
            
            # Predict noise
            noise_pred = self.denoising_net(x, t_batch, condition)
            
            # Get alpha values
            alpha_cumprod_t = self.noise_scheduler.alphas_cumprod[t]
            
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
                alpha_cumprod_t_prev = self.noise_scheduler.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_t_prev = torch.tensor(1.0, device=self.device, dtype=self.dtype)
            
            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
            
            # Direction pointing to x_t
            dir_xt = torch.sqrt(1 - alpha_cumprod_t_prev - eta**2 * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)) * noise_pred
            
            # Compute x_{t-1}
            x = torch.sqrt(alpha_cumprod_t_prev) * x_0_pred + dir_xt
            
            # Add stochastic noise
            if eta > 0 and i < len(timesteps) - 1:
                variance = eta**2 * (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev)
                noise = torch.randn_like(x)
                x = x + torch.sqrt(variance) * noise
        
        return x

