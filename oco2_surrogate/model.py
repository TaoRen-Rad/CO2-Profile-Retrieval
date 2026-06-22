import torch.nn as nn
import torch
from .diffusion import LatentDiffusion
from .loss import mask_nan

class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_dims=[256, 128],
        dropout_rate=0.3,
        activation=torch.nn.SiLU(),
        batch_norm=True,
        layer_norm=False,
        residual=False,          # 新增：是否启用残差
    ):
        super(MLP, self).__init__()

        assert not (batch_norm and layer_norm), "batch_norm 和 layer_norm 只能开一个"

        self.residual = residual
        self.activation = activation
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate

        # 把每个 hidden layer 做成一个 block，方便加残差
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()  # 残差分支（可能是 Identity 或 Linear）

        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers = []

            # 主分支的 Linear + Norm + Act + Dropout
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            elif batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            self.blocks.append(nn.Sequential(*layers))

            # 残差分支：如果维度不同，用 Linear 投影；否则 Identity
            if residual:
                if prev_dim == hidden_dim:
                    self.shortcuts.append(nn.Identity())
                else:
                    self.shortcuts.append(nn.Linear(prev_dim, hidden_dim))
            else:
                # 不用残差时占位，forward 里会忽略
                self.shortcuts.append(nn.Identity())

            prev_dim = hidden_dim

        # 输出层不加残差，保持原逻辑
        self.out = nn.Linear(prev_dim, output_dim)

    def forward(self, x):
        h = x
        for block, shortcut in zip(self.blocks, self.shortcuts):
            if self.residual:
                # 残差: h_{l+1} = F(h_l) + P(h_l)
                h = block(h) + shortcut(h)
            else:
                # 原始: h_{l+1} = F(h_l)
                h = block(h)

        return self.out(h)

class MLPOutputMasked(nn.Module):
    def __init__(self, input_dim, output_dim, nnan_indices, full_size, 
                 hidden_dims=[256, 128], dropout_rate=0.3, masked_value=0.0,
                 activation=torch.nn.SiLU(), batch_norm=True, layer_norm=False, residual=False):
        super(MLPOutputMasked, self).__init__()
        self.network = MLP(input_dim, output_dim, hidden_dims=hidden_dims, 
                           dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm, layer_norm=layer_norm, residual=residual)
        self.nnan_indices = nnan_indices
        self.full_size = full_size
        self.masked_value = masked_value
    
    def forward(self, x):
        y = torch.ones(x.shape[0], self.full_size, dtype=x.dtype, device=x.device) * self.masked_value
        y[:, self.nnan_indices] = self.network(x)
        return y

class MLPInputMasked(nn.Module):
    def __init__(self, input_dim, output_dim, nnan_indices, full_size, 
                 hidden_dims=[256, 128], dropout_rate=0.3,
                 activation=torch.nn.SiLU(), batch_norm=True, layer_norm=False, residual=False):
        super(MLPInputMasked, self).__init__()
        self.network = MLP(input_dim, output_dim, hidden_dims=hidden_dims, 
                           dropout_rate=dropout_rate, activation=activation, batch_norm=batch_norm, layer_norm=layer_norm, residual=residual)
        self.nnan_indices = nnan_indices
        self.full_size = full_size
    
    def forward(self, x):
        return self.network(x[:, self.nnan_indices])


class LatentEncoder(nn.Module):
    """
    Encoder that maps from input features to compressed latent space
    Architecture: (geo_dim + band_state_dim) -> not_nan_number -> latent_dim
    """
    
    def __init__(self, input_dim, intermediate_dim, latent_dim, 
                 dropout_rate=0.0, activation=nn.SiLU(), batch_norm=False):
        super(LatentEncoder, self).__init__()
        self.input_dim = input_dim
        self.intermediate_dim = intermediate_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, intermediate_dim),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, intermediate_dim),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, latent_dim),
        )
    
    def forward(self, geo, ret_band):
        """
        Args:
            geo: (B, geo_dim) - geometry features
            ret_band: (B, band_state_dim) - retrieved state for specific band
        Returns:
            (B, latent_dim) - latent representation
        """
        x = torch.cat([geo, ret_band], dim=1)
        return self.encoder(x)


class SpectralDecoder(nn.Module):
    """
    Decoder that maps from latent space to full radiance
    Architecture: latent_dim -> not_nan_number -> full radiance (with masking)
    Uses the bottleneck approach for better generalization
    """
    
    def __init__(self, latent_dim, intermediate_dim, nnan_indices, full_size,
                 dropout_rate=0.0, activation=nn.SiLU(), batch_norm=False, masked_value=0.0):
        super(SpectralDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.nnan_indices = nnan_indices
        self.full_size = full_size
        self.masked_value = masked_value
        
        # Decoder network
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, intermediate_dim),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, intermediate_dim),
            activation,
            nn.Dropout(dropout_rate),
            nn.Linear(intermediate_dim, len(nnan_indices)),
        )
    
    def forward(self, latent):
        """
        Args:
            latent: (B, latent_dim) - latent representation
        Returns:
            (B, full_size) - reconstructed radiance with masking
        """
        # Decode to non-NaN indices
        decoded = self.decoder(latent)
        
        # Create full output with masking
        batch_size = latent.shape[0]
        y = torch.ones(batch_size, self.full_size, dtype=latent.dtype, device=latent.device) * self.masked_value
        y[:, self.nnan_indices] = decoded
        
        return y


class ConditionalDiffusionModel(nn.Module):
    """
    Complete conditional diffusion model for spectral emulation
    Combines encoder, diffusion process, and decoder
    """
    
    def __init__(self, geo_dim, ret_dim, band_indices, nnan_indices, rad_dim=1017,
                 latent_dim=None, num_timesteps=200, beta_start=1e-4, beta_end=0.02,
                 schedule_type='linear', denoising_hidden_dims=None, time_emb_dim=128,
                 dropout_rate=0.0, activation=nn.SiLU(), batch_norm=False,
                 masked_value=0.0, device='cpu', dtype=torch.float32):
        super(ConditionalDiffusionModel, self).__init__()
        
        self.geo_dim = geo_dim
        self.ret_dim = ret_dim
        self.band_indices = band_indices
        self.nnan_indices = nnan_indices
        self.rad_dim = rad_dim
        self.device = device
        self.dtype = dtype
        
        # Calculate dimensions
        band_state_dim = len(band_indices)
        input_dim = geo_dim + band_state_dim
        intermediate_dim = len(nnan_indices)
        
        # Default latent dimension to intermediate_dim // 5 (bottleneck approach)
        if latent_dim is None:
            latent_dim = max(intermediate_dim // 5, 50)  # Minimum 50 to avoid too small
        self.latent_dim = latent_dim
        
        # Encoder: maps (geo + ret_band) to latent space
        self.encoder = LatentEncoder(
            input_dim=input_dim,
            intermediate_dim=intermediate_dim,
            latent_dim=latent_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm
        )
        
        # Diffusion model in latent space
        # Condition on input features for better control
        condition_dim = input_dim
        
        if denoising_hidden_dims is None:
            denoising_hidden_dims = [latent_dim*2, latent_dim*4, latent_dim*4, latent_dim*2]
        
        self.diffusion = LatentDiffusion(
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            num_timesteps=num_timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type,
            denoising_hidden_dims=denoising_hidden_dims,
            time_emb_dim=time_emb_dim,
            dropout_rate=dropout_rate,
            device=device,
            dtype=dtype
        )
        
        # Decoder: maps latent space to full radiance
        self.decoder = SpectralDecoder(
            latent_dim=latent_dim,
            intermediate_dim=intermediate_dim,
            nnan_indices=nnan_indices,
            full_size=rad_dim,
            dropout_rate=dropout_rate,
            activation=activation,
            batch_norm=batch_norm,
            masked_value=masked_value
        )
        # In ConditionalDiffusionModel.__init__, add:
        self.radiance_projector = nn.Linear(
            len(nnan_indices), latent_dim,
            device=device, dtype=dtype
        )

    # Then in encode_radiance, change to:
    def encode_radiance(self, radiance):
        """Encode radiance to latent space (for training)"""
        radiance_valid = radiance[:, self.nnan_indices]
        return self.radiance_projector(radiance_valid)
    
    def forward(self, geo, ret, radiance=None, mode='train'):
        """
        Forward pass
        Args:
            geo: (B, geo_dim) - geometry features
            ret: (B, ret_dim) - retrieved state features
            radiance: (B, rad_dim) - target radiance (required for training)
            mode: 'train' or 'sample'
        Returns:
            For training: (noise_pred, noise, timesteps, condition, latent_target)
            For sampling: (B, rad_dim) - predicted radiance
        """
        # Extract band-specific retrieved state
        ret_band = ret[:, self.band_indices]
        
        # Prepare conditioning
        condition = torch.cat([geo, ret_band], dim=1)
        
        if mode == 'train':
            if radiance is None:
                raise ValueError("radiance must be provided for training mode")
            radiance, _ = mask_nan(radiance, radiance)
            
            # Encode radiance to latent space
            latent_target = self.encode_radiance(radiance)
            
            # Forward through diffusion (returns noise prediction for loss)
            noise_pred, noise, timesteps = self.diffusion(latent_target, condition)
            
            return noise_pred, noise, timesteps, condition, latent_target
        
        elif mode == 'sample':
            # Sample from diffusion model
            latent_samples = self.diffusion.sample_ddim(
                condition, num_inference_steps=20, eta=0.0, num_samples=1
            )
            
            # Decode to radiance
            radiance_pred = self.decoder(latent_samples)
            
            return radiance_pred
        
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    @torch.no_grad()
    def sample(self, geo, ret, num_inference_steps=20, eta=0.0, num_samples=1):
        """
        Generate samples using DDIM sampling
        Args:
            geo: (B, geo_dim) - geometry features
            ret: (B, ret_dim) - retrieved state features
            num_inference_steps: number of denoising steps
            eta: stochasticity (0=deterministic)
            num_samples: number of samples per input
        Returns:
            (B*num_samples, rad_dim) - predicted radiance
        """
        # Extract band-specific retrieved state
        ret_band = ret[:, self.band_indices]
        
        # Prepare conditioning
        condition = torch.cat([geo, ret_band], dim=1)
        
        # Sample from diffusion model
        latent_samples = self.diffusion.sample_ddim(
            condition, num_inference_steps=num_inference_steps, 
            eta=eta, num_samples=num_samples
        )
        
        # Decode to radiance
        radiance_pred = self.decoder(latent_samples)
        
        return radiance_pred
