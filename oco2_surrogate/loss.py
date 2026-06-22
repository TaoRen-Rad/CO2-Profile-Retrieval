import torch
import torch.nn as nn

def mask_nan(predictions, targets):
    # Create boolean mask for valid (non-NaN) values
    valid_mask = (~torch.isnan(targets) & ~torch.isnan(predictions))
    
    # Convert NaN to 0, then apply mask
    predictions = torch.nan_to_num(predictions, nan=0.0) * valid_mask
    targets = torch.nan_to_num(targets, nan=0.0) * valid_mask
    return predictions, targets
    # pred_mask = torch.isnan(predictions)
    # targ_mask = torch.isnan(targets)
    
    # predictions = torch.where(pred_mask | targ_mask, torch.zeros_like(predictions), predictions)
    # targets = torch.where(pred_mask | targ_mask, torch.zeros_like(targets), targets)
    # return predictions, targets


class MSENaNMaskLoss(nn.Module):
    def __init__(self, dtype=torch.float32):
        super(MSENaNMaskLoss, self).__init__()
        self.dtype = dtype
    
    def forward(self, predictions, targets):
        predictions, targets = mask_nan(predictions, targets)
        return torch.mean((predictions - targets) ** 2)

class CosSimNormLoss(nn.Module):
    def __init__(self, eps=1e-8, cos_weight=1.0, norm_weight=1.0, dtype=torch.float32):
        super(CosSimNormLoss, self).__init__()
        self.eps = eps
        self.cos_weight = cos_weight
        self.norm_weight = norm_weight
        self.dtype = dtype
        # Gaussian kernel parameter: sigma = sqrt(2 * 1016)
        self.sigma = torch.sqrt(torch.tensor(2 * 1016 * 0.01, dtype=dtype))
    
    def forward(self, predictions, targets):
        pred_shape_norm = torch.nn.functional.normalize(predictions, p=2, dim=1, eps=self.eps)
        target_shape_norm = torch.nn.functional.normalize(targets, p=2, dim=1, eps=self.eps)
        
        # Cosine similarity (higher is better, so we use 1 - cosine_sim as loss)
        cosine_sim = torch.sum(pred_shape_norm * target_shape_norm, dim=1)
        cosine_loss = torch.mean(1.0 - cosine_sim)
        
        # predictions, targets: (B, D)
        diff = predictions - targets                 # (B, D)
        sq = (diff ** 2).sum(dim=1)                   # (B,)  每个样本的 ||diff||^2
        gauss = torch.exp(- sq / (2.0 * (self.sigma ** 2)))  # (B,)
        norm_loss = torch.mean(1.0 - gauss)           # scalar

        return self.cos_weight * cosine_loss + self.norm_weight * norm_loss

class SingleBandLoss(nn.Module):
    """Single band loss function:
    1. MSE loss for scalar (first dimension)
    2. Cosine similarity + norm loss for spectral shape (remaining dimensions)
    """
    
    def __init__(self, reduction='mean', eps=1e-8, cos_weight=1.0, norm_weight=1.0, 
                 scalar_weight=1.0, shape_weight=1.0):
        super(SingleBandLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.cos_loss_func = CosSimNormLoss(eps=eps, cos_weight=cos_weight, norm_weight=norm_weight)
        self.scalar_weight = scalar_weight
        self.shape_weight = shape_weight
    
    def forward(self, predictions, targets):
        # Handle NaN values
        predictions, targets = mask_nan(predictions, targets)
        
        # Scalar loss (first column)
        scalar_loss = torch.mean((predictions[:, 0] - targets[:, 0]) ** 2)
        
        # Shape loss (remaining columns)
        shape_loss = self.cos_loss_func(predictions[:, 1:], targets[:, 1:])
        return self.scalar_weight * scalar_loss + self.shape_weight * shape_loss


class NewSingleBandLoss(nn.Module):
    """Single band loss function:
    1. MSE loss for scalar (first dimension)
    2. Cosine similarity + norm loss for spectral shape (remaining dimensions)
    """
    
    def __init__(self, reduction='mean', eps=1e-8, cos_weight=1.0, norm_weight=1.0):
        super(NewSingleBandLoss, self).__init__()
        self.reduction = reduction
        self.eps = eps
        self.cos_loss_func = CosSimNormLoss(eps=eps, cos_weight=cos_weight, norm_weight=norm_weight)
    
    def forward(self, predictions, targets):
        # Handle NaN values
        predictions, targets = mask_nan(predictions, targets)
        return self.cos_loss_func(predictions, targets)


class ThreeBandLoss(nn.Module):
    def __init__(self, eps=1e-8, cos_weight=1.0, norm_weight=1.0, band_weights=[1.0, 1.0, 1.0], dtype=torch.float32):
        super(ThreeBandLoss, self).__init__()
        self.eps = eps
        self.cos_weight = cos_weight
        self.norm_weight = norm_weight
        self.dtype = dtype
        self.cos_loss_func = CosSimNormLoss(eps=eps, cos_weight=cos_weight, norm_weight=norm_weight)
        self.band_weights = band_weights
    
    def forward(self, predictions, targets):
        predictions, targets = mask_nan(predictions, targets)
        loss = 0.0
        for i in range(3):
            loss += self.band_weights[i] * self.cos_loss_func(predictions[:, i*1016:(i+1)*1016], targets[:, i*1016:(i+1)*1016])
        return loss

class ScaledThreeBandLoss(nn.Module):
    def __init__(self, eps=1e-8, cos_weight=100.0, norm_weight=1.0, band_weights=[1.0, 1.0, 1.0], dtype=torch.float32):
        super(ScaledThreeBandLoss, self).__init__()
        self.eps = eps
        self.cos_weight = cos_weight
        self.norm_weight = norm_weight
        self.dtype = dtype
        self.cos_loss_func = CosSimNormLoss(eps=eps, cos_weight=cos_weight, norm_weight=norm_weight)
        self.band_weights = band_weights
    
    def forward(self, predictions, targets):
        predictions, targets = mask_nan(predictions, targets)
        loss = 0.0
        for i in range(3):
            pred = predictions[:, i*1016:(i+1)*1016]
            tar = targets[:, i*1016:(i+1)*1016]
            row_max = torch.max(pred, dim=1, keepdim=True)[0]
            pred = pred / row_max
            tar = tar / row_max
            loss += self.band_weights[i] * self.cos_loss_func(pred, tar)
        return loss


class DiffusionLoss(nn.Module):
    """
    Simple L2 loss for diffusion model in latent space
    Measures the difference between predicted and actual noise
    """
    
    def __init__(self, reduction='mean'):
        super(DiffusionLoss, self).__init__()
        self.reduction = reduction
    
    def forward(self, noise_pred, noise_target):
        """
        Args:
            noise_pred: (B, latent_dim) - predicted noise
            noise_target: (B, latent_dim) - actual noise added
        Returns:
            scalar loss
        """
        loss = torch.nn.functional.mse_loss(noise_pred, noise_target, reduction=self.reduction)
        return loss


class LatentReconstructionLoss(nn.Module):
    """
    Ensures encoder-decoder consistency by checking if we can reconstruct
    the radiance from its latent representation
    """
    
    def __init__(self, scalar_weight=10.0, shape_weight=1.0, eps=1e-8):
        super(LatentReconstructionLoss, self).__init__()
        self.scalar_weight = scalar_weight
        self.shape_weight = shape_weight
        self.eps = eps
        self.cos_loss_func = CosSimNormLoss(eps=eps, cos_weight=1.0, norm_weight=1.0)
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, rad_dim) - reconstructed radiance
            targets: (B, rad_dim) - original radiance
        Returns:
            scalar loss
        """
        # Handle NaN values
        predictions, targets = mask_nan(predictions, targets)
        
        # Scalar loss (first column - log signal)
        scalar_loss = torch.mean((predictions[:, 0] - targets[:, 0]) ** 2)
        
        # Shape loss (remaining columns - normalized spectral)
        shape_loss = self.cos_loss_func(predictions[:, 1:], targets[:, 1:])
        
        return self.scalar_weight * scalar_loss + self.shape_weight * shape_loss


class SpectralConsistencyLoss(nn.Module):
    """
    Maintains physical properties during diffusion
    Ensures spectral shapes remain physically plausible
    """
    
    def __init__(self, smoothness_weight=0.1, positivity_weight=0.1):
        super(SpectralConsistencyLoss, self).__init__()
        self.smoothness_weight = smoothness_weight
        self.positivity_weight = positivity_weight
    
    def forward(self, predictions):
        """
        Args:
            predictions: (B, rad_dim) - predicted radiance
        Returns:
            scalar loss
        """
        # Spectral smoothness: penalize large differences between adjacent wavelengths
        # Skip first column (scalar), only apply to spectral part
        spectral = predictions[:, 1:]
        
        # Compute differences between adjacent wavelengths
        diff = spectral[:, 1:] - spectral[:, :-1]
        smoothness_loss = torch.mean(diff ** 2)
        
        # Positivity: encourage positive values in spectral part
        # Since we're working with normalized radiance, this is less critical
        # but helps maintain physical plausibility
        positivity_loss = torch.mean(torch.relu(-spectral))
        
        return self.smoothness_weight * smoothness_loss + self.positivity_weight * positivity_loss


class CombinedDiffusionLoss(nn.Module):
    """
    Combined loss for diffusion model training
    Balances diffusion loss, reconstruction loss, and physical consistency
    """
    
    def __init__(self, diffusion_weight=1.0, reconstruction_weight=0.1, 
                 consistency_weight=0.01, scalar_weight=10.0, shape_weight=1.0):
        super(CombinedDiffusionLoss, self).__init__()
        self.diffusion_weight = diffusion_weight
        self.reconstruction_weight = reconstruction_weight
        self.consistency_weight = consistency_weight
        
        self.diffusion_loss = DiffusionLoss()
        self.reconstruction_loss = LatentReconstructionLoss(
            scalar_weight=scalar_weight, shape_weight=shape_weight
        )
        self.consistency_loss = SpectralConsistencyLoss()
    
    def forward(self, noise_pred, noise_target, radiance_pred=None, radiance_target=None):
        """
        Args:
            noise_pred: (B, latent_dim) - predicted noise
            noise_target: (B, latent_dim) - actual noise
            radiance_pred: (B, rad_dim) - reconstructed radiance (optional)
            radiance_target: (B, rad_dim) - target radiance (optional)
        Returns:
            total_loss, loss_dict
        """
        # Diffusion loss (primary)
        diff_loss = self.diffusion_loss(noise_pred, noise_target)
        total_loss = self.diffusion_weight * diff_loss
        
        loss_dict = {'diffusion': diff_loss.item()}
        
        # Reconstruction loss (if radiance provided)
        if radiance_pred is not None and radiance_target is not None:
            radiance_pred, radiance_target = mask_nan(radiance_pred, radiance_target)
            recon_loss = self.reconstruction_loss(radiance_pred, radiance_target)
            total_loss += self.reconstruction_weight * recon_loss
            loss_dict['reconstruction'] = recon_loss.item()
        
        # Consistency loss (if radiance predicted)
        if radiance_pred is not None:
            cons_loss = self.consistency_loss(radiance_pred)
            total_loss += self.consistency_weight * cons_loss
            loss_dict['consistency'] = cons_loss.item()
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict