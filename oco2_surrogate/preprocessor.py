import torch
import torch.nn as nn

class StandardScaler(nn.Module):
    def __init__(self):
        super(StandardScaler, self).__init__()

    def fit(self, X):
        # Pure PyTorch implementation with NaN handling
        # Create mask for non-NaN values
        mask = ~torch.isnan(X)
        
        # Calculate mean for each feature
        sum_vals = torch.where(mask, X, torch.zeros_like(X)).sum(dim=0)
        count_vals = mask.sum(dim=0).float()
        mean = sum_vals / torch.clamp(count_vals, min=1)  # Avoid division by zero
        
        # Calculate standard deviation for each feature
        # Broadcast mean to match X shape for subtraction
        diff = X - mean.unsqueeze(0)
        diff_squared = torch.where(mask, diff * diff, torch.zeros_like(diff))
        var = diff_squared.sum(dim=0) / torch.clamp(count_vals - 1, min=1)  # Sample std (ddof=1)
        scale = torch.sqrt(var)
        
        # Handle case where std is 0 (constant features)
        scale = torch.where(scale > 0, scale, torch.ones_like(scale))
        
        # Register as buffers (handle re-fitting case)
        if hasattr(self, 'mean_') and self.mean_ is not None:
            # Update existing buffers
            self.mean_.data = mean
            self.scale_.data = scale
        else:
            # First time: register as buffers
            self.register_buffer('mean_', mean)
            self.register_buffer('scale_', scale)
        
        return self
    
    def transform(self, X):
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled
    
    def inverse_transform(self, X):
        X_unscaled = X * self.scale_ + self.mean_
        return X_unscaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

class InterStandardScaler(nn.Module):
    """StandardScaler with interpolation for sparse features"""
    def __init__(self):
        super(InterStandardScaler, self).__init__()
        self.mean_ = None
        self.scale_ = None
        self.count_vals = None
    
    def fit(self, X):
        # Pure PyTorch implementation with NaN handling
        # Create mask for non-NaN values
        mask = ~torch.isnan(X)
        
        # Calculate mean for each feature
        sum_vals = torch.where(mask, X, torch.zeros_like(X)).sum(dim=0)
        count_vals = mask.sum(dim=0).float()
        self.count_vals = count_vals  # Save for potential debugging
        self.mean_ = sum_vals / torch.clamp(count_vals, min=1)  # Avoid division by zero
        
        # Calculate standard deviation for each feature
        # Broadcast mean to match X shape for subtraction
        diff = X - self.mean_.unsqueeze(0)
        diff_squared = torch.where(mask, diff * diff, torch.zeros_like(diff))
        var = diff_squared.sum(dim=0) / torch.clamp(count_vals - 1, min=1)  # Sample std (ddof=1)
        self.scale_ = torch.sqrt(var)
        
        # Handle case where std is 0 (constant features)
        self.scale_ = torch.where(self.scale_ > 0, self.scale_, torch.ones_like(self.scale_))
        
        # Post-processing: interpolate sparse positions
        n_samples = len(X)
        threshold = n_samples * 0.25
        
        # Define positions that need interpolation and valid reference positions
        needs_interp = (count_vals > 0) & (count_vals < threshold)
        valid_for_interp = count_vals >= threshold
        
        # Interpolate if needed
        if needs_interp.any() and valid_for_interp.any():
            # Get indices
            all_indices = torch.arange(len(count_vals), dtype=torch.float32, device=X.device)
            valid_indices = all_indices[valid_for_interp]
            interp_indices = all_indices[needs_interp]
            
            # Interpolate mean_
            valid_means = self.mean_[valid_for_interp]
            self.mean_[needs_interp] = self._interpolate_1d(
                interp_indices, valid_indices, valid_means
            )
            
            # Interpolate scale_
            valid_scales = self.scale_[valid_for_interp]
            self.scale_[needs_interp] = self._interpolate_1d(
                interp_indices, valid_indices, valid_scales
            )
        
        return self
    
    def _interpolate_1d(self, target_indices, reference_indices, reference_values):
        """
        Perform 1D linear interpolation/extrapolation
        
        Args:
            target_indices: indices where we need values
            reference_indices: indices where we have valid values
            reference_values: valid values at reference_indices
        
        Returns:
            interpolated values at target_indices
        """
        # Sort reference data by index
        sorted_idx = torch.argsort(reference_indices)
        ref_x = reference_indices[sorted_idx]
        ref_y = reference_values[sorted_idx]
        
        # For each target index, find neighboring reference points and interpolate
        interpolated = torch.zeros_like(target_indices)
        
        for i, target_idx in enumerate(target_indices):
            # Find the position where target_idx would be inserted
            # This gives us the right neighbor
            right_pos = torch.searchsorted(ref_x, target_idx)
            
            if right_pos == 0:
                # Extrapolate to the left using first two points
                if len(ref_x) >= 2:
                    slope = (ref_y[1] - ref_y[0]) / (ref_x[1] - ref_x[0])
                    interpolated[i] = ref_y[0] + slope * (target_idx - ref_x[0])
                else:
                    interpolated[i] = ref_y[0]
            elif right_pos == len(ref_x):
                # Extrapolate to the right using last two points
                if len(ref_x) >= 2:
                    slope = (ref_y[-1] - ref_y[-2]) / (ref_x[-1] - ref_x[-2])
                    interpolated[i] = ref_y[-1] + slope * (target_idx - ref_x[-1])
                else:
                    interpolated[i] = ref_y[-1]
            else:
                # Interpolate between two points
                left_pos = right_pos - 1
                x0, x1 = ref_x[left_pos], ref_x[right_pos]
                y0, y1 = ref_y[left_pos], ref_y[right_pos]
                
                if x1 - x0 > 0:
                    # Linear interpolation
                    t = (target_idx - x0) / (x1 - x0)
                    interpolated[i] = y0 + t * (y1 - y0)
                else:
                    interpolated[i] = y0
        
        return interpolated
    
    def transform(self, X):
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled
    
    def inverse_transform(self, X):
        X_unscaled = X * self.scale_ + self.mean_
        return X_unscaled
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

def convert_y(y_raw):
    """Convert y for single band - divide spectral data by scalar signal"""
    # y_raw shape: [N, 1017] where first column is scalar signal, rest are spectral data
    y_raw[:, 1:] = y_raw[:, 1:] / y_raw[:, 0:1]  # Divide spectral by scalar
    return y_raw

def unconvert_y(y_converted):
    """Unconvert y for single band - multiply spectral data by scalar signal"""
    # Return only the spectral part (exclude scalar)
    return y_converted[:, 1:] * y_converted[:, 0:1]