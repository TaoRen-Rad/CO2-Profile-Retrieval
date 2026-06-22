# %% [markdown]
# # Single Band MLP Training for OCO-2 Data Correlation Learning
# 
# This script trains an MLP using PyTorch to learn the correlation between input data and a single band of output data.
# Usage: python train_single_band.py --band o2|weak_co2|strong_co2
# 

# %%
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
import os
import argparse
import copy
# from tqdm.auto import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from numba import jit
import time
import pickle
import pandas as pd
from aim import Run
from oco2_surrogate.load_data import load_retrieved_data, get_state_indices
from oco2_surrogate.preprocessor import StandardScaler
from oco2_surrogate.loss import CosSimNormLoss, mask_nan
from oco2_surrogate.model import MLPOutputMasked
from oco2_surrogate import RAND_SEED
from oco2_surrogate.plot import forward_plot
from oco2_surrogate.absc import AbsorptionCoefficientModule, sh2vmr, vmr2sh
from oco2_surrogate.release import make_forward_smoke_datasets
# from LOSTorch.predict import predict as torch_los_predict

NCOLS = 120

import torch
import torch.nn as nn

class MixRelCosSimNormLoss(nn.Module):
    """
    混合损失：
      1) 光谱形状：CosSimNormLoss（在标准化空间上）
      2) 光谱幅度：物理空间上的 (绝对 + 相对) MSE
    
    predictions / targets: 标准化后的辐射 (radiance_scaler 标准化出来的)
    mean, scale: 标准化用的 mean/scale，shape = [C]
    """
    def __init__(
        self,
        mean,
        scale,
        rel_weight = 1.0,
        alpha=0.5,          # 绝对误差 vs 相对误差 权重
        cos_weight=1.0,
        norm_weight=1.0,
        eps_rel=1e-3
    ):
        super().__init__()

        mean = torch.as_tensor(mean, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32)
        # 注册为 buffer，自动跟模型一起搬到 device / 存ckpt
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale)
        self.rel_weight = rel_weight

        self.alpha = alpha
        self.eps_rel = eps_rel
        self.cos_loss_func = CosSimNormLoss(
            eps=1e-8, cos_weight=cos_weight, norm_weight=norm_weight
        )

    def forward(self, predictions, targets):
        # 统一 NaN mask
        predictions, targets = mask_nan(predictions, targets)

        # 1) 形状部分：仍在标准化空间上做 cos+norm
        cos_loss = self.cos_loss_func(predictions, targets)

        # 2) 幅度部分：反标准化到物理空间
        mean = self.mean.to(predictions.device)   # [C]
        scale = self.scale.to(predictions.device) # [C]

        preds_phys = predictions * scale + mean   # [B, C]
        targs_phys = targets * scale + mean       # [B, C]

        diff = preds_phys - targs_phys

        # 绝对误差 MSE
        abs_loss = (diff ** 2).mean()

        # 相对误差 MSE: ((ŷ - y)/( |y| + eps ))^2
        denom = targs_phys.abs() + self.eps_rel
        rel = diff / denom
        rel_loss = (rel ** 2).mean()

        amp_loss = self.alpha * abs_loss + (1.0 - self.alpha) * rel_loss

        return cos_loss + self.rel_weight * amp_loss

# Training functions
def train_epoch(model, train_loader, criterion, optimizer, scheduler, device, 
        epoch=None, aim_run=None, dtype=None, max_norm=0.5):
    model.train()
    total_loss = 0.0
    total_n = 0
    num_batches = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1, ncols=NCOLS) as batch_pbar:
        for batch_geo, batch_ret, batch_los, batch_rad in batch_pbar:
            optimizer.zero_grad()
            batch_geo = batch_geo.to(device, dtype=dtype, non_blocking=True)
            batch_ret = batch_ret.to(device, dtype=dtype, non_blocking=True)
            batch_los = batch_los.to(device, dtype=dtype, non_blocking=True)
            batch_cur_rad = batch_rad.to(device, dtype=dtype, non_blocking=True)  # No slicing needed - already sliced in dataset
            
            predictions = model(batch_geo, batch_ret, batch_los)
            loss = criterion(predictions, batch_cur_rad)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()
            
            # Update scheduler per batch
            scheduler.step()
            
            total_loss += loss.item() * len(batch_cur_rad)
            total_n += len(batch_cur_rad)
            num_batches += 1
            
            current_lr = optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({
                'Loss': f'{(total_loss/total_n):.6f}',
                'LR': f'{current_lr:.2e}'
            })
    
    avg_loss = total_loss / total_n
    
    # Log epoch average loss and learning rate to Aim (only per epoch, not per batch)
    if aim_run is not None and epoch is not None:
        aim_run.track(avg_loss, name='epoch_loss', step=epoch, context={'subset': 'train'})
        aim_run.track(current_lr, name='learning_rate', step=epoch)
    
    return avg_loss

def evaluate_model(model, test_loader, criterion, device, epoch=None, 
                   return_predictions=False, aim_run=None, subset='val', dtype=None):
    model.eval()
    total_loss = 0.0
    total_n = 0
    num_batches = 0
    
    if return_predictions:
        all_predictions = []
        all_targets = []
    
    with torch.no_grad():
        for batch_geo, batch_ret, batch_los, batch_rad in test_loader:
            batch_geo = batch_geo.to(device, dtype=dtype)
            batch_ret = batch_ret.to(device, dtype=dtype)
            batch_los = batch_los.to(device, dtype=dtype)
            batch_cur_rad = batch_rad.to(device, dtype=dtype)  # No slicing needed - already sliced in dataset
            predictions = model(batch_geo, batch_ret, batch_los)
            loss = criterion(predictions, batch_cur_rad)
            total_loss += loss.item() * len(batch_cur_rad)
            total_n += len(batch_cur_rad)
            num_batches += 1
            if return_predictions:
                all_predictions.append(predictions)
                all_targets.append(batch_cur_rad)
    
    avg_loss = total_loss / total_n
    
    # Log epoch average loss to Aim
    if aim_run is not None and epoch is not None:
        aim_run.track(avg_loss, name='epoch_loss', step=epoch, context={'subset': subset})
    
    if return_predictions:
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return avg_loss, predictions, targets
    else:
        return avg_loss


def get_nnan_indices(rad_tensor):
    full_nan_columns = torch.isnan(rad_tensor).all(dim=0)
    nnan_indices = (torch.arange(rad_tensor.shape[1])[~full_nan_columns]).tolist()
    return nnan_indices

def load_scalers(scalers_path):
    """Load scalers with backward compatibility
    
    Args:
        scalers_path: Path to scalers file
        
    Returns:
        Dictionary containing StandardScaler objects
    """
    loaded = torch.load(scalers_path, map_location='cpu')
    
    # Check if loaded data is already StandardScaler objects (old format)
    if isinstance(loaded.get("geometry"), StandardScaler):
        return loaded
    
    # Otherwise, it's state_dicts (new format), reconstruct scalers
    scalers = {
        "geometry": StandardScaler(),
        "retrieved": StandardScaler(),
        "radiance": StandardScaler()
    }
    # Manually assign buffers to avoid None initialization issues
    scalers["geometry"].register_buffer('mean_', loaded["geometry"]["mean_"])
    scalers["geometry"].register_buffer('scale_', loaded["geometry"]["scale_"])
    scalers["retrieved"].register_buffer('mean_', loaded["retrieved"]["mean_"])
    scalers["retrieved"].register_buffer('scale_', loaded["retrieved"]["scale_"])
    scalers["radiance"].register_buffer('mean_', loaded["radiance"]["mean_"])
    scalers["radiance"].register_buffer('scale_', loaded["radiance"]["scale_"])
    return scalers

def training_data(fraction, dtype):
    bands = ["o2", "weak_co2", "strong_co2"]
    print("Loading training data...")
    yymms = [f'{yy}{mm:02d}' for yy in range(17, 20) for mm in range(1, 13)]
    yymms.pop(7)  # Remove one month
    # yymms = ["1701"]
    df_geometry, df_retrieved, _, _, df_los, df_radiance = load_retrieved_data(yymms, 
        fraction=fraction, load_modeled=True, filter_outcome_flag=False, 
        filter_asia=False)
    
    all_state_indices = {
        band: get_state_indices(df_retrieved.columns, band)
        for band in bands
    }
    
    # df_los = torch_los_predict(df_geometry, df_retrieved, device=device, batch_size=64)
    
    
    train_geo_raw = torch.from_numpy(df_geometry.values).to(dtype)
    del df_geometry
    train_ret_raw = torch.from_numpy(df_retrieved.values).to(dtype)
    del df_retrieved
    train_los_raw = torch.from_numpy(df_los.values).to(dtype)
    del df_los
    train_rad_raw = torch.from_numpy(df_radiance.values).to(dtype)
    del df_radiance
    
    all_nnan_indices = {}
    for i, band in enumerate(bands):
        idx = [i * 1016 + j for j in range(1016)]
        all_nnan_indices[band] = get_nnan_indices(train_rad_raw[:, idx])
    
    geometry_scaler = StandardScaler()
    retrieved_scaler = StandardScaler()
    radiance_scaler = StandardScaler()
    train_geo_scaled = geometry_scaler.fit_transform(train_geo_raw)
    del train_geo_raw
    train_ret_scaled = retrieved_scaler.fit_transform(train_ret_raw)
    del train_ret_raw
    train_rad_scaled = radiance_scaler.fit_transform(train_rad_raw)
    del train_rad_raw
    train_los_scaled = radiance_scaler.transform(train_los_raw)
    del train_los_raw
    
    
    print("Loading test data...")
    yymms = [f'23{mm:02d}' for mm in range(1, 13)]
    # yymms = ["2001"]
    df_geometry, df_retrieved, _, _, df_los, df_radiance = load_retrieved_data(yymms, 
        fraction=0.05, load_modeled=True, filter_outcome_flag=False, 
        filter_asia=False)

    # df_los = torch_los_predict(df_geometry, df_retrieved, device=device, batch_size=128)
    test_geo_scaled = geometry_scaler.transform(torch.from_numpy(df_geometry.values).to(dtype))
    del df_geometry
    test_ret_scaled = retrieved_scaler.transform(torch.from_numpy(df_retrieved.values).to(dtype))
    df_retrieved_columns = df_retrieved.columns.values
    del df_retrieved
    test_los_scaled = radiance_scaler.transform(torch.from_numpy(df_los.values).to(dtype))
    del df_los
    test_rad_scaled = radiance_scaler.transform(torch.from_numpy(df_radiance.values).to(dtype))
    del df_radiance
    
    full_train_dataset = TensorDataset(train_geo_scaled, train_ret_scaled, train_los_scaled, train_rad_scaled)
    test_dataset = TensorDataset(test_geo_scaled, test_ret_scaled, test_los_scaled, test_rad_scaled)
    scalers = {
        "geometry": geometry_scaler,
        "retrieved": retrieved_scaler,
        "radiance": radiance_scaler
    }
    indices = {
        "all_state_indices": all_state_indices,
        "all_nnan_indices": all_nnan_indices
    }
    
    return full_train_dataset, test_dataset, indices, scalers, df_retrieved_columns

def band_inverse(predictions, targets, scaler):
    # Note: scaler has already been sliced to match the current band's radiance data
    mean_ = scaler.mean_.to(predictions.device)
    scale_ = scaler.scale_.to(predictions.device)
    predictions = predictions * scale_ + mean_
    targets = targets * scale_ + mean_
    return predictions, targets

class ForwardModel(nn.Module):
    def __init__(self, band_indices, geo_dim, ret_dim, rad_dim, nnan_indices, 
                 hidden_dims=[256, 128], dropout_rate=0.3, rad_los_diff_mean=None, rad_los_diff_std=None):
        super(ForwardModel, self).__init__()
        self.band_indices = band_indices
        self.geo_dim = geo_dim
        self.ret_dim = ret_dim
        self.rad_dim = rad_dim
        self.nnan_indices = nnan_indices
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        # Save mean and std as buffers, so they move with model.to(device)
        self.register_buffer('rad_los_diff_mean', rad_los_diff_mean)
        self.register_buffer('rad_los_diff_std', rad_los_diff_std)
        
        # Initialize preprocessor scalers as submodules for end-to-end training
        self.geometry_scaler = StandardScaler()
        self.retrieved_scaler = StandardScaler()
        self.radiance_scaler = StandardScaler()
        
        los_embed_dim = 128
        MLP_kwargs = {
            "input_dim": los_embed_dim + geo_dim + len(band_indices),
            "output_dim": len(nnan_indices),
            "nnan_indices": nnan_indices,
            "full_size": rad_dim,
            "hidden_dims": hidden_dims,
            "dropout_rate": dropout_rate,
            "masked_value": 0.0,
            "activation": torch.nn.SiLU(),
            "batch_norm": False,
            "layer_norm": False,
            "residual": True
        }
        self.network1 = MLPOutputMasked(**MLP_kwargs)
        self.network2 = MLPOutputMasked(**MLP_kwargs)
        self.scale_amp = 0.3
        self.los_encoder = nn.Linear(rad_dim, los_embed_dim)
        # nn.Sequential(
        #     nn.Linear(rad_dim, 256),
        #     nn.SiLU(),
        #     nn.Linear(256, los_embed_dim),
        # )
        # self.network_scale = MLPOutputMasked(**MLP_kwargs)
        # self.network_bias = MLPOutputMasked(**MLP_kwargs)
    
    def forward(self, geo, ret, los):
        x = torch.cat([geo, ret[:, self.band_indices]], dim=1)
        los_feat = self.los_encoder(los)  # [B, d]
        x1 = torch.cat([x, los_feat], dim=1)
        # 1) 乘性项：限制在 1 附近，避免 OOD 发散/塌缩
        scale = 1.0 + self.scale_amp * torch.tanh(self.network1(x1))  # [B, C]

        # 2) 加性残差：让它弱依赖 LOS（通过低维 embedding）

        delta = self.network2(x1) * self.rad_los_diff_std + self.rad_los_diff_mean  # [B, C]

        return scale * los + delta
    
    def predict(self, geo_raw, ret_raw, los_raw):
        """End-to-end prediction with preprocessing included"""
        geo_scaled = self.geometry_scaler.transform(geo_raw)
        ret_scaled = self.retrieved_scaler.transform(ret_raw)
        los_scaled = self.radiance_scaler.transform(los_raw)
        rad_scaled = self.forward(geo_scaled, ret_scaled, los_scaled)
        rad_raw = self.radiance_scaler.inverse_transform(rad_scaled)
        return rad_raw
        

def parse_args():
    parser = argparse.ArgumentParser(description='Train single band OCO-2 MLP')
    parser.add_argument('--band', type=str, required=True, 
                       choices=['o2', 'weak_co2', 'strong_co2'],
                       help='Band to train: o2, weak_co2, or strong_co2')
    parser.add_argument('--dry_run', action='store_true', default=False,
                       help='Whether to run a dry run')
    parser.add_argument('--sample_new', action='store_true', default=False,
                       help='Whether to run a dry run')
    parser.add_argument('--smoke_test', action='store_true',
                       help='Use tiny synthetic fixture data and CPU-safe training defaults.')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override the number of training epochs.')
    parser.add_argument('--max_rows', type=int, default=None,
                       help='Limit rows loaded in smoke-test mode.')
    parser.add_argument('--no_aim', action='store_true',
                       help='Disable Aim logging.')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory for checkpoints and plots.')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use, e.g. cpu or cuda.')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    band = args.band
    sample_new = args.sample_new
    fraction = 0.5
    dry_run = args.dry_run
    
    band2bandidx = {
        "o2": 0,
        "weak_co2": 1,
        "strong_co2": 2
    }
    
    bandidx = band2bandidx[band]
    # Setup
    plt.rcParams.update({"figure.dpi": 120})
    requested_device = args.device or ("cpu" if args.smoke_test else "cuda")
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    dtype = torch.float32
    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Training band: {band}")
    
    # Create output directory with band name
    base_name = __file__.split('/')[-1].split('.')[0] + f"_{band}"
    status_dir = args.output_dir or f'status/{base_name}'
    data_dir = "status/train_forward_o2"
    os.makedirs(status_dir, exist_ok=True)
    
    # Load training data (2017-2020)
    if args.smoke_test:
        full_train_dataset, test_dataset, indices, scalers, df_retrieved_columns = make_forward_smoke_datasets(
            max_rows=args.max_rows,
            dtype=dtype,
        )
        scaler_states = {
            "geometry": scalers["geometry"].state_dict(),
            "retrieved": scalers["retrieved"].state_dict(),
            "radiance": scalers["radiance"].state_dict()
        }
        torch.save(full_train_dataset, f"{status_dir}/full_train_dataset.pth")
        torch.save(test_dataset, f"{status_dir}/test_dataset.pth")
        torch.save(indices, f"{status_dir}/indices.pth")
        torch.save(scaler_states, f"{status_dir}/scalers.pth")
        torch.save(df_retrieved_columns, f"{status_dir}/df_retrieved_columns.pth")
    elif sample_new:
        full_train_dataset, test_dataset, indices, scalers, df_retrieved_columns = training_data(fraction, dtype)
        torch.save(full_train_dataset, f"{status_dir}/full_train_dataset.pth")
        torch.save(test_dataset, f"{status_dir}/test_dataset.pth")
        torch.save(indices, f"{status_dir}/indices.pth")
        # Save scaler state_dicts instead of whole objects
        scaler_states = {
            "geometry": scalers["geometry"].state_dict(),
            "retrieved": scalers["retrieved"].state_dict(),
            "radiance": scalers["radiance"].state_dict()
        }
        torch.save(scaler_states, f"{status_dir}/scalers.pth")
        torch.save(df_retrieved_columns, f"{status_dir}/df_retrieved_columns.pth")
    else:
        full_train_dataset = torch.load(f"{data_dir}/full_train_dataset.pth")
        test_dataset = torch.load(f"{data_dir}/test_dataset.pth")
        indices = torch.load(f"{data_dir}/indices.pth")
        # Load scalers with backward compatibility
        scalers = load_scalers(f"{data_dir}/scalers.pth")
        df_retrieved_columns = torch.load(f"{data_dir}/df_retrieved_columns.pth")
    
    
    band_indices = indices["all_state_indices"][band]
    specie_indices = []
    pressure_indices = []
    temperature_indices = []
    broadener_indices = []
    remaining_indices = []
    for i in range(len(band_indices)):
        column_name = df_retrieved_columns[band_indices[i]]
        if column_name.startswith("co2_profile_"):
            specie_indices.append(i)
        elif column_name.startswith("vector_pressure_levels_"):
            pressure_indices.append(i)
        elif column_name.startswith("temperature_profile_"):
            temperature_indices.append(i)
        elif column_name.startswith("h2o_profile_"):
            broadener_indices.append(i)
        else:
            remaining_indices.append(i)

    for name in df_retrieved_columns[band_indices]:
        print(name)
    # ===== Optimization: Pre-slice radiance data for current band =====
    print(f"Slicing radiance data for band {band}...")
    
    # Calculate radiance indices for current band
    rad_indices = [bandidx * 1016 + i for i in range(1016)]
    
    # Process training dataset
    train_geo = full_train_dataset.tensors[0]
    train_ret = full_train_dataset.tensors[1]
    train_los = full_train_dataset.tensors[2]
    train_rad = full_train_dataset.tensors[3]
    train_los = train_los[:, rad_indices].contiguous()
    train_rad = train_rad[:, rad_indices].contiguous()  # Slice and make contiguous
    
    rad_los_diff = train_rad - train_los
    # Compute mean and std, ignoring NaN values (places may contain nan)
    rad_los_diff_mean = torch.nanmean(rad_los_diff, dim=0)
    rad_los_diff_std = torch.from_numpy(np.nanstd(rad_los_diff.cpu().numpy(), axis=0)).to(rad_los_diff.device)

    full_train_dataset = TensorDataset(train_geo, train_ret, train_los, train_rad)
    
    # Process test dataset
    test_geo = test_dataset.tensors[0]
    test_ret = test_dataset.tensors[1]
    test_los = test_dataset.tensors[2]
    test_los = test_los[:, rad_indices].contiguous()
    test_rad = test_dataset.tensors[3]
    test_rad = test_rad[:, rad_indices].contiguous()  # Slice and make contiguous
    test_dataset = TensorDataset(test_geo, test_ret, test_los, test_rad)
    
    
    # Also slice the radiance scaler parameters to match the sliced data
    radiance_scaler = scalers["radiance"]
    radiance_scaler.mean_ = radiance_scaler.mean_[rad_indices]
    radiance_scaler.scale_ = radiance_scaler.scale_[rad_indices]
    print(f"Scaler parameters also sliced to match radiance shape")
    
    retrieved_scaler = scalers["retrieved"]
    ret_mean = retrieved_scaler.mean_[band_indices]
    ret_std = retrieved_scaler.scale_[band_indices]
    
    # Delete original large tensors to free memory
    del train_rad, test_rad
    # ===== End of radiance slicing =====
    
    total_size = len(full_train_dataset)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    torch.manual_seed(RAND_SEED)
    train_dataset, val_dataset = torch.utils.data.random_split(full_train_dataset, [train_size, val_size])
    
    # Delete full_train_dataset to free memory
    del full_train_dataset
    
    band_indices = indices["all_state_indices"][band]
    nnan_indices = indices["all_nnan_indices"][band]
    geo_dim = len(train_dataset[0][0])
    ret_dim = len(train_dataset[0][1])
    rad_dim = 1016
    
    ret_names = df_retrieved_columns[band_indices]
    specie_names = ret_names[specie_indices]
    pressure_names = ret_names[pressure_indices]
    temperature_names = ret_names[temperature_indices]
    broadener_names = ret_names[broadener_indices]
    remaining_names = ret_names[remaining_indices]
    print(f"specie_names: {specie_names}")
    print(f"pressure_names: {pressure_names}")
    print(f"temperature_names: {temperature_names}")
    print(f"broadener_names: {broadener_names}")
    print(f"remaining_names: {remaining_names}")
    
    not_nan_number = len(nnan_indices)
    # hidden_dims = [(geo_dim + ret_dim)//2] + [not_nan_number] * 4 + [not_nan_number//5]
    if args.smoke_test:
        hidden_dims = [32, 32]
    else:
        hidden_dims = [(geo_dim + ret_dim)//2] + [not_nan_number] * 4 + [not_nan_number//5]
    # hidden_dims = [not_nan_number] * 4 + [not_nan_number//5]
    dropout_rate = 0.0
    n_components = 50
    
    model_config = {
        "band_indices": band_indices,
        "geo_dim": geo_dim,
        "ret_dim": ret_dim,
        "rad_dim": rad_dim,
        "nnan_indices": nnan_indices,
        "hidden_dims": hidden_dims,
        "dropout_rate": dropout_rate,
        "rad_los_diff_mean": rad_los_diff_mean,
        "rad_los_diff_std": rad_los_diff_std
    }
    
    print(f"geo_dim: {geo_dim}, ret_dim: {ret_dim}, rad_dim: {rad_dim}, not nan number: {not_nan_number}, hidden_dims: {hidden_dims}, dropout_rate: {dropout_rate}")
    # Initialize model with dummy tau statistics (will be updated after dry run)
    model = ForwardModel(**model_config)
    # Transfer scaler states to model's submodules
    # Directly copy the buffers from fitted scalers
    model.geometry_scaler.register_buffer('mean_', scalers["geometry"].mean_)
    model.geometry_scaler.register_buffer('scale_', scalers["geometry"].scale_)
    model.retrieved_scaler.register_buffer('mean_', scalers["retrieved"].mean_)
    model.retrieved_scaler.register_buffer('scale_', scalers["retrieved"].scale_)
    model.radiance_scaler.register_buffer('mean_', scalers["radiance"].mean_)
    model.radiance_scaler.register_buffer('scale_', scalers["radiance"].scale_)
    model.to(device, dtype=dtype)
    with open(f"{status_dir}/model_config.pkl", "wb") as f:
        pickle.dump(model_config, f)
    
    batch_size = 8 if args.smoke_test else 1024
    # dataloader_kwargs = {
    #     "batch_size": batch_size,
    #     "shuffle": True,
    #     "num_workers": 8,
    #     "persistent_workers": True,
    #     "pin_memory": True,
    #     "prefetch_factor": 2
    # }
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": True,
        # "num_workers": 8,
        # "persistent_workers": True,
        # "pin_memory": True,
        # "prefetch_factor": 2
    }
    train_loader = DataLoader(train_dataset, **dataloader_kwargs)
    val_loader = DataLoader(val_dataset, **dataloader_kwargs)
    test_loader = DataLoader(test_dataset, **dataloader_kwargs)
    
    criterion = MixRelCosSimNormLoss(radiance_scaler.mean_, radiance_scaler.scale_, rel_weight=10.0)
    # criterion = MSENaNMaskLoss()
    weight_decay = 5e-4
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    
    # Learning rate scheduler - Cosine Annealing with Warm Restarts
    T_0_epochs = 1 if args.smoke_test else 25  # Initial restart period in epochs
    T_mult = 1  # Multiplication factor for restart period (must be integer >= 1)
    eta_min = 1e-7  # Minimum learning rate
    
    # Convert T_0 from epochs to steps (batches)
    steps_per_epoch = len(train_loader)
    T_0_steps = T_0_epochs * steps_per_epoch
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0_steps, T_mult=T_mult, eta_min=eta_min
    )
    
    geometric_series_sum = lambda a0, r, n: int(a0 * (1 - r**n) / (1 - r) if r != 1 else a0 * n)
    n_cycles = 1 if args.smoke_test else 10
    n_epochs = geometric_series_sum(T_0_epochs, T_mult, n_cycles)
    if args.epochs is not None:
        n_epochs = args.epochs
    
    # Initialize Aim run
    run = None
    if not args.no_aim:
        run = Run(
            repo='.',  # Current directory as aim repo
            experiment=base_name
        )
    
    max_norm = 0.5  # Reduced from 1.0 for better stability
    
    # Log hyperparameters
    if run is not None:
        run['hparams'] = {
            'band': band,
            'fraction': fraction,
            'batch_size': batch_size,
            'hidden_dims': hidden_dims,
            'dropout_rate': dropout_rate,
            'learning_rate': 1e-3,
            'weight_decay': weight_decay,
            'T_0_epochs': T_0_epochs,
            'T_mult': T_mult,
            'eta_min': eta_min,
            'n_cycles': n_cycles,
            'n_epochs': n_epochs,
            'geo_dim': geo_dim,
            'ret_dim': ret_dim,
            'rad_dim': rad_dim,
            'num_band_indices': len(band_indices),
            'num_nnan_indices': len(nnan_indices),
            'device': str(device),
            'dtype': str(dtype),
            'max_norm': max_norm
        }
    
    # Collect tau features from training data to compute statistics
    model.eval()
    all_tau_features = []
    model.to(device, dtype=dtype)
    # Reinitialize optimizer and scheduler with new model
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=weight_decay)
    steps_per_epoch = len(train_loader)
    T_0_steps = T_0_epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0_steps, T_mult=T_mult, eta_min=eta_min
    )
    
    # Run dry evaluation with reinitialized model
    print("Running dry run evaluation...")
    dry_loss, dry_predictions, dry_targets = evaluate_model(model, test_loader, criterion, device, 0, return_predictions=True, aim_run=run, subset='test', dtype=dtype)
    dry_predictions, dry_targets = band_inverse(dry_predictions, dry_targets, scalers["radiance"])
    if not args.smoke_test:
        forward_plot(dry_predictions, dry_targets, status_dir, "dry_run")
    print("Dry run done")
    
    if dry_run:
        exit()
    
    
    print("Starting training...")
    start_time = time.time()
    
    epoch_pbar = tqdm(range(n_epochs), desc="Training Progress", leave=True, position=0, 
                     dynamic_ncols=False, mininterval=0.3, ncols=NCOLS)
    
    min_val_loss = 1e10
    min_test_loss = 1e10
    
    check_epochs = [geometric_series_sum(T_0_epochs, T_mult, i) - 1 for i in range(1, n_cycles + 1)]
    
    for epoch in epoch_pbar:
        # Training
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, epoch, aim_run=run, dtype=dtype, max_norm=max_norm)
        val_loss = evaluate_model(model, val_loader, criterion, device, epoch, aim_run=run, subset='val', dtype=dtype)
        if epoch in check_epochs:
            test_loss, test_predictions, test_targets = evaluate_model(model, test_loader, criterion, device, epoch, return_predictions=True, aim_run=run, subset='test', dtype=dtype)
            test_predictions, test_targets = band_inverse(test_predictions, test_targets, scalers["radiance"])
            if not args.smoke_test:
                forward_plot(test_predictions, test_targets, status_dir, f"check_{epoch}")
            torch.save(model.state_dict(), f'{status_dir}/mlp_model_{epoch}.pth')
        else:
            test_loss = evaluate_model(model, test_loader, criterion, device, epoch, aim_run=run, subset='test', dtype=dtype)
        
        # Update progress bar
        epoch_pbar.set_postfix({
            'Train': f'{train_loss:.6f}',
            'Val': f'{val_loss:.6f}',
            'Test': f'{test_loss:.6f}',
        })
        
        # Save best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), f'{status_dir}/mlp_model_best.pth')
        if test_loss < min_test_loss:
            min_test_loss = test_loss
            torch.save(model.state_dict(), f'{status_dir}/mlp_model_best_test.pth')
    
    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds")
    print(f"Final Train Loss: {train_loss:.6f}")
    print(f"Final Validation Loss: {val_loss:.6f}")
    print(f"Final Test Loss: {test_loss:.6f}")
    print(f"Best Validation Loss: {min_val_loss:.6f}")
    print(f"Best Test Loss: {min_test_loss:.6f}")
    
    # Load best model for evaluation
    model.load_state_dict(torch.load(f'{status_dir}/mlp_model_best.pth'))
    model.to(device, dtype=dtype)  # Ensure loaded model has correct dtype
    
    final_test_loss, final_test_predictions, final_test_targets = evaluate_model(model, test_loader, criterion, device, epoch, return_predictions=True, aim_run=run, subset='final_test', dtype=dtype)
    final_test_predictions, final_test_targets = band_inverse(final_test_predictions, final_test_targets, scalers["radiance"])
    if not args.smoke_test:
        forward_plot(final_test_predictions, final_test_targets, status_dir, "final_test")
    
    # Log final metrics to Aim
    if run is not None:
        run.track(min_val_loss, name='best_val_loss', step=n_epochs-1)
        run.track(min_test_loss, name='best_test_loss', step=n_epochs-1)
        run.track(final_test_loss, name='final_test_loss', step=n_epochs-1)
    
    # Close the Aim run
        run.close()
    
    
    
    
