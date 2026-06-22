import argparse
import os
from typing import Dict, List
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from aim import Run
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from oco2_surrogate import RAND_SEED
from oco2_surrogate.forward_models import OCO2ForwardModel
from oco2_surrogate.load_data import load_retrieved_data
from oco2_surrogate.plot import plot_co2_profiles_with_uncertainty_raw, plot_xco2_scatter_raw
from oco2_surrogate.preprocessor import StandardScaler
from oco2_surrogate.model import MLP
from oco2_surrogate.release import load_smoke_inverse_dfs


TRAIN_YYMMS = [f"{yy}{mm:02d}" for yy in range(17, 20) for mm in range(1, 13)]
TRAIN_YYMMS.pop(7)
# TRAIN_YYMMS = ["1701"]
TEST_YYMMS = [f"22{mm:02d}" for mm in range(1, 13)]
# TEST_YYMMS = ["2301"]
CO2_SHIFT_RANGE = (5e-6, 7e-6)
NCOLS = 120
TRAIN_BUNDLE_KEYS = [
    "df_geo_train",
    "df_retr_train",
    "df_apriori_train",
    "df_wf_train",
    "df_rad_train",
]
TEST_BUNDLE_KEYS = [
    "df_geo_test",
    "df_retr_test",
    "df_apriori_test",
    "df_wf_test",
    "df_rad_test",
]

import torch
import torch.nn as nn


class GaussianNLLLoss(nn.Module):
    def __init__(self, weight):
        """
        length:   L, number of dimensions
        weight:   [L] tensor or None. If None, defaults to ones.
        """
        super().__init__()

        # 注册为 buffer（模型的一部分，但不训练）
        self.register_buffer("weight", weight.view(1, -1))  # shape [1, L]

    def forward(self, mu, log_sigma, target):
        """
        mu:        [B, L]
        log_sigma: [B, L]
        target:    [B, L]
        """
        sigma = torch.exp(log_sigma)
        gaussian = torch.distributions.Normal(mu, sigma)

        # NLL per element: [B, L]
        nll = -gaussian.log_prob(target)

        # 自动 broadcast weight: [1, L] -> [B, L]
        nll = nll * self.weight

        # reduction
        return nll.mean()


def unconvert_retr_pred(retr_pred: torch.Tensor, retr_dim: int, retr_scaler: StandardScaler):
    retr_pred_mean = retr_pred[:, :retr_dim]
    retr_pred_log_sigma = retr_pred[:, retr_dim:]
    retr_pred_mean = retr_pred_mean * retr_scaler.scale_ + retr_scaler.mean_
    retr_pred_sigma = torch.exp(retr_pred_log_sigma)
    retr_pred_sigma = retr_pred_sigma * retr_scaler.scale_
    return retr_pred_mean, retr_pred_sigma


def parse_args():
    parser = argparse.ArgumentParser(description="Inverse retrieval training")
    parser.add_argument("--train_fraction", type=float, default=0.5)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2048)
    # parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_dir", type=str, default="status/inverse_training_data")
    parser.add_argument("--forward_base_name", type=str, default="train_forward")
    parser.add_argument("--forward_data_dir", type=str, default="status/train_forward_o2")
    parser.add_argument("--forward_checkpoints", type=int, nargs="+", default=[149, 174, 199, 224, 249])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--sample_new", action="store_true", help="Regenerate integrated dataframes with augmentation")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Use tiny synthetic fixture data and CPU-safe defaults")
    parser.add_argument("--epochs", type=int, default=None, help="Override the number of training epochs")
    parser.add_argument("--max_rows", type=int, default=None, help="Limit rows loaded in smoke-test mode")
    parser.add_argument("--no_aim", action="store_true", help="Disable Aim logging")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for checkpoints and plots")
    return parser.parse_args()


def normalize_apriori_name(name: str) -> str:
    return name.replace("_apriori_", "_").replace("_apriori", "")


# def build_apriori_scaler(
#     apriori_raw: torch.Tensor,
#     retrieved_scaler: StandardScaler,
#     apriori_columns: List[str],
#     retrieved_columns: List[str],
# ) -> StandardScaler:
#     scaler = StandardScaler().fit(apriori_raw)
#     lookup = {normalize_apriori_name(col): idx for idx, col in enumerate(retrieved_columns)}
#     for apr_idx, col in enumerate(apriori_columns):
#         canon = normalize_apriori_name(col)
#         ret_idx = lookup.get(canon)
#         if ret_idx is not None:
#             scaler.mean_[apr_idx] = retrieved_scaler.mean_[ret_idx]
#             scaler.scale_[apr_idx] = retrieved_scaler.scale_[ret_idx]
#     return scaler


def tensor_from_df(df, dtype):
    return torch.from_numpy(df.to_numpy(copy=True)).to(dtype)


def dataframe_from_tensor(tensor: torch.Tensor, columns: List[str]):
    return pd.DataFrame(tensor.detach().cpu().numpy(), columns=columns)


def save_dataframe_bundle(bundle: Dict[str, pd.DataFrame], target_dir: str):
    os.makedirs(target_dir, exist_ok=True)
    for name, df in bundle.items():
        df.to_parquet(os.path.join(target_dir, f"{name}.parquet"), index=True)


def load_dataframe_bundle(target_dir: str, keys: List[str]) -> Dict[str, pd.DataFrame]:
    return {key: pd.read_parquet(os.path.join(target_dir, f"{key}.parquet")) for key in keys}


def dataframe_from_tensor(tensor: torch.Tensor, columns: List[str]):
    return pd.DataFrame(tensor.detach().cpu().numpy(), columns=columns)


def build_radiance_features(rad_tensor: torch.Tensor, scaler: StandardScaler):
    rad_scaled = scaler.transform(rad_tensor)
    rad_scaled = torch.nan_to_num(rad_scaled, nan=0.0)
    nan_mask = torch.isnan(rad_tensor).float()
    return torch.cat([rad_scaled, nan_mask], dim=1)


def get_profile_indices(columns: List[str], prefix: str, length: int = 20):
    return [columns.index(f"{prefix}_{i}") for i in range(length)]


def load_forward_model(args, geometry_columns, retrieved_columns, dtype):
    forward_model = OCO2ForwardModel(
        base_name=args.forward_base_name,
        data_dir=args.forward_data_dir,
        checkpoints=args.forward_checkpoints,
        geometry_column_names=geometry_columns,
        retrieved_column_names=retrieved_columns,
        dtype=dtype,
    )
    return forward_model


def compute_xco2_from_profiles(co2_phys: torch.Tensor, wf_phys: torch.Tensor):
    # co2_phys in VMR, convert to ppm for XCO2 computation
    return torch.sum(wf_phys * (co2_phys), dim=1, keepdim=True)


def augment_with_shift(
    geo_raw,
    apriori_raw,
    retrieved_raw,
    wf_raw,
    rad_raw,
    column_info,
    forward_model,
    shift_range,
    batch_size,
    noise_ratio,
    device,
    dtype,
):
    geo_aug = []
    apr_aug = []
    rad_aug = []
    retr_aug = []
    wf_aug = []

    forward_model = forward_model.to(device, dtype=dtype)
    forward_model.eval()

    co2_idx = column_info["co2_indices"]
    apr_co2_idx = column_info["apriori_co2_indices"]
    
    tbar = tqdm(range(0, len(geo_raw), batch_size), desc="Augmenting", ncols=NCOLS)

    for start in tbar:
        end = min(start + batch_size, len(geo_raw))
        geo_batch_raw = geo_raw[start:end]
        ret_batch_raw = retrieved_raw[start:end]
        apr_batch_raw = apriori_raw[start:end]
        wf_batch_raw = wf_raw[start:end]
        rad_batch_raw = rad_raw[start:end]

        geo_batch_device = geo_batch_raw.to(device, dtype=dtype)
        ret_shift_values = ret_batch_raw.new_empty((ret_batch_raw.size(0), 1)).uniform_(
            shift_range[0], shift_range[1]
        )
        # apr_shift_values = ret_shift_values + apr_batch_raw.new_empty((apr_batch_raw.size(0), 1)).uniform_(
        #     -5e-7, 5e-7
        # )
        apr_shift_values = apr_batch_raw.new_empty((apr_batch_raw.size(0), 1)).uniform_(
            shift_range[0], shift_range[1]
        )
        ret_shift_raw = ret_batch_raw.clone()
        ret_shift_raw[:, co2_idx] += ret_shift_values
        ret_shift_device = ret_shift_raw.to(device, dtype=dtype)

        with torch.no_grad():
            rad_batch = forward_model(geo_batch_device, ret_shift_device).cpu()
        
        # Apply Gaussian noise multiplier (mean=1.0, std=0.003)
        noise_multiplier = torch.normal(mean=1.0, std=noise_ratio, size=rad_batch.shape)
        rad_batch = rad_batch * noise_multiplier
        
        # Preserve NaN values from original rad_raw
        nan_mask = torch.isnan(rad_batch_raw)
        rad_batch[nan_mask] = float('nan')
        
        # mask = (~torch.isnan(rad_batch)) & (torch.rand_like(rad_batch) < 0.001)
        # rad_batch[mask] = float('nan')

        apr_shift_raw = apr_batch_raw.clone()
        apr_shift_raw[:, apr_co2_idx] += apr_shift_values

        geo_aug.append(geo_batch_raw.cpu())
        apr_aug.append(apr_shift_raw.cpu())
        rad_aug.append(rad_batch)
        retr_aug.append(ret_shift_raw.cpu())
        wf_aug.append(wf_batch_raw.cpu())

    geo_aug = torch.cat(geo_aug, dim=0)
    apr_aug = torch.cat(apr_aug, dim=0)
    rad_aug = torch.cat(rad_aug, dim=0)
    retr_aug = torch.cat(retr_aug, dim=0)
    wf_aug = torch.cat(wf_aug, dim=0)
    
    

    return {
        "geo": geo_aug,
        "apriori": apr_aug,
        "rad": rad_aug,
        "retrieved": retr_aug,
        "wf": wf_aug,
    }


def expand_to_retrieved_dim(tensor: torch.Tensor, total_dim: int, indices: List[int]) -> torch.Tensor:
    expanded = torch.zeros(tensor.size(0), total_dim, device=tensor.device, dtype=tensor.dtype)
    expanded[:, indices] = tensor
    return expanded


def create_retr_weight_tensor(retr_dim, co2_indices, pressure_indices, h2o_indices, device, dtype):
    """Create element-wise weight tensor for retrieval loss."""
    weights = torch.ones(retr_dim, device=device, dtype=dtype) * 0.1
    weights[co2_indices] = 2.0
    weights[pressure_indices] = 2.0
    weights[h2o_indices] = 2.0
    weights = weights / weights.mean()
    return weights


class InverseRetrievalModel(nn.Module):
    def __init__(
        self,
        geo_dim,
        apriori_dim,
        rad_dim,
        retr_dim,
        co2_indices,
        pressure_indices,
        h2o_indices,
        hidden_dim=512,
        dtype=torch.float32,
        co2_mean=None,
        co2_scale=None,
        wf_mean=None,
        wf_scale=None,
    ):
        super().__init__()
        self.retr_dim = retr_dim
        
        MLP_kwarrgs = {"dropout_rate": 0.0, "activation": nn.SiLU(), "batch_norm": False, "layer_norm": False, "residual": True}
        # self.geo_encoder = nn.Linear(geo_dim, geo_dim // 2)
        # self.apriori_encoder = nn.Linear(apriori_dim, apriori_dim // 2)
        # self.rad_encoder = nn.Linear(rad_dim, rad_dim // 2)
        self.geo_encoder = MLP(geo_dim, geo_dim // 2, hidden_dims = [geo_dim], **MLP_kwarrgs)
        self.apriori_encoder = MLP(apriori_dim, apriori_dim // 2, hidden_dims = [apriori_dim], **MLP_kwarrgs)
        self.rad_encoder = MLP(rad_dim, rad_dim // 6, hidden_dims = [rad_dim//3], 
                               dropout_rate = 0.2, activation = nn.SiLU(), batch_norm = False, layer_norm = False, residual = True)
        fusion_dim = geo_dim  // 2 + apriori_dim // 2 + rad_dim // 6
        self.fusion = MLP(fusion_dim, hidden_dim, hidden_dims = [hidden_dim, hidden_dim], **MLP_kwarrgs)
        
        # Unified retrieval head with apriori integration
        self.retr_dim = retr_dim
        self.retr_head = nn.Linear(hidden_dim + apriori_dim, retr_dim * 2)
        
        # WF head uses sliced pressure and h2o predictions
        # self.wf_head = nn.Linear(len(pressure_indices) + len(h2o_indices), len(pressure_indices))
        wf_input_dim = len(pressure_indices) + len(h2o_indices)
        wf_output_dim = len(pressure_indices)
        self.wf_head = MLP(wf_input_dim, wf_output_dim, hidden_dims = [wf_input_dim], **MLP_kwarrgs)

        # Store indices as buffers
        self.register_buffer("co2_indices", torch.tensor(co2_indices, dtype=torch.long))
        self.register_buffer("pressure_indices", torch.tensor(pressure_indices, dtype=torch.long))
        self.register_buffer("h2o_indices", torch.tensor(h2o_indices, dtype=torch.long))
        
        self.register_buffer("co2_mean", co2_mean)
        self.register_buffer("co2_scale", co2_scale)
        self.register_buffer("wf_mean", wf_mean)
        self.register_buffer("wf_scale", wf_scale)

    def forward(self, geo, apriori, rad):
        geo_feat = self.geo_encoder(geo)
        apr_feat = self.apriori_encoder(apriori)
        rad_feat = self.rad_encoder(rad)
        fused = torch.cat([geo_feat, apr_feat, rad_feat], dim=1)
        fused = self.fusion(fused)
        
        # Leverage apriori at final layer
        retr_input = torch.cat([fused, apriori], dim=1)
        retr_pred = self.retr_head(retr_input)  # first retr_pred is the mean, second is the log_sigma
        retr_pred[:, :self.retr_dim] = retr_pred[:, :self.retr_dim] + apriori
        
        # Slice out pressure and h2o for wf computation
        pressure_pred = retr_pred[:, self.pressure_indices]
        h2o_pred = retr_pred[:, self.h2o_indices]
        
        # Compute wf using sliced predictions
        wf_input = torch.cat([pressure_pred, h2o_pred], dim=1)
        wf_pred = self.wf_head(wf_input)
        
        return retr_pred, wf_pred

    def compute_xco2(self, retr_mean, wf_pred):
        """Compute XCO2 mean only (for plotting/compatibility)."""
        co2_mean_scaled = retr_mean[:, self.co2_indices]
        co2_mean_phys = co2_mean_scaled * self.co2_scale + self.co2_mean
        wf_phys = wf_pred * self.wf_scale + self.wf_mean
        return compute_xco2_from_profiles(co2_mean_phys, wf_phys)

    def compute_xco2_stats(self, retr_mean, retr_log_sigma, wf_pred):
        """
        retr_mean:      [B, retr_dim] (scaled space)
        retr_log_sigma: [B, retr_dim] (scaled space)
        wf_pred:        [B, n_levels] (scaled space)

        return:
            xco2_mean_phys:  [B, 1]
            xco2_sigma_phys: [B, 1]
        """
        # 1. 取出 CO2 部分的 mean / sigma（scaled）
        co2_mean_scaled = retr_mean[:, self.co2_indices]                 # [B, L]
        co2_sigma_scaled = torch.exp(retr_log_sigma[:, self.co2_indices])  # [B, L]

        # 2. 从 scaled -> physical
        co2_mean_phys = co2_mean_scaled * self.co2_scale + self.co2_mean   # [B, L]
        co2_sigma_phys = co2_sigma_scaled * self.co2_scale                 # [B, L]

        # 3. WF 从 scaled -> physical
        wf_phys = wf_pred * self.wf_scale + self.wf_mean                   # [B, L]

        # 4. XCO2 = sum(w_i * c_i)
        xco2_mean_phys = torch.sum(wf_phys * co2_mean_phys, dim=1, keepdim=True)  # [B, 1]

        # var = sum( (w_i * sigma_i)^2 )
        xco2_var_phys = torch.sum((wf_phys * co2_sigma_phys) ** 2, dim=1, keepdim=True)
        xco2_sigma_phys = torch.sqrt(torch.clamp(xco2_var_phys, min=1e-18))
        # xco2_sigma_phys = torch.sum(wf_phys * co2_sigma_phys, dim=1, keepdim=True)

        return xco2_mean_phys, xco2_sigma_phys


def train_epoch(model, loader, optimizer, scheduler,
                criterion_retr, criterion_wf, criterion_xco2,
                xco2_mean, xco2_std, device):
    model.train()
    total_loss = 0.0
    num_samples = 0
    tbar = tqdm(loader, desc="Training", ncols=NCOLS, leave=False)

    for batch in tbar:
        geo, apr, rad, retr, wf, xco2 = [b.to(device) for b in batch]
        optimizer.zero_grad()

        retr_pred, wf_pred = model(geo, apr, rad)
        retr_pred_mean = retr_pred[:, :model.retr_dim]
        retr_pred_log_sigma = retr_pred[:, model.retr_dim:]

        # 1) retr loss
        retr_loss = criterion_retr(retr_pred_mean, retr_pred_log_sigma, retr)

        # 2) wf loss (仍然 MSE)
        wf_loss = criterion_wf(wf_pred, wf)

        # 3) xco2 的 mean / sigma（物理空间）
        xco2_mean_phys, xco2_sigma_phys = model.compute_xco2_stats(
            retr_pred_mean, retr_pred_log_sigma, wf_pred
        )   # [B,1], [B,1]

        # 4) 物理 -> 标准化空间
        xco2_mu_norm = (xco2_mean_phys - xco2_mean) / xco2_std        # [B,1]
        xco2_sigma_norm = xco2_sigma_phys / xco2_std                  # [B,1]
        xco2_log_sigma_norm = torch.log(xco2_sigma_norm + 1e-12)

        xco2_target_norm = (xco2 - xco2_mean) / xco2_std              # [B,1]

        # 5) XCO2 的 Gaussian NLL loss
        xco2_loss = criterion_xco2(xco2_mu_norm, xco2_log_sigma_norm, xco2_target_norm)

        # 总 loss
        loss = retr_loss + wf_loss + xco2_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * geo.size(0)
        num_samples += geo.size(0)

        current_lr = optimizer.param_groups[0]['lr']
        tbar.set_postfix(
            loss=f"{total_loss / max(num_samples, 1):.2e}",
            lr=f"{current_lr:.2e}"
        )

    return total_loss / max(num_samples, 1)


def evaluate(model, loader, criterion_retr, criterion_wf, criterion_xco2,
             xco2_mean, xco2_std, device, return_preds=False,
             eval_type: str="Validating"):
    model.eval()
    total_loss = 0.0
    num_samples = 0
    preds = []
    targets = []
    
    xco2_sigma_physs = []
    with torch.no_grad():
        # tbar = tqdm(loader, desc=eval_type, ncols=NCOLS)
        for batch in loader:
            geo, apr, rad, retr, wf, xco2 = [b.to(device) for b in batch]
            retr_pred, wf_pred = model(geo, apr, rad)
            retr_pred_mean = retr_pred[:, :model.retr_dim]
            retr_pred_log_sigma = retr_pred[:, model.retr_dim:]

            # retr loss
            retr_loss = criterion_retr(retr_pred_mean, retr_pred_log_sigma, retr)

            # wf loss
            wf_loss = criterion_wf(wf_pred, wf)

            # xco2 stats in physical space
            xco2_mean_phys, xco2_sigma_phys = model.compute_xco2_stats(
                retr_pred_mean, retr_pred_log_sigma, wf_pred
            )
            xco2_sigma_physs.append(xco2_sigma_phys.cpu())

            xco2_mu_norm = (xco2_mean_phys - xco2_mean) / xco2_std
            xco2_sigma_norm = xco2_sigma_phys / xco2_std
            xco2_log_sigma_norm = torch.log(xco2_sigma_norm + 1e-12)
            xco2_target_norm = (xco2 - xco2_mean) / xco2_std

            xco2_loss = criterion_xco2(xco2_mu_norm, xco2_log_sigma_norm, xco2_target_norm)

            loss = retr_loss + wf_loss + xco2_loss
            total_loss += loss.item() * geo.size(0)
            num_samples += geo.size(0)

            if return_preds:
                # 这里为了和后面画图兼容，返回 mean（物理空间）的 xco2
                preds.append((retr_pred.cpu(), wf_pred.cpu(), xco2_mean_phys.cpu()))
                targets.append((retr.cpu(), wf.cpu(), xco2.cpu()))
            # tbar.set_postfix(loss=f"{total_loss / max(num_samples, 1):.2e}")
    avg_loss = total_loss / max(num_samples, 1)
    xco2_sigma_physs = torch.cat(xco2_sigma_physs, dim=0)
    print(torch.mean(xco2_sigma_physs), torch.max(xco2_sigma_physs), torch.min(xco2_sigma_physs))
    if return_preds:
        return avg_loss, preds, targets
    return avg_loss


def main():
    args = parse_args()
    base_name = __file__.split('/')[-1].split('.')[0]
    status_dir = args.output_dir or f'status/{base_name}'
    os.makedirs(status_dir, exist_ok=True)
    requested_device = "cpu" if args.smoke_test and args.device == "cuda" else args.device
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        requested_device = "cpu"
    device = torch.device(requested_device)
    dtype = torch.float32

    torch.manual_seed(RAND_SEED)

    if args.smoke_test:
        print("Loading synthetic smoke-test data...")
        train_dfs, test_dfs = load_smoke_inverse_dfs(max_rows=args.max_rows)
        df_geo_train = train_dfs["df_geo_train"]
        df_retr_train = train_dfs["df_retr_train"]
        df_apriori_train = train_dfs["df_apriori_train"]
        df_wf_train = train_dfs["df_wf_train"]
        df_rad_train = train_dfs["df_rad_train"]

        df_geo_test = test_dfs["df_geo_test"]
        df_retr_test = test_dfs["df_retr_test"]
        df_apriori_test = test_dfs["df_apriori_test"]
        df_wf_test = test_dfs["df_wf_test"]
        df_rad_test = test_dfs["df_rad_test"]
    elif args.sample_new:
        print("Loading and processing new data...")
        train_rows = load_retrieved_data(
            TRAIN_YYMMS,
            fraction=args.train_fraction,
            load_modeled=True,
            filter_outcome_flag=True,
            filter_asia=False,
        )
        test_rows = load_retrieved_data(
            TEST_YYMMS,
            fraction=args.test_fraction,
            load_modeled=False,
            filter_outcome_flag=True,
            filter_asia=False,
        )

        (
            df_geo_train,
            df_retr_train,
            df_apriori_train,
            df_wf_train,
            _,
            df_rad_train,
        ) = train_rows
        (
            df_geo_test,
            df_retr_test,
            df_apriori_test,
            df_wf_test,
            _,
            df_rad_test,
        ) = test_rows

        base_train_geo = tensor_from_df(df_geo_train, dtype)
        base_train_retr = tensor_from_df(df_retr_train, dtype)
        base_train_apriori = tensor_from_df(df_apriori_train, dtype)
        base_train_wf = tensor_from_df(df_wf_train, dtype)
        base_train_rad = tensor_from_df(df_rad_train, dtype)

        column_info_aug = {
            "co2_indices": get_profile_indices(df_retr_train.columns.tolist(), "co2_profile", 20),
            "pressure_indices": get_profile_indices(df_retr_train.columns.tolist(), "vector_pressure_levels", 20),
            "h2o_indices": get_profile_indices(df_retr_train.columns.tolist(), "h2o_profile", 20),
            "apriori_co2_indices": get_profile_indices(df_apriori_train.columns.tolist(), "co2_profile_apriori", 20),
        }

        forward_model = load_forward_model(args, df_geo_train.columns.tolist(), df_retr_train.columns.tolist(), dtype)
        aug_raw = augment_with_shift(
            geo_raw=base_train_geo,
            apriori_raw=base_train_apriori,
            retrieved_raw=base_train_retr,
            wf_raw=base_train_wf,
            rad_raw=base_train_rad,
            column_info=column_info_aug,
            forward_model=forward_model,
            shift_range=CO2_SHIFT_RANGE,
            batch_size=128,
            noise_ratio=0.003,
            device=device,
            dtype=dtype,
        )
        del forward_model

        df_geo_aug = dataframe_from_tensor(aug_raw["geo"], df_geo_train.columns.tolist())
        df_apriori_aug = dataframe_from_tensor(aug_raw["apriori"], df_apriori_train.columns.tolist())
        df_retr_aug = dataframe_from_tensor(aug_raw["retrieved"], df_retr_train.columns.tolist())
        df_wf_aug = dataframe_from_tensor(aug_raw["wf"], df_wf_train.columns.tolist())
        df_rad_aug = dataframe_from_tensor(aug_raw["rad"], df_rad_train.columns.tolist())

        df_geo_train = pd.concat([df_geo_train, df_geo_aug], ignore_index=False)
        df_apriori_train = pd.concat([df_apriori_train, df_apriori_aug], ignore_index=False)
        df_retr_train = pd.concat([df_retr_train, df_retr_aug], ignore_index=False)
        df_wf_train = pd.concat([df_wf_train, df_wf_aug], ignore_index=False)
        df_rad_train = pd.concat([df_rad_train, df_rad_aug], ignore_index=False)

        train_bundle = {
            "df_geo_train": df_geo_train,
            "df_retr_train": df_retr_train,
            "df_apriori_train": df_apriori_train,
            "df_wf_train": df_wf_train,
            "df_rad_train": df_rad_train,
        }
        test_bundle = {
            "df_geo_test": df_geo_test,
            "df_retr_test": df_retr_test,
            "df_apriori_test": df_apriori_test,
            "df_wf_test": df_wf_test,
            "df_rad_test": df_rad_test,
        }
        save_dataframe_bundle(train_bundle, os.path.join(status_dir, "train_dataframes"))
        save_dataframe_bundle(test_bundle, os.path.join(status_dir, "test_dataframes"))
        print("Integrated dataframes saved to parquet.")
    else:
        print("Loading integrated dataframes...")
        train_dfs = load_dataframe_bundle(os.path.join(args.data_dir, "train_dataframes"), TRAIN_BUNDLE_KEYS)
        test_dfs = load_dataframe_bundle(os.path.join(args.data_dir, "test_dataframes"), TEST_BUNDLE_KEYS)

        df_geo_train = train_dfs["df_geo_train"]
        df_retr_train = train_dfs["df_retr_train"]
        df_apriori_train = train_dfs["df_apriori_train"]
        df_wf_train = train_dfs["df_wf_train"]
        df_rad_train = train_dfs["df_rad_train"]

        df_geo_test = test_dfs["df_geo_test"]
        df_retr_test = test_dfs["df_retr_test"]
        df_apriori_test = test_dfs["df_apriori_test"]
        df_wf_test = test_dfs["df_wf_test"]
        df_rad_test = test_dfs["df_rad_test"]
        print("Integrated dataframes loaded from parquet.")
        
    # n_df_train = len(df_geo_train)

    # train_geo_raw = tensor_from_df(df_geo_train.iloc[:n_df_train//2, :], dtype)
    # train_retr_raw = tensor_from_df(df_retr_train.iloc[:n_df_train//2, :], dtype)
    # train_apriori_raw = tensor_from_df(df_apriori_train.iloc[:n_df_train//2, :], dtype)
    # train_wf_raw = tensor_from_df(df_wf_train.iloc[:n_df_train//2, :], dtype)
    # train_rad_raw = tensor_from_df(df_rad_train.iloc[:n_df_train//2, :], dtype)

    train_geo_raw = tensor_from_df(df_geo_train, dtype)
    train_retr_raw = tensor_from_df(df_retr_train, dtype)
    train_apriori_raw = tensor_from_df(df_apriori_train, dtype)
    train_wf_raw = tensor_from_df(df_wf_train, dtype)
    train_rad_raw = tensor_from_df(df_rad_train, dtype)

    geometry_scaler = StandardScaler().fit(train_geo_raw)
    retrieved_scaler = StandardScaler().fit(train_retr_raw)
    # apriori_scaler = build_apriori_scaler(
    #     train_apriori_raw, retrieved_scaler, df_apriori_train.columns.tolist(), df_retr_train.columns.tolist()
    # )
    radiance_scaler = StandardScaler().fit(train_rad_raw)
    wf_scaler = StandardScaler().fit(train_wf_raw)

    train_geo_scaled = geometry_scaler.transform(train_geo_raw)
    train_apriori_scaled = retrieved_scaler.transform(train_apriori_raw)
    train_rad_feat = build_radiance_features(train_rad_raw, radiance_scaler)
    train_retr_scaled = retrieved_scaler.transform(train_retr_raw)
    train_wf_scaled = wf_scaler.transform(train_wf_raw)

    co2_indices = get_profile_indices(df_retr_train.columns.tolist(), "co2_profile", 20)
    pressure_indices = get_profile_indices(df_retr_train.columns.tolist(), "vector_pressure_levels", 20)
    h2o_indices = get_profile_indices(df_retr_train.columns.tolist(), "h2o_profile", 20)
    apriori_co2_indices = get_profile_indices(df_apriori_train.columns.tolist(), "co2_profile_apriori", 20)

    # Unified retrieval target
    retr_target = train_retr_scaled
    wf_target = train_wf_scaled
    xco2_target = compute_xco2_from_profiles(train_retr_raw[:, co2_indices], train_wf_raw)
    
    # Compute XCO2 normalization statistics
    xco2_mean = torch.mean(xco2_target).to(device)
    xco2_std = torch.std(xco2_target).to(device)

    scalers = {
        "geometry": geometry_scaler,
        "retrieved": retrieved_scaler,
        "radiance": radiance_scaler,
        "wf": wf_scaler,
    }
    column_info = {
        "co2_indices": co2_indices,
        "pressure_indices": pressure_indices,
        "h2o_indices": h2o_indices,
        "apriori_co2_indices": apriori_co2_indices,
    }

    total_size = len(train_geo_scaled)
    train_size = int(0.9 * total_size)
    val_size = total_size - train_size
    dataset = TensorDataset(
        train_geo_scaled,
        train_apriori_scaled,
        train_rad_feat,
        retr_target,
        wf_target,
        xco2_target,
    )
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=min(args.batch_size, len(train_dataset)) if args.smoke_test else args.batch_size,
        shuffle=True,
        num_workers=0 if args.smoke_test else 4,
        pin_memory=False if args.smoke_test else True,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Prepare test tensors (without augmentation)
    test_geo_raw = tensor_from_df(df_geo_test, dtype)
    test_apriori_raw = tensor_from_df(df_apriori_test, dtype)
    test_rad_raw = tensor_from_df(df_rad_test, dtype)
    test_retr_raw = tensor_from_df(df_retr_test, dtype)
    test_wf_raw = tensor_from_df(df_wf_test, dtype)

    test_geo_scaled = geometry_scaler.transform(test_geo_raw)
    test_apriori_scaled = retrieved_scaler.transform(test_apriori_raw)
    test_rad_feat = build_radiance_features(test_rad_raw, radiance_scaler)
    test_retr_scaled = retrieved_scaler.transform(test_retr_raw)
    test_wf_scaled = wf_scaler.transform(test_wf_raw)

    test_retr = test_retr_scaled
    test_xco2 = compute_xco2_from_profiles(test_retr_raw[:, co2_indices], test_wf_raw)

    test_dataset = TensorDataset(
        test_geo_scaled,
        test_apriori_scaled,
        test_rad_feat,
        test_retr,
        test_wf_scaled,
        test_xco2,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    model = InverseRetrievalModel(
        geo_dim=train_geo_scaled.shape[1],
        apriori_dim=train_apriori_scaled.shape[1],
        rad_dim=train_rad_feat.shape[1],
        retr_dim=train_retr_scaled.shape[1],
        co2_indices=co2_indices,
        pressure_indices=pressure_indices,
        h2o_indices=h2o_indices,
        hidden_dim=64 if args.smoke_test else 1024,
        co2_mean=retrieved_scaler.mean_[co2_indices],
        co2_scale=retrieved_scaler.scale_[co2_indices],
        wf_mean=wf_scaler.mean_,
        wf_scale=wf_scaler.scale_,
    ).to(device, dtype=dtype)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler - Cosine Annealing with Warm Restarts
    T_0_epochs = 1 if args.smoke_test else 20  # Initial restart period in epochs
    T_mult = 1  # Multiplication factor for restart period
    eta_min = 1e-6  # Minimum learning rate
    n_cycles = 1 if args.smoke_test else 15
    steps_per_epoch = len(train_loader)
    T_0_steps = T_0_epochs * steps_per_epoch
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0_steps, T_mult=T_mult, eta_min=eta_min
    )
    
    geometric_series_sum = lambda a0, r, n: int(a0 * (1 - r**n) / (1 - r) if r != 1 else a0 * n)
    n_epochs = geometric_series_sum(T_0_epochs, T_mult, n_cycles)
    if args.epochs is not None:
        n_epochs = args.epochs
    cycle_final_epochs = [1]+ [geometric_series_sum(T_0_epochs, T_mult, i) for i in range(1, n_cycles + 1)]
    
    # Create retrieval weight tensor
    retr_weights = create_retr_weight_tensor(
        train_retr_scaled.shape[1],
        co2_indices,
        pressure_indices,
        h2o_indices,
        device,
        dtype
    )
    
    criterion_retr = GaussianNLLLoss(retr_weights)
    criterion_wf = nn.MSELoss()
    
    # XCO2: L = 1, weight 就 [1.0] 即可
    xco2_weight = torch.ones(1, device=device, dtype=dtype)
    criterion_xco2 = GaussianNLLLoss(xco2_weight)

    run = None
    if not args.no_aim:
        run = Run(repo=".", experiment=base_name)
    hparams = vars(args).copy()
    hparams.update({
        'T_0_epochs': T_0_epochs,
        'T_mult': T_mult,
        'eta_min': eta_min,
    })
    if run is not None:
        run["hparams"] = hparams

    best_val_loss = float("inf")
    best_test_loss = float("inf")
    retrieved_dim = train_retr_scaled.shape[1]
    
    tbar = tqdm(range(n_epochs), desc="Training inverse model", ncols=NCOLS)

    train_loss = 0.0
    val_loss = 0.0
    test_loss = 0.0
    
    for epoch in tbar:
        tbar.set_postfix(
            {
                "train": f"{train_loss:.2e}", 
                "val": f"{val_loss:.2e}", 
                "test": f"{test_loss:.2e}",
             }
        )
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            criterion_retr, criterion_wf, criterion_xco2,
            xco2_mean, xco2_std, device
        )
        val_loss, val_preds, val_targets = evaluate(
            model, val_loader,
            criterion_retr, criterion_wf, criterion_xco2,
            xco2_mean, xco2_std, device,
            return_preds=True
        )
        # if epoch % 25 == 0:
        test_loss, test_preds, test_targets = evaluate(
            model, test_loader,
            criterion_retr, criterion_wf, criterion_xco2,
            xco2_mean, xco2_std, device,
            return_preds=True
        )
        
        if run is not None:
            run.track(train_loss, name="loss", step=epoch, context={"subset": "train"})
            run.track(val_loss, name="loss", step=epoch, context={"subset": "val"})
            run.track(test_loss, name="loss", step=epoch, context={"subset": "test"})

        if (epoch + 1) in cycle_final_epochs:
            torch.save(model.state_dict(), os.path.join(status_dir, f"{epoch:04d}.pth"))
            plot_data = [] if args.smoke_test else [(val_preds, val_targets), (test_preds, test_targets)]
            plot_names = ["val", "test"]
            for (preds, targets), name in zip(plot_data, plot_names):
                retr_pred_batches = torch.cat([p[0] for p in preds], dim=0)
                retr_target_batches = torch.cat([t[0] for t in targets], dim=0)
                wf_pred_batches = torch.cat([p[1] for p in preds], dim=0)
                wf_target_batches = torch.cat([t[1] for t in targets], dim=0)
                xco2_pred_raw = torch.cat([p[2] for p in preds], dim=0)
                xco2_target_raw = torch.cat([t[2] for t in targets], dim=0)
                
                retr_pred_mean_raw, retr_pred_sigma_raw = unconvert_retr_pred(retr_pred_batches, retrieved_dim, retrieved_scaler)
                retr_target_mean_raw = retr_target_batches * retrieved_scaler.scale_ + retrieved_scaler.mean_
                co2_pred_mean_raw = retr_pred_mean_raw[:, co2_indices]
                co2_pred_sigma_raw = retr_pred_sigma_raw[:, co2_indices]
                co2_target_raw = retr_target_mean_raw[:, co2_indices]
                
                plot_co2_profiles_with_uncertainty_raw(
                    co2_pred_mean_raw.cpu().numpy(),
                    co2_target_raw.cpu().numpy(),
                    co2_pred_sigma_raw.cpu().numpy(),
                    status_dir,
                    f"{name}_epoch_{epoch}",
                    "pred",
                )
                plot_xco2_scatter_raw(
                    xco2_pred_raw.cpu().numpy(),
                    xco2_target_raw.cpu().numpy(),
                    status_dir,
                    f"{name}_epoch_{epoch}",
                )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(status_dir, "inverse_model_best_val.pth"))
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), os.path.join(status_dir, "inverse_model_best_test.pth"))

    torch.save(
        {
            "geometry": geometry_scaler.state_dict(),
            "retrieved": retrieved_scaler.state_dict(),
            "radiance": radiance_scaler.state_dict(),
            "wf": wf_scaler.state_dict(),
            "co2_indices": co2_indices,
            "pressure_indices": pressure_indices,
            "h2o_indices": h2o_indices,
            "apriori_co2_indices": apriori_co2_indices,
        },
        os.path.join(status_dir, "scalers.pth"),
    )

    if run is not None:
        run.track(best_val_loss, name="best_val_loss", step=n_epochs - 1)
        run.track(best_test_loss, name="best_test_loss", step=n_epochs - 1)
        run.close()


if __name__ == "__main__":
    main()
