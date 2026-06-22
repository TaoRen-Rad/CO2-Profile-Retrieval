from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
import numpy as np
from . import RAND_SEED
from scipy.stats import gaussian_kde
from typing import Dict

def plot_radiance(plot_label, plot_pred):
    rad_width = 0.5
    x_label = "Band Spectral Index"
    gs = GridSpec(2, 1, left=0.1, right=0.99, bottom=0.1, top=0.98, wspace=0.05)
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(plot_label, label='Target', alpha=0.8, linewidth=rad_width, color="blue")
    ax1.plot(plot_pred, label='Prediction', alpha=0.8, linewidth=rad_width, linestyle='--', color="red")
    # ax1.set_xlabel(x_label)
    ax1.set_xticklabels([])
    ax1.set_ylabel('Radiance [-]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, np.nanmax(plot_label) * 1.1)
    
    # Relative error plot
    ax2 = fig.add_subplot(gs[1])
    scale = np.nanmax(plot_label)
    ax2.plot((plot_pred - plot_label) / scale * 100.0, 
            alpha=0.8, linewidth=rad_width, color="black")
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Relative Error [%]')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 0.5)
    return fig

def plot_unscaled_radiance(plot_label, plot_pred):
    rad_width = 0.5
    x_label = "Band Spectral Index"
    gs = GridSpec(1, 2, left=0.05, right=0.99, bottom=0.2, top=0.92, wspace=0.2)
    fig = plt.figure(figsize=(8, 2))
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(plot_label, label='Target', alpha=0.8, linewidth=rad_width, color="blue")
    ax1.plot(plot_pred, label='Prediction', alpha=0.8, linewidth=rad_width, linestyle='--', color="red")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel('Radiance [-]')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # ax1.set_ylim(0, np.nanmax(plot_label) * 1.1)
    
    # Relative error plot
    ax2 = fig.add_subplot(gs[0, 1])
    scale = np.nanmax(plot_label) - np.nanmin(plot_label)
    ax2.plot((plot_pred - plot_label) / scale * 100.0, 
            alpha=0.8, linewidth=rad_width, color="black")
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.8)
    ax2.set_xlabel(x_label)
    ax2.set_ylabel('Relative Error [%]')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2, 2)
    return fig

def plot_radiance_scatter(target, prediction, need_value=False):
    flatten_pred = prediction.flatten()
    flatten_target = target.flatten()
    flatten_pred = flatten_pred[~np.isnan(flatten_target)]
    flatten_target = flatten_target[~np.isnan(flatten_target)]
    indices = np.random.choice(len(flatten_pred), 10000, replace=False)
    flatten_pred = flatten_pred[indices]
    flatten_target = flatten_target[indices]
    
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    
    # Test scatter
    ax.scatter(flatten_target, flatten_pred, alpha=0.4, s=2, c='red')
    ax.plot([flatten_target.min(), flatten_target.max()], [flatten_target.min(), flatten_target.max()], 
                'k--', lw=2, label='Perfect Prediction')
    
    rmse = np.sqrt(np.mean((flatten_pred - flatten_target) ** 2))
    me = np.mean(flatten_pred - flatten_target)
    ax.text(0.05, 0.95, f"RMSE: {rmse:.3e}\nME: {me:.3e}", transform=ax.transAxes, fontsize=12,
            verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.8))
    ax.set_xlabel('Target Radiance')
    ax.set_ylabel('Prediction Radiance')
    ax.set_aspect('equal')
    ax.legend()
    # ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if need_value:
        return fig, rmse, me
    else:
        return fig
    

def forward_plot(predictions, targets, status_dir, name_prefix):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mask = np.isnan(targets)
    predictions[mask] = np.nan
    
    
    n_examples = 10
    np.random.seed(RAND_SEED)
    indices = np.random.choice(len(predictions), n_examples, replace=False)
    for i, idx in enumerate(indices):
        fig = plot_radiance(targets[idx], predictions[idx])
        fig.savefig(f"{status_dir}/{name_prefix}_{i}.png")
    
    
    # get the flatten pred and target, remove nan, sample 10000 points

    fig = plot_radiance_scatter(targets, predictions)
    fig.savefig(f"{status_dir}/{name_prefix}_scatter.png")
    plt.close("all")
    
def kde_plot(ax, x, y, plot_setup):
    """Create KDE density plot"""
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    scatter = ax.scatter(x, y, c=z, **plot_setup)
    return scatter

def plot_radiance_comparison(rad_pred, rad_label, scaler, status_dir, prefix, n_samples=10):
    """Plot radiance comparison between predictions and labels"""
    plt.close('all')
    
    # Inverse transform
    unscaled_rad_pred = rad_pred.to("cpu")
    unscaled_rad_label = scaler.inverse_transform(rad_label).to("cpu")
    
    idxs = np.random.randint(0, len(unscaled_rad_label), n_samples)
    
    for i in range(n_samples):
        idx = idxs[i]
        plot_label = unscaled_rad_label[idx, :].detach().cpu().numpy()
        plot_predict = unscaled_rad_pred[idx, :].detach().cpu().numpy()
        plot_predict[np.isnan(plot_label)] = np.nan
        
        for j in range(3):
            plot_radiance(plot_label[j*1016:(j+1)*1016], plot_predict[j*1016:(j+1)*1016])
        
            plt.savefig(f"{status_dir}/{prefix}_radiance_{j}_{i}.png", bbox_inches='tight')
    plt.close('all')

def plot_co2_profiles(ret_pred, ret_label, co2_indices, scaler, status_dir, prefix, n_plots=4, n_cols=7):
    """Plot CO2 profiles comparison"""
    plt.close('all')
    
    # Inverse transform
    ret_pred_unscaled = scaler.inverse_transform(ret_pred.to("cpu"))
    ret_label_unscaled = scaler.inverse_transform(ret_label.to("cpu"))
    
    co2_profile_pred = ret_pred_unscaled[:, co2_indices].detach().cpu().numpy() * 1e6
    co2_profile_label = ret_label_unscaled[:, co2_indices].detach().cpu().numpy() * 1e6
    
    for plot_idx in range(n_plots):
        fig, axs = plt.subplots(1, n_cols, figsize=[n_cols, 3])
        for i in range(len(axs)):
            idx = np.random.randint(0, len(ret_pred_unscaled))
            co2_pred = co2_profile_pred[idx, :]
            co2_label = co2_profile_label[idx, :]
            line1, = axs[i].plot(co2_label, np.arange(20), label="Label", color="blue")
            line2, = axs[i].plot(co2_pred, np.arange(20), label="Pred", color="red", linestyle="--")
            if i != 0:
                axs[i].set_yticklabels([])
        
        axs[3].set_xlabel("CO$_2$ Profile [ppm]")
        fig.legend([line1, line2], ["Label", "Pred"], loc="upper center", ncol=2)
        plt.savefig(f"{status_dir}/{prefix}_co2_profiles_{plot_idx}.png", bbox_inches='tight')
    
        plt.close('all')
        
def plot_co2_profiles_with_uncertainty(ret_pred, ret_label, co2_indices, co2_uncert, scaler, status_dir, prefix, n_plots=4, n_cols=7):
    """Plot CO2 profiles comparison"""
    plt.close('all')
    
    # Inverse transform
    ret_pred_unscaled = scaler.inverse_transform(ret_pred.to("cpu"))
    ret_label_unscaled = scaler.inverse_transform(ret_label.to("cpu"))
    
    co2_profile_pred = ret_pred_unscaled[:, co2_indices].detach().cpu().numpy() * 1e6
    co2_profile_label = ret_label_unscaled[:, co2_indices].detach().cpu().numpy() * 1e6
    co2_uncert = co2_uncert*1e6
    
    for plot_idx in range(n_plots):
        fig, axs = plt.subplots(1, n_cols, figsize=[n_cols, 3])
        for i in range(len(axs)):
            idx = np.random.randint(0, len(ret_pred_unscaled))
            co2_pred = co2_profile_pred[idx, :]
            co2_label = co2_profile_label[idx, :]
            line1, = axs[i].plot(co2_label, np.arange(20), label="Label", color="blue")
            axs[i].fill_betweenx(np.arange(20), co2_label - co2_uncert[idx, :], co2_label + co2_uncert[idx, :], color="blue", alpha=0.2)
            line2, = axs[i].plot(co2_pred, np.arange(20), label="Pred", color="red", linestyle="--")
            axs[i].xaxis.set_major_locator(MaxNLocator(2))
            if i != 0:
                axs[i].set_yticklabels([])
        
        axs[3].set_xlabel("CO$_2$ Profile [ppm]")
        fig.legend([line1, line2], ["Label", "Pred"], loc="upper center", ncol=2)
        plt.savefig(f"{status_dir}/{prefix}_co2_profiles_{plot_idx}.png", bbox_inches='tight')
    
        plt.close('all')


def plot_co2_profiles_with_uncertainty_raw(co2_profile_pred, co2_profile_label, co2_uncert, status_dir, prefix, uncert_target, n_plots=4, n_cols=7):
    """Plot CO2 profiles comparison"""
    plt.close('all')
    
    co2_profile_pred *= 1e6
    co2_profile_label *= 1e6
    co2_uncert *= 1e6
    
    for plot_idx in range(n_plots):
        fig, axs = plt.subplots(1, n_cols, figsize=[n_cols, 3])
        for i in range(len(axs)):
            idx = np.random.randint(0, len(co2_profile_pred))
            co2_pred = co2_profile_pred[idx, :]
            co2_label = co2_profile_label[idx, :]
            line1, = axs[i].plot(co2_label, np.arange(20), label="Label", color="blue")
            if uncert_target == "label":
                axs[i].fill_betweenx(np.arange(20), co2_label - co2_uncert[idx, :], co2_label + co2_uncert[idx, :], color="blue", alpha=0.2)
            elif uncert_target == "pred":
                axs[i].fill_betweenx(np.arange(20), co2_pred - co2_uncert[idx, :], co2_pred + co2_uncert[idx, :], color="red", alpha=0.2)
            line2, = axs[i].plot(co2_pred, np.arange(20), label="Pred", color="red", linestyle="--")
            axs[i].xaxis.set_major_locator(MaxNLocator(2))
            if i != 0:
                axs[i].set_yticklabels([])
        
        axs[3].set_xlabel("CO$_2$ Profile [ppm]")
        fig.legend([line1, line2], ["Label", "Pred"], loc="upper center", ncol=2)
        plt.savefig(f"{status_dir}/{prefix}_co2_profiles_{plot_idx}.png", bbox_inches='tight')
    
        plt.close('all')

def plot_co2_profiles_with_two_uncertainty_raw(co2_profile_pred, co2_profile_label, co2_uncert_dict: Dict[str, np.ndarray], status_dir, prefix, n_plots=4, n_cols=7):
    """Plot CO2 profiles comparison"""
    plt.close('all')
    
    co2_profile_pred = co2_profile_pred * 1e6
    co2_profile_label = co2_profile_label * 1e6
    
    for plot_idx in range(n_plots):
        fig, axs = plt.subplots(1, n_cols, figsize=[n_cols, 3])
        for i in range(len(axs)):
            idx = np.random.randint(0, len(co2_profile_pred))
            co2_pred = co2_profile_pred[idx, :]
            co2_label = co2_profile_label[idx, :]
            line1, = axs[i].plot(co2_label, np.arange(20), label="Label", color="blue")
            if "label" in co2_uncert_dict:
                co2_uncert = co2_uncert_dict["label"] * 1e6
                axs[i].fill_betweenx(np.arange(20), co2_label - co2_uncert[idx, :], co2_label + co2_uncert[idx, :], color="blue", alpha=0.2)
            if "pred" in co2_uncert_dict:
                co2_uncert = co2_uncert_dict["pred"] * 1e6
                axs[i].fill_betweenx(np.arange(20), co2_pred - co2_uncert[idx, :], co2_pred + co2_uncert[idx, :], color="red", alpha=0.2)
            line2, = axs[i].plot(co2_pred, np.arange(20), label="Pred", color="red", linestyle="--")
            axs[i].xaxis.set_major_locator(MaxNLocator(2))
            if i != 0:
                axs[i].set_yticklabels([])
        
        axs[3].set_xlabel("CO$_2$ Profile [ppm]")
        fig.legend([line1, line2], ["Label", "Pred"], loc="upper center", ncol=2)
        plt.savefig(f"{status_dir}/{prefix}_co2_profiles_{plot_idx}.png", bbox_inches='tight')
    
        plt.close('all')

def plot_xco2_scatter(ret_pred, ret_label, wf_pred, wf_label, co2_indices, scalers, status_dir, prefix):
    """Plot XCO2 scatter plot with KDE"""
    plt.close('all')
    
    # Inverse transform
    ret_pred_unscaled = scalers["retrieved"].inverse_transform(ret_pred.to("cpu"))
    ret_label_unscaled = scalers["retrieved"].inverse_transform(ret_label.to("cpu"))
    
    wf_pred_unscaled = scalers["wf"].inverse_transform(wf_pred.to("cpu")).detach().cpu().numpy()
    wf_label_unscaled = scalers["wf"].inverse_transform(wf_label.to("cpu")).detach().cpu().numpy()
    
    co2_profile_pred = ret_pred_unscaled[:, co2_indices].detach().cpu().numpy() * 1e6
    co2_profile_label = ret_label_unscaled[:, co2_indices].detach().cpu().numpy() * 1e6
    
    xco2_pred = np.sum(wf_pred_unscaled * co2_profile_pred, axis=1)
    xco2_label = np.sum(wf_label_unscaled * co2_profile_label, axis=1)
    
    idx = np.random.randint(0, len(xco2_pred), 10000)
    xco2_pred = xco2_pred[idx]
    xco2_label = xco2_label[idx]
    
    fig, ax = plt.subplots(1, 1, figsize=[3, 3])
    kde_plot(ax, xco2_label, xco2_pred, {"s": 1, "alpha": 0.5})
    
    rmse = np.sqrt(np.mean((xco2_pred - xco2_label) ** 2))
    mean_error = np.mean(xco2_pred - xco2_label)
    
    vmin, vmax = ax.get_xlim()
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.plot([vmin, vmax], [vmin, vmax], color="black", linestyle="--")
    ax.text(vmin, vmax, f"RMSE: {rmse:.3f} ppm\nMean Error: {mean_error:.3f} ppm", fontsize=10, ha="left", va="top")
    
    ax.set_xlabel("Label [ppm]")
    ax.set_ylabel("Pred [ppm]")
    
    plt.savefig(f"{status_dir}/{prefix}_xco2.png", bbox_inches='tight')
    plt.close('all')


def plot_xco2_scatter_raw(xco2_pred, xco2_label, status_dir, prefix):
    """Plot XCO2 scatter plot with KDE"""
    plt.close('all')
    xco2_pred = xco2_pred * 1e6
    xco2_label = xco2_label * 1e6
    
    idx = np.random.randint(0, len(xco2_pred), 10000)
    xco2_pred = xco2_pred[idx]
    xco2_label = xco2_label[idx]
    
    fig, ax = plt.subplots(1, 1, figsize=[3, 3])
    kde_plot(ax, xco2_label, xco2_pred, {"s": 1, "alpha": 0.5})
    
    rmse = np.sqrt(np.mean((xco2_pred - xco2_label) ** 2))
    mean_error = np.mean(xco2_pred - xco2_label)
    
    vmin, vmax = ax.get_xlim()
    ax.set_xlim(vmin, vmax)
    ax.set_ylim(vmin, vmax)
    ax.plot([vmin, vmax], [vmin, vmax], color="black", linestyle="--")
    ax.text(vmin, vmax, f"RMSE: {rmse:.3f} ppm\nMean Error: {mean_error:.3f} ppm", fontsize=10, ha="left", va="top")
    
    ax.set_xlabel("Label [ppm]")
    ax.set_ylabel("Pred [ppm]")
    
    plt.savefig(f"{status_dir}/{prefix}_xco2.png", bbox_inches='tight')
    plt.close('all')
