import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import List, Optional, Union, Tuple
import os
from pathlib import Path
import arviz as az

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif']
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 12
mpl.rcParams['pdf.fonttype'] = 42  
mpl.rcParams['ps.fonttype'] = 42   


def _calculate_grid_layout(n_plots: int) -> Tuple[int, int]:
    if n_plots <= 3:
        return n_plots, 1
    elif n_plots <= 6:
        return 2, 3
    elif n_plots <= 9:
        return 3, 3
    elif n_plots <= 12:
        return 3, 4
    elif n_plots <= 16:
        return 4, 4
    elif n_plots <= 20:
        return 4, 5
    elif n_plots <= 25:
        return 5, 5
    else:
        ncols = int(np.ceil(np.sqrt(n_plots)))
        nrows = int(np.ceil(n_plots / ncols))
        return nrows, ncols


def plot_parameter_traces(
    mcmc_results: List,
    param_name: str,
    output_path: str,
    true_value: Optional[float] = None,
    format: str = 'pdf',
    dpi: int = 300,
    figsize: Optional[Tuple[float, float]] = None,
    title: Optional[str] = None,
) -> None:
    n_reps = len(mcmc_results)

    nrows, ncols = _calculate_grid_layout(n_reps)

    if figsize is None:
        width = 7.0 
        height = 1.8 * nrows  
        figsize = (width, height)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    if title is None:
        param_display = param_name.replace('_', ' ').replace('epsilon', 'ε')
        title = f'Trace Plots: {param_display}'
    fig.suptitle(title, fontsize=12, y=0.995)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  

    for rep_idx, idata in enumerate(mcmc_results):
        ax = axes[rep_idx]

        if param_name not in idata.posterior:
            ax.text(0.5, 0.5, f'Rep {rep_idx+1}\nNo data',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        traces = idata.posterior[param_name].values  
        n_chains, n_draws = traces.shape

        for chain_idx in range(n_chains):
            color = colors[chain_idx % len(colors)]
            ax.plot(traces[chain_idx, :], color=color, alpha=0.7,
                   linewidth=0.5, label=f'Chain {chain_idx+1}' if rep_idx == 0 else '')

        if true_value is not None:
            ax.axhline(true_value, color='black', linestyle='--',
                      linewidth=1.0, alpha=0.6, label='True value' if rep_idx == 0 else '')

        try:
            rhat_dict = az.rhat(idata)
            if param_name in rhat_dict:
                rhat_val = float(rhat_dict[param_name].values)
                rhat_color = 'green' if rhat_val < 1.01 else ('orange' if rhat_val < 1.05 else 'red')
                ax.text(0.98, 0.98, f'R̂={rhat_val:.3f}',
                       transform=ax.transAxes,
                       ha='right', va='top',
                       fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=rhat_color, alpha=0.3))
        except:
            pass

        ax.set_title(f'Rep {rep_idx+1}', fontsize=9)
        ax.set_xlabel('Iteration', fontsize=9)
        if rep_idx % ncols == 0: 
            ax.set_ylabel(param_name.replace('_', ' '), fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)

        if rep_idx == 0 and n_chains > 1:
            ax.legend(loc='upper left', fontsize=7, framealpha=0.8)

    for idx in range(n_reps, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = f"{output_path}.{format}"
    plt.savefig(output_file, format=format, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved trace plot to: {output_file}")


def plot_r_and_p_epsilon_traces(
    mcmc_results: List,
    output_path: str,
    true_r: Optional[float] = None,
    true_p_epsilon: Optional[float] = None,
    format: str = 'pdf',
    dpi: int = 300,
    figsize: Optional[Tuple[float, float]] = None,
) -> None:
    n_reps = len(mcmc_results)

    if figsize is None:
        width = 7.0  
        height = 1.5 * n_reps  
        figsize = (width, height)

    fig, axes = plt.subplots(n_reps, 2, figsize=figsize, squeeze=False)

    fig.suptitle('Trace Plots: r and p_ε', fontsize=12, y=0.995)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for rep_idx, idata in enumerate(mcmc_results):
        ax_r = axes[rep_idx, 0]
        if 'r' in idata.posterior:
            traces_r = idata.posterior['r'].values  
            n_chains, n_draws = traces_r.shape

            for chain_idx in range(n_chains):
                ax_r.plot(traces_r[chain_idx, :],
                         color=colors[chain_idx % len(colors)],
                         alpha=0.7, linewidth=0.5)

            if true_r is not None:
                ax_r.axhline(true_r, color='black', linestyle='--',
                           linewidth=1.0, alpha=0.6)

            try:
                rhat_dict = az.rhat(idata)
                if 'r' in rhat_dict:
                    rhat_val = float(rhat_dict['r'].values)
                    rhat_color = 'green' if rhat_val < 1.01 else ('orange' if rhat_val < 1.05 else 'red')
                    ax_r.text(0.98, 0.98, f'R̂={rhat_val:.3f}',
                            transform=ax_r.transAxes,
                            ha='right', va='top', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=rhat_color, alpha=0.3))
            except:
                pass

        ax_r.set_title(f'Rep {rep_idx+1}: r', fontsize=9)
        ax_r.set_xlabel('Iteration', fontsize=9)
        ax_r.set_ylabel('r', fontsize=9)
        ax_r.tick_params(labelsize=8)
        ax_r.grid(True, alpha=0.3, linewidth=0.5)

        ax_p = axes[rep_idx, 1]
        if 'p_epsilon' in idata.posterior:
            traces_p = idata.posterior['p_epsilon'].values 
            n_chains, n_draws = traces_p.shape

            for chain_idx in range(n_chains):
                ax_p.plot(traces_p[chain_idx, :],
                         color=colors[chain_idx % len(colors)],
                         alpha=0.7, linewidth=0.5)

            if true_p_epsilon is not None:
                ax_p.axhline(true_p_epsilon, color='black', linestyle='--',
                           linewidth=1.0, alpha=0.6)

            try:
                rhat_dict = az.rhat(idata)
                if 'p_epsilon' in rhat_dict:
                    rhat_val = float(rhat_dict['p_epsilon'].values)
                    rhat_color = 'green' if rhat_val < 1.01 else ('orange' if rhat_val < 1.05 else 'red')
                    ax_p.text(0.98, 0.98, f'R̂={rhat_val:.3f}',
                            transform=ax_p.transAxes,
                            ha='right', va='top', fontsize=8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=rhat_color, alpha=0.3))
            except:
                pass

        ax_p.set_title(f'Rep {rep_idx+1}: p_ε', fontsize=9)
        ax_p.set_xlabel('Iteration', fontsize=9)
        ax_p.set_ylabel('p_ε', fontsize=9)
        ax_p.tick_params(labelsize=8)
        ax_p.grid(True, alpha=0.3, linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    output_file = f"{output_path}.{format}"
    plt.savefig(output_file, format=format, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved combined trace plot to: {output_file}")


def plot_density_comparison(
    mcmc_results: List,
    param_name: str,
    output_path: str,
    true_value: Optional[float] = None,
    format: str = 'pdf',
    dpi: int = 300,
    figsize: Tuple[float, float] = (7, 5),
) -> None:
    fig, ax = plt.subplots(figsize=figsize)

    for rep_idx, idata in enumerate(mcmc_results):
        if param_name in idata.posterior:
            samples = idata.posterior[param_name].values.flatten()
            ax.hist(samples, bins=30, alpha=0.3, density=True,
                   label=f'Rep {rep_idx+1}')

    if true_value is not None:
        ax.axvline(true_value, color='red', linestyle='--', linewidth=2,
                  label='True value')

    ax.set_xlabel(param_name.replace('_', ' '))
    ax.set_ylabel('Density')
    ax.set_title(f'Posterior Distributions: {param_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = f"{output_path}.{format}"
    plt.savefig(output_file, format=format, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved density plot to: {output_file}")


def create_trace_plots_for_condition(
    mcmc_results: List,
    sim_link: str,
    r_value: Optional[float],
    est_link: str,
    N: int,
    I: int,
    R: int,
    output_dir: str,
    true_r: Optional[float] = None,
    true_p_epsilon: Optional[float] = None,
    true_tau_epsilon: Optional[float] = None,
    formats: List[str] = ['pdf'],
    dpi: int = 300,
) -> None:
    figures_dir = Path(output_dir) / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    r_str = str(r_value).replace('.', 'p') if r_value is not None else 'noR'
    base_name = f"traces_sim_{sim_link}_r_{r_str}_est_{est_link}_N{N}_I{I}_R{R}"

    has_r = any('r' in idata.posterior for idata in mcmc_results)
    has_p_epsilon = any('p_epsilon' in idata.posterior for idata in mcmc_results)
    has_tau_epsilon = any('tau_epsilon' in idata.posterior for idata in mcmc_results)

    for fmt in formats:
        if has_r and est_link in ['plogit', 'splogit']:
            output_path = figures_dir / f"{base_name}_r"
            plot_parameter_traces(
                mcmc_results=mcmc_results,
                param_name='r',
                output_path=str(output_path),
                true_value=true_r,
                format=fmt,
                dpi=dpi,
            )

        if has_p_epsilon and est_link == 'splogit':
            output_path = figures_dir / f"{base_name}_p_epsilon"
            plot_parameter_traces(
                mcmc_results=mcmc_results,
                param_name='p_epsilon',
                output_path=str(output_path),
                true_value=true_p_epsilon,
                format=fmt,
                dpi=dpi,
            )

        if has_tau_epsilon and est_link == 'splogit':
            output_path = figures_dir / f"{base_name}_tau_epsilon"
            plot_parameter_traces(
                mcmc_results=mcmc_results,
                param_name='tau_epsilon',
                output_path=str(output_path),
                true_value=true_tau_epsilon,
                format=fmt,
                dpi=dpi,
            )

        if has_r and has_p_epsilon and est_link == 'splogit':
            output_path = figures_dir / f"{base_name}_both"
            plot_r_and_p_epsilon_traces(
                mcmc_results=mcmc_results,
                output_path=str(output_path),
                true_r=true_r,
                true_p_epsilon=true_p_epsilon,
                format=fmt,
                dpi=dpi,
            )

    print(f"✓ All trace plots saved to: {figures_dir}")
