from __future__ import annotations
from dataclasses import dataclass
import os
from typing import Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from codes.simulate import Params
    from codes.recovery import RecoveryMetrics, ModelSelectionResults


@dataclass
class SaveConfig:
    output_dir: str
    sim_link: str
    r: float
    est_link: str
    N: int
    I: int
    R: int
    save_true_params: bool = False
    save_simulated_data: bool = False
    save_mcmc_results: bool = False
    save_log_likelihood: bool = False
    save_recovery_metrics: bool = False
    save_model_selection: bool = False

    def get_data_dir(self) -> str:
        r_str = str(self.r).replace('.', 'p') if self.sim_link in ['plogit', 'splogit'] else 'noR'
        dir_path = os.path.join(
            self.output_dir,
            'data',
            f'sim_{self.sim_link}_r_{r_str}_est_{self.est_link}'
        )
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def get_simulated_data_path(self) -> str:
        return os.path.join(
            self.get_data_dir(),
            f'simulated_data_N{self.N}_I{self.I}_R{self.R}.npz'
        )

    def get_mcmc_path(self, rep_idx: int) -> str:
        ext = 'nc' if self.save_log_likelihood else 'npz'
        return os.path.join(
            self.get_data_dir(),
            f'mcmc_rep{rep_idx:03d}_N{self.N}_I{self.I}.{ext}'
        )

    def get_true_params_path(self) -> str:
        return os.path.join(
            self.get_data_dir(),
            f'true_params_N{self.N}_I{self.I}.npz'
        )

    def get_recovery_metrics_path(self) -> str:
        return os.path.join(
            self.get_data_dir(),
            f'recovery_metrics_N{self.N}_I{self.I}_R{self.R}.npz'
        )

    def get_model_selection_path(self) -> str:
        return os.path.join(
            self.get_data_dir(),
            f'model_selection_N{self.N}_I{self.I}_R{self.R}.npz'
        )


def save_simulated_data(
    y_list: list[np.ndarray],
    save_path: str,
) -> None:
    y_array = np.array(y_list)
    np.savez(save_path, y=y_array)


def save_true_params(
    true_params: 'Params',
    save_path: str,
) -> None:
    save_dict = {
        'sim_link': true_params.sim_link,
        'a': true_params.a,
        'b': true_params.b,
        'mu': true_params.mu,
        'r': true_params.r,
    }

    if true_params.epsilon is not None:
        save_dict['epsilon'] = true_params.epsilon
    if true_params.z is not None:
        save_dict['z'] = true_params.z
    if true_params.p_epsilon is not None:
        save_dict['p_epsilon'] = true_params.p_epsilon
    if true_params.xi is not None:
        save_dict['xi'] = true_params.xi
    if true_params.lambda_skew is not None:
        save_dict['lambda_skew'] = true_params.lambda_skew
    if true_params.hermite_coeffs is not None:
        save_dict['hermite_coeffs'] = true_params.hermite_coeffs
    if true_params.alpha1 is not None:
        save_dict['alpha1'] = true_params.alpha1
    if true_params.alpha2 is not None:
        save_dict['alpha2'] = true_params.alpha2

    np.savez(save_path, **save_dict)


def save_recovery_metrics(
    metrics: 'RecoveryMetrics',
    save_path: str,
) -> None:
    save_dict = {}

    for param_name in ['a', 'b', 'mu', 'r']:
        param_metrics = getattr(metrics, f'metrics_{param_name}', None)
        if param_metrics is not None:
            metrics_dict = param_metrics.as_dict()

            for metric_name, metric_values in metrics_dict.items():
                key = f'{param_name}_{metric_name}'
                save_dict[key] = metric_values

                metric_array = np.asarray(metric_values)
                if metric_array.ndim == 0:
                    scalar_val = float(metric_array)
                    save_dict[f'{param_name}_{metric_name}_mean'] = scalar_val
                    save_dict[f'{param_name}_{metric_name}_2.5%'] = scalar_val
                    save_dict[f'{param_name}_{metric_name}_97.5%'] = scalar_val
                elif metric_array.size > 0:
                    save_dict[f'{param_name}_{metric_name}_mean'] = np.mean(metric_array)
                    save_dict[f'{param_name}_{metric_name}_2.5%'] = np.percentile(metric_array, 2.5)
                    save_dict[f'{param_name}_{metric_name}_97.5%'] = np.percentile(metric_array, 97.5)

    theta_table = metrics.get_theta_metrics_table()
    if theta_table is not None:
        for _, row in theta_table.iterrows():
            metric_name = row['Metric'].lower()
            save_dict[f'theta_{metric_name}_mean'] = row['Mean']
            save_dict[f'theta_{metric_name}_2.5%'] = row['2.5%']
            save_dict[f'theta_{metric_name}_97.5%'] = row['97.5%']

    np.savez(save_path, **save_dict)


def save_model_selection(
    model_sel: 'ModelSelectionResults',
    save_path: str,
) -> None:
    save_dict = {
        'dic_mean': model_sel.dic_mean,
        'dic_std': model_sel.dic_std,
        'lpml_mean': model_sel.lpml_mean,
        'lpml_std': model_sel.lpml_std,
    }

    if hasattr(model_sel, 'dic_list') and model_sel.dic_list is not None:
        save_dict['dic_list'] = np.array(model_sel.dic_list)
    if hasattr(model_sel, 'lpml_list') and model_sel.lpml_list is not None:
        save_dict['lpml_list'] = np.array(model_sel.lpml_list)

    np.savez(save_path, **save_dict)
