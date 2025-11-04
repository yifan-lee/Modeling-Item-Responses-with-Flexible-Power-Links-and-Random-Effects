from __future__ import annotations

from typing import Optional, Literal, Union
from dataclasses import dataclass
import hashlib
import os
import time
import numpy as np
import pandas as pd
import arviz as az



try:
    from codes.save_utils import (
        SaveConfig,
        save_simulated_data as save_sim_data_func,
        save_true_params as save_true_params_func,
        save_recovery_metrics as save_recovery_metrics_func,
        save_model_selection as save_model_selection_func,
    )
    from codes.simulate import (
        HyperParams,
        sample_params,
        generate_responses,
        LinkType,
        Params,
    )
    from codes.mcmc import run_mcmc, save_mcmc_posterior
    from codes.metrics import compute_mcmc_metrics, MCMCMetrics
    from codes.model_selection import compute_dic, compute_lpml
    from codes.plots import create_trace_plots_for_condition
except ModuleNotFoundError:
    from save_utils import (
        SaveConfig,
        save_simulated_data as save_sim_data_func,
        save_true_params as save_true_params_func,
        save_recovery_metrics as save_recovery_metrics_func,
        save_model_selection as save_model_selection_func,
    )
    from simulate import (
        HyperParams,
        sample_params,
        generate_responses,
        LinkType,
        Params,
    )
    from mcmc import run_mcmc, save_mcmc_posterior
    from metrics import compute_mcmc_metrics, MCMCMetrics
    from model_selection import compute_dic, compute_lpml
    from plots import create_trace_plots_for_condition


@dataclass(frozen=True)
class SeedGen:
    master_seed: int | str = 20250101

    def _key(self) -> bytes:
        from decimal import Decimal
        s = str(self.master_seed).encode("utf-8")
        return hashlib.blake2b(s, digest_size=32).digest()

    @staticmethod
    def _norm(v) -> str:
        from decimal import Decimal
        if isinstance(v, float):
            return str(Decimal(str(v)))
        return str(v)

    def _derive(self, *tags, bits: int = 32) -> int:
        h = hashlib.blake2b(key=self._key(), digest_size=8)
        for t in tags:
            h.update(b"|")
            h.update(self._norm(t).encode("utf-8"))
        n = int.from_bytes(h.digest(), "big")
        if bits >= 64:
            return n
        mask = (1 << bits) - 1
        return n & mask

    def seed_param(self, model: str, r: float, sim_idx: int) -> int:
        return self._derive("param", model, r, sim_idx, bits=32)

    def seed_data(self, model: str, r: float, sim_idx: int, group_id: int) -> int:
        return self._derive("data", model, r, sim_idx, group_id, bits=32)

    def seed_estimate(self, est_model: str, src_model: str, r: float, sim_idx: int, est_idx: int) -> int:
        return self._derive("estimate", est_model, src_model, r, sim_idx, est_idx, bits=32)




def step1_sample_parameters(
    N: int,
    I: int,
    sim_link: LinkType,
    sim_r: Optional[float] = None,
    sim_p_epsilon: Optional[float] = None,
    sim_hyper: Optional[HyperParams] = None,
    seed: int = 42,
) -> Params:
    if sim_hyper is None:
        sim_hyper = HyperParams()

    true_params = sample_params(
        N=N,
        I=I,
        hyper=sim_hyper,
        link=sim_link,
        r=sim_r,
        p_epsilon=sim_p_epsilon,
        seed=seed,
    )

    return true_params



def step2_simulate_responses(
    true_params: Params,
    sim_link: LinkType,
    seed: int = 42,
    return_prob: bool = False,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    result = generate_responses(
        params=true_params,
        link=sim_link,
        seed=seed,
        return_prob=return_prob,
    )

    return result


def _run_single_mcmc_worker(args):
    y_i, seed_i, est_link, sampler, draws, tune, chains, mcmc_kwargs = args

    try:
        idata = run_mcmc(
            y=y_i,
            link=est_link,
            sampler=sampler,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=1,  # Always 1 in parallel mode
            random_seed=seed_i,
            progressbar=False,
            silence_output=True,
            **mcmc_kwargs,
        )
        return {'success': True, 'idata': idata, 'error': None}
    except Exception as e:
        return {'success': False, 'idata': None, 'error': str(e)}


@dataclass
class MCMCResults:
    idatas: list[az.InferenceData]
    y_list: list[np.ndarray]
    est_link: LinkType

    @property
    def n_replications(self) -> int:
        return len(self.idatas)

    def __getitem__(self, idx: int) -> tuple[az.InferenceData, np.ndarray]:
        return self.idatas[idx], self.y_list[idx]

    def plot_trace(
        self,
        var_names: Optional[Union[str, list[str]]] = None,
        replication: int = 0,
        indices: Optional[Union[int, list[int]]] = None,
        figsize: Optional[tuple[int, int]] = None,
        **kwargs
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")

        if replication < 0 or replication >= self.n_replications:
            raise ValueError(f"replication must be in [0, {self.n_replications-1}], got {replication}")

        idata = self.idatas[replication] 

        if var_names is None:
            var_names_to_plot = None
            coords = None
        else:
            if isinstance(var_names, str):
                var_names = [var_names]

            var_names_to_plot = []
            coords = {}

            for var in var_names:
                var = var.strip()

                if var in {'theta', 'ability'}:
                    var = 'mu'
                elif var in {'discrimination'}:
                    var = 'a'
                elif var in {'difficulty'}:
                    var = 'b'

                if var not in idata.posterior:  
                    available_vars = list(idata.posterior.data_vars)  
                    raise ValueError(f"Variable '{var}' not found in posterior. Available: {available_vars}")

                var_names_to_plot.append(var)

                if var in ['a', 'b', 'mu'] and indices is not None:
                    if isinstance(indices, int):
                        indices_list = [indices]
                    else:
                        indices_list = list(indices)

                    if var == 'a' or var == 'b':
                        dim_name = 'item'
                    else: 
                        dim_name = 'person'

                    if dim_name not in coords:
                        coords[dim_name] = indices_list
                    else:
                        coords[dim_name] = sorted(set(coords[dim_name]) | set(indices_list))

        if coords:
            axes = az.plot_trace(idata, var_names=var_names_to_plot, coords=coords, figsize=figsize, **kwargs)
        else:
            axes = az.plot_trace(idata, var_names=var_names_to_plot, figsize=figsize, **kwargs)

        plt.tight_layout()
        return axes

    def plot_posterior(
        self,
        var_names: Optional[Union[str, list[str]]] = None,
        replication: int = 0,
        indices: Optional[Union[int, list[int]]] = None,
        figsize: Optional[tuple[int, int]] = None,
        **kwargs
    ):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install it with: pip install matplotlib")

        if replication < 0 or replication >= self.n_replications:
            raise ValueError(f"replication must be in [0, {self.n_replications-1}], got {replication}")

        idata = self.idatas[replication]

        if var_names is None:
            var_names_to_plot = None
            coords = None
        else:
            if isinstance(var_names, str):
                var_names = [var_names]

            var_names_to_plot = []
            coords = {}

            for var in var_names:
                var = var.strip()
                if var in {'theta', 'ability'}:
                    var = 'mu'
                elif var in {'discrimination'}:
                    var = 'a'
                elif var in {'difficulty'}:
                    var = 'b'

                if var not in idata.posterior:  
                    available_vars = list(idata.posterior.data_vars)  
                    raise ValueError(f"Variable '{var}' not found in posterior. Available: {available_vars}")

                var_names_to_plot.append(var)

                if var in ['a', 'b', 'mu'] and indices is not None:
                    if isinstance(indices, int):
                        indices_list = [indices]
                    else:
                        indices_list = list(indices)

                    if var == 'a' or var == 'b':
                        dim_name = 'item'
                    else:
                        dim_name = 'person'

                    if dim_name not in coords:
                        coords[dim_name] = indices_list
                    else:
                        coords[dim_name] = sorted(set(coords[dim_name]) | set(indices_list))

        if coords:
            axes = az.plot_posterior(idata, var_names=var_names_to_plot, coords=coords, figsize=figsize, **kwargs)
        else:
            axes = az.plot_posterior(idata, var_names=var_names_to_plot, figsize=figsize, **kwargs)

        plt.tight_layout()
        return axes


def step3_run_mcmc(
    y: Union[np.ndarray, list[np.ndarray]],
    est_link: LinkType,
    sampler: Literal["nuts", "hmc", "gibbs", "slice", "mh"] = "nuts",
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
    seed: Union[int, list[int]] = 42,
    show_progress: bool = True,
    n_jobs: int = 1,
    verbose: bool = True,
    save_config: Optional['SaveConfig'] = None,
    **mcmc_kwargs
) -> MCMCResults:
    if isinstance(y, np.ndarray):
        y_list = [y]
        single_input = True
    else:
        y_list = list(y)
        single_input = False

    R = len(y_list)

    step3_specific_params = {'n_jobs', 'verbose', 'show_progress', 'progressbar'}
    filtered_mcmc_kwargs = {k: v for k, v in mcmc_kwargs.items() if k not in step3_specific_params}

    if isinstance(seed, int):
        seed_list = [seed + i for i in range(R)]
    else:
        seed_list = list(seed)
        if len(seed_list) != R:
            raise ValueError(f"Number of seeds ({len(seed_list)}) must match number of y arrays ({R})")

    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1

    actual_cores = 1 if n_jobs > 1 else cores

    if verbose and R > 1:
        if n_jobs > 1:
            print(f"\nRunning {R} MCMC replications in parallel with {n_jobs} workers...")
            print(f"  (Each MCMC uses cores=1 to avoid conflicts)")
        else:
            print(f"\nRunning {R} MCMC replications sequentially...")
            print(f"  (Each MCMC uses cores={actual_cores})")

    idatas = []

    if n_jobs == 1 or R == 1:
        from tqdm import tqdm

        show_rep_progress = show_progress and R > 1
        iterator = tqdm(range(R), desc="Replications", unit="rep", ncols=80) if show_rep_progress else range(R)

        for i in iterator:
            try:
                idata = run_mcmc(
                    y=y_list[i],
                    link=est_link,
                    sampler=sampler,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=actual_cores,
                    random_seed=seed_list[i],
                    progressbar=show_progress and (R == 1 or i == 0), 
                    silence_output=not show_progress or (R > 1 and i > 0),  
                    **filtered_mcmc_kwargs,
                )
                idatas.append(idata)
            except Exception as e:
                if verbose:
                    print(f"\n  ERROR in replication {i+1}: {e}")
                    print(f"  Skipping this replication.")
    else:
        from multiprocessing import Pool
        from tqdm import tqdm

        args_list = [
            (y_list[i], seed_list[i], est_link, sampler, draws, tune, chains, filtered_mcmc_kwargs)
            for i in range(R)
        ]

        with Pool(processes=n_jobs) as pool:
            if show_progress:
                results = []
                pbar = tqdm(
                    pool.imap(_run_single_mcmc_worker, args_list),
                    total=R,
                    desc="Replications",
                    unit="rep",
                    ncols=80,
                )
                for result in pbar:
                    results.append(result)
            else:
                results = pool.map(_run_single_mcmc_worker, args_list)

        for i, result in enumerate(results):
            if result['success']:
                idatas.append(result['idata'])
            elif verbose:
                print(f"\n  ERROR in replication {i+1}: {result['error']}")

    if verbose and R > 1:
        print(f"\n✓ Completed {len(idatas)}/{R} replications successfully")

    if save_config is not None and save_config.save_mcmc_results:
        if verbose:
            save_type = 'nc' if save_config.save_log_likelihood else 'npz'
            print(f"\nSave MCMC results as {save_type}...")

        for rep_idx, idata in enumerate(idatas):
            save_path = save_config.get_mcmc_path(rep_idx)
            save_mcmc_posterior(
                idata,
                save_path,
                save_full=save_config.save_log_likelihood
            )

        if verbose:
            print(f"✓ Save {len(idatas)} MCMC results to: {save_config.get_data_dir()}")

    return MCMCResults(
        idatas=idatas,
        y_list=[y_list[i] for i in range(len(idatas))],  
        est_link=est_link,
    )

@dataclass
class RecoveryMetrics:
    metrics_a: MCMCMetrics
    metrics_b: MCMCMetrics
    metrics_mu: MCMCMetrics
    metrics_r: Optional[MCMCMetrics] = None
    metrics_p_epsilon: Optional[MCMCMetrics] = None

    def print_summary(self):
        print("\n" + "=" * 80)
        print("RECOVERY METRICS SUMMARY")
        print("=" * 80)

        print("\nITEM DISCRIMINATION (a):")
        print(f"  Bias: {self.metrics_a.bias.mean():.4f} ± {self.metrics_a.bias.std():.4f}")
        print(f"  MSE:  {self.metrics_a.mse.mean():.4f} ± {self.metrics_a.mse.std():.4f}")
        print(f"  SD: {self.metrics_a.sd.mean():.4f} ± {self.metrics_a.sd.std():.4f}")
        print(f"  SE: {self.metrics_a.se.mean():.4f} ± {self.metrics_a.se.std():.4f}")
        print(f"  CP:   {self.metrics_a.cp.mean():.3f}")
        print(f"  Rhat: {self.metrics_a.rhat.mean():.4f} (max: {self.metrics_a.rhat.max():.4f})")

        print("\nITEM DIFFICULTY (b):")
        print(f"  Bias: {self.metrics_b.bias.mean():.4f} ± {self.metrics_b.bias.std():.4f}")
        print(f"  MSE:  {self.metrics_b.mse.mean():.4f} ± {self.metrics_b.mse.std():.4f}")
        print(f"  SD: {self.metrics_b.sd.mean():.4f} ± {self.metrics_b.sd.std():.4f}")
        print(f"  SE: {self.metrics_b.se.mean():.4f} ± {self.metrics_b.se.std():.4f}")
        print(f"  CP:   {self.metrics_b.cp.mean():.3f}")
        print(f"  Rhat: {self.metrics_b.rhat.mean():.4f} (max: {self.metrics_b.rhat.max():.4f})")

        print("\nPERSON ABILITY (theta):")
        print(f"  Bias: {self.metrics_mu.bias.mean():.4f} ± {self.metrics_mu.bias.std():.4f}")
        print(f"  MSE:  {self.metrics_mu.mse.mean():.4f} ± {self.metrics_mu.mse.std():.4f}")
        print(f"  SD: {self.metrics_mu.sd.mean():.4f} ± {self.metrics_mu.sd.std():.4f}")
        print(f"  SE: {self.metrics_mu.se.mean():.4f} ± {self.metrics_mu.se.std():.4f}")
        print(f"  CP:   {self.metrics_mu.cp.mean():.3f}")
        print(f"  Rhat: {self.metrics_mu.rhat.mean():.4f} (max: {self.metrics_mu.rhat.max():.4f})")

        if self.metrics_r is not None:
            print("\nLINK PARAMETER (r):")
            print(f"  Bias: {float(self.metrics_r.bias):.4f}")
            print(f"  MSE:  {float(self.metrics_r.mse):.4f}")
            print(f"  SD: {self.metrics_r.sd:.4f}")
            print(f"  SE: {self.metrics_r.se:.4f}")
            print(f"  CP:   {float(self.metrics_r.cp):.3f}")
            print(f"  Rhat: {float(self.metrics_r.rhat):.4f}")

        if self.metrics_p_epsilon is not None:
            print("\nSKEW PROBABILITY (p_epsilon):")
            print(f"  Bias: {float(self.metrics_p_epsilon.bias):.4f}")
            print(f"  MSE:  {float(self.metrics_p_epsilon.mse):.4f}")
            print(f"  SD: {self.metrics_p_epsilon.sd:.4f}")
            print(f"  SE: {self.metrics_p_epsilon.se:.4f}")
            print(f"  CP:   {float(self.metrics_p_epsilon.cp):.3f}")
            print(f"  Rhat: {float(self.metrics_p_epsilon.rhat):.4f}")

        print("=" * 80)

    def get_theta_metrics_table(self):
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for get_theta_metrics_table(). Install it with: pip install pandas")

        bias_arr = np.asarray(self.metrics_mu.bias).flatten()
        mse_arr = np.asarray(self.metrics_mu.mse).flatten()
        se_arr = np.asarray(self.metrics_mu.se).flatten()
        sd_arr = np.asarray(self.metrics_mu.sd).flatten()
        cp_arr = np.asarray(self.metrics_mu.cp).flatten()
        rhat_arr = np.asarray(self.metrics_mu.rhat).flatten()

        data = {
            'Metric': ['Bias', 'MSE', 'SE', 'SD', 'CP', 'Rhat'],
            'Mean': [
                bias_arr.mean(),
                mse_arr.mean(),
                se_arr.mean(),
                sd_arr.mean(),
                cp_arr.mean(),
                rhat_arr.mean()
            ],
            '2.5%': [
                np.percentile(bias_arr, 2.5),
                np.percentile(mse_arr, 2.5),
                np.percentile(se_arr, 2.5),
                np.percentile(sd_arr, 2.5),
                np.percentile(cp_arr, 2.5),
                np.percentile(rhat_arr, 2.5)
            ],
            '97.5%': [
                np.percentile(bias_arr, 97.5),
                np.percentile(mse_arr, 97.5),
                np.percentile(se_arr, 97.5),
                np.percentile(sd_arr, 97.5),
                np.percentile(cp_arr, 97.5),
                np.percentile(rhat_arr, 97.5)
            ]
        }

        return pd.DataFrame(data)


def step4_compute_metrics(
    true_params: Params,
    mcmc_results: Union[az.InferenceData, MCMCResults],
    est_link: Optional[LinkType] = None,
    credible_interval: float = 0.95,
) -> RecoveryMetrics:
    N = true_params.mu.shape[0]
    I = true_params.a.shape[0]

    if isinstance(mcmc_results, MCMCResults):
        idatas = mcmc_results.idatas
        if est_link is None:
            est_link = mcmc_results.est_link
    elif isinstance(mcmc_results, az.InferenceData):
        idatas = [mcmc_results]
        if est_link is None:
            raise ValueError("est_link must be provided when mcmc_results is InferenceData")
    else:
        raise TypeError(f"mcmc_results must be InferenceData or MCMCResults, got {type(mcmc_results)}")

    a_draws_list = []
    b_draws_list = []
    mu_draws_list = []
    r_draws_list = []
    p_epsilon_draws_list = []

    a_rhat_list = []
    b_rhat_list = []
    mu_rhat_list = []
    r_rhat_list = []
    p_epsilon_rhat_list = []

    for idata in idatas:
        post = idata.posterior  

        a_draws = post["a"].values.reshape(-1, I)
        b_draws = post["b"].values.reshape(-1, I)
        mu_draws = post["mu"].values.reshape(-1, N)

        a_draws_list.append(a_draws)
        b_draws_list.append(b_draws)
        mu_draws_list.append(mu_draws)


        rhat_dict = az.rhat(idata)
        a_rhat_list.append(rhat_dict["a"].values)   
        b_rhat_list.append(rhat_dict["b"].values)  
        mu_rhat_list.append(rhat_dict["mu"].values)  


        if est_link in {"plogit", "splogit"}:
            r_draws = post["r"].values.reshape(-1) 
            r_draws_list.append(r_draws)
            r_rhat_list.append(rhat_dict["r"].values) 

        if est_link == "splogit":
            if "p_epsilon" in post:
                p_epsilon_draws = post["p_epsilon"].values.reshape(-1) 
                p_epsilon_draws_list.append(p_epsilon_draws)
                p_epsilon_rhat_list.append(rhat_dict["p_epsilon"].values)  

    metrics_a = compute_mcmc_metrics(
        replications=a_draws_list,
        true_value=true_params.a,
        credible_interval=credible_interval,
        rhat_values=a_rhat_list,
    )

    metrics_b = compute_mcmc_metrics(
        replications=b_draws_list,
        true_value=true_params.b,
        credible_interval=credible_interval,
        rhat_values=b_rhat_list,
    )

    metrics_mu = compute_mcmc_metrics(
        replications=mu_draws_list,
        true_value=true_params.mu,
        credible_interval=credible_interval,
        rhat_values=mu_rhat_list,
    )

    metrics_r = None
    if r_draws_list:
        metrics_r = compute_mcmc_metrics(
            replications=r_draws_list,
            true_value=true_params.r,
            credible_interval=credible_interval,
            rhat_values=r_rhat_list,
        )

    metrics_p_epsilon = None
    if p_epsilon_draws_list:
        if hasattr(true_params, 'p_epsilon') and true_params.p_epsilon is not None:
            metrics_p_epsilon = compute_mcmc_metrics(
                replications=p_epsilon_draws_list,
                true_value=true_params.p_epsilon,
                credible_interval=credible_interval,
                rhat_values=p_epsilon_rhat_list,
            )

    return RecoveryMetrics(
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        metrics_mu=metrics_mu,
        metrics_r=metrics_r,
        metrics_p_epsilon=metrics_p_epsilon,
    )


@dataclass
class ModelSelectionMetrics:
    dic_values: list[float]
    pd_values: list[float]
    lpml_values: list[float]

    @property
    def n_replications(self) -> int:
        return len(self.dic_values)

    @property
    def dic_mean(self) -> float:
        return float(np.mean(self.dic_values))

    @property
    def dic_std(self) -> float:
        return float(np.std(self.dic_values))

    @property
    def lpml_mean(self) -> float:
        return float(np.mean(self.lpml_values))

    @property
    def lpml_std(self) -> float:
        return float(np.std(self.lpml_values))

    @property
    def pd_mean(self) -> float:
        return float(np.mean(self.pd_values))

    @property
    def pd_std(self) -> float:
        return float(np.std(self.pd_values))

    def print_summary(self):
        print("\n" + "=" * 80)
        print("MODEL SELECTION CRITERIA")
        print("=" * 80)

        if self.n_replications == 1:
            print(f"DIC:  {self.dic_values[0]:10.2f}  (lower is better)")
            print(f"pD:   {self.pd_values[0]:10.2f}  (effective # of parameters)")
            print(f"LPML: {self.lpml_values[0]:10.2f}  (higher is better)")
        else:
            print(f"Based on {self.n_replications} replications:")
            print(f"DIC:  {self.dic_mean:10.2f} ± {self.dic_std:8.2f}  (lower is better)")
            print(f"pD:   {self.pd_mean:10.2f} ± {self.pd_std:8.2f}  (effective # of parameters)")
            print(f"LPML: {self.lpml_mean:10.2f} ± {self.lpml_std:8.2f}  (higher is better)")

        print("=" * 80)


def _compute_model_selection_worker(args):
    y, idata, est_link, thin, rep_idx = args

    try:
        dic_results = compute_dic(y, idata, est_link)

        lpml_results = compute_lpml(y, idata, est_link, thin=thin)

        return {
            'success': True,
            'dic': dic_results["dic"],
            'pd': dic_results["pd"],
            'lpml': lpml_results["lpml"],
            'error': None,
            'rep_idx': rep_idx
        }
    except Exception as e:
        return {
            'success': False,
            'dic': None,
            'pd': None,
            'lpml': None,
            'error': str(e),
            'rep_idx': rep_idx
        }


def step5_compute_model_selection(
    mcmc_results: Union[tuple[np.ndarray, az.InferenceData], MCMCResults],
    est_link: Optional[LinkType] = None,
    thin: int = 1,
    n_jobs: int = 1,
    verbose: bool = False,
) -> ModelSelectionMetrics:
    if isinstance(mcmc_results, MCMCResults):
        y_list = mcmc_results.y_list
        idatas = mcmc_results.idatas
        if est_link is None:
            est_link = mcmc_results.est_link
    elif isinstance(mcmc_results, tuple) and len(mcmc_results) == 2:
        y, idata = mcmc_results
        y_list = [y] if isinstance(y, np.ndarray) else list(y)
        idatas = [idata] if isinstance(idata, az.InferenceData) else list(idata)
        if est_link is None:
            raise ValueError("est_link must be provided when mcmc_results is a tuple")
    else:
        raise TypeError(f"mcmc_results must be MCMCResults or tuple(y, idata), got {type(mcmc_results)}")

    if len(y_list) != len(idatas):
        raise ValueError(f"Number of y arrays ({len(y_list)}) must match number of idatas ({len(idatas)})")

    R = len(y_list)

    if n_jobs == -1:
        import os
        n_jobs = os.cpu_count() or 1

    if verbose and R > 1:
        if n_jobs > 1:
            print(f"\nComputing model selection criteria for {R} replications in parallel with {n_jobs} workers...")
        else:
            print(f"\nComputing model selection criteria for {R} replications sequentially...")

    dic_values = []
    pd_values = []
    lpml_values = []

    if n_jobs == 1 or R == 1:
        from tqdm import tqdm
        iterator = tqdm(range(R), desc="Computing DIC/LPML", unit="rep", ncols=80) if (verbose and R > 1) else range(R)

        for i in iterator:
            try:
                dic_results = compute_dic(y_list[i], idatas[i], est_link)
                dic_values.append(dic_results["dic"])
                pd_values.append(dic_results["pd"])

                lpml_results = compute_lpml(y_list[i], idatas[i], est_link, thin=thin)
                lpml_values.append(lpml_results["lpml"])

            except Exception as e:
                if verbose:
                    print(f"\n  WARNING: Failed to compute DIC/LPML for replication {i+1}: {e}")
                
                continue
    else:
        from multiprocessing import Pool
        from tqdm import tqdm

        args_list = [
            (y_list[i], idatas[i], est_link, thin, i)
            for i in range(R)
        ]

        with Pool(processes=n_jobs) as pool:
            if verbose:
                results = []
                pbar = tqdm(
                    pool.imap(_compute_model_selection_worker, args_list),
                    total=R,
                    desc="Computing DIC/LPML",
                    unit="rep",
                    ncols=80,
                )
                for result in pbar:
                    results.append(result)
            else:
                results = pool.map(_compute_model_selection_worker, args_list)

        for result in results:
            if result['success']:
                dic_values.append(result['dic'])
                pd_values.append(result['pd'])
                lpml_values.append(result['lpml'])
            elif verbose:
                print(f"\n  WARNING: Failed to compute DIC/LPML for replication {result['rep_idx']+1}: {result['error']}")

    if verbose and R > 1:
        print(f"✓ Computed DIC/LPML for {len(dic_values)}/{R} replications")

    return ModelSelectionMetrics(
        dic_values=dic_values,
        pd_values=pd_values,
        lpml_values=lpml_values,
    )


def run_multiple_replications(
    true_params: Params,
    sim_link: LinkType,
    est_link: LinkType,
    R: int,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
    sampler: Literal["nuts", "hmc", "gibbs", "slice", "mh"] = "nuts",
    seed: int = 42,
    n_jobs: int = 1,
    show_progress: bool = True,
    verbose: bool = True,
    compute_metrics: bool = True,
    compute_model_selection: bool = True,
    **mcmc_kwargs
) -> tuple[MCMCResults, Optional[RecoveryMetrics], Optional[ModelSelectionMetrics]]:
    if verbose:
        print(f"\n{'='*80}")
        print(f"Running {R} replications")
        print(f"Simulation: {sim_link}, Estimation: {est_link}")
        print(f"N={true_params.mu.shape[0]}, I={true_params.a.shape[0]}")
        print(f"{'='*80}")

    if verbose:
        print(f"\nStep 2: Generating {R} response datasets...")

    y_list = []
    for i in range(R):
        y = step2_simulate_responses(
            true_params=true_params,
            sim_link=sim_link,
            seed=seed + i,
        )
        y_list.append(y)

    if verbose:
        print(f"✓ Generated {R} datasets")

    if verbose:
        print(f"\nStep 3: Running MCMC on {R} datasets...")

    mcmc_results = step3_run_mcmc(
        y=y_list,
        est_link=est_link,
        sampler=sampler,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        seed=seed + R,  
        n_jobs=n_jobs,
        show_progress=show_progress,
        verbose=verbose,
        **mcmc_kwargs,
    )

    recovery_metrics = None
    if compute_metrics:
        if verbose:
            print(f"\nStep 4: Computing recovery metrics...")

        recovery_metrics = step4_compute_metrics(
            true_params=true_params,
            mcmc_results=mcmc_results,
        )

        if verbose:
            recovery_metrics.print_summary()

    model_selection = None
    if compute_model_selection:
        if verbose:
            print(f"\nStep 5: Computing model selection criteria...")

        model_selection = step5_compute_model_selection(
            mcmc_results=mcmc_results,
            n_jobs=n_jobs,  
            verbose=verbose,
        )

        if verbose:
            model_selection.print_summary()

    return mcmc_results, recovery_metrics, model_selection


def run_single_recovery(
    N: int,
    I: int,
    sim_link: LinkType,
    est_link: LinkType,
    sim_r: Optional[float] = None,
    sim_p_epsilon: Optional[float] = None,
    sim_hyper: Optional[HyperParams] = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 4,
    sampler: Literal["nuts", "hmc", "gibbs", "slice", "mh"] = "nuts",
    seed: int = 42,
    show_progress: bool = True,
    compute_model_selection: bool = True,
    **mcmc_kwargs
) -> tuple[RecoveryMetrics, Optional[ModelSelectionMetrics], MCMCResults]:
    true_params = step1_sample_parameters(
        N=N, I=I,
        sim_link=sim_link,
        sim_r=sim_r,
        sim_p_epsilon=sim_p_epsilon,
        sim_hyper=sim_hyper,
        seed=seed,
    )

    y = step2_simulate_responses(
        true_params=true_params,
        sim_link=sim_link,
        seed=seed + 1,  
    )

    mcmc_results = step3_run_mcmc(
        y=y,
        est_link=est_link,
        sampler=sampler,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        seed=seed + 2,  
        show_progress=show_progress,
        **mcmc_kwargs,
    )

    recovery_metrics = step4_compute_metrics(
        true_params=true_params,
        mcmc_results=mcmc_results,
        est_link=est_link,
    )

    model_selection_metrics = None
    if compute_model_selection:
        model_selection_metrics = step5_compute_model_selection(
            mcmc_results=mcmc_results,
            est_link=est_link,
        )

    return recovery_metrics, model_selection_metrics, mcmc_results

def run_batch_recovery(
    sim_links: list[str],
    r_values: list[float],
    est_links: list[str],
    N: int = 200,
    I: int = 40,
    R: int = 32,  
    sim_hyper: Optional[HyperParams] = None,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    sampler: str = 'nuts',
    n_jobs: int = -1,
    master_seed: int = 615,
    sim_idx: int = 0,
    est_idx: int = 0,

    show_progress:bool = True,
    progressbar:bool = False,
    output_dir: str = './outputs',
    output_prefix: str = 'recovery',
    save_true_params: bool = False,        
    save_simulated_data: bool = False,     
    save_mcmc_results: bool = False,       
    save_log_likelihood: bool = False,     
    save_recovery_metrics: bool = False,   
    save_model_selection: bool = False,   
    save_trace_plots: bool = False,        
    trace_plot_formats: list[str] = ['pdf'],  
    trace_plot_dpi: int = 300,            
    verbose: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)

    
    sg = SeedGen(master_seed=master_seed)

    
    if sim_hyper is None:
        sim_hyper = HyperParams()

    
    if trace_plot_formats is None:
        trace_plot_formats = ['pdf']

    
    results = []

    
    total_combinations = len(sim_links) * len(r_values) * len(est_links)
    current_combination = 0

    
    overall_start_time = time.time()

    
    for sim_link in sim_links:
        for r in r_values:
            if sim_link in ['plogit', 'splogit']:
                if r == 0:
                    if verbose:
                        print(f"\n⊘ Skipping invalid combination: sim_link={sim_link}, r={r} (plogit/splogit requires r ≠ 0)")
                    continue
            else:
                if r != 0:
                    if verbose:
                        print(f"\n⊘ Skipping invalid combination: sim_link={sim_link}, r={r} (non-plogit/splogit requires r = 0)")
                    continue

            sim_seed = sg.seed_param(sim_link, r, sim_idx)

            if sim_link in ['plogit', 'splogit']:
                true_params = step1_sample_parameters(
                    N=N, I=I,
                    sim_hyper=sim_hyper,
                    sim_link=sim_link,
                    sim_r=r,
                    seed=sim_seed,
                )
            else:
                true_params = step1_sample_parameters(
                    N=N, I=I,
                    sim_hyper=sim_hyper,
                    sim_link=sim_link,
                    seed=sim_seed,
                )

            y_list = [
                step2_simulate_responses(
                    true_params=true_params,
                    sim_link=sim_link,
                    seed=sg.seed_data(sim_link, r, sim_idx, group_id=i),
                )
                for i in range(R)
            ]

            for est_link in est_links:
                current_combination += 1

                if verbose:
                    print("\n" + "=" * 80)
                    print(f"Combination {current_combination}/{total_combinations}")
                    print(f"sim_link={sim_link}, r={r}, est_link={est_link}")
                    print("=" * 80)

                start_time = time.time()

                est_seed = sg.seed_estimate(est_link, src_model=sim_link, r=r, sim_idx=sim_idx, est_idx=est_idx)

                save_cfg = SaveConfig(
                    output_dir=output_dir,
                    sim_link=sim_link,
                    r=r,
                    est_link=est_link,
                    N=N, I=I, R=R,
                    save_true_params=save_true_params,
                    save_simulated_data=save_simulated_data,
                    save_mcmc_results=save_mcmc_results,
                    save_log_likelihood=save_log_likelihood,
                    save_recovery_metrics=save_recovery_metrics,
                    save_model_selection=save_model_selection,
                )

                try:
                    if save_cfg.save_true_params and est_link == est_links[0]:
                        save_true_params_func(
                            true_params=true_params,
                            save_path=save_cfg.get_true_params_path(),
                        )
                        if verbose:
                            print(f"  → True parameters are saved: {save_cfg.get_true_params_path()}")

                    if save_cfg.save_simulated_data and est_link == est_links[0]:
                        save_sim_data_func(
                            y_list=y_list,
                            save_path=save_cfg.get_simulated_data_path(),
                        )
                        if verbose:
                            print(f"  → Simulation data are saved: {save_cfg.get_simulated_data_path()}")

                    mcmc_kwargs = {
                        'mu_loga': np.log(0.5),
                    }

                    if est_link == 'splogit':
                        mcmc_kwargs['marginalize_skew'] = False
                    elif est_link == 'skewprobit':
                        mcmc_kwargs['mu_lambda'] = 0.0
                        mcmc_kwargs['tau_lambda'] = 4.0
                    elif est_link in ['glogit','rh']:
                        mcmc_kwargs['hermite_order'] = 2

                    mcmc_results = step3_run_mcmc(
                        y=y_list,
                        est_link=est_link,
                        sampler=sampler,
                        draws=draws,
                        tune=mcmc_kwargs.pop('tune', tune),
                        chains=chains,
                        cores=cores,
                        n_jobs=n_jobs,
                        seed=est_seed,
                        show_progress=show_progress,
                        progressbar=progressbar,  
                        verbose=False,
                        save_config=save_cfg,  
                        **mcmc_kwargs,
                    )

                    elapsed_time = time.time() - start_time

                    if verbose:
                        print(f"✓ MCMC Finished, elapsed_time: {elapsed_time:.1f}s ({elapsed_time/R:.1f}s per replication)")

                    metrics = None
                    theta_table = None
                    r_cp = None
                    p_epsilon_cp = None

                    if sim_link == est_link:
                        metrics = step4_compute_metrics(
                            true_params=true_params,
                            mcmc_results=mcmc_results,
                        )

                        theta_table = metrics.get_theta_metrics_table()

                        if sim_link in ['plogit', 'splogit'] and metrics.metrics_r is not None:
                            r_cp = float(metrics.metrics_r.cp)
                        else:
                            r_cp = None

                        if sim_link == 'splogit' and metrics.metrics_p_epsilon is not None:
                            p_epsilon_cp = float(metrics.metrics_p_epsilon.cp)
                        else:
                            p_epsilon_cp = None

                        if verbose:
                            print(f"✓ Recovery calculation finished")
                            print(f"  Theta Rhat Mean: {theta_table.loc[theta_table['Metric']=='Rhat', 'Mean'].values[0]:.4f}")
                            if r_cp is not None:
                                print(f"  r CP: {r_cp:.3f}")

                        if save_cfg.save_recovery_metrics:
                            save_recovery_metrics_func(
                                metrics=metrics,
                                save_path=save_cfg.get_recovery_metrics_path(),
                            )
                            if verbose:
                                print(f"  → Recovery are saved: {save_cfg.get_recovery_metrics_path()}")

                    model_sel = step5_compute_model_selection(
                        mcmc_results=mcmc_results,
                        thin=1,
                        n_jobs=n_jobs,
                        verbose=False,
                    )

                    if verbose:
                        print(f"✓ Model Selection Finished")
                        print(f"  DIC Mean: {model_sel.dic_mean:.2f}")
                        print(f"  LPML Mean: {model_sel.lpml_mean:.2f}")

                    if save_cfg.save_model_selection:
                        save_model_selection_func(
                            model_sel=model_sel,
                            save_path=save_cfg.get_model_selection_path(),
                        )
                        if verbose:
                            print(f"  → Model Selection Saved: {save_cfg.get_model_selection_path()}")

                    if save_trace_plots:
                        true_r_val = true_params.r if hasattr(true_params, 'r') else None
                        true_p_epsilon_val = true_params.p_epsilon if hasattr(true_params, 'p_epsilon') else None
                        true_tau_epsilon_val = sim_hyper.tau_epsilon if sim_link == 'splogit' else None

                        create_trace_plots_for_condition(
                            mcmc_results=mcmc_results.idatas,  
                            sim_link=sim_link,
                            r_value=r if sim_link in ['plogit', 'splogit'] else None,
                            est_link=est_link,
                            N=N,
                            I=I,
                            R=R,
                            output_dir=output_dir,
                            true_r=true_r_val,
                            true_p_epsilon=true_p_epsilon_val,
                            true_tau_epsilon=true_tau_epsilon_val,
                            formats=trace_plot_formats,
                            dpi=trace_plot_dpi,
                        )
                        if verbose:
                            print(f"✓ Trace plots saved {output_dir}/figures/")

                    result_row = {
                        'sim_link': sim_link,
                        'r': r if sim_link in ['plogit', 'splogit'] else np.nan,
                        'est_link': est_link,
                        'N': N,
                        'I': I,
                        'R': R,
                        'master_seed': master_seed,
                        'sim_idx': sim_idx,
                        'est_idx': est_idx,
                        'sim_seed': sim_seed,
                        'est_seed': est_seed,
                        'elapsed_time': elapsed_time,
                        'dic_mean': model_sel.dic_mean,
                        'dic_std': model_sel.dic_std,
                        'lpml_mean': model_sel.lpml_mean,
                        'lpml_std': model_sel.lpml_std,
                    }

                    if metrics is not None and theta_table is not None:
                        for _, row in theta_table.iterrows():
                            metric_name = row['Metric'].lower()
                            result_row[f'theta_{metric_name}_mean'] = row['Mean']
                            result_row[f'theta_{metric_name}_2.5%'] = row['2.5%']
                            result_row[f'theta_{metric_name}_97.5%'] = row['97.5%']

                        for metric_name, metric_values in metrics.metrics_a.as_dict().items():
                            result_row[f'a_{metric_name}_mean'] = np.mean(metric_values)
                            result_row[f'a_{metric_name}_2.5%'] = np.percentile(metric_values, 2.5)
                            result_row[f'a_{metric_name}_97.5%'] = np.percentile(metric_values, 97.5)

                        for metric_name, metric_values in metrics.metrics_b.as_dict().items():
                            result_row[f'b_{metric_name}_mean'] = np.mean(metric_values)
                            result_row[f'b_{metric_name}_2.5%'] = np.percentile(metric_values, 2.5)
                            result_row[f'b_{metric_name}_97.5%'] = np.percentile(metric_values, 97.5)

                        for metric_name, metric_values in metrics.metrics_mu.as_dict().items():
                            result_row[f'mu_{metric_name}_mean'] = np.mean(metric_values)
                            result_row[f'mu_{metric_name}_2.5%'] = np.percentile(metric_values, 2.5)
                            result_row[f'mu_{metric_name}_97.5%'] = np.percentile(metric_values, 97.5)

                        if metrics.metrics_r is not None:
                            for metric_name, metric_value in metrics.metrics_r.as_dict().items():
                                if metric_name == 'cp':
                                    r_cp = float(metric_value)
                                result_row[f'r_{metric_name}'] = metric_value

                        if metrics.metrics_p_epsilon is not None:
                            for metric_name, metric_value in metrics.metrics_p_epsilon.as_dict().items():
                                if metric_name == 'cp':
                                    p_epsilon_cp = float(metric_value)
                                result_row[f'p_epsilon_{metric_name}'] = metric_value

                    result_row['r_cp'] = r_cp
                    result_row['p_epsilon_cp'] = p_epsilon_cp

                    results.append(result_row)

                    r_str = str(r).replace('.', 'p') if sim_link in ['plogit', 'splogit'] else 'noR'
                    combination_filename = f"{output_prefix}_sim_{sim_link}_r_{r_str}_est_{est_link}_N{N}_I{I}_R{R}.csv"
                    combination_filepath = os.path.join(output_dir, combination_filename)

                    combination_df = pd.DataFrame([result_row])
                    combination_df.to_csv(combination_filepath, index=False)

                    if verbose:
                        print(f"  → Results saved: {combination_filename}")

                except Exception as e:
                    if verbose:
                        print(f"✗ Error: {e}")

                    result_row = {
                        'sim_link': sim_link,
                        'r': r if sim_link in ['plogit', 'splogit'] else np.nan,
                        'est_link': est_link,
                        'N': N,
                        'I': I,
                        'R': R,
                        'master_seed': master_seed,
                        'sim_idx': sim_idx,
                        'est_idx': est_idx,
                        'sim_seed': sim_seed,
                        'est_seed': est_seed,
                        'elapsed_time': np.nan,
                        'dic_mean': np.nan,
                        'lpml_mean': np.nan,
                        'error': str(e),
                    }
                    results.append(result_row)

                    r_str = str(r).replace('.', 'p') if sim_link in ['plogit', 'splogit'] else 'noR'
                    combination_filename = f"{output_prefix}_sim_{sim_link}_r_{r_str}_est_{est_link}_N{N}_I{I}_R{R}.csv"
                    combination_filepath = os.path.join(output_dir, combination_filename)

                    combination_df = pd.DataFrame([result_row])
                    combination_df.to_csv(combination_filepath, index=False)

                    if verbose:
                        print(f"  → Error is saved: {combination_filename}")

    overall_elapsed = time.time() - overall_start_time

    if verbose:
        print("\n" + "=" * 80)
        print("Batch recovery study completed!")
        print("=" * 80)
        print(f"Total elapsed time: {overall_elapsed:.1f}s ({overall_elapsed/60:.1f} minutes)")
        print(f"Average per combination: {overall_elapsed/total_combinations:.1f}s")
        print(f"Saved {total_combinations} individual CSV files to: {output_dir}")
        print("=" * 80)

    results_df = pd.DataFrame(results)

    return results_df


def run_model_comparison(
    y: np.ndarray,
    est_links: list[str],
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int = 1,
    sampler: str = 'nuts',
    seed: int = 12345,
    output_dir: str = './outputs',
    output_prefix: str = 'model_comparison',
    save_mcmc_results: bool = False,
    save_log_likelihood: bool = False,
    progressbar: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)

    if y.ndim != 2:
        raise ValueError(f"y must be 2D array, got shape {y.shape}")
    N, I = y.shape

    if verbose:
        print("=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"Response data: N={N}, I={I}")
        print(f"Models to compare: {est_links}")
        print(f"MCMC settings: draws={draws}, tune={tune}, chains={chains}")
        print("=" * 80)

    results = []

    for model_idx, est_link in enumerate(est_links):
        if verbose:
            print(f"\n[{model_idx+1}/{len(est_links)}] Fitting model: {est_link}")

        start_time = time.time()

        if save_mcmc_results:
            save_cfg = SaveConfig(
                output_dir=output_dir,
                sim_link=est_link,  
                r=1.0,  
                est_link=est_link,
                N=N, I=I, R=1,
                save_true_params=False,
                save_simulated_data=False,
                save_mcmc_results=True,
                save_log_likelihood=save_log_likelihood,
                save_recovery_metrics=False,
                save_model_selection=False,
            )
        else:
            save_cfg = None

        mcmc_kwargs = {
            'mu_loga': np.log(0.5),
        }

        if est_link == 'splogit':
            mcmc_kwargs['marginalize_skew'] = False
        elif est_link == 'skewprobit':
            mcmc_kwargs['mu_lambda'] = 0.0
            mcmc_kwargs['tau_lambda'] = 4.0
        elif est_link in ['glogit', 'rh']:
            mcmc_kwargs['hermite_order'] = 2

        try:
            mcmc_results = step3_run_mcmc(
                y=[y],  
                est_link=est_link, 
                sampler=sampler,  
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                n_jobs=1,  
                seed=seed,  
                show_progress=progressbar, 
                progressbar=progressbar,
                verbose=verbose,
                save_config=save_cfg,
                **mcmc_kwargs,
            )

            elapsed_time = time.time() - start_time

            if verbose:
                print(f"  ✓ MCMC Finished, elapsed_time: {elapsed_time:.1f}s")

            model_sel = step5_compute_model_selection(
                mcmc_results=mcmc_results,
                thin=1,
                n_jobs=1,
                verbose=verbose,  
            )

            if verbose:
                print(f"  DIC: {model_sel.dic_mean:.2f} (pD={model_sel.pd_mean:.1f})")
                print(f"  LPML: {model_sel.lpml_mean:.2f}")

            result_row = {
                'est_link': est_link,
                'N': N,
                'I': I,
                'draws': draws,
                'tune': tune,
                'chains': chains,
                'dic': model_sel.dic_mean,
                'pD': model_sel.pd_mean,
                'lpml': model_sel.lpml_mean,
                'elapsed_time': elapsed_time,
            }
            results.append(result_row)

        except Exception as e:
            if verbose:
                print(f"  ✗ ERROR: {str(e)}")
            result_row = {
                'est_link': est_link,
                'N': N,
                'I': I,
                'draws': draws,
                'tune': tune,
                'chains': chains,
                'dic': np.nan,
                'pD': np.nan,
                'lpml': np.nan,
                'elapsed_time': np.nan,
            }
            results.append(result_row)

    results_df = pd.DataFrame(results)

    output_filename = f"{output_prefix}_N{N}_I{I}.csv"
    output_filepath = os.path.join(output_dir, output_filename)
    results_df.to_csv(output_filepath, index=False)

    if verbose:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        print(results_df.to_string(index=False))
        print("=" * 80)
        print(f"Results saved to: {output_filepath}")

        if not results_df['dic'].isna().all():
            best_dic_idx = results_df['dic'].idxmin()
            best_lpml_idx = results_df['lpml'].idxmax()
            print(f"\nBest model (lowest DIC): {results_df.loc[best_dic_idx, 'est_link']} "
                  f"(DIC={results_df.loc[best_dic_idx, 'dic']:.2f})")
            print(f"Best model (highest LPML): {results_df.loc[best_lpml_idx, 'est_link']} "
                  f"(LPML={results_df.loc[best_lpml_idx, 'lpml']:.2f})")

    return results_df

