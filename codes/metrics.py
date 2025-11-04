from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray


@dataclass(frozen=True)
class MCMCMetrics:

    bias: np.ndarray
    mse: np.ndarray
    sd: np.ndarray
    se: np.ndarray
    cp: np.ndarray
    rhat: np.ndarray

    def as_dict(self) -> Mapping[str, np.ndarray]:
        return {
            "bias": self.bias,
            "mse": self.mse,
            "sd": self.sd,
            "se": self.se,
            "cp": self.cp,
            "rhat": self.rhat,
        }


def compute_mcmc_metrics(
    replications: Sequence[ArrayLike],
    true_value: ArrayLike | float,
    credible_interval: float = 0.95,
    rhat_values: ArrayLike | None = None,
) -> MCMCMetrics:
    draws = _stack_replications(replications)
    R, M = draws.shape[:2]
    if M < 2:
        raise ValueError("Each replication must contain at least two draws.")

    param_shape = draws.shape[2:]
    true_arr = np.asarray(true_value, dtype=float)
    if true_arr.shape not in (param_shape, ()):
        try:
            true_arr = np.broadcast_to(true_arr, param_shape)
        except ValueError as err:
            raise ValueError(
                f"true_value shape {true_arr.shape} is not broadcastable to {param_shape}"
            ) from err

    draws = draws.astype(float, copy=False)
    true_arr = true_arr.astype(float, copy=False)

    posterior_means = draws.mean(axis=1)  
    bias = (posterior_means - true_arr).mean(axis=0)
    mse = ((posterior_means - true_arr) ** 2).mean(axis=0)

    within_sd = draws.std(axis=1, ddof=1)
    sd = within_sd.mean(axis=0)

    if R >= 2:
        se = posterior_means.std(axis=0, ddof=1)
    else:
        se = np.full(param_shape, np.nan)

    if R >= 2:
        cp = _coverage_probability(draws, true_arr, credible_interval)
    else:
        cp = np.full(param_shape, np.nan)

    if rhat_values is not None:
        try:
            rhat_stacked = np.stack([np.asarray(rh) for rh in rhat_values], axis=0)
            rhat = rhat_stacked.mean(axis=0)
        except Exception:
            rhat = np.full(param_shape, np.nan)
    else:
        rhat = np.full(param_shape, np.nan)

    return MCMCMetrics(
        bias=np.asarray(bias),
        mse=np.asarray(mse),
        sd=np.asarray(sd),
        se=np.asarray(se),
        cp=np.asarray(cp),
        rhat=np.asarray(rhat),
    )

def _stack_replications(replications: Sequence[ArrayLike]) -> np.ndarray:
    if not replications:
        raise ValueError("replications must contain at least one element.")

    rep_arrays = [np.asarray(rep) for rep in replications]
    first_shape = rep_arrays[0].shape
    if len(first_shape) == 0:
        raise ValueError("Each replication must contain posterior draws (at least 1-D).")

    for idx, rep in enumerate(rep_arrays):
        if rep.shape != first_shape:
            raise ValueError(
                f"Replication {idx} has shape {rep.shape}, expected {first_shape}."
            )

    stacked = np.stack(rep_arrays, axis=0)
    if stacked.ndim < 2:
        raise ValueError("Replications must provide at least two dimensions (R, M).")
    return stacked


def _coverage_probability(
    draws: np.ndarray,
    true_value: np.ndarray,
    credible_interval: float,
) -> np.ndarray:
    if not (0.0 < credible_interval < 1.0):
        raise ValueError("credible_interval must be between 0 and 1.")

    alpha = 1.0 - credible_interval
    R = draws.shape[0]
    flat_draws = draws.reshape(R, draws.shape[1], -1)  # (R, M, n_params)
    flat_truth = true_value.reshape(-1) if true_value.shape else np.array([true_value])

    if flat_truth.size != flat_draws.shape[-1]:
        raise ValueError(
            f"true_value has {flat_truth.size} elements, expected {flat_draws.shape[-1]}."
        )

    coverage_counts = np.zeros(flat_draws.shape[-1], dtype=int)

    for r in range(R):
        replication = flat_draws[r]
        for p in range(flat_draws.shape[-1]):
            lower, upper = _highest_density_interval(replication[:, p], alpha)
            if lower <= flat_truth[p] <= upper:
                coverage_counts[p] += 1

    cp = coverage_counts / R
    return cp.reshape(true_value.shape if true_value.shape else ())


def _highest_density_interval(draws: np.ndarray, alpha: float) -> tuple[float, float]:
    sorted_draws = np.sort(np.asarray(draws, dtype=float))
    n = sorted_draws.size
    if n == 0:
        raise ValueError("Cannot compute HPD interval from an empty sample.")

    mass = 1.0 - alpha
    interval_size = int(np.floor(mass * n))
    interval_size = min(max(interval_size, 1), n - 1)

    widths = sorted_draws[interval_size:] - sorted_draws[: n - interval_size]
    min_idx = int(np.argmin(widths))
    low = float(sorted_draws[min_idx])
    high = float(sorted_draws[min_idx + interval_size])
    return low, high


__all__ = ["MCMCMetrics", "compute_mcmc_metrics"]
