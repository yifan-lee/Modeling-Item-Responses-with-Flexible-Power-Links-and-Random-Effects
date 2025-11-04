from __future__ import annotations

from typing import Optional, Literal
import numpy as np
import arviz as az
from scipy import stats
from scipy.special import logsumexp

try:
    from codes.simulate import (
        inv_logit,
        inv_cloglog,
        inv_loglog,
        inv_plogit,
        inv_splogit,
        inv_lpe,
        inv_grg,
        inv_skewprobit,
        inv_rh,
        inv_glogit,
        LinkType,
    )
except ModuleNotFoundError:
    from simulate import (
        inv_logit,
        inv_cloglog,
        inv_loglog,
        inv_plogit,
        inv_splogit,
        inv_lpe,
        inv_grg,
        inv_skewprobit,
        inv_rh,
        inv_glogit,
        LinkType,
    )


LINK_FUNCTIONS = {
    "logit": inv_logit,
    "cloglog": inv_cloglog,
    "loglog": inv_loglog,
    "plogit": inv_plogit,
    "splogit": inv_splogit,
    "lpe": inv_lpe,
    "grg": inv_grg,
    "skewprobit": inv_skewprobit,
    "rh": inv_rh,
    "glogit": inv_glogit,
}


def compute_log_likelihood(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    mu: np.ndarray,
    link: LinkType,
    r: Optional[float] = None,
    xi: Optional[float] = None,
    lambda_skew: Optional[float] = None,
    hermite_coeffs: Optional[np.ndarray] = None,
    alpha1: Optional[float] = None,
    alpha2: Optional[float] = None,
) -> float:
    N, I = y.shape

    eta = (mu[:, None] - b[None, :]) * a[None, :]

    inv_link = LINK_FUNCTIONS[link.lower()]

    if link.lower() in {"plogit", "splogit"}:
        if r is None:
            raise ValueError(f"{link} requires r parameter")
        p = inv_link(eta, r)
    elif link.lower() == "lpe":
        if xi is None:
            raise ValueError("lpe requires xi parameter")
        p = inv_link(eta, xi)
    elif link.lower() == "skewprobit":
        if lambda_skew is None:
            raise ValueError("skewprobit requires lambda_skew parameter")
        p = inv_link(eta, lambda_skew)
    elif link.lower() == "rh":
        if hermite_coeffs is None:
            raise ValueError("rh requires hermite_coeffs parameter")
        p = inv_link(eta, hermite_coeffs)
    elif link.lower() == "glogit":
        if alpha1 is None or alpha2 is None:
            raise ValueError("glogit requires alpha1 and alpha2 parameters")
        p = inv_link(eta, alpha1, alpha2)
    else:
        p = inv_link(eta)

    p = np.clip(p, 1e-10, 1 - 1e-10)

    log_lik = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))

    return float(log_lik)


def compute_log_likelihood_matrix(
    y: np.ndarray,
    idata: az.InferenceData,
    link: LinkType,
    thin: int = 1,
) -> np.ndarray:
    post = idata.posterior

    a_samples = post["a"].values.reshape(-1, post["a"].shape[-1]) 
    b_samples = post["b"].values.reshape(-1, post["b"].shape[-1]) 
    mu_samples = post["mu"].values.reshape(-1, post["mu"].shape[-1]) 

    if thin > 1:
        a_samples = a_samples[::thin]
        b_samples = b_samples[::thin]
        mu_samples = mu_samples[::thin]

    M = a_samples.shape[0]
    N, I = y.shape

    r_samples = None
    xi_samples = None
    lambda_skew_samples = None
    hermite_coeffs_samples = None
    alpha1_samples = None
    alpha2_samples = None

    if link.lower() in {"plogit", "splogit"}:
        if "r" not in post:
            raise ValueError(f"Parameter 'r' not found in posterior for {link} model")
        r_samples = post["r"].values.reshape(-1)
        if thin > 1:
            r_samples = r_samples[::thin]
        if np.any(np.isnan(r_samples)) or np.any(np.isinf(r_samples)):
            raise ValueError(f"NaN or Inf values found in r posterior samples for {link} model")

    elif link.lower() == "lpe":
        xi_samples = post["xi"].values.reshape(-1)
        if thin > 1:
            xi_samples = xi_samples[::thin]

    elif link.lower() == "skewprobit":
        lambda_skew_samples = post["lambda_skew"].values.reshape(-1)
        if thin > 1:
            lambda_skew_samples = lambda_skew_samples[::thin]

    elif link.lower() == "rh":
        hermite_c2 = post["hermite_c2"].values.reshape(-1)
        if thin > 1:
            hermite_c2 = hermite_c2[::thin]
        hermite_coeffs_samples = np.zeros((len(hermite_c2), 3))
        hermite_coeffs_samples[:, 2] = hermite_c2

    elif link.lower() == "glogit":
        alpha1_samples = post["alpha1"].values.reshape(-1)
        alpha2_samples = post["alpha2"].values.reshape(-1)
        if thin > 1:
            alpha1_samples = alpha1_samples[::thin]
            alpha2_samples = alpha2_samples[::thin]

    log_lik_matrix = np.zeros((M, N, I))

    inv_link = LINK_FUNCTIONS[link.lower()]

    for m in range(M):
        eta = (mu_samples[m, :, None] - b_samples[m, None, :]) * a_samples[m, None, :]

        if link.lower() in {"plogit", "splogit"}:
            p = inv_link(eta, r_samples[m])
        elif link.lower() == "lpe":
            p = inv_link(eta, xi_samples[m])
        elif link.lower() == "skewprobit":
            p = inv_link(eta, lambda_skew_samples[m])
        elif link.lower() == "rh":
            p = inv_link(eta, hermite_coeffs_samples[m])
        elif link.lower() == "glogit":
            p = inv_link(eta, alpha1_samples[m], alpha2_samples[m])
        else:
            p = inv_link(eta)

        p = np.clip(p, 1e-10, 1 - 1e-10)

        log_lik_matrix[m] = y * np.log(p) + (1 - y) * np.log(1 - p)

    return log_lik_matrix



def compute_dic(
    y: np.ndarray,
    idata: az.InferenceData,
    link: LinkType,
) -> dict[str, float]:
    post = idata.posterior

    a_samples = post["a"].values.reshape(-1, post["a"].shape[-1])
    b_samples = post["b"].values.reshape(-1, post["b"].shape[-1])
    mu_samples = post["mu"].values.reshape(-1, post["mu"].shape[-1])

    M = a_samples.shape[0]

    if np.any(np.isnan(a_samples)) or np.any(np.isinf(a_samples)):
        raise ValueError(f"NaN or Inf values found in 'a' posterior samples")
    if np.any(np.isnan(b_samples)) or np.any(np.isinf(b_samples)):
        raise ValueError(f"NaN or Inf values found in 'b' posterior samples")
    if np.any(np.isnan(mu_samples)) or np.any(np.isinf(mu_samples)):
        raise ValueError(f"NaN or Inf values found in 'mu' posterior samples")

    a_mean = a_samples.mean(axis=0)
    b_mean = b_samples.mean(axis=0)
    mu_mean = mu_samples.mean(axis=0)

    r_mean = None
    xi_mean = None
    lambda_skew_mean = None
    hermite_coeffs_mean = None
    alpha1_mean = None
    alpha2_mean = None

    if link.lower() in {"plogit", "splogit"}:
        if "r" not in post:
            raise ValueError(f"Parameter 'r' not found in posterior for {link} model")
        r_samples = post["r"].values.reshape(-1)
        if np.any(np.isnan(r_samples)) or np.any(np.isinf(r_samples)):
            raise ValueError(f"NaN or Inf values found in r posterior samples for {link} model")
        r_mean = float(r_samples.mean())

    elif link.lower() == "lpe":
        xi_samples = post["xi"].values.reshape(-1)
        xi_mean = float(xi_samples.mean())

    elif link.lower() == "skewprobit":
        lambda_skew_samples = post["lambda_skew"].values.reshape(-1)
        lambda_skew_mean = float(lambda_skew_samples.mean())

    elif link.lower() == "rh":
        hermite_c2 = post["hermite_c2"].values.reshape(-1)
        hermite_coeffs_mean = np.array([0.0, 0.0, float(hermite_c2.mean())])

    elif link.lower() == "glogit":
        alpha1_samples = post["alpha1"].values.reshape(-1)
        alpha2_samples = post["alpha2"].values.reshape(-1)
        alpha1_mean = float(alpha1_samples.mean())
        alpha2_mean = float(alpha2_samples.mean())

    log_lik_at_mean = compute_log_likelihood(
        y, a_mean, b_mean, mu_mean, link,
        r=r_mean, xi=xi_mean, lambda_skew=lambda_skew_mean,
        hermite_coeffs=hermite_coeffs_mean,
        alpha1=alpha1_mean, alpha2=alpha2_mean,
    )
    deviance_at_mean = -2.0 * log_lik_at_mean

    deviances = []
    for m in range(M):
        r_m = None
        xi_m = None
        lambda_skew_m = None
        hermite_coeffs_m = None
        alpha1_m = None
        alpha2_m = None

        if link.lower() in {"plogit", "splogit"}:
            r_m = float(r_samples[m])
        elif link.lower() == "lpe":
            xi_m = float(xi_samples[m])
        elif link.lower() == "skewprobit":
            lambda_skew_m = float(lambda_skew_samples[m])
        elif link.lower() == "rh":
            hermite_coeffs_m = np.array([0.0, 0.0, float(hermite_c2[m])])
        elif link.lower() == "glogit":
            alpha1_m = float(alpha1_samples[m])
            alpha2_m = float(alpha2_samples[m])

        log_lik_m = compute_log_likelihood(
            y, a_samples[m], b_samples[m], mu_samples[m], link,
            r=r_m, xi=xi_m, lambda_skew=lambda_skew_m,
            hermite_coeffs=hermite_coeffs_m,
            alpha1=alpha1_m, alpha2=alpha2_m,
        )
        deviances.append(-2.0 * log_lik_m)

    deviance_mean = float(np.mean(deviances))

    pd = deviance_mean - deviance_at_mean

    dic = deviance_mean + pd

    return {
        "dic": dic,
        "pd": pd,
        "deviance_mean": deviance_mean,
        "deviance_at_mean": deviance_at_mean,
    }


def compute_lpml(
    y: np.ndarray,
    idata: az.InferenceData,
    link: LinkType,
    thin: int = 1,
) -> dict[str, float]:
    log_lik_matrix = compute_log_likelihood_matrix(y, idata, link, thin=thin)

    M, N, I = log_lik_matrix.shape

    cpo_log = np.zeros((N, I))

    for i in range(N):
        for j in range(I):
            log_lik_ij = log_lik_matrix[:, i, j] 

            cpo_log[i, j] = -logsumexp(-log_lik_ij) + np.log(M)

    lpml = float(np.sum(cpo_log))

    cpo = np.exp(cpo_log)

    return {
        "lpml": lpml,
        "cpo_mean": float(cpo.mean()),
        "cpo_min": float(cpo.min()),
        "cpo_max": float(cpo.max()),
        "cpo_matrix": cpo,  
    }


def compute_model_selection_criteria(
    y: np.ndarray,
    idata: az.InferenceData,
    link: LinkType,
    thin: int = 1,
) -> dict[str, float | dict]:
    dic_results = compute_dic(y, idata, link)
    lpml_results = compute_lpml(y, idata, link, thin=thin)

    return {
        "dic": dic_results,
        "lpml": lpml_results,
    }


def print_model_selection_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("MODEL SELECTION CRITERIA")
    print("=" * 70)

    dic = results["dic"]
    print("\nDeviance Information Criterion (DIC):")
    print("-" * 70)
    print(f"  DIC:                    {dic['dic']:12.2f}  (lower is better)")
    print(f"  Effective # params (pD): {dic['pd']:12.2f}")
    print(f"  Mean deviance (D̄):      {dic['deviance_mean']:12.2f}")
    print(f"  Deviance at mean D(θ̄):  {dic['deviance_at_mean']:12.2f}")

    lpml = results["lpml"]
    print("\nLog Pseudo-Marginal Likelihood (LPML):")
    print("-" * 70)
    print(f"  LPML:                   {lpml['lpml']:12.2f}  (higher is better)")
    print(f"  Mean CPO:               {lpml['cpo_mean']:12.6f}")
    print(f"  Min CPO:                {lpml['cpo_min']:12.6f}")
    print(f"  Max CPO:                {lpml['cpo_max']:12.6f}")

    print("=" * 70)


__all__ = [
    "compute_log_likelihood",
    "compute_log_likelihood_matrix",
    "compute_dic",
    "compute_lpml",
    "compute_model_selection_criteria",
    "print_model_selection_summary",
]
