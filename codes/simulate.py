from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np
import os

LinkType = Literal["logit", "cloglog", "loglog", "plogit", "splogit",
                    "lpe", "grg", "skewprobit", "rh", "glogit"]



@dataclass(frozen=True)
class HyperParams:
    mu_loga: float = np.log(0.5)
    tau_loga: float = 25.0
    lower_a: float = 0.25
    upper_a: float = 0.75
    mu_b: float = 0.0
    tau_b: float = 1.0
    mu_theta: float = 0.0
    tau_theta: float = 1.0
    mu_log_r: float = 0.0
    tau_log_r: float = 1.0
    tau_epsilon: float = 25.0
    p_epsilon: float = 0.25
    alpha_xi: float = 2.0
    beta_xi: float = 1.0
    mu_lambda: float = 0.0
    tau_lambda: float = 1.0
    hermite_order: int = 2
    mu_alpha1: float = 0.0
    tau_alpha1: float = 1.0
    mu_alpha2: float = 0.0
    tau_alpha2: float = 1.0



@dataclass
class Params:
    sim_link: str
    a: np.ndarray      
    b: np.ndarray      
    mu: np.ndarray    
    r: float = 1.0
    epsilon: Optional[np.ndarray] = None 
    z: Optional[np.ndarray] = None       
    p_epsilon: Optional[float] = None    
    xi: Optional[float] = None            
    lambda_skew: Optional[float] = None  
    hermite_coeffs: Optional[np.ndarray] = None 
    alpha1: Optional[float] = None       
    alpha2: Optional[float] = None       


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


def inv_logit(eta: np.ndarray, r: Optional[float] = None) -> np.ndarray:
    return _sigmoid(eta)


def inv_cloglog(eta: np.ndarray, r: Optional[float] = None) -> np.ndarray:
    return 1.0 - np.exp(-np.exp(np.clip(eta, -20, 20)))


def inv_loglog(eta: np.ndarray, r: Optional[float] = None) -> np.ndarray:
    return np.exp(-np.exp(np.clip(-eta, -20, 20)))


def inv_plogit(eta: np.ndarray, r: float) -> np.ndarray:
    if r <= 0:
        raise ValueError("r must be positive for plogit")
    
    if r <= 1.0:
        return _sigmoid(eta / r) ** r
    else:
        return 1.0 - _sigmoid(-r * eta) ** (1.0 / r)


def inv_splogit(eta: np.ndarray, r: float) -> np.ndarray:
    return inv_plogit(eta, r)


def inv_lpe(eta: np.ndarray, xi: float) -> np.ndarray:
    if xi <= 0:
        raise ValueError("xi must be positive for lpe")

    logit_p = _sigmoid(eta)
    return np.power(logit_p, xi)


def inv_grg(eta: np.ndarray, r: Optional[float] = None) -> np.ndarray:
    gumbel_max = np.exp(-np.exp(np.clip(-eta, -20, 20)))
    reverse_gumbel = 1.0 - np.exp(-np.exp(np.clip(eta, -20, 20)))
    return 0.5 * (gumbel_max + reverse_gumbel)


def _skew_normal_cdf(x: np.ndarray, alpha: float) -> np.ndarray:
    from scipy.stats import norm
    phi_x = norm.cdf(x)
    phi_alpha_x = norm.cdf(alpha * x)
    return 2.0 * phi_x * phi_alpha_x


def inv_skewprobit(eta: np.ndarray, lambda_skew: float) -> np.ndarray:
    return _skew_normal_cdf(eta, lambda_skew)


def _hermite_poly(x: np.ndarray, n: int) -> np.ndarray:
    from numpy.polynomial.hermite import hermval
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    return hermval(x, coeffs)


def inv_rh(eta: np.ndarray, hermite_coeffs: np.ndarray) -> np.ndarray:
    from scipy.stats import norm

    if hermite_coeffs is None or len(hermite_coeffs) == 0:
        return norm.cdf(eta)
    correction = 0.0
    for k, c_k in enumerate(hermite_coeffs):
        if k == 0:
            continue
        correction += c_k * _hermite_poly(eta, k)

    eta_corrected = eta + correction
    return norm.cdf(eta_corrected)


def _h_alpha_positive(eta: np.ndarray, alpha: float) -> np.ndarray:
    eps = 1e-10

    if abs(alpha) < eps:
        return eta
    elif alpha > 0:
        return (np.exp(alpha * eta) - 1.0) / alpha
    else:
        max_eta = (1.0 / abs(alpha)) - eps
        eta_clipped = np.clip(eta, None, max_eta)
        return -np.log(1.0 - alpha * eta_clipped) / alpha


def _h_alpha_negative(eta: np.ndarray, alpha: float) -> np.ndarray:
    eps = 1e-10
    eta_abs = np.abs(eta)

    if abs(alpha) < eps:
        return eta
    elif alpha > 0:
        return -(np.exp(alpha * eta_abs) - 1.0) / alpha
    else:
        max_eta_abs = (1.0 / abs(alpha)) - eps
        eta_abs_clipped = np.clip(eta_abs, None, max_eta_abs)
        return np.log(1.0 - alpha * eta_abs_clipped) / alpha


def inv_glogit(eta: np.ndarray, alpha1: float, alpha2: float) -> np.ndarray:
    h_eta = np.zeros_like(eta, dtype=float)
    positive_mask = eta > 0
    nonpositive_mask = ~positive_mask
    if np.any(positive_mask):
        h_eta[positive_mask] = _h_alpha_positive(eta[positive_mask], alpha1)
    if np.any(nonpositive_mask):
        h_eta[nonpositive_mask] = _h_alpha_negative(eta[nonpositive_mask], alpha2)
    return _sigmoid(h_eta)


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


def sample_params(
    N: int,
    I: int,
    hyper: HyperParams,
    link: LinkType = "logit",
    r: Optional[float] = None,
    p_epsilon: Optional[float] = None,
    seed: Optional[int] = None
) -> Params:
    rng = np.random.default_rng(seed)
    loga = rng.normal(
        loc=hyper.mu_loga,
        scale=np.sqrt(1.0 / hyper.tau_loga),
        size=I
    )
    loga = np.clip(
        loga,
        np.log(hyper.lower_a + 1e-9),
        np.log(hyper.upper_a - 1e-9)
    )
    a = np.exp(loga).astype(np.float32)
    b = rng.normal(
        loc=hyper.mu_b,
        scale=np.sqrt(1.0 / hyper.tau_b),
        size=I
    ).astype(np.float32)
    mu = rng.normal(
        loc=hyper.mu_theta,
        scale=np.sqrt(1.0 / hyper.tau_theta),
        size=N
    ).astype(np.float32)
    if link in {"plogit", "splogit"}:
        if r is None:
            log_r = rng.normal(
                loc=hyper.mu_log_r,
                scale=np.sqrt(1.0 / hyper.tau_log_r)
            )
            r = float(np.exp(log_r))
        else:
            r = float(r)
    else:
        r = 1.0
    if link == "splogit":
        epsilon = rng.normal(
            loc=0.0,
            scale=np.sqrt(1.0 / hyper.tau_epsilon),
            size=(N, I)
        ).astype(np.float32)
        p_eps = p_epsilon if p_epsilon is not None else hyper.p_epsilon
        z = rng.binomial(
            n=1,
            p=p_eps,
            size=(N, I)
        ).astype(np.int8)
    else:
        epsilon = None
        z = None
        p_eps = None
    xi = None
    lambda_skew = None
    hermite_coeffs = None
    alpha1 = None
    alpha2 = None

    if link == "lpe":
        xi = float(rng.gamma(shape=hyper.alpha_xi, scale=1.0/hyper.beta_xi))

    elif link == "skewprobit":
        lambda_skew = float(rng.normal(loc=hyper.mu_lambda, scale=np.sqrt(1.0/hyper.tau_lambda)))

    elif link == "rh":
        hermite_coeffs = np.zeros(hyper.hermite_order + 1)
        for k in range(2, len(hermite_coeffs)):
            hermite_coeffs[k] = rng.normal(0, 0.1)

    elif link == "glogit":
        alpha1 = float(rng.normal(loc=hyper.mu_alpha1, scale=np.sqrt(1.0/hyper.tau_alpha1)))
        alpha2 = float(rng.normal(loc=hyper.mu_alpha2, scale=np.sqrt(1.0/hyper.tau_alpha2)))

    return Params(
        a=a, b=b, mu=mu, r=r, epsilon=epsilon, z=z, p_epsilon=p_eps,
        xi=xi, lambda_skew=lambda_skew, hermite_coeffs=hermite_coeffs,
        alpha1=alpha1, alpha2=alpha2,
        sim_link=link
    )

def save_params(file_path, params):
        np.savez(
            file_path,
            a=params.a,
            b=params.b,
            mu=params.mu,
            r=params.r,
            epsilon=params.epsilon,
            z=params.z,
            p_epsilon=params.p_epsilon,
            xi=params.xi,
            lambda_skew=params.lambda_skew,
            hermite_coeffs=params.hermite_coeffs,
            alpha1=params.alpha1,
            alpha2=params.alpha2,
        )
        print(f"✅ Parameters saved to {file_path}")

def generate_responses(
    params: Params,
    link: LinkType,
    seed: Optional[int] = None,
    return_prob: bool = False
) -> np.ndarray:
    if link not in LINK_FUNCTIONS:
        raise ValueError(f"Unknown link '{link}'. Choose from: {list(LINK_FUNCTIONS.keys())}")
    
    N, I = len(params.mu), len(params.a)
    eta = (params.mu[:, None] - params.b[None, :]) * params.a[None, :]
    if link == "splogit":
        if params.epsilon is None or params.z is None:
            raise ValueError("splogit requires epsilon and z in params")
        eta = eta + params.z * params.epsilon
    inv_link = LINK_FUNCTIONS[link]
    if link in {"plogit", "splogit"}:
        p = inv_link(eta, params.r)
    elif link == "lpe":
        if params.xi is None:
            raise ValueError("lpe requires xi in params")
        p = inv_link(eta, params.xi)
    elif link == "skewprobit":
        if params.lambda_skew is None:
            raise ValueError("skewprobit requires lambda_skew in params")
        p = inv_link(eta, params.lambda_skew)
    elif link == "rh":
        if params.hermite_coeffs is None:
            raise ValueError("rh requires hermite_coeffs in params")
        p = inv_link(eta, params.hermite_coeffs)
    elif link == "glogit":
        if params.alpha1 is None or params.alpha2 is None:
            raise ValueError("glogit requires alpha1 and alpha2 in params")
        p = inv_link(eta, params.alpha1, params.alpha2)
    else:
        p = inv_link(eta)
    p = np.clip(p, 1e-9, 1 - 1e-9)
    p = p.astype(np.float32)
    
    if return_prob:
        return p
    rng = np.random.default_rng(seed)
    y = rng.binomial(n=1, p=p, size=(N, I)).astype(np.int8)
    return y


def simulate_multiple_responses(
    true_params: Params,
    sim_link: LinkType,
    R: int,
    seed_list: list[int],
    return_prob: bool = False,
) -> list[np.ndarray]:
    if len(seed_list) != R:
        raise ValueError(f"The length of seed_list ({len(seed_list)}) must be equal to R ({R}).")

    if return_prob:
        raise NotImplementedError("return_prob is not yet implemented for batch generation.")

    y_list = []
    for i in range(R):
        y = generate_responses(
            params=true_params,
            link=sim_link,
            seed=seed_list[i],
            return_prob=False,
        )
        y_list.append(y)

    return y_list


def save_simulated_data(
    y_list: list[np.ndarray],
    true_params: Params,
    save_path: str,
) -> None:
    N = true_params.mu.shape[0]
    I = true_params.a.shape[0]
    sim_link = true_params.sim_link
    sim_r = true_params.r
    sim_p_epsilon = true_params.p_epsilon
    R = len(y_list)
    for i in range(R):
        folder_name = f"simulation_N{N}_I{I}_{sim_link}_r{sim_r}_p{sim_p_epsilon}".replace('.', 'p')
        folder_path = os.path.join(save_path,folder_name)
        os.makedirs(folder_path, exist_ok=True)
        filename = f"simulation_N{N}_I{I}_simlink_{sim_link}_r{sim_r}_p{sim_p_epsilon}_rep{i}".replace('.', 'p')+".npz"
        file_path = os.path.join(folder_path,filename)
        y = y_list[i]
        np.savez(file_path, y=y)
    


if __name__ == "__main__":
    pass
