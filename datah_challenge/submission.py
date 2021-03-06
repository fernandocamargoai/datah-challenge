from typing import List, Tuple, Callable, Optional

import numpy as np
import tweedie
from gluonts.model.forecast import SampleForecast, QuantileForecast
from scipy.optimize import fmin_l_bfgs_b
from scipy.special import gammaln, factorial, psi
from scipy.stats import norm, beta, gamma, nbinom, poisson
from sklearn.preprocessing import MinMaxScaler
from statsmodels.distributions import ECDF


def pool_forecast_transform_fn(
    input_: Tuple[SampleForecast, int],
    forecast_transform_fn: Callable[[SampleForecast, int], List[float]],
) -> List[float]:
    forecast, stock = input_
    return forecast_transform_fn(forecast, stock)


def calculate_out_of_stock_days_from_samples(
    forecast: SampleForecast, stock: int, total_days: int = 30
) -> np.ndarray:
    sample_days = np.apply_along_axis(
        np.searchsorted, 1, np.cumsum(forecast.samples, axis=1) >= stock, True
    )
    sample_days[sample_days == total_days] -= 1
    return sample_days + 1


def calculate_out_out_stock_days_from_quantiles(
    forecast: QuantileForecast, stock: int, total_days: int = 30
) -> np.ndarray:
    quantile_days = np.apply_along_axis(
        np.searchsorted, 1, np.cumsum(forecast.forecast_array, axis=1) >= stock, True
    )
    quantile_days[quantile_days == total_days] -= 1
    return quantile_days + 1


def old_cdf_to_probas(cdf: List[float]) -> List[float]:
    prob_array = np.array(np.ediff1d(cdf, to_begin=cdf[0]))
    return list(prob_array / np.sum(prob_array))


def cdf_fn_to_probas(
    cdf_fn: Callable[[float], float], total_days: int = 30
) -> List[float]:
    prob_array = np.ediff1d([cdf_fn(i) for i in range(0, total_days + 1)])
    return list(prob_array / np.sum(prob_array))


def _calculate_std(sample_days: np.ndarray, fixed_std: Optional[float], std_scaler: Optional[MinMaxScaler]):
    if fixed_std:
        std = fixed_std
    elif std_scaler:
        std = std_scaler.transform(sample_days.std().reshape(-1, 1))[0][0]
    else:
        std = sample_days.std()
    return std


def apply_tweedie(
    sample_days: np.ndarray,
    phi: float = 2.0,
    power: float = 1.3,
    fixed_std: float = None,
    std_scaler: MinMaxScaler = None,
    total_days: int = 30,
) -> List[float]:
    mu = sample_days.mean()
    if phi < 0:
        sigma = _calculate_std(sample_days, fixed_std, std_scaler)
        phi = (sigma ** 2) / mu ** power
    distro = tweedie.tweedie(p=power, mu=mu, phi=phi)
    return cdf_fn_to_probas(distro.cdf, total_days=total_days)


def apply_normal(
    sample_days: np.ndarray,
    fixed_std: float = None,
    std_scaler: MinMaxScaler = None,
    total_days: int = 30,
) -> List[float]:
    distro = norm(sample_days.mean(), _calculate_std(sample_days, fixed_std, std_scaler))

    return cdf_fn_to_probas(distro.cdf, total_days=total_days)


def apply_ecdf(sample_days: np.ndarray, total_days: int = 30) -> List[float]:
    ecdf = ECDF(sample_days)
    return cdf_fn_to_probas(ecdf, total_days=total_days)


def apply_beta(
    sample_days: np.ndarray, fixed_std: float = None,
    std_scaler: MinMaxScaler = None, total_days: int = 30
) -> List[float]:
    mu = sample_days.mean() / total_days
    sigma = _calculate_std(sample_days, fixed_std, std_scaler) / total_days

    a = mu ** 2 * ((1 - mu) / sigma ** 2 - 1 / mu)
    b = a * (1 / mu - 1)

    distro = beta(a, b)
    return cdf_fn_to_probas(distro.cdf, total_days=total_days)


# X is a numpy array representing the data
# initial params is a numpy array representing the initial values of
# size and prob parameters
def _fit_nbinom(X: np.ndarray, initial_params=None) -> Tuple[float, float]:
    infinitesimal = np.finfo(np.float).eps

    def log_likelihood(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        # MLE estimate based on the formula on Wikipedia:
        # http://en.wikipedia.org/wiki/Negative_binomial_distribution#Maximum_likelihood_estimation
        result = (
            np.sum(gammaln(X + r))
            - np.sum(np.log(factorial(X)))
            - N * (gammaln(r))
            + N * r * np.log(p)
            + np.sum(X * np.log(1 - (p if p < 1 else 1 - infinitesimal)))
        )

        return -result

    def log_likelihood_deriv(params, *args):
        r, p = params
        X = args[0]
        N = X.size

        pderiv = (N * r) / p - np.sum(X) / (1 - (p if p < 1 else 1 - infinitesimal))
        rderiv = np.sum(psi(X + r)) - N * psi(r) + N * np.log(p)

        return np.array([-rderiv, -pderiv])

    if initial_params is None:
        # reasonable initial values (from fitdistr function in R)
        m = np.mean(X)
        v = np.var(X)
        size = (m ** 2) / (v - m) if v > m else 10

        # convert mu/size parameterization to prob/size
        p0 = size / ((size + m) if size + m != 0 else 1)
        r0 = size
        initial_params = np.array([r0, p0])

    bounds = [(infinitesimal, None), (infinitesimal, 1)]
    optimres = fmin_l_bfgs_b(
        log_likelihood,
        x0=initial_params,
        # fprime=log_likelihood_deriv,
        args=(X,),
        approx_grad=1,
        bounds=bounds,
    )

    params = optimres[0]
    return (params[0], params[1])


def apply_fitted_negative_binomial(
    sample_days: np.ndarray, total_days: int = 30
) -> List[float]:
    distro = nbinom(*_fit_nbinom(sample_days))
    return cdf_fn_to_probas(distro.cdf, total_days=total_days)


def apply_negative_binomial(
    sample_days: np.ndarray,
    fixed_std: float = None,
    std_scaler: MinMaxScaler = None,
    total_days: int = 30,
) -> List[float]:
    mu = sample_days.mean()
    sigma = _calculate_std(sample_days, fixed_std, std_scaler)

    var = sigma ** 2

    r = (mu ** 2) / (var - mu) if var > mu else total_days
    p = r / ((r + mu) if r + mu != 0 else 1)

    distro = nbinom(r, p)
    return cdf_fn_to_probas(distro.cdf, total_days=total_days)


def apply_poisson(sample_days: np.ndarray, total_days: int = 30) -> List[float]:
    distro = poisson(sample_days.mean())
    return cdf_fn_to_probas(distro.cdf, total_days=total_days)


def apply_fitted_gamma(sample_days: np.ndarray, total_days: int = 30) -> List[float]:
    shape, loc, scale = gamma.fit(sample_days)
    distro = gamma(shape, loc, scale)
    return cdf_fn_to_probas(distro.cdf, total_days=total_days)
