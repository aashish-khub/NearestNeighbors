# This is a naive attempt at translating Tathagatha's R code in the ./synthetic_data_and_TSNN_R directory
# THIS CODE MAY CONTAIN BUGS AS A RESULT OF THE TRANSLATION!!! PLEASE CROSS-CHECK WITH TATHAGATHA'S R CODE AND USE WITH CAUTION!!!



# ||||||  _ _ _   _   ___ _  _  _  _  _  __  ||||||
# L|L|L| | | | | / \ | o \ \| || || \| |/ _| L|L|L|
#        | V V || o ||   / \\ || || \\ ( |_n
# ()()()  \_n_/ |_n_||_|\\_|\_||_||_|\_|\__/ ()()()

#   _ _  _  _  __ _  _  _  __  _  _    _  __   _  ___  ___  __
#  | | || \| |/ _/ \| \| |/ _|/ \| |  | ||  \ / \|_ _|| __||  \
#  | U || \\ ( (( o ) \\ |\_ ( o ) |_ | || o ) o || | | _| | o )
#  |___||_|\_|\__\_/|_|\_||__/\_/|___||_||__/|_n_||_| |___||__/

#    __ _  __  ___    _   _ _  ___   _   __   ||
#   / _/ \|  \| __|  / \ | U || __| / \ |  \  L|
#  ( (( o ) o ) _|  | o ||   || _| | o || o )
#   \__\_/|__/|___| |_n_||_n_||___||_n_||__/  ()


import numpy as np

def generate_sim_data(N: int, Ts: int, rho: float, seed: int, sig_noise: float = 1, snr: float|None = None) -> dict:
    """Generate simulated data with latent variables.

    Args:
        N: Number of rows
        Ts: Number of columns
        rho: Parameter for data generation
        seed: Random seed for reproducibility
        sig_noise: Standard deviation of noise
        snr: Signal-to-noise ratio (if provided, overrides sig_noise)

    Returns:
        Dictionary containing generated data, latent variables, and noise parameters

    """
    np.random.seed(seed)

    U = np.random.uniform(size=N) - 0.5
    V = np.random.uniform(size=Ts) - 0.5

    Y = np.abs(U[:, np.newaxis] + V) ** rho * np.sign(U[:, np.newaxis] + V)
    data_true = Y

    if snr is not None:
        signal_var = np.mean(Y**2)
        sig_noise = np.sqrt(signal_var / snr)

    Y += sig_noise * np.random.normal(size=(N, Ts))
    data_noise = Y

    return {
        'data_true': data_true,
        'data_noise': data_noise,
        'row_latent': U,
        'col_latent': V,
        'noise_sd': sig_noise
    }

def mcar(data: np.ndarray, seed: int, missing_prob: float = 0.5) -> dict:
    """Generate missing completely at random (MCAR) pattern in data.

    Args:
        data: Input data array
        seed: Random seed for reproducibility
        missing_prob: Probability of an entry being missing

    Returns:
        Dictionary containing observed data with NaNs and a binary mask of observed entries

    """
    np.random.seed(seed)
    N, Ts = data.shape

    missing_mask = np.random.binomial(1, missing_prob, size=(N, Ts)) == 1
    obs_data = data.copy()
    obs_data[missing_mask] = np.nan
    As = 1 - missing_mask

    return {'obs_data': obs_data, 'As': As}

def mnar(data: np.ndarray, seed: int, obs_prob_mat: np.ndarray) -> dict:
    """Generate missing not at random (MNAR) pattern in data.

    Args:
        data: Input data array
        seed: Random seed for reproducibility
        obs_prob_mat: Matrix of observation probabilities for each entry

    Returns:
        Dictionary containing observed data with NaNs and a binary mask of observed entries

    """
    np.random.seed(seed)
    N, Ts = data.shape

    missing_mask = np.random.binomial(1, obs_prob_mat, size=(N, Ts)) == 0
    obs_data = data.copy()
    obs_data[missing_mask] = np.nan
    As = 1 - missing_mask

    return {'obs_data': obs_data, 'As': As}

def deter_obsv_prob_maker(data: np.ndarray, seed: int, weights: tuple = (0.2, 0.8)) -> np.ndarray:
    """Generate a matrix of observation probabilities for missing not at random (MNAR) pattern.

    Args:
        data: Input data array
        seed: Random seed for reproducibility
        weights: Tuple of weights for the mixture components (default: (0.2, 0.8))

    Returns:
        Matrix of observation probabilities for each entry

    """
    np.random.seed(seed)
    N, Ts = data.shape

    component_1 = np.zeros(N * Ts)
    component_2 = np.where(data.flatten() > 0, 0.6, 0.4)

    component_choice = np.random.choice([0, 1], size=N*Ts, p=weights)
    probs_from_mix_dist = np.where(component_choice == 0, component_1, component_2).reshape(N, Ts)

    return probs_from_mix_dist

def synthetic_data_gen(N: int, Ts: int, rho: float, sig_noise: float, seed: int = 1,
                       snr: float|None = None, mode: str = "mcar",
                       miss_prob: float = 0.5, mnar_deter: bool = True) -> dict:
    """Generate synthetic data with missing values.

    Args:
        N: Number of rows
        Ts: Number of columns
        rho: Parameter for data generation
        sig_noise: Standard deviation of noise
        seed: Random seed for reproducibility
        snr: Signal-to-noise ratio (if provided, overrides sig_noise)
        mode: Missing data mechanism ("mcar" or "mnar")
        miss_prob: Probability of missing values (for MCAR)
        mnar_deter: Whether to use deterministic observation probabilities for MNAR

    Returns:
        Dictionary containing observed data, mask of observed entries, true data, and latent variables

    """
    if snr is not None:
        data_items = generate_sim_data(N=N, Ts=Ts, rho=rho, seed=seed, snr=snr)
    else:
        data_items = generate_sim_data(N=N, Ts=Ts, rho=rho, seed=seed, sig_noise=sig_noise)

    if mode.lower() == "mcar":
        observed_data = mcar(data_items['data_noise'], seed, miss_prob)['obs_data']
    elif mode.lower() == "mnar":
        if mnar_deter:
            miss_prob_mat = deter_obsv_prob_maker(data_items['data_true'], seed)
        else:
            miss_prob_mat = deter_obsv_prob_maker(data_items['data_true'], seed, weights=(0, 1))
        observed_data = mnar(data_items['data_noise'], seed, miss_prob_mat)['obs_data']
    else:
        raise ValueError("Error: Mode must be either MCAR or MNAR")

    return {
        'observed_data': observed_data,
        'observed_entries': ~np.isnan(observed_data),
        'full_data_true': data_items['data_true'],
        'full_data_noise': data_items['data_noise'],
        'row_latent': data_items['row_latent'],
        'col_latent': data_items['col_latent'],
        'noise_sd': data_items['noise_sd']
    }

if __name__ == "__main__":
    N = 100
    Ts = 100
    rho = 0.5
    sig_noise = 1
    seed = 1
    snr = 10
    mode = "mcar"
    miss_prob = 0.5
    mnar_deter = True

    synthetic_data = synthetic_data_gen(N, Ts, rho, sig_noise, seed, snr, mode, miss_prob, mnar_deter)
    for key, value in synthetic_data.items():
        print("="*50)
        print("{}:\n{}\n".format(key, value))
