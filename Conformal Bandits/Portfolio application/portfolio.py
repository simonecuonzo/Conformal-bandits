import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from types import SimpleNamespace
import matplotlib.patches as mpatches
import os
from math import sqrt, log


plt.rcParams.update({

    "axes.titlesize": 18,   # titoli
    "axes.labelsize": 18,   # label assi
    "xtick.labelsize": 18,  # numeri asse x
    "ytick.labelsize": 18,  # numeri asse y
    "legend.fontsize": 12   # legenda
})



def get_three_portfolios(start="2015-01-01",
                         end="2025-01-01",
                         freq="D",
                         assets=None,
                         mk_window=252,
                         use_log_returns=True):
    """
    Returns the returns of three portfolios: MV, Sell-All, Equally-Weighted.

    """

    
    csv_path = "etf_prices.csv"
  
    prices = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

    prices = prices.sort_index().dropna(how="all")

    print(prices)



    # ---- resample to weekly if requested ----
    if freq == "W":
        # take last close of the week
        prices = prices.resample("W-FRI").last().dropna()

    # ---- compute returns ----
    if use_log_returns:
        R = np.log(prices / prices.shift(1)).dropna()
    else:
        R = prices.pct_change().dropna()

    T, n = R.shape
    Rv = R.values
    idx = R.index

    # containers for portfolio returns (fractional, same domain as use_log_returns)
    r_mv = np.zeros(T)
    r_sa = np.zeros(T)   # always zero (sell-all)
    r_ew = np.zeros(T)

    # equal weights (rebalance every step)
    ew_w = np.ones(n) / n

    # helper: convert row returns to gross multipliers
    def gross_from_row(r_row):
        if use_log_returns:
            return np.exp(r_row)         # per-asset gross
        else:
            return 1.0 + r_row

    # try cvxopt for nonnegative min-variance; else closed-form
    def min_var_weights(window_returns):
        # estimate covariance
        Sigma = np.cov(window_returns.T)
        Sigma = np.asarray(Sigma, float)

        # numerical guard
        # small diag jitter for stability
        eps = 1e-8
        Sigma = Sigma + eps * np.eye(Sigma.shape[0])

        # First try CVXOPT with w>=0, sum w = 1
        try:
            import cvxopt
            from cvxopt import matrix, solvers
            solvers.options['show_progress'] = False


            # maximize mu^T w  - (1/2) w^T Σ w
            #mu_hat = window_returns.mean(axis=0)
            #P = matrix(Sigma)                # (1/2) w^T P w
            #q = matrix(-mu_hat)              # -mu in min form



            P = matrix(Sigma)
            q = matrix(np.zeros(n))

            G = matrix(-np.eye(n))               # w >= 0  -> -I w <= 0
            h = matrix(np.zeros(n))
            A = matrix(np.ones((1, n)))
            b = matrix(1.0)
            sol = solvers.qp(P, q, G, h, A, b)
            w = np.array(sol['x']).flatten()
            # numerical cleanup
            w[w < 0] = 0.0
            s = w.sum()
            w = w / s if s > 0 else np.ones(n)/n
            return w
        except Exception:
            # Closed-form (allows shorts): w ∝ Σ^{-1} 1
            one = np.ones(n)
            try:
                invSigma = np.linalg.pinv(Sigma)
                w = invSigma @ one
                s = w.sum()
                w = w / s if s != 0 else np.ones(n)/n
                # clip to nonnegative and renormalize for robustness
                w = np.clip(w, 0, None)
                s = w.sum()
                w = w / s if s > 0 else np.ones(n)/n
                return w
            except Exception:
                return np.ones(n) / n

    # ---- roll through time ----
    for k in range(T):
        # weights at the start of period k (to earn return of period k)
        # EW is constant rebalanced
        w_ew = ew_w

        # MV needs a window; if insufficient history, fall back to EW
        if k >= mk_window:
            window = Rv[k - mk_window : k, :]
            w_mv = min_var_weights(window)
        else:
            w_mv = ew_w

        # compute period-k portfolio return from weights and asset grosses
        g = gross_from_row(Rv[k, :])         # per-asset gross
        # exact portfolio gross under instantaneous rebalance to w:
        port_g_mv = float(np.dot(w_mv, g))
        port_g_ew = float(np.dot(w_ew, g))

        if use_log_returns:
            r_mv[k] = np.log(port_g_mv)
            r_ew[k] = np.log(port_g_ew)
        else:
            r_mv[k] = port_g_mv - 1.0
            r_ew[k] = port_g_ew - 1.0

        # Sell-All is zero return by construction
        r_sa[k] = 0.0

    out = pd.DataFrame({
        "Arm 1": r_mv,   # MV
        "Arm 2": r_sa,   # Sell-All
        "Arm 3": r_ew    # Equally-Weighted
    }, index=idx)

    return out



def detect_regimes_hmm(start="2015-01-01", end="2025-01-01", n_states=3):
    """
    Estimates a HMM model on the S&P 500 returns and identifies market regimes.
    Returns a time series with labels 'Bear', 'Neutral', and 'Bull'.
    """
    

   
    csv_path = "sp500_prices.csv"

    
    data = pd.read_csv(csv_path, index_col="Date", parse_dates=True)

    
    data = data.sort_index().dropna(how="all")

    print(data)

    ret = np.log(data / data.shift(1)).dropna()
    X = ret.values.reshape(-1, 1)

    # --- Fit HMM ---
    model = GaussianHMM(
        n_components=n_states, covariance_type="full",
        n_iter=500, random_state=42
    )
    model.fit(X)
    hidden_states = model.predict(X)

    # --- Order states by mean returns ---
    means = model.means_.flatten()
    order = np.argsort(means)
    if n_states == 3:
        labels = {order[0]: "Bear", order[1]: "Neutral", order[2]: "Bull"}
    else:
        labels = {order[0]: "Bear", order[1]: "Bull"}

    regime_series = pd.Series(
        [labels[s] for s in hidden_states],
        index=ret.index,
        name="Regime"
    )

    print("Regime means:", means)
    print("Ordered labels:", labels)

    return regime_series



def merge_assets_with_regime(df_assets, regime_series):

    """
    Aligns the three real assets with the market regime estimated via HMM.

    Output:
    DataFrame with columns: 'Arm 1', 'Arm 2', 'Arm 3', 'Regime'
    """

    # Alignment on common time indices
    df_merged = df_assets.merge(regime_series, left_index=True, right_index=True, how="inner")
    print(f"→ Dati uniti su {len(df_merged)} date comuni.")
    print(df_merged["Regime"].value_counts())

    return df_merged

# --- Step 1: Download real assets ---

df_real = get_three_portfolios(start="2015-01-01", end="2025-01-01", freq="D")

# --- Step 2: Estimate market regimes ---
regimes = detect_regimes_hmm(start="2015-01-01", end="2025-01-01", n_states=3)

# --- Step 3: Merge asset + regime ---
df_real_with_regime = merge_assets_with_regime(df_real, regimes)

# --- Step 4: View result ---
print(df_real_with_regime.head())




###########################








def rolling_qr_aci_with_scaling(y_obs, t_local, alpha_t,
                                scores_history,
                                lookback=240,
                                eta=0.01, alpha_target=0.1,
                                use_scaling=False, dt=1):
    """
    Quantile conformal forecasting with ACI
    """


    
    window_data = np.asarray(y_obs[:-1], float)
    tau_low, tau_high,tau_med = alpha_target/2 , 1 - alpha_target/2 , 0.5
    q_low = np.quantile(window_data, tau_low)
    q_high = np.quantile(window_data, tau_high)
    q_med = np.quantile(window_data, tau_med)

    
   
    y_t = float(y_obs[-1])
    
 
    recent_scores = scores_history
    
    nc = len(recent_scores)
    if nc == 0:
        qn = 0.0
    else:
        qn_idx = int(np.ceil((1 - alpha_t) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[qn_idx])
    

    # predictive interval
    L_t = q_low - qn
    U_t = q_high + qn
    mu_med_t = q_med
    
 
    score_t = max(q_low - y_t, y_t - q_high)
    scores_history.append(float(score_t))           
    
    
    # Update ACI
    covered = 1 if (L_t <= y_t <= U_t) else 0
    err = 1 - covered
    alpha_t_new = alpha_t + eta * (alpha_target - err)
    alpha_t_new = float(np.clip(alpha_t_new, 1e-4, 0.5))


    # ESI
    den = max(np.abs(L_t), 1e-9)
    ESI_t = (U_t) / den

    return mu_med_t, L_t, U_t, ESI_t, alpha_t_new, scores_history




def run_ucb_cp_partial(df, *,
                       alpha_target=0.1, eta=0.01,
                       gamma1=0.6, beta=1.0,
                       use_scaling=True,
                       regime_mode=None,
                       lam=0.5,
                       lookback=250,epsilon=0.05):
    """
    UCB-CP in PARTIAL INFORMATION setting (version with regime_mode):
    - regime_mode = None → standard behavior
    - regime_mode = "score" → scoring function modified based on regime
    """
    arms = [c for c in df.columns if c != "Regime"]
    K, T = len(arms), len(df)


    mu_act  = {a: 0.0 for a in arms}
    L_act   = {a: -np.inf for a in arms}
    U_act   = {a:  np.inf for a in arms}
    ESI_act = {a: 0.0 for a in arms}
    N_sel   = {a: 0 for a in arms}
    alpha_k = {a: alpha_target for a in arms}
    dt_arm  = {a: 0 for a in arms}


    y_obs = {a: [] for a in arms}
    t_obs = {a: [] for a in arms}
    
    
    T0=lookback
    
    
    scores_hist = {a: [] for a in arms}
    rows = []

   
    for t in range(T):
        regime_t = df["Regime"].iloc[t]

        # --- Warm-up ---
        if t < (3 * (T0+1)):
            k_t = arms[t % K]
        else:
            scores = {}
         

            for a in arms:
                

                if np.isinf(L_act[a]) or np.isinf(U_act[a]):
                    scores[a] = np.inf
                else:
                    # ===============================
                    # Adaptive scoring based on regime
                    # ===============================
                    if regime_mode == "score":

                        


                        if regime_t == "Bull":
                            scores[a] = U_act[a]
                        elif regime_t == "Neutral":
                            scores[a] = U_act[a]
                        else:  # Bear
                            scores[a] = -np.abs(L_act[a])
                    else:
                        # Classic formula (without adaptation)
                        scores[a] =  U_act[a]

        
       
            # --- Epsilon-greedy selection ---
            if np.random.rand() > epsilon:
                # exploit → choose the weapon with the maximum score
                k_t = max(scores, key=scores.get)
            else:
                # explore → choose an arm other than the best
                best_arm = max(scores, key=scores.get)
                other_arms = [a for a in arms if a != best_arm]
                k_t = np.random.choice(other_arms)

        # --- Observed Reward  ---
        r_t = df[k_t].iloc[t]



        # Update observations
        y_obs[k_t].append(r_t)
        t_obs[k_t].append(t)

       
        for a in arms:
            dt_arm[a] = 0 if a == k_t else dt_arm[a] + 1



        
        if len(y_obs[k_t]) >= (lookback + 1):
            mu, L, U, esi, alpha_new, scores_hist[k_t] = rolling_qr_aci_with_scaling(
                y_obs=np.array(y_obs[k_t], dtype=float),
                t_local=len(y_obs[k_t]) - 1,
                alpha_t=alpha_k[k_t],
                scores_history=scores_hist[k_t],
                lookback=lookback,
                eta=eta,
                alpha_target=alpha_target,
                use_scaling=use_scaling,
                dt=dt_arm[k_t]
            )
            mu_act[k_t], L_act[k_t], U_act[k_t], ESI_act[k_t] = mu, L, U, esi
            alpha_k[k_t] = float(np.clip(alpha_new, 1e-4, 0.5))


        # Logging
        rows.append({
            "t": t, "Regime": regime_t, "chosen_arm": k_t, "reward": r_t,
            **{f"mu_{a}": mu_act[a] for a in arms},
            **{f"L_{a}": L_act[a] for a in arms},
            **{f"U_{a}": U_act[a] for a in arms},
            **{f"ESI_{a}": ESI_act[a] for a in arms},
            **{f"alpha_{a}": alpha_k[a] for a in arms},
            **{f"dt_{a}": dt_arm[a] for a in arms},
        })

   
    log_df = pd.DataFrame(rows)
    mask_after = log_df["t"] >= (3* (T0+1) + 1)
    log_after = log_df.loc[mask_after].copy()

    return log_df, log_after, {
        "mu": mu_act, "L": L_act, "U": U_act, "ESI": ESI_act,
        "alpha": alpha_k, "dt": dt_arm,
        "lookback": lookback, "y_obs": y_obs, "t_obs": t_obs
    }




def run_ucb1_partial(df, T0, t_axis):
    """
    UCB1 with warm-up consistent with UCB-CP.
   
    """
    arms = [c for c in df.columns if c != "Regime"]
    K = len(arms)

    rewards_sum = {a: 0.0 for a in arms}
    pulls = {a: 0 for a in arms}
    y_obs = {a: [] for a in arms}
    rows = []

    # --- Warm-up  ---
    for idx in range(min(3 * (T0+1), len(t_axis))):
        t = t_axis[idx]
        a = arms[idx % K]
        regime_t = df["Regime"].iloc[t]
        r = df[a].iloc[t]


        rewards_sum[a] += r
        pulls[a] += 1
        y_obs[a].append(r)

        mu_vals = {f"mu_{arm}": rewards_sum[arm] / max(pulls[arm], 1) for arm in arms}
        N_vals  = {f"N_{arm}": pulls[arm] for arm in arms}

        rows.append({
            "t": t,
            "Regime": regime_t, "chosen_arm": a,
            "reward": r,
            **mu_vals,
            **N_vals
        })
     

    # --- CLASSIC UCB1 (decision phase) ---
    for idx in range(3 * (T0+1), len(t_axis)):
        t = t_axis[idx]
        regime_t = df["Regime"].iloc[t]

       
        scores = {
            a: (rewards_sum[a] / max(pulls[a], 1))
               + sqrt(2.0 * log(max(t, 2)) / max(pulls[a], 1))
            for a in arms
        }
        k = max(scores, key=scores.get)
        r = df[k].iloc[t]



        rewards_sum[k] += r
        pulls[k] += 1
        y_obs[k].append(r)

        mu_vals = {f"mu_{arm}": rewards_sum[arm] / max(pulls[arm], 1) for arm in arms}
        N_vals  = {f"N_{arm}": pulls[arm] for arm in arms}

        rows.append({
            "t": t,
            "Regime": regime_t, "chosen_arm": k,
            "reward": r,
            **mu_vals,
            **N_vals
        })


    # --- Logging finale ---
    log_df = pd.DataFrame(rows)
    log_after = log_df.loc[log_df["t"] >= (3 * (T0+1)+1)].copy()

    return log_df, log_after, pulls




def run_ucb1_partial_comb(df, T0, t_axis, lam=2.0):
    """
    UCB1 with risk-aware scoring

    """
    arms = [c for c in df.columns if c != "Regime"]
    K = len(arms)

    rewards_sum = {a: 0.0 for a in arms}
    pulls = {a: 0 for a in arms}
    y_obs = {a: [] for a in arms}
    rows = []

    # --- Warm-up round robin ---
    for idx in range(min(3 * (T0+1), len(t_axis))):
        t = t_axis[idx]
        a = arms[idx % K]
        regime_t = df["Regime"].iloc[t]
        r = df[a].iloc[t]

        rewards_sum[a] += r
        pulls[a] += 1
        y_obs[a].append(r)

        mu_vals = {f"mu_{arm}": rewards_sum[arm] / max(pulls[arm], 1) for arm in arms}
        N_vals  = {f"N_{arm}": pulls[arm] for arm in arms}

        rows.append({
            "t": t,
            "Regime": regime_t, "chosen_arm": a,
            "reward": r,
            **mu_vals,
            **N_vals
        })
       

    # --- UCB1 con regime-aware scoring ---
    for idx in range(3 * (T0+1), len(t_axis)):
        t = t_axis[idx]
        regime_t = df["Regime"].iloc[t]

        scores = {}
        for a in arms:
            mean_a = rewards_sum[a] / max(pulls[a], 1)
            sigma_a = np.std(y_obs[a]) if len(y_obs[a]) > 1 else 0.0
            expl = sqrt(2.0 * log(max(t, 2)) / max(pulls[a], 1))


            adj_mean = (1-lam)* mean_a -  lam*sigma_a

            scores[a] = adj_mean + expl

        # select the arm with the highest score
        k = max(scores, key=scores.get)
        r = df[k].iloc[t]

        rewards_sum[k] += r
        pulls[k] += 1
        y_obs[k].append(r)
        mu_vals = {f"mu_{arm}": rewards_sum[arm] / max(pulls[arm], 1) for arm in arms}
        N_vals  = {f"N_{arm}": pulls[arm] for arm in arms}

        rows.append({
            "t": t,
            "Regime": regime_t, "chosen_arm": k,
            "reward": r,
            **mu_vals,
            **N_vals
        })
    

    # --- Logging finale ---
    log_df = pd.DataFrame(rows)
    log_after = log_df.loc[log_df["t"] >= (3 * (T0+1)+1)].copy()

    return log_df, log_after, pulls







def run_ucb1_partial_regime_score_comb(df, T0, t_axis, lam=2.0):
    """
    UCB1 with regime-aware scoring:
    - In 'Bear' regimes, the exploitation term (mean) is penalized
    with a weighted standard deviation.
    """
    arms = [c for c in df.columns if c != "Regime"]
    K = len(arms)

    rewards_sum = {a: 0.0 for a in arms}
    pulls = {a: 0 for a in arms}
    y_obs = {a: [] for a in arms}
    rows = []

    # --- Warm-up round robin ---
    for idx in range(min(3 * (T0+1), len(t_axis))):
        t = t_axis[idx]
        a = arms[idx % K]
        regime_t = df["Regime"].iloc[t]
        r = df[a].iloc[t]

        rewards_sum[a] += r
        pulls[a] += 1
        y_obs[a].append(r)

        mu_vals = {f"mu_{arm}": rewards_sum[arm] / max(pulls[arm], 1) for arm in arms}
        N_vals  = {f"N_{arm}": pulls[arm] for arm in arms}

        rows.append({
            "t": t,
            "Regime": regime_t, "chosen_arm": a,
            "reward": r,
            **mu_vals,
            **N_vals
        })
        

    # --- UCB1 con regime-aware scoring ---
    for idx in range(3 * (T0+1), len(t_axis)):
        t = t_axis[idx]
        regime_t = df["Regime"].iloc[t]

        scores = {}
        for a in arms:
            mean_a = rewards_sum[a] / max(pulls[a], 1)
            sigma_a = np.std(y_obs[a]) if len(y_obs[a]) > 1 else 0.0
            expl = sqrt(2.0 * log(max(t, 2)) / max(pulls[a], 1))

            # Regime-dependent exploitation term
            if regime_t == "Bull":
                adj_mean = mean_a
            elif regime_t == "Neutral":
                adj_mean = mean_a
            else:  # Bear
                adj_mean = (1-lam)* mean_a - lam*sigma_a

            scores[a] = adj_mean + expl

        # Seleziono il braccio con score massimo
        k = max(scores, key=scores.get)
        r = df[k].iloc[t]

        rewards_sum[k] += r
        pulls[k] += 1
        y_obs[k].append(r)

        mu_vals = {f"mu_{arm}": rewards_sum[arm] / max(pulls[arm], 1) for arm in arms}
        N_vals  = {f"N_{arm}": pulls[arm] for arm in arms}

        rows.append({
            "t": t,
            "Regime": regime_t, "chosen_arm": k,
            "reward": r,
            **mu_vals,
            **N_vals
        })
       

    # --- Logging finale ---
    log_df = pd.DataFrame(rows)
    log_after = log_df.loc[log_df["t"] >= (3 * (T0+1)+1)].copy()

    return log_df, log_after, pulls



# ===================================================
# Calculating financial metrics
# ===================================================

def calculate_metrics(cumret, mar_per_period=0.0):
    W = pd.Series(cumret, copy=True).astype(float).dropna()
    n = W.shape[0]
    if n < 2:
        return dict(total_return=np.nan, annualized_return=np.nan, sharpe=np.nan,
                    maxdd=np.nan, calmar=np.nan)

    W0, WT = W.iloc[0], W.iloc[-1]
    if W0 <= 0 or WT <= 0:
        return dict(total_return=np.nan, annualized_return=np.nan, sharpe=np.nan,
                    maxdd=np.nan, calmar=np.nan)

    total_return = WT / W0 - 1.0
    annualized_return = (WT / W0) ** (252.0 / n) - 1.0

    rets = W.pct_change().dropna().values
    mu = np.nanmean(rets)
    sigma = np.nanstd(rets, ddof=0)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.finfo(float).eps
    sharpe = np.sqrt(252.0) * (mu / sigma)

    hwm = np.maximum.accumulate(W.values)
    drawdown = W.values / hwm - 1.0
    maxdd = -np.min(drawdown)
    calmar = np.nan if maxdd == 0 else annualized_return / maxdd

    return dict(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe=sharpe,
        maxdd=maxdd,
        calmar=calmar
    )
    




    




def compute_cumwealth(rewards):
    """Transforms a series of percentage returns into cumulative wealth."""
    return (1 + rewards).cumprod()


def simulate_once(seed=None,
                  alpha_target=0.20,
                  eta=0.005,
                  lam=0.5,
                  warmup=50,
                  epsilon=0.05):
    """
    Runs a single simulation of ONLY CP-based strategies:
    - CP-UCB
    - CP-RegimeAware
    Returns:
    - cum_wealth_df: DataFrame (T_eff × n_CP_strategies) with cumulative wealth.
    """

    if seed is not None:
        np.random.seed(seed)

    lookback = warmup

    # --- CP-UCB (no regime) ---
    log_cp_std, log_cp_after_std, _ = run_ucb_cp_partial(
        df_real_with_regime,
        alpha_target=alpha_target,
        eta=eta,
        regime_mode=None,
        lam=lam,
        lookback=lookback,
        epsilon=epsilon
    )

    # --- CP-RegimeAware ---
    log_cp_score, log_cp_after_score, _ = run_ucb_cp_partial(
        df_real_with_regime,
        alpha_target=alpha_target,
        eta=eta,
        regime_mode="score",
        lam=lam,
        lookback=lookback,
        epsilon=epsilon
    )



    # 2) Ritorni CP-based
    strategies = {
        "CP-UCB": log_cp_after_std["reward"].to_numpy(),
        "CP-RegimeAware": log_cp_after_score["reward"].to_numpy(),
        
    }

    # 3) Wealth cumulato
    cum_wealth_df = pd.DataFrame({
        name: compute_cumwealth(rets) for name, rets in strategies.items()
    })

   
    min_len = min(len(v) for v in cum_wealth_df.values.T)
    cum_wealth_df = cum_wealth_df.iloc[:min_len]

    
    start_idx = 3 * (warmup + 1) + 1
    dates = df_real_with_regime.index[start_idx: start_idx + min_len]
    cum_wealth_df.index = dates

    return cum_wealth_df





# =========================
#  MONTE CARLO (only CP-based)
# =========================

n_mc = 2#500
base_seed = 100

alpha_target = 0.10
eta          = 0.005
lam          = 0.5
epsilon      = 0.03
warmup       = 210


B            = 0.5
xi           = 2.0
gamma        = 0.99

strategy_names = None
all_wealth_paths = {}
all_metrics = {}

for mc in range(n_mc):
    seed = base_seed + mc
    cum_wealth_df = simulate_once(
        seed=seed,
        alpha_target=alpha_target,
        eta=eta,
        lam=lam,
        warmup=warmup,
        epsilon=epsilon,
    )
    
    if strategy_names is None:
        strategy_names = list(cum_wealth_df.columns)   # solo CP-based
        all_wealth_paths = {name: [] for name in strategy_names}
        all_metrics = {name: [] for name in strategy_names}
    
   
    for name in strategy_names:
        path = cum_wealth_df[name].values.astype(float)
        all_wealth_paths[name].append(path)

        m = calculate_metrics(path)
        all_metrics[name].append(m)





########################################################################

for name in strategy_names:
    all_wealth_paths[name] = np.vstack(all_wealth_paths[name])

dates_full = cum_wealth_df.index
T_eff_full = all_wealth_paths[strategy_names[0]].shape[1]



start_eval_date = pd.to_datetime("2018-01-01")


mask_eval = dates_full >= start_eval_date


dates_mc = dates_full[mask_eval]
T_eff = dates_mc.shape[0]


for name in strategy_names:  
    M = all_wealth_paths[name]          # shape: (n_mc, T_eff_full)
    M = M[:, mask_eval]                 # only hold from 2018 onwards
    M = M / M[:, [0]]                   # rebase: wealth = 1 at the first date >= 2018
    all_wealth_paths[name] = M

########################################################################

# =========================
#  MEAN + CI 95% CUMULATIVE WEALTH
# =========================
z = 1.96  # per 95%

mean_curves = {}
ci_lower = {}
ci_upper = {}

for name in strategy_names:
    M = all_wealth_paths[name]         # (n_mc, T_eff)
    mean_curve = M.mean(axis=0)
    std_curve  = M.std(axis=0, ddof=1)
    se_curve   = std_curve / np.sqrt(n_mc)
 
    lower = mean_curve - z * se_curve
    upper = mean_curve + z * se_curve
    
    #########################
    # CI WITH EMPIRICAL QUANTILES
    #alpha=0.05
    #lower = np.quantile(M, alpha/2, axis=0)
    #upper = np.quantile(M, 1 - alpha/2, axis=0)
    #########################

    mean_curves[name] = mean_curve
    ci_lower[name] = lower
    ci_upper[name] = upper



# =========================
#  DETERMINISTIC STRATEGIES (UCB1 + HOLD) - one run only
# =========================



lookback = warmup
start_idx = 3 * (warmup + 1) + 1

# --- UCB1 (standard) ---
log_ucb1, log_ucb1_after, pulls_ucb1 = run_ucb1_partial(
    df_real_with_regime, lookback, np.arange(len(df_real_with_regime))
)

# --- UCB1 (score-based penalty) ---
log_ucb1_comb, log_ucb1_after_comb, pulls_ucb1_comb = run_ucb1_partial_comb(
    df_real_with_regime, lookback, t_axis=np.arange(len(df_real_with_regime)), lam=lam
)

# --- UCB1 (score-based penalty + regime) ---
log_ucb1_r_comb, log_ucb1_after_r_comb, pulls_ucb1_r_comb = run_ucb1_partial_regime_score_comb(
    df_real_with_regime, lookback, t_axis=np.arange(len(df_real_with_regime)), lam=lam
)



# Ritorni delle deterministic
strategies_det_returns = {
    "UCB1": log_ucb1_after["reward"].to_numpy(),
    "MV-UCB1": log_ucb1_after_comb["reward"].to_numpy(),
    "Regime-Aware MV-UCB1": log_ucb1_after_r_comb["reward"].to_numpy(),
}

# BUY & HOLD (deterministiche)
arm_to_portfolio = {
    "Arm 1": "MV",
    "Arm 2": "SA",
    "Arm 3": "EW"
}

for col in ["Arm 1", "Arm 2", "Arm 3"]:
    portfolio_name = arm_to_portfolio[col]
    strategies_det_returns[f"Hold {portfolio_name}"] = (
        df_real_with_regime[col].iloc[start_idx:].to_numpy()
    )


########################################################################################################


det_wealth = {}
for name, rets in strategies_det_returns.items():
    rets = np.asarray(rets, float)


    wealth_full = compute_cumwealth(rets)

   
    wealth_full = wealth_full[:T_eff_full]

  
    wealth_sub = wealth_full[mask_eval]

 
    wealth_rebased = wealth_sub / wealth_sub[0]

    det_wealth[name] = wealth_rebased

    
    mean_curves[name] = wealth_rebased
    ci_lower[name] = wealth_rebased
    ci_upper[name] = wealth_rebased

########################################################################################################



output_dir = "./roll_results_partial"
os.makedirs(output_dir, exist_ok=True)




import matplotlib.patches as mpatches

def add_regime_background(ax, df_regime, alpha=0.15):
    """
    Adds colored bands to the chart based on regimes,
    using actual DATEs as x-coordinates.

    df_regime must have a 'Regime' column
    and a datetime index (or the same index as the chart).
    """

    colors = {"Bull": "green", "Neutral": "gold", "Bear": "red"}

    regimes = df_regime["Regime"].values
    dates   = df_regime.index

    start_idx = 0
    current_regime = regimes[0]

    # scansiona i cambi di regime
    for t in range(1, len(regimes)):
        if regimes[t] != current_regime:
            # usa le DATE per axvspan
            ax.axvspan(dates[start_idx], dates[t],
                       color=colors[current_regime], alpha=alpha)
            start_idx = t
            current_regime = regimes[t]

    # ultimo segmento
    ax.axvspan(dates[start_idx], dates[-1],
               color=colors[current_regime], alpha=alpha)

    # legenda delle bande
    patches = [mpatches.Patch(color=c, alpha=alpha, label=lab)
               for lab, c in colors.items()]
    ax.legend(handles=patches, loc="upper left", frameon=False)




# stocastich strategies (CP-based): the ones you did MC on
cp_strategies = set(strategy_names)          # es: {"CP-UCB", "CP-RegimeAware"}

# deterministic strategies : UCB1-based + Hold (from det_wealth)
det_strategies = set(det_wealth.keys())     # es: {"UCB1", "MV-UCB1", ..., "Hold MV", ...}

# subset di strategie Hold
hold_strategies = {name for name in det_strategies if "Hold" in name}

# deterministic non-Hold (
det_bandit_strategies = det_strategies - hold_strategies


strategy_names_all = list(cp_strategies) + list(det_bandit_strategies) + list(hold_strategies)








# ==============================
# MC plot: average wealth + CI + regime bands
# ==============================
plt.figure(figsize=(14, 7))
ax = plt.gca()


df_regime_plot_mc = df_real_with_regime.loc[dates_mc]
add_regime_background(ax, df_regime_plot_mc, alpha=0.10)

for name in strategy_names_all:
    y_mean = mean_curves[name]

    if name in hold_strategies:
        # Strategie buy & hold: tratteggiato, niente CI
        ax.plot(dates_mc, y_mean, "--", alpha=0.9, label=name)

    elif name in det_bandit_strategies:
        
        ax.plot(dates_mc, y_mean, lw=1.8, label=name)
        # (ci_lower == ci_upper per queste, se vuoi NON fai fill)

    elif name in cp_strategies:
        # CP-based: media + CI (MC vera)
        ax.plot(dates_mc, y_mean, lw=2.2, label=f"{name} (mean)")
        ax.fill_between(
            dates_mc,
            ci_lower[name],
            ci_upper[name],
            alpha=0.2,
            label=f"{name} 95% CI"
        )

ax.set_title("Cumulative Wealth (MC mean) with Market Regimes — CP-based vs UCB1 vs Hold",
             fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Wealth")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")
plt.tight_layout()
#plt.show()

# 🔽 SALVA IL PLOT 
output_path = os.path.join(output_dir, "Cumulative_Wealth_with_Market_Regimes.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')


print(f"✅ Salvato: {output_path}")






# =========================
# METRICS + 95% CI: CP with MC, single-value deterministic
# =========================

metric_names = ["total_return", "annualized_return", "sharpe", "maxdd", "calmar"]
metrics_summary = {}


for name in strategy_names:  # CP-UCB, CP-RegimeAware
    M = all_wealth_paths[name]  

    summary = {}
    for m in metric_names:
        vals = []
        for run in range(n_mc):
            w = M[run, :]
            m_dict = calculate_metrics(w)
            vals.append(float(m_dict[m]))

        vals = np.array(vals)
        mean_val = np.nanmean(vals)
        std_val  = np.nanstd(vals, ddof=1)
        #se_val   = std_val / np.sqrt(n_mc)
        #lower    = mean_val - z * se_val
        #upper    = mean_val + z * se_val

        summary[f"{m}_mean"]  = mean_val
        summary[f"{m}_std"]  = std_val
        #summary[f"{m}_lower"] = lower
        #summary[f"{m}_upper"] = upper

    metrics_summary[name] = summary

# --- deterministic (UCB1 + Hold)  ---
for name, wealth in det_wealth.items():
    m_dict = calculate_metrics(wealth)

    summary = {}
    for m in metric_names:
        val = float(m_dict[m])
        summary[f"{m}_mean"]  = val



    metrics_summary[name] = summary

metrics_ci_df = pd.DataFrame(metrics_summary).T.round(4)

print("\n=== Performance Metrics (dal 2018 in poi) ===")
print(metrics_ci_df)
metrics_ci_df.to_csv(f"performance_metrics_partial_{lam}.csv", index=True)






def rolling_caviar_aci_with_scaling(
    y_obs, t_local, alpha_t,
    scores_history,
    lookback=240,
    eta=0.01, alpha_target=0.10,
    use_scaling=True, dt=1,
    model="AS", G=10.0
):


    
    window_data = np.asarray(y_obs[:-1], float)   
    


    y_t = float(y_obs[-1])


    tau_low, tau_high = alpha_target/2.0, 1.0 - alpha_target/2.0


   
    ########## PREDICTOR: EMPIRICAL QUANTILE
    q_low_t = np.quantile(window_data, tau_low)
    q_high_t = np.quantile(window_data, tau_high)
    
       
    
    recent_scores = scores_history
    
    
    nc = len(recent_scores)
    if nc == 0:
        qn = 0.0
    else:
        qn_idx = int(np.ceil((1 - alpha_t) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[qn_idx])

    # 4) predictive interval
    L_t = float(q_low_t  - qn)
    U_t = float(q_high_t + qn)
    


       
    score_t = max(q_low_t - y_t, y_t - q_high_t)
    scores_history.append(float(score_t))          
    
    
    # 6) ACI update
    covered = 1 if (L_t <= y_t <= U_t) else 0
    err = 1 - covered
    alpha_t_new = float(np.clip(alpha_t + eta * (alpha_target - err), 1e-4, 0.5))

    # 7) ESI
    den = max(abs(L_t), 1e-9)
    ESI_t = float(U_t / den)


    return float(L_t), float(U_t), float(ESI_t), float(alpha_t_new), scores_history


def run_ucb_cp_full(df, *,
                    lookback=240,
                    alpha_target=0.1, eta=0.01,
                    gamma1=0.6, beta=1.0,
                    use_scaling=True,
                    caviar_model="AS", G=10.0,
                    regime_mode = "score",lam=0.5):


    arms = [c for c in df.columns if c != "Regime"]
    K, T = len(arms), len(df)
    T0 = lookback

   

    L_act   = {a: np.nan for a in arms}
    U_act   = {a: np.nan for a in arms}
    ESI_act = {a: np.nan for a in arms}
    alpha_k = {a: float(alpha_target) for a in arms}

    scores_hist = {a: [] for a in arms}

    y_obs = {a: [] for a in arms}

    rows = []

   
    for t in range(T):
        regime_t = df["Regime"].iloc[t]
        # --- Warm-up ---
        if t < (3 * T0):
            k_t = arms[t % K]
        else:
            # --- Decision ---
            scores = {}
            for a in arms:
                if np.isinf(L_act[a]) or np.isinf(U_act[a]):
                    scores[a] = np.inf
                else:
                    if regime_mode == "score":
                        if regime_t == "Bear":
                            scores[a] = L_act[a]
                        else:  # Bull / Neutral
                            scores[a] =  U_act[a]
                    else:
                        scores[a] = U_act[a]
            k_t = max(scores, key=scores.get)


        # Observe all the rewards  (FULL INFO)
        for a in arms:
            y_obs[a].append(float(df[a].iloc[t]))

        
        r_t = y_obs[k_t][-1]




        #FULL INFO : UPDATE AL THE ARMS!!!!
        for a in arms:

            if len(y_obs[a]) >= (T0 + 1):
                L, U, esi, alpha_new, scores_hist[a] = rolling_caviar_aci_with_scaling(
                    y_obs=np.array(y_obs[a], dtype=float),
                    t_local=len(y_obs[a]) - 1,
                    alpha_t=alpha_k[a],
                    scores_history=scores_hist[a],
                    eta=eta,
                    alpha_target=alpha_target
                )






                L_act[a], U_act[a], ESI_act[a] =  L, U, esi
                alpha_k[a] = float(np.clip(alpha_new, 1e-4, 0.5))

                
       





        # Logging
        rows.append({
            "t": t,
            "chosen_arm": k_t,
            "reward": r_t,

            **{f"L_{a}":  L_act[a]  for a in arms},
            **{f"U_{a}":  U_act[a]  for a in arms},
            **{f"ESI_{a}": ESI_act[a] for a in arms},
            **{f"alpha_{a}": alpha_k[a] for a in arms},
        })

    # === Output ===
    log_df = pd.DataFrame(rows)
    mask_after = log_df["t"] >= (3*lookback + 1)
    log_after = log_df.loc[mask_after].copy()

    state = {
        "L": L_act, "U": U_act, "ESI": ESI_act,
        "alpha": alpha_k, "T0": T0, "y_obs": y_obs
    }

    return log_df, log_after, state







# ===================================================
# PARAMETRI
# ===================================================


T0 = lookback=210
caviar_model="AS"
# ===================================================
# 1️⃣ Esegui tutte le strategie
# ===================================================
log_cp_score_full_solo_u, log_cp_after_score_full_solo_u, _ = run_ucb_cp_full( # SOLO U
    df_real_with_regime, lookback=lookback, caviar_model=caviar_model, G=10.0, regime_mode=None
)
log_cp_score_full_regime_l, log_cp_after_score_full_regime_l, _ = run_ucb_cp_full( # L se risk
    df_real_with_regime, lookback=lookback, caviar_model=caviar_model, G=10.0, regime_mode="score"
)




# ===================================================
# 2️⃣ Calcolo wealth (cumulative returns)
# ===================================================

def compute_cumwealth(rewards):
    """
    Trasforma una serie di ritorni percentuali in wealth cumulato.
    """
    rewards = np.asarray(rewards, float)
    rewards = rewards[~np.isnan(rewards)]  # rimuove eventuali NaN
    if len(rewards) == 0:
        return np.array([])
    return (1 + rewards).cumprod()



strategies = {

    "UCB-CP (full)": log_cp_after_score_full_solo_u["reward"].to_numpy(),
   
    "CP-RegimeAware (full)": log_cp_after_score_full_regime_l["reward"].to_numpy(),

    
}


arm_to_portfolio = {
    "Arm 1": "MV",
    "Arm 2": "SA",
    "Arm 3": "EW"
}

for i, col in enumerate(["Arm 1", "Arm 2", "Arm 3"], 1):
    portfolio_name = arm_to_portfolio[col]
    strategies[f"Hold {portfolio_name}"] = df_real_with_regime[col].iloc[(3 *T0 + 1):].to_numpy()





for name, arr in strategies.items():
    print(f"{name:25s} → {len(arr)} osservazioni")


min_len = min(len(v) for v in strategies.values())
print(f"\n✅ Lunghezza uniforme impostata a: {min_len} osservazioni")


start_idx = 3 * T0 + 1
dates_full = df_real_with_regime.index[start_idx : start_idx + min_len]


start_eval_date = pd.to_datetime("2018-01-01")
mask_eval = dates_full >= start_eval_date
dates_eval = dates_full[mask_eval]

print(f"✅ Finestra di valutazione: da {dates_eval[0].date()} a {dates_eval[-1].date()}")
print(f"✅ Numero di osservazioni nella finestra: {len(dates_eval)}")


strategies_aligned = {}
for k, v in strategies.items():
    v = np.asarray(v, float)[:min_len]  
    v = v[mask_eval]                    
    strategies_aligned[k] = v


cum_wealth_df = pd.DataFrame({
    name: compute_cumwealth(rets) for name, rets in strategies_aligned.items()
}, index=dates_eval)



######################################################################################################



def calculate_metrics(cumret, mar_per_period=0.0):
    """
    Calculate basic metrics for a strategy:
    - Total return
    - Annualized return
    - Sharpe ratio (approximate)
    - Maximum drawdown
    - Calmar ratio
    """
    W = pd.Series(cumret, copy=True).astype(float).dropna()
    n = len(W)
    if n < 2:
        return dict(total_return=np.nan, annualized_return=np.nan, sharpe=np.nan,
                    maxdd=np.nan, calmar=np.nan)

    W0, WT = W.iloc[0], W.iloc[-1]
    if W0 <= 0 or WT <= 0:
        return dict(total_return=np.nan, annualized_return=np.nan, sharpe=np.nan,
                    maxdd=np.nan, calmar=np.nan)

    # Rendimento totale
    total_return = WT / W0 - 1.0
    # Rendimento annualizzato (assumendo 252 periodi all’anno)
    annualized_return = (WT / W0) ** (252.0 / n) - 1.0

    # Serie di ritorni giornalieri
    rets = W.pct_change().dropna().values
    mu = np.nanmean(rets)
    sigma = np.nanstd(rets, ddof=0)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = np.finfo(float).eps
    sharpe = np.sqrt(252.0) * (mu / sigma)

    # Max drawdown
    hwm = np.maximum.accumulate(W.values)
    drawdown = W.values / hwm - 1.0
    maxdd = -np.min(drawdown)
    calmar = np.nan if maxdd == 0 else annualized_return / maxdd

    return dict(
        total_return=total_return,
        annualized_return=annualized_return,
        sharpe=sharpe,
        maxdd=maxdd,
        calmar=calmar
    )

# Applica a tutte le strategie
metrics_df = pd.DataFrame({
    name: calculate_metrics(wealth) for name, wealth in cum_wealth_df.items()
}).T.round(4)


import os


output_dir = "./AS_roll_results_full"
os.makedirs(output_dir, exist_ok=True)





plt.figure(figsize=(14, 7))
for name in cum_wealth_df.columns:
    if "Hold" in name:
        plt.plot(cum_wealth_df.index, cum_wealth_df[name], "--", alpha=0.7, lw=1.5, label=name)
    else:
        plt.plot(cum_wealth_df.index, cum_wealth_df[name], lw=2.2, label=name)

plt.title("Cumulative Wealth — UCB-CP Variants vs Hold (from 2018)", fontsize=14)
plt.xlabel("Date")
plt.ylabel("Cumulative Wealth")
plt.grid(True, ls="--", alpha=0.5)
plt.legend()
# 🔽 SALVA IL PLOT 
output_path = os.path.join(output_dir, "AS_Cumulative_Wealth.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')


print(f"✅ Salvato: {output_path}")

######################################################################################################




print("\n=== 📊 Performance Metrics ===")
print(metrics_df)
# salvataggio su file CSV
metrics_df.to_csv(f"AS_performance_metrics_full_{lam}.csv", index=True)






def add_regime_background(ax, df_regime, alpha=0.15):
    """
    Adds colored bands to the chart based on market conditions.
    Uses actual DATES as coordinates on the x-axis.
    """
    colors = {"Bull": "green", "Neutral": "gold", "Bear": "red"}

    regimes = df_regime["Regime"].values
    dates   = df_regime.index

    start_idx = 0
    current_regime = regimes[0]

    for t in range(1, len(regimes)):
        if regimes[t] != current_regime:
            ax.axvspan(dates[start_idx], dates[t],
                       color=colors[current_regime], alpha=alpha)
            start_idx = t
            current_regime = regimes[t]

  
    ax.axvspan(dates[start_idx], dates[-1],
               color=colors[current_regime], alpha=alpha)

    patches = [mpatches.Patch(color=c, alpha=alpha, label=lab)
               for lab, c in colors.items()]
    ax.legend(handles=patches, loc="upper left", frameon=False)


    
plt.figure(figsize=(14, 7))
ax = plt.gca()

for name in cum_wealth_df.columns:
    if "Hold" in name:
        ax.plot(cum_wealth_df.index, cum_wealth_df[name], "--", alpha=0.7, label=name)
    else:
        ax.plot(cum_wealth_df.index, cum_wealth_df[name], lw=2, label=name)

# --- Add regime bands ONLY on the dates present in the wealth ---
df_regime_plot = df_real_with_regime.loc[cum_wealth_df.index]
add_regime_background(ax, df_regime_plot)

ax.set_title("Cumulative Wealth with Market Regimes — UCB-CP Variants vs Hold (from 2018)",
             fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative Wealth")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend(loc="upper left")
plt.tight_layout()

output_path = os.path.join(output_dir, "AS_Cumulative_Wealth_with_Market_Regimes.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight')


print(f"✅ Salvato: {output_path}")