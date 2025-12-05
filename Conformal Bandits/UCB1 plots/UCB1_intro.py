import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

import pandas as pd
from math import sqrt, log
import os


# ------------------------------
# 1) IID Normal
# ------------------------------
def sim_normal_iid(T=500, mu=0.0, sigma=0.02, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.normal(mu, sigma, T)



def run_ucb1_partial(df, T0, t_axis):
    arms = list(df.columns)
    K = len(arms)

    rewards_sum = {a: 0.0 for a in arms}
    pulls = {a: 0 for a in arms}
    rows = []

    # Warm-up
    for idx in range(min(3*(T0+1), len(t_axis))):
        t = t_axis[idx]
        a = arms[idx % K]
        r = float(df[a].iloc[t])
        rewards_sum[a] += r
        pulls[a] += 1

        mu_vals = {f"mu_{arm}": rewards_sum[arm] / max(pulls[arm], 1) for arm in arms}
        N_vals  = {f"N_{arm}": pulls[arm] for arm in arms}

        rows.append({
            "t": t,
            "chosen_arm": a,
            "reward": r,
            **mu_vals,
            **N_vals
        })

    # UCB1 decision phase
    for idx in range(3*(T0+1), len(t_axis)):
        t = t_axis[idx]
        scores = {}

        for a in arms:
            mu_a = rewards_sum[a] / max(pulls[a], 1)
            exploration = sqrt(2.0 * log(t) / max(pulls[a], 1))
            scores[a] = mu_a + exploration

        k = max(scores, key=scores.get)
        r = float(df[k].iloc[t])
        rewards_sum[k] += r
        pulls[k] += 1

        mu_vals = {f"mu_{arm}": rewards_sum[arm] / max(pulls[arm], 1) for arm in arms}
        N_vals  = {f"N_{arm}": pulls[arm] for arm in arms}

        rows.append({
            "t": t,
            "chosen_arm": k,
            "reward": r,
            **mu_vals,
            **N_vals
        })

    log_df = pd.DataFrame(rows)
    log_after = log_df.loc[log_df["t"] >= (3*(T0+1) + 1)].copy()
    return log_df, log_after, pulls




def montecarlo_comparison(dataset_generator,
                          N_mc=100, T=2000,
                          warm_up=50, name='gaussian',
                          method_list=["UCB1_0.01","UCB1_0.05","UCB1_0.1"]):

    regrets = {m: {"pseudo_regret": []} for m in method_list}
    bestarm_selection = {m: [] for m in method_list}

    output_dir = "./results_rolling"
    os.makedirs(output_dir, exist_ok=True)

    sd = 0.1

    # =====================================================
    # Monte Carlo Loop
    # =====================================================
    for m in method_list:

        if m=="UCB1_0.01":
            mu=0.01
        elif m=="UCB1_0.05":
            mu=0.05
        elif m=="UCB1_0.1":
            mu=0.1

        true_means = {"Arm 1": mu, "Arm 2": 0.0, "Arm 3": 0.0}

        for sim in range(N_mc):

            np.random.seed(sim)
            df = dataset_generator(T=T, mu=mu, sd=sd)

            arms = list(df.columns)
            T0 = warm_up
            t_axis = df.index.to_numpy()
            log_df, log_after, pulls = run_ucb1_partial(df, T0, t_axis)

            rewards = log_after["reward"].to_numpy()
            chosen = log_after["chosen_arm"].to_numpy()

            # true means
            mu_true = true_means
            mu_ser = pd.Series(mu_true)
            mu_star = mu_ser.max()
            best_arms_true  = set(mu_ser.index[mu_ser == mu_star])

            mu_chosen = np.array([mu_true[a] for a in chosen])
            regret_pseudo = np.cumsum(mu_star - mu_chosen)

            regrets[m]["pseudo_regret"].append(regret_pseudo)

            # cumulative share of best-arm selections
            chosen_best_vec  = np.isin(chosen, list(best_arms_true)).astype(int)
            perc_best_t  = np.cumsum(chosen_best_vec) / np.arange(1, len(chosen_best_vec)+1)
            bestarm_selection[m].append(perc_best_t)


    # ---------------------------
    # Helper for mean + CI
    # ---------------------------
    def mean_and_ci_empirical(curves, alpha=0.05):
        cleaned = [np.asarray(r, dtype=float) for r in curves]
        min_len = min(len(r) for r in cleaned)
        M = np.vstack([r[:min_len] for r in cleaned])

        M_runs = M.shape[0]
        mean_curve = np.mean(M, axis=0)
        std_curve  = np.std(M, axis=0, ddof=1)
        se_curve   = std_curve / np.sqrt(M_runs)

        z = 1.96
        lower = mean_curve - z * se_curve
        upper = mean_curve + z * se_curve
        return mean_curve, lower, upper


    # =====================================================
    # SAVE MEAN + CI FOR EACH METHOD
    # =====================================================
    results_list = []

    for m in method_list:
        curves = regrets[m]["pseudo_regret"]
        if len(curves) == 0:
            continue

        mean_curve, lower_ci, upper_ci = mean_and_ci_empirical(curves)

        for t in range(len(mean_curve)):
            results_list.append({
                "method": m,
                "t": t,
                "mean": mean_curve[t],
                "lower": lower_ci[t],
                "upper": upper_ci[t]
            })

    df_results = pd.DataFrame(results_list)

    csv_path = os.path.join(output_dir, f"pseudo_regret_CI_{name}.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV: {csv_path}")

        # =====================================================
    # SAVE MEAN + CI FOR BEST-ARM SELECTION
    # =====================================================
    bestarm_list = []

    for m in method_list:
        curves = bestarm_selection[m]
        if len(curves) == 0:
            continue

        mean_curve, lower_ci, upper_ci = mean_and_ci_empirical(curves)

        for t in range(len(mean_curve)):
            bestarm_list.append({
                "method": m,
                "t": t,
                "mean": mean_curve[t],
                "lower": lower_ci[t],
                "upper": upper_ci[t]
            })

    df_bestarm = pd.DataFrame(bestarm_list)

    csv_bestarm_path = os.path.join(output_dir, f"bestarm_CI_{name}.csv")
    df_bestarm.to_csv(csv_bestarm_path, index=False)
    print(f"📄 Saved CSV: {csv_bestarm_path}")



    # =====================================================
    # Plot Pseudo-Regret
    # =====================================================
    plt.figure(figsize=(10, 5))

    for m in method_list:
        curves = regrets[m]["pseudo_regret"]
        if len(curves) > 0:
            mean_curve, lower_ci, upper_ci = mean_and_ci_empirical(curves)
            plt.plot(mean_curve, lw=2, label=m)
            plt.fill_between(
                np.arange(len(mean_curve)),
                lower_ci, upper_ci,
                alpha=0.15
            )

    plt.title("Pseudo regret over time")
    plt.xlabel("Time step")
    plt.ylabel("Cumulative Regret")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"pseudo_regret_plot_{name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {output_path}")



    # =====================================================
    # Plot Best-Arm Selection
    # =====================================================
    plt.figure(figsize=(10, 5))

    for m in method_list:
        curves = bestarm_selection[m]
        if len(curves) > 0:
            mean_curve, lower_ci, upper_ci = mean_and_ci_empirical(curves)
            plt.plot(mean_curve, lw=2, label=m)
            plt.fill_between(
                np.arange(len(mean_curve)),
                lower_ci, upper_ci,
                alpha=0.15
            )

    plt.title("Cumulative Percentage of Best-Arm Selections")
    plt.xlabel("Time step")
    plt.ylabel("Share of Best-Arm Selections")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"bestarm_selection_plot_{name}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved: {output_path}")

    return regrets, bestarm_selection




# =====================================================
# Example generator
# =====================================================

def generate_three_gauss(T=2000, mu=0, sd=1):
    arm1 = np.random.normal(mu, sd, T)
    arm2 = np.random.normal(0, sd, T)
    arm3 = np.random.normal(0, sd, T)
    return pd.DataFrame({"Arm 1": arm1, "Arm 2": arm2, "Arm 3": arm3})


# =====================================================
# RUN
# =====================================================

warm_up = 1
T = 2000
N_mc = 500

_, _ = montecarlo_comparison(
    generate_three_gauss,
    N_mc=N_mc, T=T,
    warm_up=warm_up,
    name='gaussian_UCB1'
)



####################### ISTOGRAMMI #######################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_dir = "./results_rolling"
# Fix the seed for reproducibility (optional)
np.random.seed(12345)

T = 2000  # number of observations

def generate_three_gauss(T, mu=0, sd=1):
    """
    Generate a DataFrame with three arms:
    - Arm 1 ~ N(mu, sd)
    - Arm 2 ~ N(0, sd)
    - Arm 3 ~ N(0, sd)
    """
    arm1 = np.random.normal(mu, sd, T)
    arm2 = np.random.normal(0,  sd, T)
    arm3 = np.random.normal(0,  sd, T)
    return pd.DataFrame({"Arm 1": arm1, "Arm 2": arm2, "Arm 3": arm3})

# List of (mu, sd) combinations
param_list = [
    (0.01, 0.1),
    (0.05, 0.1),
    (0.10, 0.1)
]

colors = ["tab:blue", "tab:orange", "tab:green"]



for mu, sd in param_list:
    df = generate_three_gauss(T=T, mu=mu, sd=sd)
    # Save CSV
    csv_filename = f"three_arms_mu{mu}_sd{sd}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    print(f"📄 Saved CSV: {csv_path}")

    plt.figure(figsize=(7, 5))

    # Plot the histograms of the three arms on the same graph
    for col, color in zip(df.columns, colors):
        plt.hist(
            df[col],
            bins=30,
            alpha=0.5,
            density=True,
            label=col,
            color=color
        )

    plt.title(f"Histograms of the three arms – mu={mu}, sd={sd}")
    plt.xlabel("Value")
    plt.ylabel("Estimated density")
    plt.legend()
    plt.tight_layout()
    plt.show()