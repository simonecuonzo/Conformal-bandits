import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


output_dir = "./results_rolling"


label_map = {
    "UCB-CP": "CP-BANDIT (λ=0)",
    "UCB-CP (comb)": "CP-BANDIT (λ=0.7)",
    "UCB-CP (L)": "CP-BANDIT (λ=1)",
    "UCB-CP (UL)": "CP-BANDIT (λ=0.5)",
    "UCB1": "UCB1",
   
}


plt.rcParams.update({

    "axes.titlesize": 18,   # titoli
    "axes.labelsize": 18,   # label assi
    "xtick.labelsize": 18,  # numeri asse x
    "ytick.labelsize": 18,  # numeri asse y
    "legend.fontsize": 12   # legenda
})


# =====================================================
# Funzione: mean + CI gaussiana (CLT) sulle curve MC
# =====================================================
def mean_and_ci_empirical(curves, alpha=0.05):
    """
    curves: lista di array 1D (una per simulazione MC).
    Restituisce:
      - media nel tempo
      - lower CI
      - upper CI
    """
    cleaned = [np.asarray(r, dtype=float) for r in curves if len(r) > 0]
    if len(cleaned) == 0:
        return None, None, None

    min_len = min(len(r) for r in cleaned)
    M = np.vstack([r[:min_len] for r in cleaned])  # shape = (N_mc, T)

    M_runs = M.shape[0]
    mean_curve = np.mean(M, axis=0)
    std_curve  = np.std(M, axis=0, ddof=1)
    se_curve   = std_curve / np.sqrt(M_runs)

    z = 1.96
    lower = mean_curve - z * se_curve
    upper = mean_curve + z * se_curve

    return mean_curve, lower, upper



scenario_names = ["gaussian",'t-student','skew-t']  




for name in scenario_names:
    print(f"\n====================")
    print(f"📂 Carico scenario: {name}")
    print(f"====================")

    SUMMARY_FILE = os.path.join(output_dir, f"summary_{name}.pkl")
    REGRETS_FILE = os.path.join(output_dir, f"regrets_{name}.pkl")
    BESTSEL_FILE = os.path.join(output_dir, f"bestarm_selection_{name}.pkl")

    if not (os.path.exists(SUMMARY_FILE) and os.path.exists(REGRETS_FILE) and os.path.exists(BESTSEL_FILE)):
        print(f"⚠️ File per scenario '{name}' non trovati, salto.")
        continue

    with open(SUMMARY_FILE, "rb") as f:
        summary = pickle.load(f)
    with open(REGRETS_FILE, "rb") as f:
        regrets = pickle.load(f)
    with open(BESTSEL_FILE, "rb") as f:
        bestarm_selection = pickle.load(f)

    # metodi = chiavi del dict regrets
    method_list = list(regrets.keys())

    """# =====================================================
    # 📈 Plot 1 — Classical Regret
    # =====================================================
    plt.figure(figsize=(10, 5))

    for m in method_list:
        curves = regrets[m]["classical_regret"]
        if len(curves) == 0:
            continue

        mean_curve, lower_ci, upper_ci = mean_and_ci_empirical(curves)
        if mean_curve is None:
            continue

        x = np.arange(len(mean_curve))
        label = label_map.get(m, m)  # usa label custom se presente, altrimenti m
        plt.plot(x, mean_curve, lw=2, label=label)
        plt.fill_between(x, lower_ci, upper_ci, alpha=0.15)

    plt.title(f"Classical regret over time ({name})")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"classical_regret_plot_{name}_from_pickle.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {out_path}")
    plt.close()"""

    # =====================================================
    # 📈 Plot 2 — Pseudo Regret
    # =====================================================
    plt.figure(figsize=(10, 5))

    for m in method_list:
        curves = regrets[m]["pseudo_regret"]
        if len(curves) == 0:
            continue

        mean_curve, lower_ci, upper_ci = mean_and_ci_empirical(curves)
        if mean_curve is None:
            continue

        x = np.arange(len(mean_curve))
        label = label_map.get(m, m)
        plt.plot(x, mean_curve, lw=2, label=label)
        plt.fill_between(x, lower_ci, upper_ci, alpha=0.15)

    #plt.title(f"Regret over time ({name})")
    plt.xlabel("Round")
    plt.ylabel("Cumulative Regret")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"Regret_plot_{name}_from_pickle.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {out_path}")
    plt.close()

    # =====================================================
    # 📈 Plot 3 — Best-Arm Selection Percentage
    # =====================================================
    plt.figure(figsize=(10, 5))

    for m in method_list:
        curves = bestarm_selection[m]
        if len(curves) == 0:
            continue

        mean_curve, lower_ci, upper_ci = mean_and_ci_empirical(curves)
        if mean_curve is None:
            continue

        x = np.arange(len(mean_curve))
        label = label_map.get(m, m)
        plt.plot(x, mean_curve, lw=2, label=label)
        plt.fill_between(x, lower_ci, upper_ci, alpha=0.15)

    #plt.title(f"Cumulative Percentage of Best-Arm Selections ({name})")
    plt.xlabel("Round")
    plt.ylabel("Best-Arm Selections")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"bestarm_selection_plot_{name}_from_pickle.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"✅ Salvato: {out_path}")
    plt.close()
