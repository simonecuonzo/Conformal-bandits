import numpy as np
import matplotlib.pylab as plt
import seaborn as sns

from scipy.special import gamma
from scipy.stats import t, uniform

import pandas as pd
from scipy.stats import t as student_t

from math import sqrt, log


class SkewStudent(object):

    """Skewed Student distribution class.

    Attributes
    ----------
    eta : float
        Degrees of freedom. :math:`2 < \eta < \infty`
    lam : float
        Skewness. :math:`-1 < \lambda < 1`

    Methods
    -------
    pdf
        Probability density function (PDF)
    cdf
        Cumulative density function (CDF)
    ppf
        Inverse cumulative density function (ICDF)
    rvs
        Random variates with mean zero and unit variance

    """

    def __init__(self, eta=10., lam=-.1):
        """Initialize the class.

        Parameters
        ----------
        eta : float
            Degrees of freedom. :math:`2 < \eta < \infty`
        lam : float
            Skewness. :math:`-1 < \lambda < 1`

        """
        self.eta = eta
        self.lam = lam

    def __const_a(self):
        """Compute a constant.

        Returns
        -------
        a : float

        """
        return 4*self.lam*self.__const_c()*(self.eta-2)/(self.eta-1)

    def __const_b(self):
        """Compute b constant.

        Returns
        -------
        b : float

        """
        return (1 + 3*self.lam**2 - self.__const_a()**2)**.5

    def __const_c(self):
        """Compute c constant.

        Returns
        -------
        c : float

        """
        return gamma((self.eta+1)/2) \
            / ((np.pi*(self.eta-2))**.5*gamma(self.eta/2))

    def pdf(self, arg):
        """Probability density function (PDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        Returns
        -------
        array
            PDF values. Same shape as the input.

        """
        c = self.__const_c()
        a = self.__const_a()
        b = self.__const_b()

        return b*c*(1 + 1/(self.eta-2) \
            *((b*arg+a)/(1+np.sign(arg+a/b)*self.lam))**2)**(-(self.eta+1)/2)

    def loglikelihood(self, param, arg):
        """Probability density function (PDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        Returns
        -------
        array
            PDF values. Same shape as the input.

        """
        self.eta, self.lam = param

        return -np.log(self.pdf(arg)).sum()

    def cdf(self, arg):
        """Cumulative density function (CDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate CDF at

        Returns
        -------
        array
            CDF values. Same shape as the input.

        """
        a = self.__const_a()
        b = self.__const_b()

        y = (b*arg+a)/(1+np.sign(arg+a/b)*self.lam) * (1-2/self.eta)**(-.5)
        cond = arg < -a/b

        return cond * (1-self.lam) * t.cdf(y, self.eta) \
            + ~cond * (-self.lam + (1+self.lam) * t.cdf(y, self.eta))

    def ppf(self, arg):
        """Inverse cumulative density function (ICDF).

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at. Must belong to (0, 1)

        Returns
        -------
        array
            ICDF values. Same shape as the input.

        """
        arg = np.atleast_1d(arg)

        a = self.__const_a()
        b = self.__const_b()

        cond = arg < (1-self.lam)/2

        ppf1 = t.ppf(arg / (1-self.lam), self.eta)
        ppf2 = t.ppf(.5 + (arg - (1-self.lam)/2) / (1+self.lam), self.eta)
        ppf = -999.99*np.ones_like(arg)
        ppf = np.nan_to_num(ppf1) * cond \
            + np.nan_to_num(ppf2) * np.logical_not(cond)
        ppf = (ppf * (1+np.sign(arg-(1-self.lam)/2)*self.lam) \
            * (1-2/self.eta)**.5 - a)/b

        if ppf.shape == (1, ):
            return float(ppf)
        else:
            return ppf

    def rvs(self, size=1):
        """Random variates with mean zero and unit variance.

        Parameters
        ----------
        size : int or tuple
            Size of output array

        Returns
        -------
        array
            Array of random variates

        """
        return self.ppf(uniform.rvs(size=size))

    def plot_pdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot probability density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate PDF at

        """
        scale = (self.eta/(self.eta-2))**.5
        plt.plot(arg, t.pdf(arg, self.eta, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.pdf(arg), label='skew-t distribution')
        plt.legend()
        plt.show()

    def plot_cdf(self, arg=np.linspace(-2, 2, 100)):
        """Plot cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate CDF at

        """
        scale = (self.eta/(self.eta-2))**.5
        plt.plot(arg, t.cdf(arg, self.eta, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.cdf(arg), label='skew-t distribution')
        plt.legend()
        plt.show()

    def plot_ppf(self, arg=np.linspace(.01, .99, 100)):
        """Plot inverse cumulative density function.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at

        """
        scale = (self.eta/(self.eta-2))**.5
        plt.plot(arg, t.ppf(arg, self.eta, scale=1/scale),
                 label='t distribution')
        plt.plot(arg, self.ppf(arg), label='skew-t distribution')
        plt.legend()
        plt.show()

    def plot_rvspdf(self, arg=np.linspace(-2, 2, 100), size=1000):
        """Plot kernel density estimate of a random sample.

        Parameters
        ----------
        arg : array
            Grid of point to evaluate ICDF at. Must belong to (0, 1)

        """
        rvs = self.rvs(size=size)
        xrange = [arg.min(), arg.max()]
        sns.kdeplot(rvs, clip=xrange, label='kernel')
        plt.plot(arg, self.pdf(arg), label='true pdf')
        plt.xlim(xrange)
        plt.legend()
        plt.show()




# ------------------------------
# 1) IID Normal
# ------------------------------
def sim_normal_iid(T=500, mu=0.0, sigma=0.02, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    return np.random.normal(mu, sigma, T)



# ------------------------------
# 3) IID Student-t
# ------------------------------
def sim_t_iid(T=500, df=3, scale=0.02, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    return student_t.rvs(df=df, loc=0, scale=scale, size=T)



# ------------------------------
# 5) IID Skewed Student-t
# ------------------------------
def sim_skewt_iid(T, df=5, lam=0.6, scale=0.02, random_state=None):
    """
    Simula una serie i.i.d. dalla Skewed Student-t (Hansen 1994).

    Parameters
    ----------
    T : int
        Lunghezza della serie.
    df : float
        Gradi di libertà (eta).
    lam : float
        Parametro di asimmetria (lambda).
    scale : float
        Scala (deviazione standard target).
    random_state : int or None
        Seed per la riproducibilità.

    Returns
    -------
    np.ndarray
        Serie di lunghezza T (i.i.d. skew-t).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Istanzia la distribuzione skew-t
    sts = SkewStudent(eta=df, lam=lam)

    # Genera T campioni i.i.d. (media 0, varianza 1)
    data = sts.rvs(size=T)

    # Applica la scala desiderata
    return scale * np.array(data)





def rolling_qr_aci_with_scaling(y_obs, t_local, alpha_t,
                                scores_history,
                                lookback=240,
                                eta=0.01, alpha_target=0.2,
                                use_scaling=False, dt=1):
    """
    Rolling quantile conformal forecasting con ACI e scaling opzionale.
    Versione corretta: usa score reali del passato (non ricalcolati).
    """


    window_data = np.asarray(y_obs[:-1], float)


    tau_low, tau_high = alpha_target/2.0, 1.0 - alpha_target/2.0

    q_low = np.quantile(window_data, tau_low)
    q_high = np.quantile(window_data, tau_high)
    q_med = np.quantile(window_data,  0.5)

    # 5) Ora calcolo lo score corrente e lo APPENDO
    y_t = float(y_obs[-1])

    recent_scores = scores_history


    nc = len(recent_scores)
    if nc == 0:
        qn = 0.0
    else:
        qn_idx = int(np.ceil((1 - alpha_t) * (nc + 1)) - 1)
        qn_idx = int(np.clip(qn_idx, 0, nc - 1))
        qn = float(np.sort(recent_scores)[qn_idx])


    # Intervallo predittivo
    L_t = q_low - qn
    U_t = q_high + qn
    mu_med_t = q_med


    # 5) Ora calcolo lo score corrente e lo APPENDO
    score_t = max(q_low - y_t, y_t - q_high)
    scores_history.append(float(score_t))           
    # Aggiornamento ACI
    covered = 1 if (L_t <= y_t <= U_t) else 0
    err = 1 - covered
    alpha_t_new = alpha_t + eta * (alpha_target - err)
    alpha_t_new = float(np.clip(alpha_t_new, 1e-4, 0.5))


    # Skewness Index
    den = max(np.abs(L_t), 1e-9)
    ESI_t = (U_t) / den

    return mu_med_t, L_t, U_t, ESI_t, alpha_t_new, scores_history







def run_ucb_cp_partial(df, *,
                       alpha_target=0.1, eta=0.01,
                       gamma1=0.6, beta=1.0,
                       use_scaling=True,
                       regime_mode=None,
                       lam=0.5,
                       lookback=250,
                       warm_up=50,epsilon_0 = 0.05,decay=0.7):
    """
    UCB-CP in setting PARTIAL INFORMATION (versione con regime_mode):
      - regime_mode = None     → comportamento standard
      - regime_mode = "score"  → funzione di scoring modificata in base al regime
    """
    arms = [c for c in df.columns if c != "Regime"]
    K, T = len(arms), len(df)

    # Stati
    mu_act  = {a: 0.0 for a in arms}
    L_act   = {a: -np.inf for a in arms}
    U_act   = {a:  np.inf for a in arms}
    ESI_act = {a: 0.0 for a in arms}
    N_sel   = {a: 0 for a in arms}
    alpha_k = {a: alpha_target for a in arms}
    dt_arm  = {a: 0 for a in arms}

    # Serie osservate
    y_obs = {a: [] for a in arms}
    t_obs = {a: [] for a in arms}

    # Memoria rolling degli score conformali per ogni braccio
    scores_hist = {a: [] for a in arms}
    rows = []
    T0=warm_up
    # Ciclo principale
    for t in range(T):
    

        # --- Warm-up ---
      
        if t < (3 * (T0+1)):   
            k_t = arms[t % K]
        else:
            scores = {}
           

            for a in arms:


                if np.isinf(L_act[a]) or np.isinf(U_act[a]):
                    
                    scores[a] = np.inf
                else:

                    # Formula classica (senza adattamento)
                    scores[a] =  U_act[a]

          
            epsilon=0.5*t**(-decay)


            # --- Epsilon-greedy selection ---
            if np.random.rand() > epsilon:
                # sfrutta → scegli l’arm con score massimo
                k_t = max(scores, key=scores.get)
            else:
                # esplora → scegli un arm diverso dal best
                best_arm = max(scores, key=scores.get)
                other_arms = [a for a in arms if a != best_arm]
                k_t = np.random.choice(other_arms)
    


        # --- Reward osservato ---
        r_t = df[k_t].iloc[t]



        # Aggiorna osservazioni
        y_obs[k_t].append(r_t)
        t_obs[k_t].append(t)

        # Aggiorna tempo da ultima selezione
        for a in arms:
            dt_arm[a] = 0 if a == k_t else dt_arm[a] + 1



        
        if len(y_obs[k_t]) >=  (T0+1):
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
            "t": t,
            "chosen_arm": k_t, "reward": r_t,
            **{f"mu_{a}": mu_act[a] for a in arms},
            **{f"L_{a}": L_act[a] for a in arms},
            **{f"U_{a}": U_act[a] for a in arms},
            **{f"ESI_{a}": ESI_act[a] for a in arms},
            **{f"alpha_{a}": alpha_k[a] for a in arms},
            **{f"dt_{a}": dt_arm[a] for a in arms},
        })

    # === Riepilogo finale ===
    log_df = pd.DataFrame(rows)

    mask_after = log_df["t"] >= (3 *(T0+1) +1)
    log_after = log_df.loc[mask_after].copy()

    return log_df, log_after, {
        "mu": mu_act, "L": L_act, "U": U_act, "ESI": ESI_act,
        "alpha": alpha_k, "dt": dt_arm,
        "lookback": lookback, "y_obs": y_obs, "t_obs": t_obs
    }



def run_ucb_cp_partial_comb(df, *,
                       alpha_target=0.1, eta=0.01,
                       gamma1=0.6, beta=1.0,
                       use_scaling=True,
                       regime_mode=None,
                       lam=0.5,
                       lookback=250,
                       warm_up=50,epsilon_0 = 0.05,decay=0.7):
  
    """
    UCB-CP in setting PARTIAL INFORMATION (versione con regime_mode):
      - regime_mode = None     → comportamento standard
      - regime_mode = "score"  → funzione di scoring modificata in base al regime
    """
    arms = [c for c in df.columns if c != "Regime"]
    K, T = len(arms), len(df)

    # Stati
    mu_act  = {a: 0.0 for a in arms}
    L_act   = {a: -np.inf for a in arms}
    U_act   = {a:  np.inf for a in arms}
    ESI_act = {a: 0.0 for a in arms}
    N_sel   = {a: 0 for a in arms}
    alpha_k = {a: alpha_target for a in arms}
    dt_arm  = {a: 0 for a in arms}

    # Serie osservate
    y_obs = {a: [] for a in arms}
    t_obs = {a: [] for a in arms}

    # Memoria rolling degli score conformali per ogni braccio
    scores_hist = {a: [] for a in arms}
    rows = []
    T0=warm_up
    # Ciclo principale
    for t in range(T):
      

        # --- Warm-up ---
     
        if t < (3 * (T0+1)): 
            k_t = arms[t % K]
        else:
            scores = {}
         

            for a in arms:


                if np.isinf(L_act[a]) or np.isinf(U_act[a]):
                 
                    scores[a] = np.inf
                else:
  
                    # Formula classica (senza adattamento)
                    scores[a] = (1-lam)*U_act[a]+  lam*L_act[a]
           
            epsilon=0.5*t**(-decay)


            # --- Epsilon-greedy selection ---
            if np.random.rand() > epsilon:
                # sfrutta → scegli l’arm con score massimo
                k_t = max(scores, key=scores.get)
            else:
                # esplora → scegli un arm diverso dal best
                best_arm = max(scores, key=scores.get)
                other_arms = [a for a in arms if a != best_arm]
                k_t = np.random.choice(other_arms)
       

        # --- Reward osservato ---
        r_t = df[k_t].iloc[t]



        # Aggiorna osservazioni
        y_obs[k_t].append(r_t)
        t_obs[k_t].append(t)

        # Aggiorna tempo da ultima selezione
        for a in arms:
            dt_arm[a] = 0 if a == k_t else dt_arm[a] + 1



       
        if len(y_obs[k_t]) >=  (T0+1):
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
            "t": t,
            "chosen_arm": k_t, "reward": r_t,
            **{f"mu_{a}": mu_act[a] for a in arms},
            **{f"L_{a}": L_act[a] for a in arms},
            **{f"U_{a}": U_act[a] for a in arms},
            **{f"ESI_{a}": ESI_act[a] for a in arms},
            **{f"alpha_{a}": alpha_k[a] for a in arms},
            **{f"dt_{a}": dt_arm[a] for a in arms},
        })

    # === Riepilogo finale ===
    log_df = pd.DataFrame(rows)
   
    mask_after = log_df["t"] >= (3 *(T0+1) +1)
    log_after = log_df.loc[mask_after].copy()

    return log_df, log_after, {
        "mu": mu_act, "L": L_act, "U": U_act, "ESI": ESI_act,
        "alpha": alpha_k, "dt": dt_arm,
        "lookback": lookback, "y_obs": y_obs, "t_obs": t_obs
    }



def run_ucb_cp_partial_UL(df, *,
                       alpha_target=0.1, eta=0.01,
                       gamma1=0.6, beta=1.0,
                       use_scaling=True,
                       regime_mode=None,
                       lam=0.5,
                       lookback=250,
                       warm_up=50,epsilon_0 = 0.05,decay=0.7):
  
    """
    UCB-CP in setting PARTIAL INFORMATION (versione con regime_mode):
      - regime_mode = None     → comportamento standard
      - regime_mode = "score"  → funzione di scoring modificata in base al regime
    """
    arms = [c for c in df.columns if c != "Regime"]
    K, T = len(arms), len(df)

    # Stati
    mu_act  = {a: 0.0 for a in arms}
    L_act   = {a: -np.inf for a in arms}
    U_act   = {a:  np.inf for a in arms}
    ESI_act = {a: 0.0 for a in arms}
    N_sel   = {a: 0 for a in arms}
    alpha_k = {a: alpha_target for a in arms}
    dt_arm  = {a: 0 for a in arms}

    # Serie osservate
    y_obs = {a: [] for a in arms}
    t_obs = {a: [] for a in arms}

    # Memoria rolling degli score conformali per ogni braccio
    scores_hist = {a: [] for a in arms}
    rows = []
    T0=warm_up
    # Ciclo principale
    for t in range(T):
       
        # --- Warm-up ---
       
        if t < (3 * (T0+1)):
            k_t = arms[t % K]
        else:
            scores = {}
        

            for a in arms:


                if np.isinf(L_act[a]) or np.isinf(U_act[a]):
                  
                    scores[a] = np.inf
                else:

                    # Formula classica (senza adattamento)
                    scores[a] = 0.5*U_act[a]+  0.5*L_act[a]
            
            epsilon=0.5*t**(-decay)


            # --- Epsilon-greedy selection ---
            if np.random.rand() > epsilon:
                # sfrutta → scegli l’arm con score massimo
                k_t = max(scores, key=scores.get)
            else:
                # esplora → scegli un arm diverso dal best
                best_arm = max(scores, key=scores.get)
                other_arms = [a for a in arms if a != best_arm]
                k_t = np.random.choice(other_arms)

        # --- Reward osservato ---
        r_t = df[k_t].iloc[t]



        # Aggiorna osservazioni
        y_obs[k_t].append(r_t)
        t_obs[k_t].append(t)

        # Aggiorna tempo da ultima selezione
        for a in arms:
            dt_arm[a] = 0 if a == k_t else dt_arm[a] + 1


        if len(y_obs[k_t]) >=  (T0+1):
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
            "t": t,
            "chosen_arm": k_t, "reward": r_t,
            **{f"mu_{a}": mu_act[a] for a in arms},
            **{f"L_{a}": L_act[a] for a in arms},
            **{f"U_{a}": U_act[a] for a in arms},
            **{f"ESI_{a}": ESI_act[a] for a in arms},
            **{f"alpha_{a}": alpha_k[a] for a in arms},
            **{f"dt_{a}": dt_arm[a] for a in arms},
        })

    # === Riepilogo finale ===
    log_df = pd.DataFrame(rows)
   
    mask_after = log_df["t"] >= (3 *(T0+1) +1)
    log_after = log_df.loc[mask_after].copy()

    return log_df, log_after, {
        "mu": mu_act, "L": L_act, "U": U_act, "ESI": ESI_act,
        "alpha": alpha_k, "dt": dt_arm,
        "lookback": lookback, "y_obs": y_obs, "t_obs": t_obs
    }


def run_ucb_cp_partial_L(df, *,
                       alpha_target=0.1, eta=0.01,
                       gamma1=0.6, beta=1.0,
                       use_scaling=True,
                       regime_mode=None,
                       lam=0.5,
                       lookback=250,
                       warm_up=50,epsilon_0 = 0.05,decay=0.7):
  
    """
    UCB-CP in setting PARTIAL INFORMATION (versione con regime_mode):
      - regime_mode = None     → comportamento standard
      - regime_mode = "score"  → funzione di scoring modificata in base al regime
    """
    arms = [c for c in df.columns if c != "Regime"]
    K, T = len(arms), len(df)

    # Stati
    mu_act  = {a: 0.0 for a in arms}
    L_act   = {a: -np.inf for a in arms}
    U_act   = {a:  np.inf for a in arms}
    ESI_act = {a: 0.0 for a in arms}
    N_sel   = {a: 0 for a in arms}
    alpha_k = {a: alpha_target for a in arms}
    dt_arm  = {a: 0 for a in arms}

    # Serie osservate
    y_obs = {a: [] for a in arms}
    t_obs = {a: [] for a in arms}

    # Memoria rolling degli score conformali per ogni braccio
    scores_hist = {a: [] for a in arms}
    rows = []
    T0=warm_up
    # Ciclo principale
    for t in range(T):
      

        # --- Warm-up ---
        if t < (3 * (T0+1)):
            k_t = arms[t % K]
        else:
            scores = {}
           

            for a in arms:


                if np.isinf(L_act[a]) or np.isinf(U_act[a]):
                    scores[a] = np.inf
                else:

                    scores[a] = L_act[a]

           
            epsilon=0.5*t**(-decay)


            # --- Epsilon-greedy selection ---
            if np.random.rand() > epsilon:
                # sfrutta → scegli l’arm con score massimo
                k_t = max(scores, key=scores.get)
            else:
                # esplora → scegli un arm diverso dal best
                best_arm = max(scores, key=scores.get)
                other_arms = [a for a in arms if a != best_arm]
                k_t = np.random.choice(other_arms)

        # --- Reward osservato ---
        r_t = df[k_t].iloc[t]



        # Aggiorna osservazioni
        y_obs[k_t].append(r_t)
        t_obs[k_t].append(t)

        # Aggiorna tempo da ultima selezione
        for a in arms:
            dt_arm[a] = 0 if a == k_t else dt_arm[a] + 1


        if len(y_obs[k_t]) >= (T0+1):
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
            "t": t,
            "chosen_arm": k_t, "reward": r_t,
            **{f"mu_{a}": mu_act[a] for a in arms},
            **{f"L_{a}": L_act[a] for a in arms},
            **{f"U_{a}": U_act[a] for a in arms},
            **{f"ESI_{a}": ESI_act[a] for a in arms},
            **{f"alpha_{a}": alpha_k[a] for a in arms},
            **{f"dt_{a}": dt_arm[a] for a in arms},
        })

    # === Riepilogo finale ===
    log_df = pd.DataFrame(rows)
    mask_after = log_df["t"] >= (3 *(T0+1) +1)
    log_after = log_df.loc[mask_after].copy()

    return log_df, log_after, {
        "mu": mu_act, "L": L_act, "U": U_act, "ESI": ESI_act,
        "alpha": alpha_k, "dt": dt_arm,
        "lookback": lookback, "y_obs": y_obs, "t_obs": t_obs
    }






def run_ucb1_partial(df, T0, t_axis):
    """
    UCB1 con partial information:
    - Osserva solo il reward del braccio selezionato.
    - Restituisce log_df e log_after con mu_{a} e N_{a} per ogni braccio.
    """
    arms = list(df.columns)
    K = len(arms)

    rewards_sum = {a: 0.0 for a in arms}
    pulls = {a: 0 for a in arms}
    rows = []

    # --- Warm-up ---

    for idx in range(min(3 *(T0+1), len(t_axis))):
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

    # --- decisional phase UCB1 ---
   
    for idx in range(3 *(T0+1), len(t_axis)):
        t = t_axis[idx]
        scores = {}

        for a in arms:
            mu_a = rewards_sum[a] / max(pulls[a], 1)
            
            exploration = 0.1*sqrt(2.0 *log(t) / max(pulls[a], 1))
            scores[a] = mu_a + exploration

        # Selezione del braccio
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

    # --- Output finale ---
    log_df = pd.DataFrame(rows)
   
    log_after = log_df.loc[log_df["t"] >= (3 *(T0+1) + 1)].copy()

    return log_df, log_after, pulls




def montecarlo_comparison(dataset_generator,
                          true_means,
                          N_mc=100, T=2000,
                          lookback=250,
                          alpha_target=0.1, eta=0.005,
                          use_scaling=False,
                          lam=0.7,
                          warm_up=50, name='gaussian',epsilon_0=0.05,decay=0.7,
                          method_list=[
                              "UCB-CP",
                              "UCB-CP (comb)",
                              "UCB-CP (L)",
                              "UCB-CP (UL)",
                              "UCB1"
                          ]):





    # Containers
 

    regrets = {m: {"classical_regret": [], "pseudo_regret": []
               }
           for m in method_list}
    bestarm_selection = {m: [] for m in method_list}
    worst_selections = {m: [] for m in method_list}
    best_selections = {m: [] for m in method_list}
    coverage_results = {m: [] for m in method_list}
    width_results = {m: [] for m in method_list}
    
    
    warm_up=warm_up
    lam=lam 

    method_functions = {
        "UCB-CP": run_ucb_cp_partial,
        "UCB-CP (comb)": run_ucb_cp_partial_comb,
        "UCB-CP (L)": run_ucb_cp_partial_L,
        "UCB-CP (UL)": run_ucb_cp_partial_UL,
        "UCB1": run_ucb1_partial
    }

    # =====================================================
    # Monte Carlo Loop
    # =====================================================
    for sim in range(N_mc):
        np.random.seed(sim)
        df = dataset_generator(T=T)
        arms = list(df.columns)
 

        for method in method_list:
            fn = method_functions.get(method, None)
            if fn is None:
                continue

            # --- Run strategy ---
            if method == "UCB1":
                T0 = warm_up
                t_axis = df.index.to_numpy()
                log_df, log_after, pulls = fn(df, T0, t_axis)
            else:
                log_df, log_after, state = fn(
                    df,
                    alpha_target=alpha_target, eta=eta,
                    use_scaling=use_scaling,
                    lam=lam,
                    lookback=lookback,
                    warm_up=warm_up,
                    epsilon_0 = epsilon_0,
                    decay=decay
                )


            rewards = log_after["reward"].to_numpy()
            chosen = log_after["chosen_arm"].to_numpy()
          

            
            mu_true = true_means

         

            mu_ser = pd.Series(mu_true)

            mu_star  = mu_ser.max()                     
            mu_worst = mu_ser.min()                      

            best_arms_true  = set(mu_ser.index[mu_ser == mu_star])  
            worst_arms_true = set(mu_ser.index[mu_ser == mu_worst])  

            


            mu_chosen = np.array([mu_true[a] for a in chosen])


            
            regret_classical = np.cumsum(mu_star - rewards)


          
            regret_pseudo = np.cumsum(mu_star - mu_chosen)

       
            regrets[method]["classical_regret"].append(regret_classical)
            regrets[method]["pseudo_regret"].append(regret_pseudo)
           
            chosen_best_vec  = np.isin(chosen, list(best_arms_true)).astype(int)
            chosen_worst_vec = np.isin(chosen, list(worst_arms_true)).astype(int)

            perc_best_t  = np.cumsum(chosen_best_vec)  / np.arange(1, len(chosen_best_vec)  + 1)
            #perc_worst_t = np.cumsum(chosen_worst_vec) / np.arange(1, len(chosen_worst_vec) + 1)

            bestarm_selection[method].append(perc_best_t)



        
            best_share  = chosen_best_vec.mean()
            worst_share = chosen_worst_vec.mean()

            worst_selections[method].append(worst_share)
            best_selections[method].append(best_share)
            
            
            
            if "UCB-CP" in method:
                coverages, widths = {}, {}
                

                for a in df.columns:
                     
                      t_all   = log_after["t"].to_numpy()
                      obs_all = df[a].iloc[t_all].to_numpy()             
                      Ls_all  = log_after[f"L_{a}"].to_numpy()
                      Us_all  = log_after[f"U_{a}"].to_numpy()

                     
                      chosen_mask = (log_after["chosen_arm"].to_numpy() == a)

                      
                      valid_mask = (~np.isnan(Ls_all)) & (~np.isnan(Us_all)) & chosen_mask

                      obs = obs_all[valid_mask]
                      Ls  = Ls_all[valid_mask]
                      Us  = Us_all[valid_mask]

                     
                      if len(obs) == 0:
                          
                          continue

                      
                      covered_t = ((obs >= Ls) & (obs <= Us)).astype(float)

                    
                      cov_rate = covered_t.mean()                         
                      int_width = Us - Ls
                      coverages[a] = cov_rate
                      widths[a] = {"mean": np.mean(int_width), "median": np.median(int_width)}



                coverage_results[method].append(coverages)
                width_results[method].append(widths)



            if method == "UCB1":
                ucb1_df = log_after.copy()
                ucb1_coverages = {}
                ucb1_widths = {}        
           

                for a in arms:
                    
                    mask_arm = (ucb1_df["chosen_arm"] == a)
                    log_a = ucb1_df[mask_arm].copy()
                    if len(log_a) == 0:
                       
                       
                        ucb1_coverages[a] = np.nan
                        ucb1_widths[a] = {"mean": np.nan, "median": np.nan}
                        continue

                    rewards_a = log_a["reward"].to_numpy()   
                    pulls = np.maximum(
                        1,
                        log_a.get(f"N_{a}", pd.Series(np.ones(len(log_a)))).to_numpy()
                    )
                    mu_hat = log_a.get(f"mu_{a}", pd.Series(np.zeros(len(log_a)))).to_numpy()

                    t_for_log = np.maximum(log_a["t"].to_numpy(), 2)
                    expl = 0.1*np.sqrt(2.0 * np.log(t_for_log) / pulls)
                    L_ucb = mu_hat - expl
                    U_ucb = mu_hat + expl

                    
                    covered_t = ((rewards_a >= L_ucb) & (rewards_a <= U_ucb)).astype(float)
                   
                    ucb1_coverages[a] = float(np.nanmean(covered_t)) if covered_t.size > 0 else np.nan

                    
                    int_width = U_ucb - L_ucb
                    if int_width.size > 0:
                        ucb1_widths[a] = {
                            "mean": float(np.nanmean(int_width)),
                            "median": float(np.nanmedian(int_width))
                        }
                    else:
                        ucb1_widths[a] = {"mean": np.nan, "median": np.nan}

                coverage_results[method].append(ucb1_coverages)
                width_results[method].append(ucb1_widths)         


    # =====================================================
    # Aggregazione finale
    # =====================================================

    summary = {}
    for m in method_list:


        summary[m] = {

            "best_selection_mean": np.mean(best_selections[m]),
            "best_selection_std": np.std(best_selections[m]),
            "worst_selection_mean": np.mean(worst_selections[m]),
            "worst_selection_std": np.std(worst_selections[m]),
        }

    
        if "UCB-CP" in m:
            
            cover_df = pd.DataFrame(coverage_results[m])

           
            summary[m]["coverage_by_arm_mean"] = cover_df.mean().to_dict()
            summary[m]["coverage_by_arm_std"]  = cover_df.std().to_dict()

            width_dict = {}
            for a in df.columns:
                
                arm_widths = [
                    sim.get(a, {"mean": np.nan, "median": np.nan})
                    for sim in width_results[m]
                ]
                width_df = pd.DataFrame(arm_widths)

                width_dict[a] = {
                    
                    "mean_width_mean":   width_df["mean"].mean(),
                    "mean_width_std":    width_df["mean"].std(),
                  
                    "median_width_mean": width_df["median"].mean(),
                    "median_width_std":  width_df["median"].std(),
                }

            summary[m]["width_by_arm"] = width_dict

       
        if m == "UCB1":
            cover_df = pd.DataFrame(coverage_results[m])

            summary[m]["coverage_by_arm_mean"] = cover_df.mean().to_dict()
            summary[m]["coverage_by_arm_std"]  = cover_df.std().to_dict()

            width_dict = {}
            for a in df.columns:
                arm_widths = [
                    sim.get(a, {"mean": np.nan, "median": np.nan})
                    for sim in width_results[m]
                ]
                width_df = pd.DataFrame(arm_widths)

                width_dict[a] = {
                    "mean_width_mean":   width_df["mean"].mean(),
                    "mean_width_std":    width_df["mean"].std(),
                    "median_width_mean": width_df["median"].mean(),
                    "median_width_std":  width_df["median"].std(),
                }

            summary[m]["width_by_arm"] = width_dict

    
    
    # Save results
    import os
    import pickle
    
    output_dir = "./results_rolling"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n====================")
    print(f"🔹 Scenario: {name}")
    print(f"====================")

    # file pickle for this scenario
    SUMMARY_FILE = os.path.join(output_dir, f"summary_{name}.pkl")
    REGRETS_FILE = os.path.join(output_dir, f"regrets_{name}.pkl")
    BESTSEL_FILE = os.path.join(output_dir, f"bestarm_selection_{name}.pkl")

    # Save results
    with open(SUMMARY_FILE, "wb") as f:
        pickle.dump(summary, f)
    with open(REGRETS_FILE, "wb") as f:
        pickle.dump(regrets, f)
    with open(BESTSEL_FILE, "wb") as f:
        pickle.dump(bestarm_selection, f)

    print("✅ Risultati MC salvati:")
    print(f"  - {SUMMARY_FILE}")
    print(f"  - {REGRETS_FILE}")
    print(f"  - {BESTSEL_FILE}")



    return summary, regrets, bestarm_selection









alpha_target = 0.20
eta          = 0.005
lookback = 250
warm_up = 1

lam = 0.7

decay = 0.4

epsilon_0 = 0.05


use_scaling  = False

T = 2000 
N_mc = 1000



sd=0.01


#scenario 1
#mu=0.005

#scenario 2
#mu=0.001

#scenario 3
#mu=0.01

#scenario 4
mu=0.05



def generate_three_gauss(T=T):
  arm1 = np.random.normal(mu, sd, T)
  arm2 = np.random.normal(0, sd, T)
  arm3 = np.random.normal(0, sd, T)
  return pd.DataFrame({"Arm 1": arm1, "Arm 2": arm2, "Arm 3": arm3})


# --- Student-t arms ---
from scipy.stats import t as student_t
def generate_three_student(T=T, df=3):
  arm1 = mu + student_t.rvs(3, loc=0, scale=sd, size=T)
  arm2 =0 + student_t.rvs(3, loc=0, scale=sd, size=T)
  arm3 = 0 + student_t.rvs(3, loc=0, scale=sd, size=T)
  return pd.DataFrame({"Arm 1": arm1, "Arm 2": arm2, "Arm 3": arm3})


# --- Skew-t arms ---
def generate_three_skewt(T=T):
  arm1 = mu + sim_skewt_iid(T, df=3, lam=0.3, scale=sd)
  arm2 =0 + sim_skewt_iid(T, df=3, lam=-0.5, scale=sd)
  arm3 = 0 + sim_skewt_iid(T, df=3, lam=0.6, scale=sd)
  return pd.DataFrame({"Arm 1": arm1, "Arm 2": arm2, "Arm 3": arm3})








# ----------------------------
# Run su diversi dataset
# ----------------------------


true_means = {"Arm 1": mu, "Arm 2": 0.0, "Arm 3": 0.0}

print('=== Gaussian Dataset ===')
summary_gauss,_,_ = montecarlo_comparison(
    generate_three_gauss,true_means,
    N_mc=N_mc, T=T,
    lookback=lookback, alpha_target=alpha_target, eta=eta,lam=lam,warm_up=warm_up,name='gaussian',epsilon_0=epsilon_0,decay=decay
)
print('=== Student-t Dataset ===')
summary_student,_,_ = montecarlo_comparison(
    generate_three_student,true_means,
    N_mc=N_mc, T=T,
    lookback=lookback, alpha_target=alpha_target, eta=eta,lam=lam,warm_up=warm_up,name='t-student',epsilon_0=epsilon_0,decay=decay
)
print('=== Skewed-Student-t Dataset ===')
summary_skewt,_,_ = montecarlo_comparison(
    generate_three_skewt,true_means,
    N_mc=N_mc, T=T,
    lookback=lookback, alpha_target=alpha_target, eta=eta,lam=lam,warm_up=warm_up,name='skew-t',epsilon_0=epsilon_0,decay=decay
)


def print_summary(name, summary):
    """
    Include:
      - coverage (mean ± std) per arm
      - interval width (mean ± std) per arm
    """
    print(f"\n=== {name} Dataset ===")


    for method, stats in summary.items():


        print(f"\n▶ {method}")
        # === Coverage (mean ± std) per arm ===
       
        if "coverage_by_arm_mean" in stats:
            cov_mean = stats["coverage_by_arm_mean"]
            cov_std  = stats.get("coverage_by_arm_std", {})
            print("\n Coverage by arm (mean ± std across MC runs):")
            for arm, cm in cov_mean.items():
                cs = cov_std.get(arm, np.nan)
                print(f"   {arm:<10}: {cm:>6.2%} (±{cs:>6.2%})")

        # === Interval width (mean ± std) per arm ===
 
        if "width_by_arm" in stats:
            print("\n Interval width by arm (mean/median ± std across MC runs):")
            for arm, w in stats["width_by_arm"].items():
                mw_mean   = w.get("mean_width_mean",   np.nan)
                mw_std    = w.get("mean_width_std",    np.nan)
                med_mean  = w.get("median_width_mean", np.nan)
                med_std   = w.get("median_width_std",  np.nan)
                print(
                    f"   {arm:<10}: "
                    f"mean={mw_mean:>8.4f} (±{mw_std:>7.4f}), "
                    f"median={med_mean:>8.4f} (±{med_std:>7.4f})"
                )

        print("\n" + "-" * 60)



# ----------------------------
# final print
# ----------------------------
print_summary("Gaussian", summary_gauss)
print_summary("Student-t", summary_student)
print_summary("Skew-t", summary_skewt)
