import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import coint
from statsmodels.stats.diagnostic import acorr_ljungbox

# ==================== Persistent State ====================
_last_pos = None
_arima_models = None      # dict: inst -> fitted statsmodels result
_ou_params = None         # dict: (hub, leaf) -> {theta, mu_s, sigma_ou}
_ou_beta = None           # dict: (hub, leaf) -> float (hedge ratio)
_ou_alpha = None          # dict: (hub, leaf) -> float (intercept)
_last_refit_t = None
_last_graph_t = None
_side_state = None        # per-instrument: -1 short / 0 flat / +1 long
_star_pairs = None        # list of (hub, leaf) tuples
_singleton_insts = None   # list of singleton instrument indices
_hub_set = None           # set of hub instrument indices

# ==================== Config ====================
REFIT_EVERY = 50
GRAPH_REFIT_EVERY = 100
MIN_TRAIN = 80
MAX_ARIMA_TRAIN = 200     # trim series for ARIMA fitting speed

ARIMA_CANDIDATES = [(1, 0, 1), (1, 0, 2), (2, 0, 1), (2, 0, 2)]
MAX_ARIMA_ORDER = 3       # upper bound for Ljung-Box order refinement
LB_MAX_EXTRA = 1          # max extra LB refinement iterations
LB_PVAL_THRESH = 0.05

MAX_STEP_FRAC = 0.18
TARGET_UTIL = 0.42
INST0_MULT = 2.0
NO_TRADE_BAND_FRAC = 0.015

Z_ENTER_LONG  =  0.55
Z_EXIT_LONG   =  0.10
Z_ENTER_SHORT = -0.55
Z_EXIT_SHORT  = -0.10
MAX_Z_FOR_SIZE = 2.5
EPS = 1e-8

COINT_PVAL_THRESH = 0.05
CORR_THRESH = 0.60
# Correlation edge weight is scaled below cointegration edge weight (max 1-pval)
# to reflect that correlation alone is a weaker mean-reversion signal.
CORR_EDGE_SCALE = 0.5
# Large bonus added to instrument 0's weighted degree so it preferentially
# becomes the hub of its cluster, exploiting its 10× position limit and
# 5× lower commission rate.
HUB_INST0_BONUS = 20.0
# Volatility estimation lookback (days)
VOL_LOOKBACK = 30
# Ljung-Box lag selection parameters
LB_LAG_MAX = 10
LB_LAG_MIN = 2
LB_LAG_DIVISOR = 5


def _safe_log(x):
    return np.log(np.maximum(x, EPS))


# ==================== Union-Find ====================
class _UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.rk = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rk[px] < self.rk[py]:
            px, py = py, px
        self.p[py] = px
        if self.rk[px] == self.rk[py]:
            self.rk[px] += 1

    def components(self):
        d = {}
        for i in range(len(self.p)):
            r = self.find(i)
            d.setdefault(r, []).append(i)
        return list(d.values())


# ==================== Graph & Cluster Building ====================
def _build_clusters(log_prc):
    """
    Compute pairwise Engle-Granger cointegration (primary) and Pearson
    correlation on log-returns (secondary) edges.  Find connected
    components via Union-Find, choose a hub per component, and return
    star pairs (hub, leaf) plus singleton instruments.
    """
    nInst = log_prc.shape[0]
    uf = _UF(nInst)
    wdeg = np.zeros(nInst)

    for i in range(nInst):
        for j in range(i + 1, nInst):
            w = 0.0
            # Primary: Engle-Granger cointegration
            try:
                _, pval, _ = coint(log_prc[i], log_prc[j], trend='c')
                if np.isfinite(pval) and pval < COINT_PVAL_THRESH:
                    w = 1.0 - pval
            except Exception:
                pass
            # Secondary: Pearson correlation on log-returns
            if w == 0.0:
                try:
                    ri = np.diff(log_prc[i])
                    rj = np.diff(log_prc[j])
                    r = float(np.corrcoef(ri, rj)[0, 1])
                    if np.isfinite(r) and abs(r) > CORR_THRESH:
                        w = (abs(r) - CORR_THRESH) * CORR_EDGE_SCALE
                except Exception:
                    pass
            if w > 0.0:
                uf.union(i, j)
                wdeg[i] += w
                wdeg[j] += w

    star_pairs = []
    singleton_insts = []

    for comp in uf.components():
        if len(comp) == 1:
            singleton_insts.append(comp[0])
            continue
        # Hub: highest weighted degree; instrument 0 gets a large bonus
        hub = max(comp, key=lambda x: wdeg[x] + (HUB_INST0_BONUS if x == 0 else 0.0))
        for leaf in comp:
            if leaf != hub:
                star_pairs.append((hub, leaf))

    return star_pairs, singleton_insts


# ==================== OU Parameter Estimation ====================
def _fit_ou_pair(y_hub, y_leaf):
    """
    Step 1 – OLS regression: y_leaf = alpha + beta * y_hub + spread.
    Step 2 – Fit Ornstein-Uhlenbeck to spread via discrete OLS:
              Δspread = a + b * spread_{t-1} + noise  →  theta = -b, mu_s = a/theta.
    Returns (alpha, beta, ou_dict) or None on failure.
    """
    n = len(y_hub)
    if n < 20:
        return None
    X = np.column_stack([np.ones(n), y_hub])
    try:
        c, _, _, _ = np.linalg.lstsq(X, y_leaf, rcond=None)
    except Exception:
        return None
    alpha, beta = float(c[0]), float(c[1])
    spread = y_leaf - alpha - beta * y_hub

    ds = np.diff(spread)
    sl = spread[:-1]
    m = len(ds)
    if m < 10:
        return None
    Xou = np.column_stack([np.ones(m), sl])
    try:
        co, _, _, _ = np.linalg.lstsq(Xou, ds, rcond=None)
    except Exception:
        return None
    a, b = float(co[0]), float(co[1])
    theta = -b
    if theta <= 0:
        return None
    mu_s = a / theta
    sigma_noise = float(np.std(ds - Xou @ co))
    sigma_ou = sigma_noise / np.sqrt(max(2.0 * theta, EPS))
    if sigma_ou < EPS:
        return None
    return alpha, beta, {'theta': theta, 'mu_s': mu_s, 'sigma_ou': sigma_ou}


# ==================== ARIMA Fitting ====================
def _fit_best_arima(y):
    """
    Fit ARIMA candidates by AIC on the last MAX_ARIMA_TRAIN observations.
    Apply Ljung-Box residual diagnostic; if residuals are autocorrelated,
    attempt a higher AR or MA order (up to LB_MAX_EXTRA extra iterations).
    """
    y = y[-MAX_ARIMA_TRAIN:]

    best_m = None
    best_aic = np.inf
    best_p, best_d, best_q = ARIMA_CANDIDATES[0]

    for order in ARIMA_CANDIDATES:
        try:
            m = ARIMA(
                y, order=order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(method_kwargs={"warn_convergence": False})
            aic = m.aic if np.isfinite(m.aic) else np.inf
            if aic < best_aic:
                best_aic = aic
                best_m = m
                best_p, best_d, best_q = order
        except Exception:
            continue

    if best_m is None:
        return None

    # Ljung-Box residual refinement
    p, d, q = best_p, best_d, best_q
    for _ in range(LB_MAX_EXTRA):
        try:
            resid = np.array(best_m.resid)
            lag = min(LB_LAG_MAX, max(LB_LAG_MIN, len(resid) // LB_LAG_DIVISOR))
            lb = acorr_ljungbox(resid, lags=[lag], return_df=True)
            lb_pval = float(lb['lb_pvalue'].iloc[0])
        except Exception:
            break
        if lb_pval >= LB_PVAL_THRESH:
            break
        # Increase AR order first, then MA order
        if p < MAX_ARIMA_ORDER:
            p += 1
        elif q < MAX_ARIMA_ORDER:
            q += 1
        else:
            break
        try:
            m2 = ARIMA(
                y, order=(p, d, q),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(method_kwargs={"warn_convergence": False})
            if np.isfinite(m2.aic) and m2.aic < best_aic:
                best_m = m2
                best_aic = m2.aic
        except Exception:
            break

    return best_m


# ==================== Hysteresis State Machine ====================
def _update_side_state(z, s):
    ns = s.copy()
    flat = s == 0
    ns[flat & (z >= Z_ENTER_LONG)] = 1
    ns[flat & (z <= Z_ENTER_SHORT)] = -1
    ns[(s == 1) & (z <= Z_EXIT_LONG)] = 0
    ns[(s == -1) & (z >= Z_EXIT_SHORT)] = 0
    return ns


def _strength(z_arr):
    z_abs = np.minimum(np.abs(z_arr), MAX_Z_FOR_SIZE) / MAX_Z_FOR_SIZE
    return np.tanh(2.0 * z_abs)


# ==================== Main Entry Point ====================
def getMyPosition(prcSoFar):
    global _last_pos, _arima_models, _ou_params, _ou_beta, _ou_alpha
    global _last_refit_t, _last_graph_t, _side_state
    global _star_pairs, _singleton_insts, _hub_set

    nInst, t = prcSoFar.shape
    cur = prcSoFar[:, -1]

    # Dollar position limits (eval.py rules)
    dlr_limits = np.full(nInst, 10_000.0)
    dlr_limits[0] = 100_000.0
    pos_limits = np.maximum((dlr_limits / np.maximum(cur, EPS)).astype(int), 1)

    # Initialise persistent arrays
    if _last_pos is None or len(_last_pos) != nInst:
        _last_pos = np.zeros(nInst, dtype=int)
    if _side_state is None or len(_side_state) != nInst:
        _side_state = np.zeros(nInst, dtype=int)

    if t < MIN_TRAIN:
        return _last_pos.copy()

    log_prc = _safe_log(prcSoFar)

    # ------------------------------------------------------------------ #
    # Graph rebuild (infrequent – cointegration is a long-run property)   #
    # ------------------------------------------------------------------ #
    need_graph = (
        _star_pairs is None
        or _last_graph_t is None
        or t - _last_graph_t >= GRAPH_REFIT_EVERY
    )
    if need_graph:
        _star_pairs, _singleton_insts = _build_clusters(log_prc)
        _hub_set = {hub for hub, _ in _star_pairs}
        _last_graph_t = t

    # ------------------------------------------------------------------ #
    # Model refit (every REFIT_EVERY steps or after graph rebuild)        #
    # ------------------------------------------------------------------ #
    need_refit = (
        _arima_models is None
        or _last_refit_t is None
        or t - _last_refit_t >= REFIT_EVERY
        or need_graph
    )
    if need_refit:
        # OU parameters for each (hub, leaf) star pair
        _ou_params, _ou_beta, _ou_alpha = {}, {}, {}
        for hub, leaf in _star_pairs:
            res = _fit_ou_pair(log_prc[hub], log_prc[leaf])
            if res is not None:
                _ou_alpha[(hub, leaf)] = res[0]
                _ou_beta[(hub, leaf)] = res[1]
                _ou_params[(hub, leaf)] = res[2]

        # ARIMA for singleton instruments
        _arima_models = {}
        for inst in _singleton_insts:
            _arima_models[inst] = _fit_best_arima(log_prc[inst])

        _last_refit_t = t

    # ------------------------------------------------------------------ #
    # Compute z-scores                                                     #
    # ------------------------------------------------------------------ #
    z = np.zeros(nInst)

    # Leaf z-scores from OU spread
    for hub, leaf in _star_pairs:
        if (hub, leaf) not in _ou_params:
            continue
        ou = _ou_params[(hub, leaf)]
        alpha = _ou_alpha[(hub, leaf)]
        beta = _ou_beta[(hub, leaf)]
        spread_cur = log_prc[leaf, -1] - alpha - beta * log_prc[hub, -1]
        # Positive spread → leaf overvalued relative to hub → want to SHORT leaf.
        # Flip sign so that z > 0 means "want to BUY" (matches state-machine convention).
        z[leaf] = -(spread_cur - ou['mu_s']) / ou['sigma_ou']

    # Singleton z-scores from ARIMA 1-step forecast
    rets = prcSoFar[:, 1:] / np.maximum(prcSoFar[:, :-1], EPS) - 1.0
    lb_vol = min(VOL_LOOKBACK, t - 1)
    vol = np.std(rets[:, -lb_vol:], axis=1)
    vol = np.where(vol < 1e-4, 1e-4, vol)

    for inst in _singleton_insts:
        m = _arima_models.get(inst)
        if m is None:
            continue
        try:
            fc_val = float(m.forecast(steps=1)[0])
            exp_ret = np.exp(fc_val - log_prc[inst, -1]) - 1.0
            z[inst] = exp_ret / vol[inst]
        except Exception:
            pass

    # Hubs get no direct z-score (their position is derived from hedging)
    for hub in _hub_set:
        z[hub] = 0.0

    # ------------------------------------------------------------------ #
    # Update hysteresis side states                                        #
    # ------------------------------------------------------------------ #
    _side_state = _update_side_state(z, _side_state)
    for hub in _hub_set:
        _side_state[hub] = 0

    # ------------------------------------------------------------------ #
    # Compute target positions                                             #
    # ------------------------------------------------------------------ #
    target = np.zeros(nInst)
    str_arr = _strength(z)

    # Leaf positions (OU spread signal) + dollar-neutral hub hedge
    for hub, leaf in _star_pairs:
        if (hub, leaf) not in _ou_params:
            continue
        side = int(_side_state[leaf])
        if side == 0:
            continue
        leaf_tgt = TARGET_UTIL * float(pos_limits[leaf]) * side * float(str_arr[leaf])
        target[leaf] += leaf_tgt
        # Hub hedge: maintain dollar-neutral position opposite to leaf.
        # dollar_hub = -beta * dollar_leaf  →  shares_hub = -beta * leaf_tgt * P_leaf / P_hub
        beta = _ou_beta[(hub, leaf)]
        target[hub] += -beta * leaf_tgt * float(cur[leaf]) / max(float(cur[hub]), EPS)

    # Singleton positions (ARIMA signal)
    for inst in _singleton_insts:
        side = int(_side_state[inst])
        if side == 0:
            continue
        target[inst] = TARGET_UTIL * float(pos_limits[inst]) * side * float(str_arr[inst])

    # Overweight instrument 0 (10× position limit, 5× lower commission)
    target[0] *= INST0_MULT

    # Clip to legal bounds
    target = np.clip(target, -pos_limits, pos_limits)

    # Turnover control: cap the per-step position change
    max_step = np.maximum((MAX_STEP_FRAC * pos_limits).astype(int), 1)
    delta = target - _last_pos
    step = np.clip(delta, -max_step, max_step)

    # No-trade band: avoid churning for tiny signals
    no_trade_band = np.maximum((NO_TRADE_BAND_FRAC * pos_limits).astype(int), 1)
    step[np.abs(step) < no_trade_band] = 0

    new_pos = np.clip(_last_pos + step, -pos_limits, pos_limits).astype(int)
    _last_pos = new_pos
    return new_pos
