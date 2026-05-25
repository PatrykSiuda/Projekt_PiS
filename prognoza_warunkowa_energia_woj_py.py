# -*- coding: utf-8 -*-
# ============================================================
# ZADANIE 5 – PROGNOZA WARUNKOWA
# ENERGIA ELEKTRYCZNA – WOJEWÓDZTWA (dane panelowe)
# ============================================================
# Model optymalny (Pooled OLS):
#   ln(ZUZYCIE) = b0 + b1*ln(DOCHOD_OS_lag1) + b2*ln(CENA)
#               + b3*URBANIZACJA + b4*LICZBA_OS + b5*POW_OS + b6*HDD
#
# Procedura:
#   1. Estymacja Pooled OLS na probie uczacej 2005-2022
#   2. Prognoza X per województwo (Z4): dochod, cena, urban,
#      liczba_os, pow_os, hdd
#   3. Podstawienie X_hat do modelu => Y_hat [ln] => exp => [GWh]
#   4. Miary jakosci dla 2023-2024 (per województwo + agregat)
#   5. Prognoza ex-ante 2025 (model na 2005-2024)
#   Porownanie: prog. warunkowa | model(actual X) | naiwna
#
# Uwaga dot. zmiennej opóznionej ln_DOCHOD_OS_lag1:
#   - test 2023: lag = ln(actual dochod_2022)  [dane historyczne]
#   - test 2024: lag = ln(Z4_pred dochod_2023) [prognoza Z4]
#   - fc   2025: lag = ln(Z4_pred dochod_2024) [prognoza Z4]
# ============================================================

import os, sys
_terminal = len(sys.argv) > 0 and sys.argv[0].endswith(".py")
if _terminal:
    try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except: pass

import numpy as np
import pandas as pd
import matplotlib
if _terminal: matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings; warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    import pmdarima as pm
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "font.family": "DejaVu Sans",
})
BLUE="#1a5c96"; RED="#c0392b"; GREEN="#27ae60"; ORANGE="#e67e22"; GRAY="#7f8c8d"
PALETTE = plt.cm.tab20.colors[:16]

TRAIN_END  = 2022
TEST_YEARS = [2023, 2024]
FC_YEAR    = 2025
X_COLS = ["ln_dochod_os_lag1", "ln_cena", "urbanizacja_pct",
          "liczba_os", "pow_os", "hdd"]

# ── 1. DANE ──────────────────────────────────────────────────
df = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_wojewodztwa.xlsx"))
df = df.sort_values(["wojewodztwo", "rok"]).reset_index(drop=True)

# Imputacja zer w dochod_os
zero_mask = df["dochod_os"] <= 0
if zero_mask.any():
    df.loc[zero_mask, "dochod_os"] = np.nan
    df["dochod_os"] = (
        df.groupby("wojewodztwo")["dochod_os"]
          .transform(lambda s: s.interpolate(method="linear",
                                             limit=3,
                                             limit_direction="both"))
    )

df["ln_dochod_os"]      = np.log(df["dochod_os"].where(df["dochod_os"] > 0))
df["ln_dochod_os_lag1"] = df.groupby("wojewodztwo")["ln_dochod_os"].shift(1)
df["ln_zuzycie"]        = np.log(df["zuzycie_energii_GWh"].where(df["zuzycie_energii_GWh"] > 0))
df["ln_cena"]           = np.log(df["cena_energii_zl_kWh"].where(df["cena_energii_zl_kWh"] > 0))

PROVINCES = sorted(df["wojewodztwo"].unique())
N_PROV    = len(PROVINCES)
n_te      = len(TEST_YEARS)

# ── 2. MIARY JAKOSCI ─────────────────────────────────────────
def eval_metrics(actual, pred):
    actual = np.array(actual, dtype=float)
    pred   = np.array(pred,   dtype=float)
    if np.any(~np.isfinite(actual)) or np.any(~np.isfinite(pred)):
        return {"ME": np.nan, "MPE%": np.nan, "MAE": np.nan,
                "MAPE%": np.nan, "RMSE": np.nan, "RMSPE%": np.nan, "Theil_U": np.nan}
    me    = np.mean(pred - actual)
    mpe   = 100.0 * np.mean((pred - actual) / actual)
    mae   = np.mean(np.abs(pred - actual))
    mape  = 100.0 * np.mean(np.abs((pred - actual) / actual))
    rmse  = np.sqrt(np.mean((pred - actual) ** 2))
    rmspe = 100.0 * rmse / np.mean(actual) if np.mean(actual) != 0 else np.nan
    theil = (np.sqrt(np.mean((pred[1:] - actual[1:]) ** 2) / np.mean(actual[1:] ** 2))
             if len(actual) > 1 else np.nan)
    return {"ME": round(me,2), "MPE%": round(mpe,2), "MAE": round(mae,2),
            "MAPE%": round(mape,2), "RMSE": round(rmse,2), "RMSPE%": round(rmspe,2),
            "Theil_U": round(theil,4) if (not isinstance(theil, float) or not np.isnan(theil)) else np.nan}

# ── 3. FUNKCJA PROGNOZOWANIA (Z4) ────────────────────────────
def _X1(t):
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    return np.column_stack([np.ones(len(t)), t])

def _X2(t):
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    return np.column_stack([np.ones(len(t)), t, t**2])

def forecast_variable(y_all, years_all, train_end, n_exante):
    years_all = np.array(years_all)
    y_all     = np.array(y_all, dtype=float)
    mask_tr   = years_all <= train_end
    mask_te   = years_all >  train_end
    y_tr = y_all[mask_tr]; y_te = y_all[mask_te]
    n_tr, n_te_ = len(y_tr), len(y_te)
    t_tr = np.arange(1, n_tr+1, dtype=float)
    t_te = np.arange(n_tr+1, n_tr+n_te_+1, dtype=float)
    t_fc = np.arange(n_tr+n_te_+1, n_tr+n_te_+n_exante+1, dtype=float)
    if np.any(~np.isfinite(y_tr)) or n_tr < 5:
        return pd.DataFrame(), None, np.nan, {}
    res = {}
    try:
        m = sm.OLS(y_tr, _X1(t_tr)).fit()
        res["OLS_lin"] = {"fitted": np.asarray(m.fittedvalues).ravel(),
                          "pred_test": np.asarray(m.predict(_X1(t_te))).ravel(),
                          "pred_fc":   np.asarray(m.predict(_X1(t_fc))).ravel()}
    except: pass
    try:
        m = sm.OLS(y_tr, _X2(t_tr)).fit()
        res["OLS_kw"] = {"fitted": np.asarray(m.fittedvalues).ravel(),
                         "pred_test": np.asarray(m.predict(_X2(t_te))).ravel(),
                         "pred_fc":   np.asarray(m.predict(_X2(t_fc))).ravel()}
    except: pass
    try:
        m = AutoReg(y_tr, lags=1, old_names=False).fit()
        res["AR(1)"] = {"fitted": np.asarray(m.fittedvalues).ravel(),
                        "pred_test": np.asarray(m.predict(start=n_tr, end=n_tr+n_te_-1)).ravel(),
                        "pred_fc":   np.asarray(m.predict(start=n_tr+n_te_, end=n_tr+n_te_+n_exante-1)).ravel()}
    except: pass
    try:
        m = AutoReg(y_tr, lags=2, old_names=False).fit()
        res["AR(2)"] = {"fitted": np.asarray(m.fittedvalues).ravel(),
                        "pred_test": np.asarray(m.predict(start=n_tr, end=n_tr+n_te_-1)).ravel(),
                        "pred_fc":   np.asarray(m.predict(start=n_tr+n_te_, end=n_tr+n_te_+n_exante-1)).ravel()}
    except: pass
    if PMDARIMA_OK:
        try:
            m = pm.auto_arima(y_tr, seasonal=False, suppress_warnings=True, stepwise=True)
            pred_all = m.predict(n_periods=n_te_+n_exante)
            res["ARIMA"] = {"fitted": m.predict_in_sample(),
                            "pred_test": pred_all[:n_te_], "pred_fc": pred_all[n_te_:]}
        except: pass
    try:
        m = ExponentialSmoothing(y_tr, trend="add", seasonal=None).fit(optimized=True)
        fc_all = np.asarray(m.forecast(n_te_+n_exante)).ravel()
        res["Holt"] = {"fitted": np.asarray(m.fittedvalues).ravel(),
                       "pred_test": fc_all[:n_te_], "pred_fc": fc_all[n_te_:]}
    except: pass
    try:
        m_t = sm.OLS(y_tr, _X1(t_tr)).fit()
        e_tr = np.asarray(m_t.resid).ravel()
        e_df = pd.DataFrame({"e_t": e_tr[1:], "e_lag": e_tr[:-1]})
        m_r = sm.OLS(e_df["e_t"], sm.add_constant(e_df["e_lag"])).fit()
        alpha_e, rho_e = m_r.params["const"], m_r.params["e_lag"]
        e_last = e_tr[-1]; e_list = []
        for _ in range(n_te_+n_exante):
            e_list.append(alpha_e + rho_e * (e_list[-1] if e_list else e_last))
        e_arr = np.array(e_list)
        res["Pawlowski"] = {
            "fitted": np.asarray(m_t.fittedvalues).ravel(),
            "pred_test": np.asarray(m_t.predict(_X1(t_te))).ravel() + e_arr[:n_te_],
            "pred_fc":   np.asarray(m_t.predict(_X1(t_fc))).ravel() + e_arr[n_te_:]}
    except: pass
    if not res: return pd.DataFrame(), None, np.nan, {}
    metrics = {}
    for mname, mdata in res.items():
        try: metrics[mname] = eval_metrics(y_te, mdata["pred_test"])
        except: pass
    if not metrics: return pd.DataFrame(), None, np.nan, res
    metrics_df  = pd.DataFrame(metrics).T
    best_method = metrics_df["RMSPE%"].abs().idxmin()
    best_rmspe  = float(metrics_df.loc[best_method, "RMSPE%"])
    return metrics_df, best_method, best_rmspe, res

# ── 4. ESTYMACJA MODELU OLS (panel) ──────────────────────────
print("="*65)
print("ZADANIE 5 – PROGNOZA WARUNKOWA  |  ENERGIA WOJEWÓDZTWA")
print("Model: Pooled OLS  ln(ZUZYCIE) ~ ln(DOCHOD_lag1) + ln(CENA)")
print("       + URBANIZACJA + LICZBA_OS + POW_OS + HDD")
print(f"Proba: 2005-{TRAIN_END}  |  Test: {TEST_YEARS[0]}-{TEST_YEARS[-1]}  |  FC: {FC_YEAR}")
print("="*65)

df_model = df.dropna(subset=X_COLS + ["ln_zuzycie"]).copy()
df_model = df_model[np.isfinite(df_model[X_COLS + ["ln_zuzycie"]]).all(axis=1)].copy()

df_tr_panel   = df_model[df_model["rok"] <= TRAIN_END].copy()
df_full_panel = df_model.copy()   # 2005-2024

y_tr_p = df_tr_panel["ln_zuzycie"]
X_tr_p = sm.add_constant(df_tr_panel[X_COLS])
model_tr = sm.OLS(y_tr_p, X_tr_p).fit()

y_full_p = df_full_panel["ln_zuzycie"]
X_full_p = sm.add_constant(df_full_panel[X_COLS])
model_full = sm.OLS(y_full_p, X_full_p).fit()

print(f"\nModel OLS – proba uczaca 2005-{TRAIN_END} (n={int(model_tr.nobs)}):")
print(f"  R2={model_tr.rsquared:.4f}  R2adj={model_tr.rsquared_adj:.4f}"
      f"  AIC={model_tr.aic:.2f}  BIC={model_tr.bic:.2f}")
for nm, co, pv in zip(model_tr.params.index, model_tr.params, model_tr.pvalues):
    sig = "***" if pv<0.01 else ("**" if pv<0.05 else ("*" if pv<0.1 else ""))
    print(f"  {nm:<25} b={co:+.4f}  p={pv:.4f}  {sig}")

# ── 5. PROGNOZA WARUNKOWA PER WOJEWÓDZTWO ────────────────────
print("\n" + "="*65)
print("PROGNOZA WARUNKOWA X (Z4) + Y – PER WOJEWÓDZTWO")
print("="*65)

# VAR_CONFIG dla zmiennych nielagowanych
VAR_OTHER = [
    {"col": "cena_energii_zl_kWh", "transform": "log",  "model_col": "ln_cena"},
    {"col": "urbanizacja_pct",     "transform": None,   "model_col": "urbanizacja_pct"},
    {"col": "liczba_os",           "transform": None,   "model_col": "liczba_os"},
    {"col": "pow_os",              "transform": None,   "model_col": "pow_os"},
    {"col": "hdd",                 "transform": None,   "model_col": "hdd"},
]

prov_results = {}

for prov in PROVINCES:
    dp    = df[df["wojewodztwo"] == prov].copy().sort_values("rok")
    dp_tr = dp[dp["rok"] <= TRAIN_END].copy()
    dp_te = dp[dp["rok"].isin(TEST_YEARS)].copy()
    years_prov = dp["rok"].values

    y_te_actual = dp_te["zuzycie_energii_GWh"].values.astype(float)

    x_test_model = {}   # model_col -> array(n_te)
    x_fc_model   = {}   # model_col -> scalar
    x_best       = {}   # col -> method name

    # ── dochod_os: specjalna obsluga opoznienia ──────────────
    y_d_all = dp["dochod_os"].values.astype(float)
    _, best_d, rmspe_d, res_d = forecast_variable(y_d_all, years_prov, TRAIN_END, 1)

    if best_d and best_d in res_d:
        actual_d_last = float(dp_tr["dochod_os"].iloc[-1])       # rzecz. dochod 2022
        pred_d2023    = float(res_d[best_d]["pred_test"][0])      # Z4 dochod 2023
        pred_d2024    = float(res_d[best_d]["pred_test"][1])      # Z4 dochod 2024
        # lag dla 2023 = rzecz. 2022, lag dla 2024 = Z4_2023
        x_test_model["ln_dochod_os_lag1"] = np.array([
            np.log(max(actual_d_last, 1e-9)),
            np.log(max(pred_d2023,    1e-9))
        ])
        # lag dla 2025 = Z4_2024
        x_fc_model["ln_dochod_os_lag1"] = np.log(max(pred_d2024, 1e-9))
        x_best["dochod_os"] = best_d
    else:
        # fallback: rzeczywiste wartosci lagowane
        x_test_model["ln_dochod_os_lag1"] = dp_te["ln_dochod_os_lag1"].values.astype(float)
        x_fc_model["ln_dochod_os_lag1"]   = np.nan
        x_best["dochod_os"] = "ACTUAL"

    # ── pozostale zmienne X ──────────────────────────────────
    for cfg in VAR_OTHER:
        col = cfg["col"]
        y_c = dp[col].values.astype(float)
        _, best_c, _, res_c = forecast_variable(y_c, years_prov, TRAIN_END, 1)
        mc = cfg["model_col"]
        if best_c and best_c in res_c:
            pt = np.asarray(res_c[best_c]["pred_test"]).ravel()
            pf = float(res_c[best_c]["pred_fc"][0])
            if cfg["transform"] == "log":
                x_test_model[mc] = np.log(np.clip(pt, 1e-9, None))
                x_fc_model[mc]   = np.log(max(pf, 1e-9))
            else:
                x_test_model[mc] = pt
                x_fc_model[mc]   = pf
            x_best[col] = best_c
        else:
            # fallback: rzeczywiste wartosci
            if mc in dp_te.columns:
                x_test_model[mc] = dp_te[mc].values.astype(float)
            else:
                x_test_model[mc] = np.full(n_te, np.nan)
            x_fc_model[mc] = np.nan
            x_best[col] = "ACTUAL"

    # ── Prognoza warunkowa (X z Z4) ──────────────────────────
    X_test_df = pd.DataFrame({
        "const": np.ones(n_te),
        **{col: x_test_model[col] for col in X_COLS}
    })
    try:
        ln_y_hat_te  = model_tr.predict(X_test_df)
        y_hat_cond   = np.exp(np.asarray(ln_y_hat_te).ravel())
    except:
        y_hat_cond = np.full(n_te, np.nan)

    # ── Benchmark: model z rzeczywistymi X ───────────────────
    dp_te_mod = df_model[
        (df_model["wojewodztwo"] == prov) & (df_model["rok"].isin(TEST_YEARS))
    ].copy()
    if len(dp_te_mod) == n_te:
        X_te_act = pd.DataFrame({
            "const": np.ones(n_te),
            **{col: dp_te_mod[col].values for col in X_COLS}
        })
        try:
            ln_y_hat_act = model_tr.predict(X_te_act)
            y_hat_act = np.exp(np.asarray(ln_y_hat_act).ravel())
        except:
            y_hat_act = np.full(n_te, np.nan)
    else:
        y_hat_act = np.full(n_te, np.nan)

    # ── Naiwna ───────────────────────────────────────────────
    y_naive = np.full(n_te, float(dp_tr["zuzycie_energii_GWh"].iloc[-1]))

    # ── Miary jakosci ─────────────────────────────────────────
    m_cond  = eval_metrics(y_te_actual, y_hat_cond)
    m_act   = eval_metrics(y_te_actual, y_hat_act)
    m_naive = eval_metrics(y_te_actual, y_naive)

    # ── Prognoza ex-ante 2025 (pelny model) ──────────────────
    X_fc_df = pd.DataFrame({
        "const": [1.0],
        **{col: [x_fc_model[col]] for col in X_COLS}
    })
    try:
        ln_y_fc25  = float(model_full.predict(X_fc_df).values[0])
        y_fc25     = np.exp(ln_y_fc25)
        pf_frame   = model_full.get_prediction(X_fc_df).summary_frame(alpha=0.05)
        y_fc25_lo  = np.exp(pf_frame["mean_ci_lower"].values[0])
        y_fc25_hi  = np.exp(pf_frame["mean_ci_upper"].values[0])
    except:
        y_fc25 = y_fc25_lo = y_fc25_hi = np.nan

    prov_results[prov] = {
        "y_actual":  y_te_actual,
        "y_cond":    y_hat_cond,
        "y_act":     y_hat_act,
        "y_naive":   y_naive,
        "y_fc25":    y_fc25,
        "y_fc25_lo": y_fc25_lo,
        "y_fc25_hi": y_fc25_hi,
        "m_cond":    m_cond,
        "m_act":     m_act,
        "m_naive":   m_naive,
        "y_hist":    dp_tr["zuzycie_energii_GWh"].values,
        "years_tr":  dp_tr["rok"].values,
        "x_best":    x_best,
    }

    rmspe_v = m_cond.get("RMSPE%", np.nan)
    ok_str  = "<= 10%" if not np.isnan(rmspe_v) and abs(rmspe_v) <= 10 else "(!)"
    fc_str  = f"{y_fc25:,.0f} GWh" if not np.isnan(y_fc25) else "N/A"
    print(f"  {prov:<22}  RMSPE%={rmspe_v:6.2f}%  {ok_str:<8}  FC{FC_YEAR}={fc_str}")

# ── 6. ZESTAWIENIE MIAR JAKOSCI ───────────────────────────────
print("\n" + "="*65)
print("MIARY JAKOSCI – PROGNOZA WARUNKOWA (test 2023-2024)")
print("="*65)

hdr = (f"  {'Województwo':<22} {'ME':>8} {'MPE%':>7} {'MAE':>8}"
       f" {'MAPE%':>7} {'RMSE':>8} {'RMSPE%':>8} {'Theil':>7}")
print(hdr); print("  " + "-"*82)
for prov in PROVINCES:
    m = prov_results[prov]["m_cond"]
    th_v = m.get("Theil_U", np.nan)
    th   = f"{th_v:.4f}" if isinstance(th_v, float) and not np.isnan(th_v) else "N/A"
    ok   = "<=" if not np.isnan(m.get("RMSPE%", np.nan)) and abs(m.get("RMSPE%", 99)) <= 10 else "(!)"
    print(f"  {prov:<22} {m.get('ME',0):>8.1f} {m.get('MPE%',0):>7.2f}"
          f" {m.get('MAE',0):>8.1f} {m.get('MAPE%',0):>7.2f}"
          f" {m.get('RMSE',0):>8.1f} {m.get('RMSPE%',0):>7.2f}{ok:>4}  {th}")

# Miary agregatu (wszystkie obserwacje laczymy)
all_actual = np.concatenate([prov_results[p]["y_actual"] for p in PROVINCES])
all_cond   = np.concatenate([prov_results[p]["y_cond"]   for p in PROVINCES])
all_naive  = np.concatenate([prov_results[p]["y_naive"]  for p in PROVINCES])
m_agg_cond  = eval_metrics(all_actual, all_cond)
m_agg_naive = eval_metrics(all_actual, all_naive)

print(f"\n  {'AGREGAT (16 woj.)':<22} {m_agg_cond.get('ME',0):>8.1f}"
      f" {m_agg_cond.get('MPE%',0):>7.2f} {m_agg_cond.get('MAE',0):>8.1f}"
      f" {m_agg_cond.get('MAPE%',0):>7.2f} {m_agg_cond.get('RMSE',0):>8.1f}"
      f" {m_agg_cond.get('RMSPE%',0):>7.2f}"
      f"  {'<=' if abs(m_agg_cond.get('RMSPE%',99))<=10 else '(!)'}")

# ── 7. WYKRESY ────────────────────────────────────────────────
# 7a. 4x4 grid: prognoza per województwo
fig, axes = plt.subplots(4, 4, figsize=(22, 18))
fig.suptitle(
    f"Zadanie 5 – Prognoza warunkowa zużycia energii per województwo "
    f"(test {TEST_YEARS[0]}–{TEST_YEARS[-1]})",
    fontsize=13, fontweight="bold"
)
for i, (ax, prov) in enumerate(zip(axes.flat, PROVINCES)):
    res = prov_results[prov]
    ax.plot(res["years_tr"], res["y_hist"], "ko-", lw=1.5, ms=4,
            label="Historia (train)", zorder=5)
    ax.plot(TEST_YEARS, res["y_actual"], "ko-", lw=1.5, ms=4, zorder=5)
    ax.plot(TEST_YEARS, res["y_cond"],   "r^--", lw=2, ms=7, zorder=7,
            label="Prog. warunkowa")
    if not np.any(np.isnan(res["y_act"])):
        ax.plot(TEST_YEARS, res["y_act"], "gs--", lw=1.2, ms=5, alpha=0.7, zorder=6,
                label="Model(actual X)")
    if not np.isnan(res["y_fc25"]):
        ax.plot(FC_YEAR, res["y_fc25"], "r*", ms=10, zorder=8,
                label=f"FC{FC_YEAR}")
    rmspe = res["m_cond"].get("RMSPE%", np.nan)
    ok_str = "OK" if not np.isnan(rmspe) and abs(rmspe) <= 10 else "(!)"
    ax.set_title(f"{prov}\nRMSPE%={rmspe:.1f}% {ok_str}",
                 fontsize=7.5, fontweight="bold", pad=3)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
    ax.tick_params(labelsize=7)
    if i == 0:
        ax.legend(fontsize=6, loc="best", ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ew_z5_01_prognoza_woj.png"), bbox_inches="tight")
plt.show(); plt.close()
print("\nZapisano: ew_z5_01_prognoza_woj.png")

# 7b. RMSPE% per województwo – bar chart
rmspe_vals = [abs(prov_results[p]["m_cond"].get("RMSPE%", np.nan)) for p in PROVINCES]
colors_bar = [GREEN if (not np.isnan(v) and v <= 10) else RED for v in rmspe_vals]

fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(range(N_PROV), rmspe_vals, color=colors_bar, alpha=0.85, edgecolor="white")
ax.axhline(10, color="red", ls="--", lw=2, label="Próg 10%")
ax.set_xticks(range(N_PROV))
ax.set_xticklabels(PROVINCES, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("RMSPE% (prognoza warunkowa)")
ax.set_title(
    f"Jakość prognozy warunkowej per województwo (test {TEST_YEARS[0]}–{TEST_YEARS[-1]})",
    fontsize=12, fontweight="bold"
)
for bar, val in zip(bars, rmspe_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8)
n_ok_bar = sum(1 for v in rmspe_vals if not np.isnan(v) and v <= 10)
ax.set_title(
    f"Jakość prognozy warunkowej per województwo (test {TEST_YEARS[0]}–{TEST_YEARS[-1]})\n"
    f"RMSPE%≤10%: {n_ok_bar}/{N_PROV} województw",
    fontsize=11, fontweight="bold"
)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ew_z5_02_miary_woj.png"), bbox_inches="tight")
plt.show(); plt.close()
print("Zapisano: ew_z5_02_miary_woj.png")

# 7c. Agregat krajowy
sum_hist = df[df["rok"] <= TRAIN_END].groupby("rok")["zuzycie_energii_GWh"].sum()
sum_te_actual = df[df["rok"].isin(TEST_YEARS)].groupby("rok")["zuzycie_energii_GWh"].sum()
sum_te_cond   = {yr: float(np.sum([prov_results[p]["y_cond"][i]
                                   for p in PROVINCES]))
                 for i, yr in enumerate(TEST_YEARS)}
fc25_sum = float(np.nansum([prov_results[p]["y_fc25"] for p in PROVINCES]))
n_fc_ok  = sum(1 for p in PROVINCES if not np.isnan(prov_results[p]["y_fc25"]))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Zadanie 5 – Prognoza warunkowa zużycia energii (agregat krajowy)",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(sum_hist.index, sum_hist.values, "ko-", lw=2, ms=6,
        label="Historia (train)", zorder=5)
ax.plot(sum_te_actual.index, sum_te_actual.values, "ko-", lw=2, ms=6, zorder=5)
ax.plot(TEST_YEARS, [sum_te_cond[yr] for yr in TEST_YEARS], "r^-", lw=2.5, ms=10,
        zorder=7, label="Prog. warunkowa (suma woj.)")
if n_fc_ok == N_PROV:
    ax.plot(FC_YEAR, fc25_sum, "r*", ms=16, zorder=8,
            label=f"FC {FC_YEAR}: {fc25_sum:,.0f} GWh")
ax.axvspan(TEST_YEARS[0]-0.5, TEST_YEARS[-1]+0.5, alpha=0.07, color="red",
           label="Okres testowy")
ax.axvline(FC_YEAR-0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("Rok"); ax.set_ylabel("Zużycie energii [GWh]")
ax.set_title("Suma 16 województw – prognoza warunkowa")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(fontsize=9)

ax2 = axes[1]
mape_vals  = [prov_results[p]["m_cond"].get("MAPE%", np.nan) for p in PROVINCES]
xp = np.arange(N_PROV); w = 0.35
ax2.bar(xp - w/2, rmspe_vals, w,
        color=[GREEN if (not np.isnan(v) and v<=10) else RED for v in rmspe_vals],
        alpha=0.85, edgecolor="white", label="RMSPE%")
ax2.bar(xp + w/2, mape_vals,  w, color=BLUE, alpha=0.5, edgecolor="white", label="MAPE%")
ax2.axhline(10, color="red", ls="--", lw=1.5, label="Próg 10%")
ax2.set_xticks(xp)
ax2.set_xticklabels([p[:7] for p in PROVINCES], rotation=45, ha="right", fontsize=7)
ax2.set_title("RMSPE% i MAPE% per województwo")
ax2.set_ylabel("Błąd [%]")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ew_z5_03_agregat.png"), bbox_inches="tight")
plt.show(); plt.close()
print("Zapisano: ew_z5_03_agregat.png")

# 7d. Tabela miar jakosci – obraz
fig, ax = plt.subplots(figsize=(18, 8))
ax.axis("off")
hdr_row = ["Województwo", "ME [GWh]", "MPE%", "MAE [GWh]", "MAPE%",
           "RMSE [GWh]", "RMSPE%", "Theil U", "Ocena"]
rows = []
for prov in PROVINCES:
    m  = prov_results[prov]["m_cond"]
    th_v = m.get("Theil_U", np.nan)
    th   = f"{th_v:.4f}" if isinstance(th_v, float) and not np.isnan(th_v) else "N/A"
    ok   = "OK (<= 10%)" if not np.isnan(m.get("RMSPE%", np.nan)) and abs(m.get("RMSPE%", 99)) <= 10 else "(!)"
    rows.append([
        prov,
        f"{m.get('ME',0):+,.0f}",
        f"{m.get('MPE%',0):+.2f}%",
        f"{m.get('MAE',0):,.0f}",
        f"{m.get('MAPE%',0):.2f}%",
        f"{m.get('RMSE',0):,.0f}",
        f"{m.get('RMSPE%',0):.2f}%",
        th, ok
    ])

th_agg = (f"{m_agg_cond['Theil_U']:.4f}"
          if isinstance(m_agg_cond.get("Theil_U"), float) and not np.isnan(m_agg_cond.get("Theil_U", np.nan))
          else "N/A")
ok_agg = "OK (<= 10%)" if abs(m_agg_cond.get("RMSPE%", 99)) <= 10 else "(!)"
rows.append([
    "AGREGAT (16 woj.)",
    f"{m_agg_cond['ME']:+,.0f}",   f"{m_agg_cond['MPE%']:+.2f}%",
    f"{m_agg_cond['MAE']:,.0f}",   f"{m_agg_cond['MAPE%']:.2f}%",
    f"{m_agg_cond['RMSE']:,.0f}",  f"{m_agg_cond['RMSPE%']:.2f}%",
    th_agg, ok_agg
])

tbl = ax.table(cellText=rows, colLabels=hdr_row, cellLoc="center",
               loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for j in range(len(hdr_row)):
    tbl[0, j].set_facecolor("#1a5c96")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(rows)+1):
    is_agg = (i == len(rows))
    clr_bg = "#e3eaf7" if is_agg else ("#f0f5ff" if i % 2 == 0 else "white")
    ok_clr = "#e8f5e9" if "OK" in rows[i-1][-1] else "#ffebee"
    for j in range(len(hdr_row)):
        cell_clr = (ok_clr if j == len(hdr_row)-1 else clr_bg)
        tbl[i, j].set_facecolor(cell_clr)
        if is_agg:
            tbl[i, j].set_text_props(fontweight="bold")
ax.set_title(
    "Miary jakości – prognoza warunkowa zużycia energii per województwo (test 2023-2024)",
    fontsize=11, fontweight="bold", pad=8
)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ew_z5_04_tabela_miar.png"), bbox_inches="tight")
plt.show(); plt.close()
print("Zapisano: ew_z5_04_tabela_miar.png")

# ── 8. PODSUMOWANIE ───────────────────────────────────────────
print("\n" + "="*65)
print("PODSUMOWANIE – PROGNOZA WARUNKOWA ENERGIA WOJEWÓDZTWA")
print("="*65)
n_ok = sum(1 for p in PROVINCES
           if not np.isnan(prov_results[p]["m_cond"].get("RMSPE%", np.nan))
           and abs(prov_results[p]["m_cond"].get("RMSPE%", 99)) <= 10)
print(f"  Województw z RMSPE% <= 10%: {n_ok}/{N_PROV}")
print(f"  Miary agregatu (prog. warunkowa, test 2023-2024):")
print(f"    RMSPE% = {m_agg_cond.get('RMSPE%',np.nan):.2f}%"
      f"  ({'OK' if abs(m_agg_cond.get('RMSPE%',99))<=10 else 'przekroczone >10%'})")
print(f"    MAPE%  = {m_agg_cond.get('MAPE%',np.nan):.2f}%")
print(f"    MAE    = {m_agg_cond.get('MAE',np.nan):,.0f} GWh")
print(f"    Theil  = {m_agg_cond.get('Theil_U', 'N/A')}")
print(f"  Prognoza ex-ante {FC_YEAR} (suma {n_fc_ok}/{N_PROV} woj.): {fc25_sum:,.0f} GWh")
print(f"  Pliki: ew_z5_01_prognoza_woj.png, ew_z5_02_miary_woj.png,")
print(f"         ew_z5_03_agregat.png, ew_z5_04_tabela_miar.png")
sys.stdout.flush()
