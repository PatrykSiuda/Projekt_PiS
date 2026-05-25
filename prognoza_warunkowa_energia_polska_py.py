# -*- coding: utf-8 -*-
# ============================================================
# ZADANIE 5 – PROGNOZA WARUNKOWA
# ENERGIA ELEKTRYCZNA – POLSKA
# ============================================================
# Model optymalny (Iteracja 2):
#   ln(ZUZYCIE) = b0 + b1*ln(PKB_pc) + b2*ln(CENA) + b3*HDD
#
# Procedura:
#   1. Estymacja OLS na probie uczacej 2004-2022
#   2. Prognoza X (Z4, najlepsza metoda): pkb_pc, cena, hdd
#   3. Podstawienie X_hat do modelu => Y_hat [ln] => exp => [GWh]
#   4. Miary jakosci dla 2023-2024 (Y_hat vs Y_rzeczywiste)
#   5. Prognoza ex-ante 2025 (model na 2004-2024)
#   Porownanie: prog. warunkowa | model(actual X) | naiwna
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
    "figure.dpi": 120, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "font.family": "DejaVu Sans",
})
BLUE="#1a5c96"; RED="#c0392b"; GREEN="#27ae60"; ORANGE="#e67e22"; GRAY="#7f8c8d"

TRAIN_END  = 2022
TEST_YEARS = [2023, 2024]
FC_YEAR    = 2025
MODEL_COLS = ["ln_pkb_pc", "ln_cena", "hdd"]   # Iteracja 2

# ── 1. DANE ───────────────────────────────────────────────────
df = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_polska.xlsx"))
df = df.sort_values("rok").reset_index(drop=True)
df["pkb_per_capita"] = df["pkb_mln_zl"] * 1e6 / df["ludnosc"]
df["ln_pkb_pc"]  = np.log(df["pkb_per_capita"])
df["ln_zuzycie"] = np.log(df["zuzycie_energii_GWh"])
df["ln_cena"]    = np.log(df["cena_energii_zl_kWh"])
YEARS_ALL = df["rok"].values
df_tr = df[df["rok"] <= TRAIN_END].copy()
df_te = df[df["rok"].isin(TEST_YEARS)].copy()

# ── 2. MIARY JAKOSCI ─────────────────────────────────────────
def eval_metrics(actual, pred):
    actual = np.array(actual, dtype=float)
    pred   = np.array(pred,   dtype=float)
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
            "Theil_U": round(theil,4) if not np.isnan(theil) else np.nan}

# ── 3. FUNKCJA PROGNOZOWANIA (Z4) ─────────────────────────────
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
    n_tr, n_te = len(y_tr), len(y_te)
    t_tr = np.arange(1, n_tr+1, dtype=float)
    t_te = np.arange(n_tr+1, n_tr+n_te+1, dtype=float)
    t_fc = np.arange(n_tr+n_te+1, n_tr+n_te+n_exante+1, dtype=float)
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
                        "pred_test": np.asarray(m.predict(start=n_tr, end=n_tr+n_te-1)).ravel(),
                        "pred_fc":   np.asarray(m.predict(start=n_tr+n_te, end=n_tr+n_te+n_exante-1)).ravel()}
    except: pass
    try:
        m = AutoReg(y_tr, lags=2, old_names=False).fit()
        res["AR(2)"] = {"fitted": np.asarray(m.fittedvalues).ravel(),
                        "pred_test": np.asarray(m.predict(start=n_tr, end=n_tr+n_te-1)).ravel(),
                        "pred_fc":   np.asarray(m.predict(start=n_tr+n_te, end=n_tr+n_te+n_exante-1)).ravel()}
    except: pass
    if PMDARIMA_OK:
        try:
            m = pm.auto_arima(y_tr, seasonal=False, suppress_warnings=True, stepwise=True)
            pred_all = m.predict(n_periods=n_te+n_exante)
            res["ARIMA"] = {"fitted": m.predict_in_sample(),
                            "pred_test": pred_all[:n_te], "pred_fc": pred_all[n_te:]}
        except: pass
    try:
        m = ExponentialSmoothing(y_tr, trend="add", seasonal=None).fit(optimized=True)
        fc_all = np.asarray(m.forecast(n_te+n_exante)).ravel()
        res["Holt"] = {"fitted": np.asarray(m.fittedvalues).ravel(),
                       "pred_test": fc_all[:n_te], "pred_fc": fc_all[n_te:]}
    except: pass
    try:
        m_t = sm.OLS(y_tr, _X1(t_tr)).fit()
        e_tr = np.asarray(m_t.resid).ravel()
        e_df = pd.DataFrame({"e_t": e_tr[1:], "e_lag": e_tr[:-1]})
        m_r = sm.OLS(e_df["e_t"], sm.add_constant(e_df["e_lag"])).fit()
        alpha_e, rho_e = m_r.params["const"], m_r.params["e_lag"]
        e_last = e_tr[-1]; e_list = []
        for _ in range(n_te+n_exante):
            e_list.append(alpha_e + rho_e * (e_list[-1] if e_list else e_last))
        e_arr = np.array(e_list)
        res["Pawlowski"] = {"fitted": np.asarray(m_t.fittedvalues).ravel(),
                            "pred_test": np.asarray(m_t.predict(_X1(t_te))).ravel() + e_arr[:n_te],
                            "pred_fc":   np.asarray(m_t.predict(_X1(t_fc))).ravel() + e_arr[n_te:]}
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

# ── 4. ESTYMACJA MODELU OLS ───────────────────────────────────
print("="*65)
print("ZADANIE 5 – PROGNOZA WARUNKOWA  |  ENERGIA POLSKA")
print("Model: ln(ZUZYCIE) = b0 + b1*ln(PKB_pc) + b2*ln(CENA) + b3*HDD")
print(f"Proba: 2004-{TRAIN_END}  |  Test: {TEST_YEARS[0]}-{TEST_YEARS[-1]}  |  FC: {FC_YEAR}")
print("="*65)

y_tr   = df_tr["ln_zuzycie"]
X_tr   = sm.add_constant(df_tr[MODEL_COLS])
model_tr = sm.OLS(y_tr, X_tr).fit()

y_full = df["ln_zuzycie"]
X_full = sm.add_constant(df[MODEL_COLS])
model_full = sm.OLS(y_full, X_full).fit()

print(f"\nModel OLS (proba uczaca 2004-{TRAIN_END}):")
print(f"  R2={model_tr.rsquared:.4f}  R2adj={model_tr.rsquared_adj:.4f}"
      f"  AIC={model_tr.aic:.2f}  BIC={model_tr.bic:.2f}")
for nm, co, pv in zip(model_tr.params.index, model_tr.params, model_tr.pvalues):
    sig = "***" if pv<0.01 else ("**" if pv<0.05 else ("*" if pv<0.1 else ""))
    print(f"  {nm:<20} b={co:+.4f}  p={pv:.4f}  {sig}")

# ── 5. PROGNOZA ZMIENNYCH X (Z4) ─────────────────────────────
print("\n" + "="*65)
print("PROGNOZA ZMIENNYCH OBJASNIAJACYCH (metoda Z4)")
print("="*65)

VAR_CONFIG = [
    {"col": "pkb_per_capita",      "transform": "log",  "model_col": "ln_pkb_pc"},
    {"col": "cena_energii_zl_kWh", "transform": "log",  "model_col": "ln_cena"},
    {"col": "hdd",                  "transform": None,   "model_col": "hdd"},
]

x_model_test = {}   # model_col -> array(n_te) in model form
x_model_fc   = {}   # model_col -> scalar in model form
x_raw_test   = {}   # col -> array(n_te) in raw units (for printing)
x_raw_fc     = {}   # col -> scalar in raw units
x_best_meth  = {}   # col -> best method name

for cfg in VAR_CONFIG:
    col = cfg["col"]
    y_x = df[col].values.astype(float)
    _, best_m, best_rmspe, results = forecast_variable(y_x, YEARS_ALL, TRAIN_END, 1)

    if best_m is None or best_m not in results:
        print(f"  {col:<25}  BLAD: brak prognozy")
        x_model_test[cfg["model_col"]] = np.full(len(TEST_YEARS), np.nan)
        x_model_fc[cfg["model_col"]]   = np.nan
        continue

    pt_raw = np.asarray(results[best_m]["pred_test"]).ravel()
    pf_raw = float(results[best_m]["pred_fc"][0])

    x_raw_test[col] = pt_raw
    x_raw_fc[col]   = pf_raw
    x_best_meth[col] = best_m

    if cfg["transform"] == "log":
        x_model_test[cfg["model_col"]] = np.log(np.clip(pt_raw, 1e-9, None))
        x_model_fc[cfg["model_col"]]   = np.log(max(pf_raw, 1e-9))
    else:
        x_model_test[cfg["model_col"]] = pt_raw
        x_model_fc[cfg["model_col"]]   = pf_raw

    ok = "OK" if abs(best_rmspe) <= 10 else "(!)"
    print(f"  {col:<25}  metoda: {best_m:<12}  RMSPE%={best_rmspe:6.2f}%  {ok}")
    for yr, rv in zip(TEST_YEARS, pt_raw):
        print(f"    prognoza {yr}: {rv:.4f}")
    print(f"    prognoza {FC_YEAR}:  {pf_raw:.4f}")

# ── 6. PROGNOZA WARUNKOWA Y ───────────────────────────────────
print("\n" + "="*65)
print("PROGNOZA WARUNKOWA Y = ZUZYCIE ENERGII [GWh]")
print("="*65)

n_te = len(TEST_YEARS)

# A) Prognoza warunkowa (X z Z4)
X_test_df = pd.DataFrame({
    "const": np.ones(n_te),
    **{col: x_model_test[col] for col in MODEL_COLS}
})
ln_y_hat_te = model_tr.predict(X_test_df)
y_hat_cond_gwh = np.exp(np.asarray(ln_y_hat_te).ravel())

# B) Prognoza modelu z rzeczywistymi X (benchmark)
X_te_actual = sm.add_constant(df_te[MODEL_COLS])
ln_y_hat_act = model_tr.predict(X_te_actual)
y_hat_act_gwh = np.exp(np.asarray(ln_y_hat_act).ravel())

# C) Prognoza naiwna: ostatnia obserwacja probki uczacej
y_naive_gwh = np.full(n_te, float(df_tr["zuzycie_energii_GWh"].iloc[-1]))

# D) Rzeczywiste Y (test)
y_te_actual_gwh = df_te["zuzycie_energii_GWh"].values.astype(float)

# E) Prognoza ex-ante 2025 (pelny model)
X_fc_df = pd.DataFrame({
    "const": [1.0],
    **{col: [x_model_fc[col]] for col in MODEL_COLS}
})
ln_y_fc25 = float(model_full.predict(X_fc_df).values[0])
y_fc25_gwh = np.exp(ln_y_fc25)
pred_frame = model_full.get_prediction(X_fc_df).summary_frame(alpha=0.05)
y_fc25_lo  = np.exp(pred_frame["mean_ci_lower"].values[0])
y_fc25_hi  = np.exp(pred_frame["mean_ci_upper"].values[0])

print(f"\n  Prognoza warunkowa (X z Z4) – test {TEST_YEARS[0]}-{TEST_YEARS[-1]}:")
for yr, yhat, yact in zip(TEST_YEARS, y_hat_cond_gwh, y_te_actual_gwh):
    err = (yhat - yact) / yact * 100
    print(f"    {yr}: prognoza={yhat:,.0f} GWh  |  rzeczywiste={yact:,.0f} GWh  |  blad={err:+.2f}%")

print(f"\n  Prognoza ex-ante {FC_YEAR}: {y_fc25_gwh:,.0f} GWh")
print(f"  95% CI: [{y_fc25_lo:,.0f} – {y_fc25_hi:,.0f}] GWh")
delta = (y_fc25_gwh / float(df["zuzycie_energii_GWh"].iloc[-1]) - 1) * 100
print(f"  Zmiana vs {int(df['rok'].iloc[-1])}: {delta:+.2f}%")

# ── 7. MIARY JAKOSCI ─────────────────────────────────────────
print("\n" + "="*65)
print("MIARY JAKOSCI PROGNOZY WARUNKOWEJ (test 2023-2024)")
print("="*65)

m_cond  = eval_metrics(y_te_actual_gwh, y_hat_cond_gwh)
m_act   = eval_metrics(y_te_actual_gwh, y_hat_act_gwh)
m_naive = eval_metrics(y_te_actual_gwh, y_naive_gwh)

miary_all = [
    ("Prog. warunkowa (X z Z4)", m_cond),
    ("Model z rzecz. X (dolna granica)", m_act),
    ("Naiwna (ostatnia obs.)", m_naive),
]

hdr = f"  {'Metoda':<38} {'ME':>8} {'MPE%':>7} {'MAE':>8} {'MAPE%':>7} {'RMSE':>9} {'RMSPE%':>8} {'Theil':>7}"
print(hdr)
print("  " + "-"*95)
for nm, m in miary_all:
    th = f"{m['Theil_U']:.4f}" if not (isinstance(m['Theil_U'], float) and np.isnan(m['Theil_U'])) else "  N/A"
    ok = " <=" if abs(m['RMSPE%']) <= 10 else " (!)"
    print(f"  {nm:<38} {m['ME']:>8.1f} {m['MPE%']:>7.2f} {m['MAE']:>8.1f}"
          f" {m['MAPE%']:>7.2f} {m['RMSE']:>9.1f} {m['RMSPE%']:>7.2f}{ok} {th}")

# ── 8. WYKRESY ────────────────────────────────────────────────
years_tr = df_tr["rok"].values
years_te = np.array(TEST_YEARS)
y_tr_gwh = df_tr["zuzycie_energii_GWh"].values
y_tr_fit = np.exp(np.asarray(model_tr.fittedvalues).ravel())

# 8a. Wykres prognozy
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Zadanie 5 – Prognoza warunkowa zużycia energii elektrycznej (Polska)",
             fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(years_tr, y_tr_gwh, "ko-", lw=2, ms=6, label="Rzeczywiste (train)", zorder=5)
ax.plot(years_te, y_te_actual_gwh, "ko-", lw=2, ms=6, zorder=5, label="Rzeczywiste (test)")
ax.plot(years_tr, y_tr_fit, "b--", lw=1.5, alpha=0.6, label="Dopasowanie OLS (train)")
ax.plot(years_te, y_hat_cond_gwh, "r^-", lw=2.5, ms=10, zorder=7,
        label="Prog. warunkowa (X z Z4)")
ax.plot(years_te, y_hat_act_gwh,  "gs--", lw=1.8, ms=8, zorder=6,
        label="Model z rzecz. X")
ax.plot(FC_YEAR, y_fc25_gwh, "r*", ms=16, zorder=8,
        label=f"FC {FC_YEAR}: {y_fc25_gwh:,.0f} GWh")
ax.fill_between([FC_YEAR-0.3, FC_YEAR+0.3], [y_fc25_lo]*2, [y_fc25_hi]*2,
                color="red", alpha=0.25, zorder=5)
ax.axvspan(TEST_YEARS[0]-0.5, TEST_YEARS[-1]+0.5, alpha=0.07, color="red",
           label="Okres testowy")
ax.axvline(FC_YEAR-0.5, color="gray", ls=":", lw=1.2)
ax.set_xlabel("Rok"); ax.set_ylabel("Zużycie energii [GWh]")
ax.set_title("Prognoza warunkowa vs wartości rzeczywiste")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.legend(fontsize=8, ncol=2)

# 8b. RMSPE/MAPE porownanie
ax2 = axes[1]
methods_bar = ["Prog. warunkowa\n(X z Z4)", "Model z\nrzeczywnymi X", "Naiwna\n(ostatnia obs.)"]
rmspe_vals  = [abs(m["RMSPE%"]) for _, m in miary_all]
mape_vals   = [m["MAPE%"] for _, m in miary_all]
cols_b = [GREEN if v <= 10 else RED for v in rmspe_vals]
xp = np.arange(len(methods_bar)); w = 0.35
bars1 = ax2.bar(xp - w/2, rmspe_vals, width=w, color=cols_b, alpha=0.85,
                edgecolor="white", label="RMSPE%")
bars2 = ax2.bar(xp + w/2, mape_vals,  width=w, color=BLUE, alpha=0.5,
                edgecolor="white", label="MAPE%")
ax2.axhline(10, color="red", ls="--", lw=2, label="Próg 10%")
ax2.set_xticks(xp); ax2.set_xticklabels(methods_bar, fontsize=9)
ax2.set_title("Miary jakości prognozy (test 2023-2024)")
ax2.set_ylabel("Błąd [%]")
for bar, val in zip(bars1, rmspe_vals):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
             f"{val:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax2.legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ep_z5_01_prognoza_warunkowa.png"), bbox_inches="tight")
plt.show(); plt.close()
print("\nZapisano: ep_z5_01_prognoza_warunkowa.png")

# 8c. Tabela miar jakosci
fig, ax = plt.subplots(figsize=(14, 3.5))
ax.axis("off")
hdr_row = ["Metoda", "ME [GWh]", "MPE%", "MAE [GWh]", "MAPE%",
           "RMSE [GWh]", "RMSPE%", "Theil U", "Ocena"]
rows = []
for nm, m in miary_all:
    th = f"{m['Theil_U']:.4f}" if not (isinstance(m['Theil_U'], float) and np.isnan(m['Theil_U'])) else "N/A"
    ok = "OK (<= 10%)" if abs(m['RMSPE%']) <= 10 else "PRZEKROCZONE"
    rows.append([nm, f"{m['ME']:+,.0f}", f"{m['MPE%']:+.2f}%", f"{m['MAE']:,.0f}",
                 f"{m['MAPE%']:.2f}%", f"{m['RMSE']:,.0f}", f"{m['RMSPE%']:.2f}%", th, ok])

tbl = ax.table(cellText=rows, colLabels=hdr_row, cellLoc="center",
               loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for j in range(len(hdr_row)):
    tbl[0, j].set_facecolor("#1a5c96")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(rows)+1):
    clr = "#e8f5e9" if "OK" in rows[i-1][-1] else "#ffebee"
    for j in range(len(hdr_row)):
        tbl[i, j].set_facecolor(clr if j == len(hdr_row)-1 else
                                 ("#f0f5ff" if i % 2 == 0 else "white"))
ax.set_title("Miary jakości – prognoza warunkowa zużycia energii (test 2023-2024)",
             fontsize=11, fontweight="bold", pad=8)
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ep_z5_02_tabela_miar.png"), bbox_inches="tight")
plt.show(); plt.close()
print("Zapisano: ep_z5_02_tabela_miar.png")

# ── 9. PODSUMOWANIE ───────────────────────────────────────────
print("\n" + "="*65)
print("PODSUMOWANIE – PROGNOZA WARUNKOWA ENERGIA POLSKA")
print("="*65)
print(f"  Model (Iteracja 2): ln(ZUZYCIE) ~ ln(PKB_pc) + ln(CENA) + HDD")
print(f"  Zmienne X prognozowane metoda Z4:")
for col, meth in x_best_meth.items():
    print(f"    {col:<25} -> {meth}")
print(f"  Miary (prog. warunkowa, test 2023-2024):")
print(f"    RMSPE% = {m_cond['RMSPE%']:.2f}%  ({'OK' if abs(m_cond['RMSPE%'])<=10 else 'przekroczone >10%'})")
print(f"    MAPE%  = {m_cond['MAPE%']:.2f}%")
print(f"    MAE    = {m_cond['MAE']:,.0f} GWh")
print(f"    Theil  = {m_cond['Theil_U']}")
print(f"  Prognoza ex-ante {FC_YEAR}: {y_fc25_gwh:,.0f} GWh")
print(f"  95% CI: [{y_fc25_lo:,.0f} – {y_fc25_hi:,.0f}] GWh")
print(f"  Zmiana vs {int(df['rok'].iloc[-1])}: {delta:+.2f}%")
print(f"  Pliki: ep_z5_01_prognoza_warunkowa.png, ep_z5_02_tabela_miar.png")
sys.stdout.flush()
