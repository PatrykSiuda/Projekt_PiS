# -*- coding: utf-8 -*-
# ============================================================
# ZADANIE 4 – PROGNOZY ZMIENNYCH OBJASNIAJACYCH
# ZUZYCIE ENERGII – DANE PANELOWE (16 WOJEWÓDZTW)
# ============================================================
# Proba uczaca  : 2004–2022  (n = 19 per województwo)
# Okres testowy : 2023–2024  (h = 2 per województwo)
# Prognoza ex-ante : 2025    (h = 1 per województwo)
# Zmienne: dochod_os, cena_energii_zl_kWh, urbanizacja_pct,
#          liczba_os, pow_os, hdd, cdd
# Metody: OLS_lin, OLS_kw, AR(1), AR(2), ARIMA, Holt, Pawlowski
# Miary: ME, MPE%, MAE, MAPE%, RMSE, RMSPE%, Theil_U
# ============================================================

import os
import sys

_terminal = len(sys.argv) > 0 and sys.argv[0].endswith(".py")
if _terminal:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib

if _terminal:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.holtwinters import ExponentialSmoothing

try:
    import pmdarima as pm
    PMDARIMA_OK = True
except ImportError:
    PMDARIMA_OK = False
    print("UWAGA: pmdarima niedostepne – ARIMA pominiete.")

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.3, "font.family": "DejaVu Sans",
})
BLUE   = "#1a5c96"; RED    = "#c0392b"; GREEN  = "#27ae60"
ORANGE = "#e67e22"; PURPLE = "#8e44ad"; GRAY   = "#7f8c8d"
PALETTE = plt.cm.tab20.colors[:16]

# ── 1. WCZYTANIE I PRZYGOTOWANIE DANYCH ──────────────────────
df = pd.read_excel(os.path.join(SCRIPT_DIR, "Zuzycie_energii_wojewodztwa.xlsx"))
df = df.sort_values(["wojewodztwo", "rok"]).reset_index(drop=True)

# Imputacja zer w dochod_os (interpolacja liniowa per województwo)
zero_mask = df["dochod_os"] <= 0
if zero_mask.any():
    df.loc[zero_mask, "dochod_os"] = np.nan
    df["dochod_os"] = (
        df.groupby("wojewodztwo")["dochod_os"]
          .transform(lambda s: s.interpolate(method="linear",
                                             limit=3,
                                             limit_direction="both"))
    )

PROVINCES = sorted(df["wojewodztwo"].unique())
YEARS_ALL = sorted(df["rok"].unique())
TRAIN_END = 2022
TEST_START = 2023
FC_YEAR   = 2025

VAR_CONFIG = [
    {"col": "dochod_os",           "label": "Dochod na osobe [zl]",
     "transform": "log", "model_col": "ln_dochod_os", "file": "ew_z4_01_dochod.png"},
    {"col": "cena_energii_zl_kWh", "label": "Cena energii [zl/kWh]",
     "transform": "log", "model_col": "ln_cena",       "file": "ew_z4_02_cena.png"},
    {"col": "urbanizacja_pct",     "label": "Urbanizacja [%]",
     "transform": None,  "model_col": "urbanizacja_pct", "file": "ew_z4_03_urban.png"},
    {"col": "liczba_os",           "label": "Liczba osob w gosp. dom.",
     "transform": None,  "model_col": "liczba_os",     "file": "ew_z4_04_liczba_os.png"},
    {"col": "pow_os",              "label": "Pow. mieszk. na osobe [m2]",
     "transform": None,  "model_col": "pow_os",        "file": "ew_z4_05_pow_os.png"},
    {"col": "hdd",                 "label": "HDD",
     "transform": None,  "model_col": "hdd",           "file": "ew_z4_06_hdd.png"},
    # CDD usuniete z modelu optymalnego: p=0.61 w pooled OLS, usuniecie poprawia AIC i BIC
]

print("=" * 65)
print("ZADANIE 4 – PROGNOZY ZMIENNYCH OBJASNIAJACYCH")
print("Energia elektryczna – 16 województw | 7 zmiennych")
print(f"Proba: 2004–{TRAIN_END}  |  Test: {TEST_START}–2024  |  Prognoza: {FC_YEAR}")
print("=" * 65)

# ── 2. MIARY JAKOSCI (wg PiS_lab_7_1) ────────────────────────
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
    return {
        "ME":      round(me,    4),
        "MPE%":    round(mpe,   4),
        "MAE":     round(mae,   4),
        "MAPE%":   round(mape,  4),
        "RMSE":    round(rmse,  4),
        "RMSPE%":  round(rmspe, 4),
        "Theil_U": round(theil, 4) if not np.isnan(theil) else np.nan,
    }

def _X1(t):
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    return np.column_stack([np.ones(len(t)), t])

def _X2(t):
    t = np.atleast_1d(np.asarray(t, dtype=float)).ravel()
    return np.column_stack([np.ones(len(t)), t, t**2])

# ── 3. FUNKCJA PROGNOZOWANIA (identyczna z polska) ────────────
def forecast_variable(y_all, years_all, train_end, n_exante):
    years_all = np.array(years_all)
    y_all     = np.array(y_all, dtype=float)
    mask_tr   = years_all <= train_end
    mask_te   = years_all >  train_end
    y_tr = y_all[mask_tr]
    y_te = y_all[mask_te]
    n_tr, n_te = len(y_tr), len(y_te)
    t_tr = np.arange(1, n_tr + 1, dtype=float)
    t_te = np.arange(n_tr + 1, n_tr + n_te + 1, dtype=float)
    t_fc = np.arange(n_tr + n_te + 1, n_tr + n_te + n_exante + 1, dtype=float)

    if np.any(~np.isfinite(y_tr)) or n_tr < 5:
        return {}, None, np.nan

    res = {}

    # OLS trend liniowy
    try:
        m = sm.OLS(y_tr, _X1(t_tr)).fit()
        res["OLS_lin"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": np.asarray(m.predict(_X1(t_te))).ravel(),
            "pred_fc":   np.asarray(m.predict(_X1(t_fc))).ravel(),
        }
    except Exception:
        pass

    # OLS trend kwadratowy
    try:
        m = sm.OLS(y_tr, _X2(t_tr)).fit()
        res["OLS_kw"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": np.asarray(m.predict(_X2(t_te))).ravel(),
            "pred_fc":   np.asarray(m.predict(_X2(t_fc))).ravel(),
        }
    except Exception:
        pass

    # AR(1)
    try:
        m = AutoReg(y_tr, lags=1, old_names=False).fit()
        res["AR(1)"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": np.asarray(m.predict(start=n_tr, end=n_tr+n_te-1)).ravel(),
            "pred_fc":   np.asarray(m.predict(start=n_tr+n_te, end=n_tr+n_te+n_exante-1)).ravel(),
        }
    except Exception:
        pass

    # AR(2)
    try:
        m = AutoReg(y_tr, lags=2, old_names=False).fit()
        res["AR(2)"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": np.asarray(m.predict(start=n_tr, end=n_tr+n_te-1)).ravel(),
            "pred_fc":   np.asarray(m.predict(start=n_tr+n_te, end=n_tr+n_te+n_exante-1)).ravel(),
        }
    except Exception:
        pass

    # ARIMA
    if PMDARIMA_OK:
        try:
            m = pm.auto_arima(y_tr, seasonal=False,
                               suppress_warnings=True, stepwise=True)
            pred_all = m.predict(n_periods=n_te + n_exante)
            res["ARIMA"] = {
                "fitted":    m.predict_in_sample(),
                "pred_test": pred_all[:n_te],
                "pred_fc":   pred_all[n_te:],
            }
        except Exception:
            pass

    # Holt
    try:
        m = ExponentialSmoothing(y_tr, trend="add", seasonal=None).fit(optimized=True)
        fc_all = np.asarray(m.forecast(n_te + n_exante)).ravel()
        res["Holt"] = {
            "fitted":    np.asarray(m.fittedvalues).ravel(),
            "pred_test": fc_all[:n_te],
            "pred_fc":   fc_all[n_te:],
        }
    except Exception:
        pass

    # Pawlowski
    try:
        m_trend = sm.OLS(y_tr, _X1(t_tr)).fit()
        e_tr    = np.asarray(m_trend.resid).ravel()
        e_df    = pd.DataFrame({"e_t": e_tr[1:], "e_lag": e_tr[:-1]})
        m_res   = sm.OLS(e_df["e_t"], sm.add_constant(e_df["e_lag"])).fit()
        alpha_e, rho_e = m_res.params["const"], m_res.params["e_lag"]
        e_last  = e_tr[-1]
        e_list  = []
        for _ in range(n_te + n_exante):
            e_next = alpha_e + rho_e * (e_list[-1] if e_list else e_last)
            e_list.append(e_next)
        e_arr = np.array(e_list)
        res["Pawlowski"] = {
            "fitted":    np.asarray(m_trend.fittedvalues).ravel(),
            "pred_test": np.asarray(m_trend.predict(_X1(t_te))).ravel() + e_arr[:n_te],
            "pred_fc":   np.asarray(m_trend.predict(_X1(t_fc))).ravel() + e_arr[n_te:],
        }
    except Exception:
        pass

    if not res:
        return {}, None, np.nan

    metrics = {}
    for mname, mdata in res.items():
        try:
            metrics[mname] = eval_metrics(y_te, mdata["pred_test"])
        except Exception:
            pass

    if not metrics:
        return res, None, np.nan

    metrics_df  = pd.DataFrame(metrics).T
    best_method = metrics_df["RMSPE%"].abs().idxmin()
    best_rmspe  = float(metrics_df.loc[best_method, "RMSPE%"])

    return res, best_method, best_rmspe

# ── 4. OBLICZENIA PER WOJEWÓDZTWO × ZMIENNA ──────────────────
#
# Przechowujemy:
#   all_rmspe[var_col][province]  = best RMSPE%
#   all_method[var_col][province] = najlepsza metoda
#   all_fc[var_col][province]     = prognoza 2025 (raw scale)
#
all_rmspe  = {cfg["col"]: {} for cfg in VAR_CONFIG}
all_method = {cfg["col"]: {} for cfg in VAR_CONFIG}
all_fc     = {cfg["col"]: {} for cfg in VAR_CONFIG}
all_results = {cfg["col"]: {} for cfg in VAR_CONFIG}

years_arr = np.array(YEARS_ALL)

for cfg in VAR_CONFIG:
    col = cfg["col"]
    print(f"\n{'='*65}")
    print(f"ZMIENNA: {cfg['label']}  ({col})")
    print(f"{'='*65}")

    for prov in PROVINCES:
        dp = df[df["wojewodztwo"] == prov].sort_values("rok")
        y_all = dp[col].values.astype(float)

        res, best_method, best_rmspe = forecast_variable(
            y_all, years_arr, TRAIN_END, 1
        )

        all_results[col][prov] = res
        all_method[col][prov]  = best_method if best_method else "brak"
        all_rmspe[col][prov]   = round(abs(best_rmspe), 2) if not np.isnan(best_rmspe) else np.nan
        all_fc[col][prov]      = (float(res[best_method]["pred_fc"][0])
                                  if best_method and best_method in res else np.nan)

        ok_str = "OK" if (not np.isnan(best_rmspe) and abs(best_rmspe) <= 10) else "(!)"
        print(f"  {prov:<25}  najlepsza: {str(best_method):<12}  "
              f"RMSPE% = {best_rmspe:6.2f}%  {ok_str}")

    sys.stdout.flush()

# ── 5. TABELE ZBIORCZE ────────────────────────────────────────
print("\n" + "=" * 65)
print("ZESTAWIENIE ZBIORCZE – RMSPE% NAJLEPSZEJ METODY")
print("=" * 65)
df_rmspe = pd.DataFrame(all_rmspe, index=PROVINCES)
df_meth  = pd.DataFrame(all_method, index=PROVINCES)
df_fc    = pd.DataFrame(all_fc, index=PROVINCES)

col_labels = [c["col"] for c in VAR_CONFIG]
df_rmspe.columns = col_labels
df_meth.columns  = col_labels

print("\n  RMSPE% per województwo × zmienna:")
print(df_rmspe.round(2).to_string())
print("\n  Najlepsza metoda per województwo × zmienna:")
print(df_meth.to_string())

n_ok = (df_rmspe <= 10).sum().sum()
n_total = df_rmspe.notna().sum().sum()
print(f"\n  Zmiennych spełniajacych próg RMSPE<=10%: {n_ok}/{n_total}")

# ── 6. WYKRESY PER ZMIENNA (4×4 siatka, najlepsza metoda) ────
for i, cfg in enumerate(VAR_CONFIG):
    col = cfg["col"]
    fig, axes = plt.subplots(4, 4, figsize=(20, 15), sharex=True)
    fig.suptitle(
        f"Prognoza zmiennej: {cfg['label']} – 16 województw\n"
        f"[Proba: 2004–{TRAIN_END} | Test: {TEST_START}–2024 | Prognoza: {FC_YEAR}]",
        fontsize=13, fontweight="bold", y=1.01
    )

    years_arr_fc  = np.array([FC_YEAR])
    years_te_arr  = years_arr[years_arr > TRAIN_END]

    for j, (ax, prov) in enumerate(zip(axes.flat, PROVINCES)):
        dp      = df[df["wojewodztwo"] == prov].sort_values("rok")
        y_all_p = dp[col].values.astype(float)
        years_p = dp["rok"].values

        ax.plot(years_p, y_all_p, "ko-", lw=1.8, ms=4, zorder=5)

        best_meth = all_method[col][prov]
        res_p     = all_results[col][prov]

        if best_meth and best_meth in res_p:
            bdata = res_p[best_meth]
            c = PALETTE[j]
            ax.plot(years_te_arr, bdata["pred_test"], "^--",
                    color=c, lw=1.5, ms=5, label=f"{best_meth}")
            ax.plot(years_arr_fc, bdata["pred_fc"], "D",
                    color=c, ms=8, zorder=6)

        rmspe_v = all_rmspe[col][prov]
        ok_sym  = "" if (not np.isnan(rmspe_v) and rmspe_v <= 10) else " (!)"
        ax.set_title(
            f"{prov}\n{best_meth}  RMSPE={rmspe_v:.1f}%{ok_sym}",
            fontsize=7.5, fontweight="bold"
        )
        ax.xaxis.set_major_locator(mticker.MultipleLocator(5))
        if j >= 12:
            ax.set_xlabel("Rok", fontsize=7)
        if j % 4 == 0:
            ax.set_ylabel(cfg["label"][:15], fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, cfg["file"]), bbox_inches="tight")
    plt.show(); plt.close()
    print(f"Zapisano: {cfg['file']}")
    sys.stdout.flush()

# ── 7. HEATMAPA RMSPE% ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Podsumowanie prognoz zmiennych objaśniajacych – 16 województw",
             fontsize=13, fontweight="bold")

# Heatmapa RMSPE%
short_cols = [c["col"].replace("cena_energii_zl_kWh", "cena").replace("urbanizacja_pct","urban")
              .replace("liczba_os","l_os").replace("pow_os","pow").replace("dochod_os","dochod")
              for c in VAR_CONFIG]
df_rmspe_plot = df_rmspe.copy()
df_rmspe_plot.columns = short_cols

ax1 = axes[0]
sns.heatmap(
    df_rmspe_plot, annot=True, fmt=".1f", cmap="RdYlGn_r",
    vmin=0, vmax=20, ax=ax1, linewidths=0.5,
    annot_kws={"size": 7}, cbar_kws={"label": "RMSPE%"}
)
ax1.set_title("RMSPE% najlepszej metody (czerwony = >10%)", fontweight="bold")
ax1.set_xlabel("Zmienna"); ax1.set_ylabel("Województwo")
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=7)

# Heatmapa najlepszej metody
method_map = {"OLS_lin": 1, "OLS_kw": 2, "AR(1)": 3, "AR(2)": 4,
              "ARIMA": 5, "Holt": 6, "Pawlowski": 7, "brak": 0}
df_meth_num = df_meth.copy()
df_meth_num.columns = short_cols
df_meth_coded = df_meth_num.apply(lambda col: col.map(
    lambda v: method_map.get(str(v), 0)))

ax2 = axes[1]
cmap_meth = plt.cm.get_cmap("tab10", 8)
sns.heatmap(
    df_meth_coded, annot=df_meth_num.values, fmt="",
    cmap=cmap_meth, vmin=0, vmax=7, ax=ax2,
    linewidths=0.5, annot_kws={"size": 6},
    cbar_kws={"label": "Metoda (numer)"}
)
ax2.set_title("Najlepsza metoda prognozy", fontweight="bold")
ax2.set_xlabel("Zmienna"); ax2.set_ylabel("Województwo")
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=8)
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=7)

# Legenda metod
from matplotlib.patches import Patch
legend_handles = [Patch(color=cmap_meth(v/7), label=f"{v}: {m}")
                  for m, v in method_map.items() if v > 0]
ax2.legend(handles=legend_handles, loc="upper right",
           bbox_to_anchor=(1.55, 1.0), fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ew_z4_07_heatmapy.png"), bbox_inches="tight")
plt.show(); plt.close()
print("Zapisano: ew_z4_07_heatmapy.png")

# ── 8. ZBIORCZY WYKRES RMSPE% PER ZMIENNA ────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
x_pos    = np.arange(len(VAR_CONFIG))
width    = 0.05
offsets  = np.linspace(-len(PROVINCES)/2*width, len(PROVINCES)/2*width, len(PROVINCES))

for j, prov in enumerate(PROVINCES):
    rmspe_vals = [all_rmspe[c["col"]].get(prov, np.nan) for c in VAR_CONFIG]
    ax.bar(x_pos + offsets[j], rmspe_vals, width=width,
           color=PALETTE[j], alpha=0.7, label=prov)

ax.axhline(10, color="red", ls="--", lw=2, label="Próg 10%")
ax.set_xticks(x_pos)
ax.set_xticklabels([c["col"].replace("_", "\n").replace("energii\nzl\nkWh", "[zl/kWh]")
                    for c in VAR_CONFIG], fontsize=8)
ax.set_ylabel("RMSPE%")
ax.set_title("RMSPE% najlepszej metody per zmienna i województwo",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=6, ncol=4, loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(SCRIPT_DIR, "ew_z4_08_rmspe_bar.png"), bbox_inches="tight")
plt.show(); plt.close()
print("Zapisano: ew_z4_08_rmspe_bar.png")

# ── 9. PROGNOZY NA 2025 – TABELA DO MODELU ────────────────────
print("\n" + "=" * 65)
print(f"PROGNOZY ZMIENNYCH NA {FC_YEAR} (RAW SCALE) – DO MODELU WARUNKOWEGO")
print("=" * 65)
df_fc_out = df_fc.copy()
df_fc_out.columns = [c["col"] for c in VAR_CONFIG]
print(df_fc_out.round(4).to_string())

# Postac modelowa (z transformacja log)
print(f"\n  Postac modelowa ({FC_YEAR}):")
for cfg in VAR_CONFIG:
    col = cfg["col"]
    if cfg["transform"] == "log":
        fc_log = df_fc_out[col].apply(lambda v: round(np.log(v), 4)
                                      if pd.notna(v) and v > 0 else np.nan)
        print(f"  {cfg['model_col']:<22} : {fc_log.to_dict()}")
    else:
        print(f"  {cfg['model_col']:<22} : {df_fc_out[col].round(4).to_dict()}")

print("\nPliki wygenerowane: ew_z4_01 .. ew_z4_08_*.png")
